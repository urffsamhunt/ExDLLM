#!/usr/bin/env python3
"""
Generate text from a DSB checkpoint.

Handles BOTH checkpoint formats that the project produces:

  1. DSBHybrid checkpoints (from scripts/train_dsb_hybrid.py) — keys include
     'hybrid'. Decode uses the discrete tagger/generator heads via iterative
     variable-length edit refinement.

  2. Plain DSB pretrain checkpoints (from scripts/train_dsb_pretrain.py) — keys
     are ['score_net','embedder','config','dim',...], NO discrete heads.
     Here decode is nearest-neighbour: reverse the SDE from the prompt to an
     output embedding, then find the corpus sentence whose embedding is closest.
     Requires --corpus <file> (default: config data.train_path).

Usage:
    # hybrid
    python scripts/generate_dsb_hybrid.py --checkpoint checkpoints_dsb_hybrid/best.pt --prompt "hi"

    # plain pretrain
    python scripts/generate_dsb_hybrid.py --checkpoint checkpoints_dsb_pretrain/best.pt \
        --prompt "the quick brown fox" --corpus data/pretrain_en.txt --k 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from dllm.dsb import DiffSchrodingerBridge, MLPScoreNet, TransformerScoreNet
from dllm.dsb_hybrid import DSBHybrid, EditConditionedScoreNet
from dllm.utils import resolve_device, set_seed


class TextEmbedder(torch.nn.Module):
    """Frozen or trainable RoBERTa embedder producing per-token embeddings."""

    def __init__(self, backbone="roberta-base", max_length=128, trainable=False,
                 gradient_checkpointing=False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        self.max_length = max_length
        self.dim = self.encoder.config.hidden_size
        self.trainable = trainable
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt",
                              return_token_type_ids=False)

    def embed_ids(self, input_ids, attention_mask):
        """Return PER-TOKEN embeddings (B, S, D), not pooled."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, S, D)

    def embed_pool(self, texts):
        """Mean-pooled document embedding (B, D) for nearest-neighbour search."""
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt",
                             return_token_type_ids=False)
        enc = {k: v.to(next(self.encoder.parameters()).device) for k, v in enc.items()}
        out = self.encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled  # (B, D)


def build_hybrid(config, device):
    """Reconstruct embedder + score net + bridge + hybrid from a config dict.

    Returns (embedder, hybrid, tokenizer).
    """
    mcfg = config["model"]
    embedder = TextEmbedder(
        backbone=mcfg["embedder"],
        max_length=mcfg["max_length"],
        trainable=False,
        gradient_checkpointing=False,
    ).to(device)
    embedder.eval()
    tokenizer = embedder.tokenizer  # HF AutoTokenizer

    cond_on = bool(config["dsb"].get("condition_on_dp1", False))
    cond_dim = embedder.dim if cond_on else 0
    score_type = mcfg.get("score_net", "mlp")
    if score_type == "transformer":
        score_net = TransformerScoreNet(
            dim=embedder.dim, hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
            cond_dim=cond_dim, num_heads=mcfg.get("num_heads", 8),
        )
    elif mcfg.get("edit_conditioned_score", False):
        score_net = EditConditionedScoreNet(
            dim=embedder.dim, num_tags=5, hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
            cond_dim=cond_dim,
        )
    else:
        score_net = MLPScoreNet(dim=embedder.dim, hidden_dim=mcfg["hidden_dim"],
                                num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
                                cond_dim=cond_dim)
    bridge = DiffSchrodingerBridge(
        dim=embedder.dim, score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"], beta_max=config["dsb"]["beta_max"],
        condition_on_dp1=bool(config["dsb"].get("condition_on_dp1", False)),
        sigma2_schedule=config["dsb"].get("sigma2_schedule", "ou"),
    ).to(device)
    hybrid = DSBHybrid(
        bridge=bridge, vocab_size=tokenizer.vocab_size,
        lambda_tag=config["training"].get("lambda_tag", 1.0),
        lambda_gen=config["training"].get("lambda_gen", 1.0),
        tag_weights=mcfg.get("tag_weights"),
        condition_heads=mcfg.get("condition_heads", False),
        time_embed_dim=mcfg.get("time_embed_dim", 128),
    ).to(device)
    return embedder, hybrid, tokenizer


def build_bridge(config, embedder, device):
    """Build just bridge + score net (for plain-DSB checkpoints)."""
    mcfg = config["model"]
    cond_on = bool(config["dsb"].get("condition_on_dp1", False))
    score_net = MLPScoreNet(dim=embedder.dim, hidden_dim=mcfg["hidden_dim"],
                            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
                            cond_dim=embedder.dim if cond_on else 0)
    bridge = DiffSchrodingerBridge(
        dim=embedder.dim, score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"], beta_max=config["dsb"]["beta_max"],
        condition_on_dp1=cond_on,
        sigma2_schedule=config["dsb"].get("sigma2_schedule", "ou"),
    ).to(device)
    return bridge


def nearest_texts(embedder, x_pooled, corpus_path, device, k=5, max_rows=200000):
    """Find the k corpus sentences whose pooled embeddings are closest to x_pooled."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()][:max_rows]
    if not lines:
        raise SystemExit(f"No lines in corpus {corpus_path}")
    seen = set()
    unique = [ln for ln in lines if not (ln in seen or seen.add(ln))]
    lines = unique
    print(f"Searching {len(lines)} corpus lines for nearest neighbors...")

    best = []  # list of (dist, text)
    batch = 256
    for i in range(0, len(lines), batch):
        chunk = lines[i:i + batch]
        emb = embedder.embed_pool(chunk).to(device)          # (B, D)
        d = 1.0 - F.cosine_similarity(x_pooled.unsqueeze(0), emb, dim=-1)  # (B,)
        for dt, txt in zip(d.tolist(), chunk):
            best.append((dt, txt))
    best.sort(key=lambda t: t[0])
    return [t for _, t in best[:k]], [dt for dt, _ in best[:k]]


def main():
    parser = argparse.ArgumentParser(description="Generate text from a DSB checkpoint")
    parser.add_argument("--checkpoint", required=True, help="DSB or DSBHybrid checkpoint (.pt)")
    parser.add_argument("--prompt", default="the quick brown fox", help="input text (DP1)")
    parser.add_argument("--sde_steps", type=int, default=100, help="reverse-SDE steps")
    parser.add_argument("--max_iterations", type=int, default=8, help="edit refine iters (hybrid only)")
    parser.add_argument("--max_len", type=int, default=None, help="decode growth cap (hybrid only)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--corpus", default=None, help="corpus file for nearest-neighbor decode (plain DSB)")
    parser.add_argument("--corrupt", action="store_true",
                        help="hybrid only: corrupt the prompt first (the model is a denoiser)")
    parser.add_argument("--k", type=int, default=5, help="nearest neighbors to show (plain DSB)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(resolve_device() if args.device is None else args.device)
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt["config"]
    fmt = "hybrid" if "hybrid" in ckpt else "plain"
    print(f"Checkpoint format: {fmt} (dim={ckpt.get('dim')})")

    embedder_sd = ckpt.get("embedder")
    embedder = TextEmbedder(
        backbone=config["model"]["embedder"],
        max_length=config["model"]["max_length"],
        trainable=False,
        gradient_checkpointing=False,
    ).to(device)
    if embedder_sd is not None:
        try:
            embedder.load_state_dict(embedder_sd)
            print("Loaded trained embedder weights.")
        except Exception as e:
            print(f"(embedder not loaded: {e})")
    embedder.eval()
    tokenizer = embedder.tokenizer

    # 1) Embed prompt -> DP1 (per-token, for the SDE).
    enc = tokenizer([args.prompt], padding=True, truncation=True,
                    max_length=config["model"]["max_length"], return_tensors="pt",
                    return_token_type_ids=False)
    ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = embedder.encoder(input_ids=ids, attention_mask=attn)
        dp1 = out.last_hidden_state.to(device)             # (1, S, D)

    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'=' * 60}\n")

    if fmt == "hybrid":
        embedder2, hybrid, _ = build_hybrid(config, device)
        if embedder_sd is not None:
            embedder2.load_state_dict(embedder_sd)
        hybrid.load_state_dict(ckpt["hybrid"])
        hybrid.eval()

        # The hybrid is a DENOISER (trained corrupted -> clean on the same
        # sentence), so the meaningful run corrupts the prompt first; from a
        # clean prompt it will (correctly) reconstruct the prompt itself.
        canvas_ids = ids[0].tolist()
        if args.corrupt:
            from dllm.dsb_hybrid import corrupt_fixed
            ccfg = config["data"]["corruption"]
            noise_pool = list(range(4, min(ccfg.get("noise_vocab_size", 100) + 4,
                                           tokenizer.vocab_size)))
            corr, _, _ = corrupt_fixed(
                canvas_ids,
                mask_prob=ccfg.get("mask_prob", 0.15),
                mask_ratio=ccfg.get("mask_ratio", 0.8),
                noise_pool=noise_pool,
                mask_id=tokenizer.mask_token_id,
            )
            canvas_ids = corr
            corr_ids = torch.tensor([corr], device=device)
            with torch.no_grad():
                out_c = embedder.encoder(input_ids=corr_ids,
                                         attention_mask=torch.ones_like(corr_ids))
            dp1 = out_c.last_hidden_state
            specials = {tokenizer.bos_token_id, tokenizer.eos_token_id,
                        tokenizer.pad_token_id, tokenizer.mask_token_id}
            shown = tokenizer.decode([t for t in corr if t not in specials]).strip()
            print(f"Corrupted canvas: {shown!r}")

        x = hybrid.bridge.sample(dp1, steps=args.sde_steps)
        with torch.no_grad():
            texts = hybrid.generate_text(
                x, tokenizer, embedder2,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                max_iterations=args.max_iterations, max_len=args.max_len,
                dp1=dp1, seed_ids=[canvas_ids],
            )
        print("Generated Text:")
        for t in texts:
            print(f"  {t}")
    else:
        # Plain DSB: load score net into a bridge, reverse SDE.
        # The plain score net is TRAINED on mean-pooled (B, D) embeddings
        # (train_dsb_pretrain.py embed_ids pools), so DP1 must be pooled too —
        # feeding per-token embeddings would be out-of-distribution.
        bridge = build_bridge(config, embedder, device)
        bridge.score_net.load_state_dict(ckpt["score_net"])
        bridge.eval()
        with torch.no_grad():
            dp1 = embedder.embed_pool([args.prompt])           # (1, D) pooled
        x = bridge.sample(dp1, steps=args.sde_steps)           # (1, D) output embedding

        corpus_path = args.corpus or config["data"].get("train_path")
        if not corpus_path or not os.path.exists(corpus_path):
            raise SystemExit(
                "Plain-DSB decode needs a corpus file (--corpus) to find the nearest "
                "training sentence to the generated embedding. Pass --corpus data/..."
            )
        x_pooled = x[0]                                        # (D,)
        texts, dists = nearest_texts(embedder, x_pooled, corpus_path, device, k=args.k)
        print("Generated embedding  ->  nearest training sentences (cosine dist):")
        for d, t in zip(dists, texts):
            print(f"  [{d:.4f}] {t}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()