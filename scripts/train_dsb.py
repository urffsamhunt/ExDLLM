#!/usr/bin/env python3
"""
Train / sample the Diffusion Schrödinger Bridge (DSB) score-based model.

The DSB transports an *input* datapoint DP1 to an *output* datapoint DP2 via
the SDE
    dx_t = beta_t * (DP2 - x_t) dt + sqrt(beta_t) dW_t
using denoising score matching against the closed-form bridge score.

DP1/DP2 are continuous vectors: each text is mean-pooled through a frozen
RoBERTa encoder. Embeddings are cached to disk so re-runs skip re-encoding.

Usage:
    # Train
    python scripts/train_dsb.py --config configs/dsb.yaml --save_dir ./checkpoints_dsb

    # Sample (generate DP2 from a DP1 text)
    python scripts/train_dsb.py --config configs/dsb.yaml --checkpoint ./checkpoints_dsb/best.pt \
        --input "the quick brown fox" --save_dir ./checkpoints_dsb
"""

import argparse
import os
import sys
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from dllm.dsb import DiffSchrodingerBridge, MLPScoreNet
from dllm.utils import set_seed, resolve_device


# ── Frozen RoBERTa embedder (mean-pooled hidden states) ──────────────────────

class TextEmbedder(nn.Module):
    """Mean-pool the last hidden state of a frozen RoBERTa encoder."""

    def __init__(self, backbone: str = "roberta-base", max_length: int = 128):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder = AutoModel.from_pretrained(backbone)
        self.max_length = max_length
        self.dim = self.encoder.config.hidden_size
        # Freeze the encoder — only the score network is trained.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def embed_texts(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(next(self.encoder.parameters()).device)
        out = self.encoder(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # (B, D)


# ── Paired dataset (input \t output per line) ────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, path: str):
        self.pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    self.pairs.append((parts[0].strip(), parts[1].strip()))
        if not self.pairs:
            raise ValueError(f"No tab-separated input/output pairs found in {path}")
        print(f"Loaded {len(self.pairs)} input/output pairs from {path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


# ── Embedding cache ──────────────────────────────────────────────────────────

def embed_pairs(dataset, embedder, cache_dir, device):
    """Embed all (input, output) pairs, caching to disk keyed by text."""
    os.makedirs(cache_dir, exist_ok=True)
    dp1_list, dp2_list = [], []
    for i, (inp, out) in enumerate(dataset.pairs):
        dp1 = _cached_embed(inp, "in", embedder, cache_dir, device)
        dp2 = _cached_embed(out, "out", embedder, cache_dir, device)
        dp1_list.append(dp1)
        dp2_list.append(dp2)
    return torch.stack(dp1_list), torch.stack(dp2_list)


def _cached_embed(text, tag, embedder, cache_dir, device):
    import hashlib
    key = hashlib.sha1(text.encode("utf-8")).hexdigest()
    path = os.path.join(cache_dir, f"{tag}_{key}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location=device)
    vec = embedder.embed_texts([text])[0].detach().cpu()
    torch.save(vec, path)
    return vec.to(device)


# ── Training loop ────────────────────────────────────────────────────────────

def train(args, config):
    device = resolve_device() if args.device is None else torch.device(args.device)
    print(f"Device: {device}")

    embedder = TextEmbedder(
        backbone=config["model"]["embedder"],
        max_length=config["model"]["max_length"],
    ).to(device)
    dim = embedder.dim

    train_ds = PairDataset(config["data"]["train_path"])
    cache_dir = config["data"]["cache_dir"]
    dp1, dp2 = embed_pairs(train_ds, embedder, cache_dir, device)
    print(f"Embedded DP1/DP2: {dp1.shape} / {dp2.shape}")

    # Optional validation pairs.
    val_dp1 = val_dp2 = None
    if config["data"].get("val_path"):
        val_ds = PairDataset(config["data"]["val_path"])
        val_dp1, val_dp2 = embed_pairs(val_ds, embedder, cache_dir, device)
    has_val = val_dp1 is not None and val_dp2 is not None

    score_net = MLPScoreNet(
        dim=dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        time_embed_dim=config["model"]["time_embed_dim"],
    ).to(device)

    bridge = DiffSchrodingerBridge(
        dim=dim,
        score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"],
        beta_max=config["dsb"]["beta_max"],
    ).to(device)

    n_params = sum(p.numel() for p in score_net.parameters())
    print(f"Score network parameters: {n_params:,}")

    tcfg = config["training"]
    optimizer = torch.optim.AdamW(
        score_net.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    total = tcfg["max_steps"]
    warmup = tcfg["warmup_steps"]
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    global_step = 0

    while global_step < total:
        perm = torch.randperm(dp1.shape[0], device=device)
        for i in range(0, dp1.shape[0], tcfg["batch_size"]):
            if global_step >= total:
                break
            idx = perm[i:i + tcfg["batch_size"]]
            b_dp1, b_dp2 = dp1[idx], dp2[idx]

            optimizer.zero_grad()
            loss = bridge.score_matching_loss(b_dp1, b_dp2)
            loss.backward()
            nn.utils.clip_grad_norm_(score_net.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % tcfg["log_every"] == 0:
                print(f"step {global_step}/{total}  loss {loss.item():.4f}  lr {scheduler.get_last_lr()[0]:.2e}")

            if global_step % tcfg["eval_every"] == 0:
                if has_val:
                    val_loss = bridge.score_matching_loss(val_dp1, val_dp2).item()
                    print(f"  [eval] val loss {val_loss:.4f}")
                else:
                    val_loss = loss.item()
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "score_net": score_net.state_dict(),
                        "config": config,
                        "dim": dim,
                        "embedder": config["model"]["embedder"],
                    }, os.path.join(args.save_dir, "best.pt"))
                    print(f"  [eval] saved best checkpoint (val {val_loss:.4f})")

    torch.save({
        "score_net": score_net.state_dict(),
        "config": config,
        "dim": dim,
        "embedder": config["model"]["embedder"],
    }, os.path.join(args.save_dir, "final.pt"))
    print("Training complete.")


# ── Sampling ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(args, config):
    device = resolve_device() if args.device is None else torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    embedder = TextEmbedder(
        backbone=ckpt["embedder"],
        max_length=config["model"]["max_length"],
    ).to(device)
    dim = ckpt["dim"]

    score_net = MLPScoreNet(
        dim=dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        time_embed_dim=config["model"]["time_embed_dim"],
    ).to(device)
    score_net.load_state_dict(ckpt["score_net"])
    score_net.eval()

    bridge = DiffSchrodingerBridge(
        dim=dim,
        score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"],
        beta_max=config["dsb"]["beta_max"],
    ).to(device)

    dp1 = embedder.embed_texts([args.input]).to(device)
    out = bridge.sample(dp1, steps=config["inference"]["sample_steps"])
    print(f"Generated DP2 embedding: {out.shape}")
    # Nearest-neighbor decode: find the closest cached training output embedding.
    cache_dir = config["data"]["cache_dir"]
    out_files = sorted(os.listdir(cache_dir))
    best_dist, best_key = float("inf"), None
    for fn in out_files:
        if not fn.startswith("out_"):
            continue
        vec = torch.load(os.path.join(cache_dir, fn), map_location=device)
        d = F.mse_loss(out[0], vec).item()
        if d < best_dist:
            best_dist, best_key = d, fn
    print(f"Closest training output embedding: {best_key} (mse {best_dist:.4f})")
    print("(Decoding a continuous embedding back to text requires a decoder or "
          "a nearest-neighbor lookup over a large output corpus.)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train/sample the DSB score-based model")
    parser.add_argument("--config", type=str, default="configs/dsb.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_dsb")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a saved checkpoint (enables sampling mode)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input text (DP1) to generate from in sampling mode")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    set_seed(args.seed)

    if args.checkpoint:
        if not args.input:
            raise SystemExit("Sampling mode requires --input <text>")
        sample(args, config)
    else:
        train(args, config)


if __name__ == "__main__":
    main()