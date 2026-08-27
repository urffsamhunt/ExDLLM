#!/usr/bin/env python3
"""
Train the DSBHybrid — Diffusion Schrödinger Bridge with discrete edit heads
(Phase 1 + Phase 2 full edit-aware SDE).

The model jointly learns:
  * continuous DP1 -> DP2 transport (denoising score matching), and
  * the discrete edit structure (tagger=head edit ops + generator=head tokens).

Two corruption schemes are supported via config ``data.corruption_scheme``:
  * "fixed"  — in-place mask/REPLACE (phase 1; exact per-position alignment).
  * "full"   — full Levenshtein edit grammar via ForwardCorruptor (phase 2;
               INSERT/DELETE/EXPAND), then aligned to the SDE canvas.

Pretraining vs fine-tuning:
    This script performs UNCONDITIONAL PRETRAINING on raw monolingual text
    (each line is corrupted->clean, no input/output pairs), same objective as
    train_dsb_pretrain.py — but it ALSO trains the discrete edit heads. The
    DSBHybrid thus learns the SDE transport AND the edit/token structure, which
    later enables edit-based text decoding. To fine-tune on a paired task
    (e.g. translation), load this checkpoint and continue on paired data in a
    separate script.

Usage:
    python scripts/train_dsb_hybrid.py --config configs/dsb_hybrid_pretrain.yaml
"""

import argparse
import os
import sys
import yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from dllm.dsb import DiffSchrodingerBridge, MLPScoreNet
from dllm.dsb_hybrid import (
    DSBHybrid,
    EditConditionedScoreNet,
    corrupt_fixed,
    corrupt_full,
)
from dllm.utils import set_seed, resolve_device


# ── Trainable RoBERTa embedder ───────────────────────────────────────────────

class TextEmbedder(nn.Module):
    def __init__(self, backbone="roberta-base", max_length=128, trainable=True,
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
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)


# ── Streaming reader (never loads the whole corpus) ─────────────────────────

def iter_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def batch_stream(path, batch_size, buffer_size=100000):
    import random
    buffer = []
    def flush():
        random.shuffle(buffer)
        for i in range(0, len(buffer), batch_size):
            yield buffer[i:i + batch_size]
    for line in iter_lines(path):
        buffer.append(line)
        if len(buffer) >= buffer_size:
            yield from flush()
            buffer.clear()
    if buffer:
        yield from flush()


def batch_clean_ids_from_texts(embedder, texts, pad_id):
    """Tokenize texts -> (clean_ids, attention_mask) on CPU."""
    enc = embedder.tokenize(texts)
    return enc["input_ids"], enc["attention_mask"]


# ── Build (noisy, tag, gen) for a batch of clean ids ────────────────────────

def build_batch_labels(
    clean_ids,           # (B, S) CPU or device
    attention_mask,      # (B, S)
    corruptor,           # ForwardCorruptor (for "full") or None
    scheme,              # "fixed" | "full"
    mask_prob, mask_ratio, noise_pool, mask_id, pad_id, S, device,
):
    """
    Produce noisy_ids, tag_labels, gen_labels (all on `device`, (B, S)).
    For "full", each row is corrupted with the Levenshtein grammar then aligned
    to the S canvas. pad positions are masked so tag/gen losses ignore them.

    Uses the batch's ACTUAL tokenized length (``attention_mask.shape[1]``) for
    the padded tensors, so they line up with ``clean_ids``/``attention_mask``
    (which the tokenizer pads only to the longest sequence in the batch, not to
    the config max_length). ``S`` is only an upper cap on growth.
    """
    B = clean_ids.shape[0]
    S_eff = attention_mask.shape[1]          # batch-wide tokenized length
    cap = min(S, S_eff)                      # don't grow past the canvas/embed width
    noisy_list, tag_list, gen_list = [], [], []
    for b in range(B):
        nz = clean_ids[b].tolist()
        # Drop pad so corruption/alignment operates on real tokens only.
        real_len = int(attention_mask[b].sum().item())
        real = nz[:real_len]
        if scheme == "fixed":
            n, tg, ge = corrupt_fixed(real, mask_prob=mask_prob, mask_ratio=mask_ratio,
                                      noise_pool=noise_pool, mask_id=mask_id)
        else:
            n, tg, ge = corrupt_full(real, corruptor)
        nn_ = torch.tensor(n[:cap] + [pad_id] * max(0, cap - len(n)))
        tg_ = torch.tensor(tg[:cap] + [-100] * max(0, cap - len(tg)))
        ge_ = torch.tensor(ge[:cap] + [-100] * max(0, cap - len(ge)))
        noisy_list.append(nn_)
        tag_list.append(tg_)
        gen_list.append(ge_)
    noisy_ids = torch.stack(noisy_list).to(device)
    tag_labels = torch.stack(tag_list).to(device)
    gen_labels = torch.stack(gen_list).to(device)
    # Overwrite pad slots (beyond real_len) with ignore labels.
    pad_mask = (attention_mask == 0).to(device)[:, :cap]
    tag_labels[pad_mask] = -100
    gen_labels[pad_mask] = -100
    return noisy_ids, tag_labels, gen_labels


def train(args, config):
    device = torch.device(resolve_device() if args.device is None else args.device)
    print(f"Device: {device}")

    embedder = TextEmbedder(
        backbone=config["model"]["embedder"],
        max_length=config["model"]["max_length"],
        trainable=config["model"].get("train_embedder", True),
        gradient_checkpointing=config["model"].get("gradient_checkpointing", True),
    ).to(device)
    dim = embedder.dim
    tokenizer = embedder.tokenizer
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    # ForwardCorruptor only needed for the full-edit scheme.
    corruptor = None
    scheme = config["data"].get("corruption_scheme", "fixed")
    if scheme == "full":
        from dllm.corruptor import ForwardCorruptor
        ccfg = config["data"]["corruption"]
        corruptor = ForwardCorruptor(
            tokenizer=tokenizer,
            replace_ratio=ccfg.get("replace_ratio", 0.15),
            delete_ratio=ccfg.get("delete_ratio", 0.05),
            insert_ratio=ccfg.get("insert_ratio", 0.05),
            expand_ratio=ccfg.get("expand_ratio", 0.05),
            mask_ratio=ccfg.get("mask_ratio", 0.80),
            noise_vocab_size=ccfg.get("noise_vocab_size", 100),
            expand_prob=ccfg.get("expand_prob", 0.10),
            insert_prob=ccfg.get("insert_prob", 0.15),
            t_skew=ccfg.get("t_skew", 2.0),
            shortage_prob=ccfg.get("shortage_prob", 0.35),
        )

    mcfg = config["model"]
    mcf = config["data"]["corruption"]
    S = config["model"]["max_length"]
    batch_size = config["training"]["batch_size"]
    mask_prob = mcf.get("mask_prob", 0.15)
    mask_ratio = mcf.get("mask_ratio", 0.8)
    noise_pool = list(range(4, min(mcf.get("noise_vocab_size", 100) + 4,
                                   tokenizer.vocab_size)))

    # Score net: MLP by default, edit-conditioned if config asks.
    if mcfg.get("edit_conditioned_score", False):
        score_net = EditConditionedScoreNet(
            dim=dim, num_tags=5, hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
        )
    else:
        score_net = MLPScoreNet(dim=dim, hidden_dim=mcfg["hidden_dim"],
                                num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"])
    bridge = DiffSchrodingerBridge(
        dim=dim, score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"], beta_max=config["dsb"]["beta_max"],
    ).to(device)
    hybrid = DSBHybrid(
        bridge=bridge, vocab_size=tokenizer.vocab_size,
        lambda_tag=config["training"].get("lambda_tag", 1.0),
        lambda_gen=config["training"].get("lambda_gen", 1.0),
        tag_weights=config["model"].get("tag_weights"),
    ).to(device)

    params = []
    for p in score_net.parameters():
        params.append(p)
    for p in hybrid.tagger.parameters():
        params.append(p)
    for p in hybrid.generator.parameters():
        params.append(p)
    if embedder.trainable:
        for p in embedder.encoder.parameters():
            params.append(p)
    n_params = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_params:,}")

    tcfg = config["training"]
    # How often the expensive interpretability diagnostics (baseline/signal/
    # reconstruction) run — every `diag_every` steps, not every log step,
    # because reconstruction_error runs a full reverse SDE and is costly.
    diag_every = int(tcfg.get("diag_every", 100))
    recon_steps = int(tcfg.get("recon_steps", 0)) or None  # 0 -> bridge default
    amp_dtype = None
    if tcfg.get("mixed_precision", True):
        if torch.cuda.is_available() or device.type == "cuda":
            amp_dtype = torch.float16
        elif device.type in ("xpu", "cpu"):
            amp_dtype = torch.bfloat16
    total = tcfg["max_steps"]
    warmup = tcfg["warmup_steps"]
    optimizer = torch.optim.AdamW(params, lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    global_step = 0

    # ── Resume support ──────────────────────────────────────────────────
    # Restore model, optimizer, scheduler, and step state so an interrupted
    # run (e.g. Kaggle's 12h cap) can continue where it left off.
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        hybrid.load_state_dict(ckpt["hybrid"])
        embedder.load_state_dict(ckpt["embedder"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        print(f"Resumed from {args.resume} (step {global_step}, best_val {best_val:.4f})")

    while global_step < total:
        for batch in batch_stream(config["data"]["train_path"], batch_size,
                                  buffer_size=config["data"].get("shuffle_buffer", 100000)):
            if global_step >= total:
                break
            clean_ids_cpu, attn_cpu = batch_clean_ids_from_texts(embedder, batch,
                                                                 tokenizer.pad_token_id)
            clean_ids = clean_ids_cpu.to(device)
            attn = attn_cpu.to(device)
            noisy_ids, tag_labels, gen_labels = build_batch_labels(
                clean_ids, attn, corruptor, scheme, mask_prob, mask_ratio,
                noise_pool, mask_id, pad_id, S, device,
            )

            # Embed the noisy (DP1) and clean (DP2) canvases.
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                dp1 = embedder.embed_ids(noisy_ids, attn)
                dp2 = embedder.embed_ids(clean_ids, attn).detach()
                # Condition the score on ground-truth tags when using the
                # edit-conditioned score net (phase 2).
                cond = tag_labels.clamp(0, 5) if mcfg.get("edit_conditioned_score", False) else None
                if scheme == "full":
                    loss, loss_dict = hybrid.loss_edit(dp1, dp2, tag_labels, gen_labels,
                                                       condition_tags=cond)
                else:
                    loss, loss_dict = hybrid.loss(dp1, dp2, clean_ids, noisy_ids, attn,
                                                  t=torch.rand(dp1.shape[0], device=device))
            loss.backward()
            nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % tcfg["log_every"] == 0:
                print(f"step {global_step}/{total}  total {loss.item():.4f}  "
                      f"[sm {loss_dict['score_matching']:.3f} tag {loss_dict['tag']:.3f} "
                      f"gen {loss_dict['gen']:.3f}]  lr {scheduler.get_last_lr()[0]:.2e}")

                # Expensive interpretability diagnostics (baseline / signal /
                # full reverse-SDE reconstruction) — run every `diag_every`
                # steps. Uses the hybrid's bridge (same SDE machinery).
                if global_step % diag_every == 0:
                    with torch.no_grad():
                        bl = hybrid.bridge.baseline_loss(dp1)
                        sig = hybrid.bridge.signal_captured(dp1, dp2)
                        recon = hybrid.bridge.reconstruction_error(dp1, dp2, steps=recon_steps)
                    print(f"  [diag] baseline {bl.item():.4f}, "
                          f"signal {sig[2]*100:.1f}%, "
                          f"recon_err {recon.item():.4f}")

            if global_step % tcfg["eval_every"] == 0:
                val_loss = loss.item()
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "hybrid": hybrid.state_dict(),
                        "embedder": embedder.state_dict(),
                        "config": config,
                        "dim": dim,
                        "embedder_name": mcfg["embedder"],
                    }, os.path.join(args.save_dir, "best.pt"))
                    print(f"  [eval] saved best (total {val_loss:.4f})")

                # Overwriteable resume checkpoint (for session-limited runs
                # like Kaggle's 12h cap): enables --resume continuation.
                torch.save({
                    "hybrid": hybrid.state_dict(),
                    "embedder": embedder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "best_val": best_val,
                    "config": config,
                    "dim": dim,
                    "embedder_name": mcfg["embedder"],
                }, os.path.join(args.save_dir, "resume.pt"))

    torch.save({
        "hybrid": hybrid.state_dict(),
        "embedder": embedder.state_dict(),
        "config": config,
        "dim": dim,
        "embedder_name": mcfg["embedder"],
    }, os.path.join(args.save_dir, "final.pt"))
    print("Pretraining complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DSBHybrid")
    parser.add_argument("--config", default="configs/dsb_hybrid_pretrain.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_dir", default="./checkpoints_dsb_hybrid")
    parser.add_argument("--resume", default=None,
                        help="Path to a resume.pt checkpoint to continue from")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    set_seed(args.seed)
    train(args, config)


if __name__ == "__main__":
    main()