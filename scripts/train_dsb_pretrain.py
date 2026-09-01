#!/usr/bin/env python3
"""
Pretrain the Diffusion Schrödinger Bridge (DSB) on raw text for language
understanding.

Self-supervised denoising objective (no input/output pairs needed):
    DP1 = embedding of a *corrupted* (masked/noised) text
    DP2 = embedding of the *clean* text
    train the bridge to transport DP1 -> DP2 via the SDE
        dx_t = beta_t * (DP2 - x_t) dt + sqrt(beta_t) dW_t

Unlike translation fine-tuning (paired input/output), each raw line is its own
target. The encoder is trained jointly with the score network so the model
learns representations that make denoising easy — i.e. language structure.

This is the PLAIN (score-matching-only) variant: it trains no discrete edit
heads, so its checkpoints have no `hybrid` key and decode via nearest-neighbor
corpus retrieval. For the version that ALSO trains the edit/vocab heads (and
saves a `hybrid` key for edit-based text decode), use train_dsb_hybrid.py
instead. Both are unconditional pretraining on raw text.

Usage:
    python scripts/train_dsb_pretrain.py --config configs/dsb_pretrain.yaml \
        --save_dir ./checkpoints_dsb_pretrain
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
from dllm.utils import set_seed, resolve_device


# ── Trainable RoBERTa embedder ───────────────────────────────────────────────

class TextEmbedder(nn.Module):
    """Mean-pool a RoBERTa encoder over token ids. Optionally trainable."""

    def __init__(self, backbone: str = "roberta-base", max_length: int = 128,
                 trainable: bool = True, gradient_checkpointing: bool = False):
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
        return self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

    def embed_ids(self, input_ids, attention_mask):
        """Mean-pool hidden states over real tokens."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled  # (B, D)


# ── Token-level corruption (dynamic, per batch) ──────────────────────────────

def corrupt_ids(input_ids, attention_mask, mask_prob, mask_ratio,
                noise_vocab_size, mask_id):
    """
    Corrupt a batch of token ids: each real token is masked with prob
    mask_prob; of those, mask_ratio become [MASK] and the rest become a random
    noise token from the top-N frequent tokens.
    """
    corrupted = input_ids.clone()
    real = attention_mask.bool()
    rand = torch.rand_like(input_ids.float())
    to_corrupt = real & (rand < mask_prob)
    is_mask = torch.rand_like(input_ids.float()) < mask_ratio
    mask_pos = to_corrupt & is_mask
    noise_pos = to_corrupt & ~is_mask
    corrupted[mask_pos] = mask_id
    noise = torch.randint(0, max(1, noise_vocab_size), input_ids.shape,
                          device=input_ids.device)
    corrupted[noise_pos] = noise[noise_pos]
    return corrupted


# ── Streaming text reader (never loads the whole corpus) ─────────────────────

def iter_lines(path):
    """Yield non-empty stripped lines from a file, one at a time (lazy)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def batch_stream(path, batch_size, buffer_size=100000):
    """
    Yield batches of lines from a file, streaming and with a finite shuffle
    buffer. Never holds more than `buffer_size` lines (plus one batch) in
    memory, so a multi-GB corpus is processed in constant memory.
    """
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


def take_batches(path, batch_size, max_batches):
    """Yield up to `max_batches` batches (no shuffle) for streaming eval."""
    n = 0
    batch = []
    for line in iter_lines(path):
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
            n += 1
            if n >= max_batches:
                return
    if batch and n < max_batches:
        yield batch


# ── Training loop ────────────────────────────────────────────────────────────

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
    mask_id = embedder.tokenizer.mask_token_id

    ccfg = config["data"]["corruption"]
    batch_size = config["training"]["batch_size"]
    train_shuffle_buffer = config["data"].get("shuffle_buffer", 100000)
    use_amp = config["training"].get("mixed_precision", True)

    # Note: NOTHING here loads the whole corpus. `batch_stream` reads the text
    # file lazily and materializes at most `shuffle_buffer` lines at a time; the
    # tokenizer/embedder only ever see one batch. The original crash was caused
    # by `load_lines` (2.9M Python str objects) + `tokenize` (dense (N, S) tensor)
    # holding the entire 775MB / 2.9M-line corpus in memory.

    cond_on_dp1 = bool(config["dsb"].get("condition_on_dp1", False))
    score_net = MLPScoreNet(
        dim=dim,
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        time_embed_dim=config["model"]["time_embed_dim"],
        cond_dim=dim if cond_on_dp1 else 0,
    ).to(device)

    bridge = DiffSchrodingerBridge(
        dim=dim,
        score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"],
        beta_max=config["dsb"]["beta_max"],
        condition_on_dp1=cond_on_dp1,
        sigma2_schedule=config["dsb"].get("sigma2_schedule", "ou"),
    ).to(device)

    # Optimize the score network AND the encoder (if trainable).
    params = list(score_net.parameters())
    if embedder.trainable:
        params += list(embedder.encoder.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_params:,}")

    tcfg = config["training"]
    # How often the expensive diagnostics (baseline/signal/reconstruction) run.
    # The core loss is logged every `log_every`; these default to every 100
    # because reconstruction_error runs a full reverse SDE and is costly.
    diag_every = int(config["training"].get("diag_every", 100))
    recon_steps = int(config["training"].get("recon_steps", 0)) or None  # 0 -> bridge default
    amp_dtype = None
    if use_amp:
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type in ("xpu", "cpu"):
            amp_dtype = torch.bfloat16
    optimizer = torch.optim.AdamW(params, lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])
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

    # ── Resume support ──────────────────────────────────────────────────
    # Restore model, optimizer, scheduler, and step state so an interrupted
    # run (e.g. Kaggle's 12h cap) can continue where it left off.
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        score_net.load_state_dict(ckpt["score_net"])
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
                                  buffer_size=train_shuffle_buffer):
            if global_step >= total:
                break
            enc = embedder.tokenize(batch)              # (B, S) padded — just this batch
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            # Corrupt on the fly (dynamic masking, like the DLLM corruptor).
            noisy_ids = corrupt_ids(
                ids, mask,
                mask_prob=ccfg["mask_prob"],
                mask_ratio=ccfg["mask_ratio"],
                noise_vocab_size=ccfg["noise_vocab_size"],
                mask_id=mask_id,
            )

            optimizer.zero_grad()
            # Mixed precision: compute the forward pass in fp16/bf16 to roughly
            # halve activation memory.
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                dp1 = embedder.embed_ids(noisy_ids, mask)  # corrupted (grad path)
                # Clean target is detached: it is a regression target, not a
                # gradient path. This halves the encoder activations kept for
                # backward (the clean pass needs no grads).
                dp2 = embedder.embed_ids(ids, mask).detach()
                loss = bridge.score_matching_loss(dp1, dp2)
            loss.backward()
            nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % tcfg["log_every"] == 0:
                # Core training loss — logged every `log_every` (cheap).
                print(f"step {global_step}/{total}  loss {loss.item():.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.2e}")

                # Expensive interpretability diagnostics (baseline / signal /
                # full reverse-SDE reconstruction) — run every `diag_every`
                # steps instead of every log step, because reconstruction_error
                # runs a full reverse SDE and is costly.
                if global_step % diag_every == 0:
                    with torch.no_grad():
                        bl = bridge.baseline_loss(dp1)
                        sig = bridge.signal_captured(dp1, dp2)
                        recon = bridge.reconstruction_error(dp1, dp2, steps=recon_steps)
                        # Identity baseline: ||DP2 - DP1||. recon_err at or above
                        # this means the bridge is not transporting at all (it
                        # would do no worse than returning the input unchanged).
                        ident = (dp2 - dp1).norm(dim=-1).mean()
                    print(f"  [diag] baseline {bl.item():.4f}, "
                          f"signal {sig[2]*100:.1f}%, "
                          f"recon_err {recon.item():.4f} "
                          f"(identity {ident.item():.4f})")

            if global_step % tcfg["eval_every"] == 0:
                val_loss = None
                val_path = config["data"].get("val_path")
                if val_path:
                    val_loss_sum = 0.0
                    val_n = 0
                    # Stream the val set in batches so we never embed more than
                    # one batch at a time. Cap how much we evaluate per checkpoint
                    # (like the DLLM config's val_max_batches) so eval stays cheap.
                    val_max = config["data"].get("val_max_batches", 100)
                    for vbatch in take_batches(val_path, batch_size, val_max):
                        venc = embedder.tokenize(vbatch)
                        vids = venc["input_ids"].to(device)
                        vmask = venc["attention_mask"].to(device)
                        vcorr = corrupt_ids(
                            vids, vmask,
                            mask_prob=ccfg["mask_prob"],
                            mask_ratio=ccfg["mask_ratio"],
                            noise_vocab_size=ccfg["noise_vocab_size"],
                            mask_id=mask_id,
                        )
                        with torch.no_grad():
                            vdp1 = embedder.embed_ids(vcorr, vmask)
                            vdp2 = embedder.embed_ids(vids, vmask)
                            val_loss_sum += bridge.score_matching_loss(vdp1, vdp2).item()
                        val_n += 1
                    val_loss = val_loss_sum / max(1, val_n)
                    print(f"  [eval] val loss {val_loss:.4f} (over {val_n} batches)")
                else:
                    val_loss = loss.item()
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "score_net": score_net.state_dict(),
                        "embedder": embedder.state_dict(),
                        "config": config,
                        "dim": dim,
                        "embedder_name": config["model"]["embedder"],
                    }, os.path.join(args.save_dir, "best.pt"))
                    print(f"  [eval] saved best checkpoint (val {val_loss:.4f})")

                # Overwriteable resume checkpoint (for session-limited runs
                # like Kaggle's 12h cap): enables --resume continuation.
                torch.save({
                    "score_net": score_net.state_dict(),
                    "embedder": embedder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "best_val": best_val,
                    "config": config,
                    "dim": dim,
                    "embedder_name": config["model"]["embedder"],
                }, os.path.join(args.save_dir, "resume.pt"))

    torch.save({
        "score_net": score_net.state_dict(),
        "embedder": embedder.state_dict(),
        "config": config,
        "dim": dim,
        "embedder_name": config["model"]["embedder"],
    }, os.path.join(args.save_dir, "final.pt"))
    print("Pretraining complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain the DSB on raw text")
    parser.add_argument("--config", type=str, default="configs/dsb_pretrain.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_dsb_pretrain")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a resume.pt checkpoint to continue from")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    set_seed(args.seed)
    train(args, config)


if __name__ == "__main__":
    main()