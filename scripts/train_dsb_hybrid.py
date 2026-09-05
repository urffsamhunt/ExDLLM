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

from dllm.dsb import DiffSchrodingerBridge, MLPScoreNet, TransformerScoreNet
from dllm.dsb_hybrid import (
    DSBHybrid,
    EditConditionedScoreNet,
    corrupt_fixed,
    corrupt_full,
    KEEP,
    DELETE,
    REPLACE,
    INSERT,
    EXPAND,
)
from dllm.utils import set_seed, resolve_device


# ── Trainable RoBERTa embedder ───────────────────────────────────────────────

class TextEmbedder(nn.Module):
    def __init__(self, backbone="roberta-base", max_length=128, trainable=True,
                 gradient_checkpointing=False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model = AutoModelForMaskedLM.from_pretrained(backbone)
        self.encoder = getattr(self.model, self.model.base_model_prefix, self.model)
        self.lm_head = getattr(self.model, "lm_head", getattr(self.model, "cls", None))
        self.max_length = max_length
        self.dim = self.model.config.hidden_size
        self.trainable = trainable
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt",
                              return_token_type_ids=False)

    def embed_ids(self, input_ids, attention_mask):
        # Return PER-TOKEN embeddings (B, S, D), not mean-pooled. The DSBHybrid
        # discrete tagger/generator heads operate on per-position structure, so
        # pooling to (B, D) would make x_t 2-D and silently skip the edit heads
        # (see DSBHybrid.loss: it returns score-matching only when x_t.dim()!=3).
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, S, D)

    def decode_logits(self, hidden_states):
        """Decode Layer 12 contextual hidden states (..., D) -> (..., V) vocabulary logits."""
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        elif hasattr(self.model, "get_output_embeddings"):
            return self.model.get_output_embeddings()(hidden_states)
        else:
            raise AttributeError("Model has no lm_head")


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
    """Tokenize texts (monolingual or TSV pairs) -> (clean_ids, attention_mask, prompt_lens) on CPU."""
    tokenizer = embedder.tokenizer
    has_pairs = any("\t" in t for t in texts)

    if not has_pairs:
        enc = embedder.tokenize(texts)
        return enc["input_ids"], enc["attention_mask"], [0] * len(texts)

    clean_ids_list = []
    prompt_lens = []
    max_len = 0
    for text in texts:
        if "\t" in text:
            src, tgt = text.split("\t", 1)
        else:
            src, tgt = text, ""

        src_ids = tokenizer.encode(src.strip(), add_special_tokens=True)
        if tgt.strip():
            tgt_ids = tokenizer.encode(tgt.strip(), add_special_tokens=False) + [tokenizer.eos_token_id]
        else:
            tgt_ids = []

        full = (src_ids + tgt_ids)[:embedder.max_length]
        clean_ids_list.append(full)
        prompt_lens.append(min(len(src_ids), len(full)))
        max_len = max(max_len, len(full))

    batch_size = len(texts)
    clean_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for b in range(batch_size):
        row = clean_ids_list[b]
        clean_ids[b, :len(row)] = torch.tensor(row, dtype=torch.long)
        attention_mask[b, :len(row)] = 1

    return clean_ids, attention_mask, prompt_lens


# ── Build (noisy, tag, gen) for a batch of clean ids ────────────────────────

def build_batch_labels(
    clean_ids,           # (B, S) CPU or device
    attention_mask,      # (B, S)
    prompt_lens,         # list of int: length of prompt region per row (0 if monolingual)
    corruptor,           # ForwardCorruptor (for "full") or None
    scheme,              # "fixed" | "full"
    mask_prob, mask_ratio, noise_pool, mask_id, pad_id, S, device,
    stutter_prob: float = 0.0,
):
    """
    Produce noisy_ids, tag_labels, gen_labels (all on `device`, (B, S)).
    If prompt_lens is provided, prompt tokens are protected with tag=KEEP and gen=-100.
    For "full", each row is corrupted with the Levenshtein grammar then aligned
    to the S canvas. pad positions are masked so tag/gen losses ignore them.
    """
    B = clean_ids.shape[0]
    S_eff = attention_mask.shape[1]          # batch-wide tokenized length
    cap = min(S, S_eff)                      # don't grow past the canvas/embed width
    noisy_list, tag_list, gen_list = [], [], []
    for b in range(B):
        nz = clean_ids[b].tolist()
        real_len = int(attention_mask[b].sum().item())
        real = nz[:real_len]
        p_len = prompt_lens[b] if prompt_lens is not None else 0

        if p_len > 0 and p_len < real_len:
            # Paired sample: Prompt section is protected
            prompt_part = real[:p_len]
            resp_part = real[p_len:]

            if scheme == "fixed":
                n_resp, tg_resp, ge_resp = corrupt_fixed(resp_part, mask_prob=mask_prob, mask_ratio=mask_ratio,
                                                         noise_pool=noise_pool, mask_id=mask_id,
                                                         stutter_prob=stutter_prob)
            else:
                n_resp, tg_resp, ge_resp = corrupt_full(resp_part, corruptor)

            n = prompt_part + n_resp
            tg = [KEEP] * len(prompt_part) + tg_resp
            ge = [-100] * len(prompt_part) + ge_resp
        else:
            # Monolingual sample: Entire sequence is corrupted
            if scheme == "fixed":
                n, tg, ge = corrupt_fixed(real, mask_prob=mask_prob, mask_ratio=mask_ratio,
                                          noise_pool=noise_pool, mask_id=mask_id,
                                          stutter_prob=stutter_prob)
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


@torch.no_grad()
def evaluate(
    hybrid, embedder, tokenizer, corruptor, scheme,
    val_batches, mask_prob, mask_ratio, noise_pool, mask_id, pad_id, S,
    device, amp_dtype, mcfg,
):
    hybrid.eval()
    embedder.eval()
    val_loss_sum = 0.0
    sm_sum = 0.0
    tag_sum = 0.0
    gen_sum = 0.0
    count = 0

    for batch in val_batches:
        clean_ids_cpu, attn_cpu, prompt_lens = batch_clean_ids_from_texts(
            embedder, batch, tokenizer.pad_token_id
        )
        clean_ids = clean_ids_cpu.to(device)
        attn = attn_cpu.to(device)
        noisy_ids, tag_labels, gen_labels = build_batch_labels(
            clean_ids, attn, prompt_lens, corruptor, scheme, mask_prob, mask_ratio,
            noise_pool, mask_id, pad_id, S, device,
        )
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            dp1 = embedder.embed_ids(noisy_ids, attn)
            dp2 = embedder.embed_ids(clean_ids, attn).detach()
            cond = tag_labels.clamp(0, 5) if mcfg.get("edit_conditioned_score", False) else None
            if scheme == "full":
                loss, loss_dict = hybrid.loss_edit(dp1, dp2, tag_labels, gen_labels,
                                                   condition_tags=cond,
                                                   attention_mask=attn,
                                                   expose_ratio=0.0)
            else:
                loss, loss_dict = hybrid.loss(dp1, dp2, clean_ids, noisy_ids, attn,
                                              t=torch.rand(dp1.shape[0], device=device),
                                              expose_ratio=0.0)
        val_loss_sum += loss.item()
        sm_sum += loss_dict.get("score_matching", 0.0)
        tag_sum += loss_dict.get("tag", 0.0)
        gen_sum += loss_dict.get("gen", 0.0)
        rout_sum = rout_sum + loss_dict.get("router", 0.0) if "rout_sum" in locals() else loss_dict.get("router", 0.0)
        count += 1

    hybrid.train()
    if embedder.trainable:
        embedder.train()

    if count == 0:
        return None
    return {
        "total": val_loss_sum / count,
        "score_matching": sm_sum / count,
        "tag": tag_sum / count,
        "router": rout_sum / count if "rout_sum" in locals() else 0.0,
        "gen": gen_sum / count,
    }


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
    stutter_prob = mcf.get("stutter_prob", 0.0)
    noise_pool = list(range(100, min(mcf.get("noise_vocab_size", 30000) + 100,
                                   tokenizer.vocab_size)))

    cond_on_dp1 = bool(config["dsb"].get("condition_on_dp1", False))
    cond_dim = dim if cond_on_dp1 else 0
    score_type = mcfg.get("score_net", "mlp")
    if score_type == "transformer":
        gated_drift = bool(config.get("dsb", {}).get("gated_drift", False) or mcfg.get("gated_drift", False))
        score_net = TransformerScoreNet(
            dim=dim, hidden_dim=mcfg["hidden_dim"], num_layers=mcfg["num_layers"],
            time_embed_dim=mcfg["time_embed_dim"], cond_dim=cond_dim,
            num_heads=mcfg.get("num_heads", 8),
            gated_drift=gated_drift,
        )
    elif mcfg.get("edit_conditioned_score", False):
        score_net = EditConditionedScoreNet(
            dim=dim, num_tags=5, hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
            cond_dim=cond_dim,
        )
    else:
        score_net = MLPScoreNet(dim=dim, hidden_dim=mcfg["hidden_dim"],
                                num_layers=mcfg["num_layers"], time_embed_dim=mcfg["time_embed_dim"],
                                cond_dim=cond_dim)
    bridge = DiffSchrodingerBridge(
        dim=dim, score_net=score_net,
        beta_schedule=config["dsb"]["beta_schedule"],
        num_steps=config["dsb"]["num_steps"],
        beta_min=config["dsb"]["beta_min"], beta_max=config["dsb"]["beta_max"],
        condition_on_dp1=cond_on_dp1,
        sigma2_schedule=config["dsb"].get("sigma2_schedule", "ou"),
        prediction_target=config["dsb"].get("prediction_target", "u"),
    ).to(device)
    embed_weight = embedder.encoder.get_input_embeddings().weight
    lm_head = getattr(embedder, "lm_head", None)
    hybrid = DSBHybrid(
        bridge=bridge, vocab_size=tokenizer.vocab_size,
        lambda_sm=config["training"].get("lambda_sm", 20.0),
        lambda_tag=config["training"].get("lambda_tag", 1.0),
        lambda_gen=config["training"].get("lambda_gen", 1.0),
        tag_weights=mcfg.get("tag_weights"),
        condition_heads=mcfg.get("condition_heads", False),
        time_embed_dim=mcfg.get("time_embed_dim", 128),
        embed_weight=embed_weight,
        tie_weights=mcfg.get("tie_weights", True),
        lm_head=lm_head,
    ).to(device)

    score_params = [p for p in score_net.parameters() if p.requires_grad]
    if embedder.trainable:
        score_params += [p for p in embedder.parameters() if p.requires_grad]
    head_params = [p for p in list(hybrid.tagger.parameters()) + list(hybrid.generator.parameters()) if p.requires_grad]
    params = score_params + head_params
    n_params = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_params:,} (score: {sum(p.numel() for p in score_params):,}, heads: {sum(p.numel() for p in head_params):,})")

    tcfg = config["training"]
    # How often the expensive interpretability diagnostics (baseline/signal/
    # reconstruction) run — every `diag_every` steps, not every log step,
    # because reconstruction_error runs a full reverse SDE and is costly.
    diag_every = int(tcfg.get("diag_every", 100))
    recon_steps = int(tcfg.get("recon_steps", 0)) or None  # 0 -> bridge default
    amp_dtype = None
    if tcfg.get("mixed_precision", True):
        if device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
        sd = ckpt["hybrid"]
        if mcfg.get("tie_weights", True) and "generator.net.3.weight" in sd:
            ckpt_tie = ckpt.get("config", {}).get("model", {}).get("tie_weights", False)
            if not ckpt_tie:
                print("  [resume] Checkpoint had untied generator weights; preserving tied pretrained word embeddings.")
                sd = {k: v for k, v in sd.items() if k != "generator.net.3.weight"}
        hybrid.load_state_dict(sd, strict=False)
        if "embedder" in ckpt:
            try:
                embedder.load_state_dict(ckpt["embedder"])
            except Exception:
                pass
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                global_step = int(ckpt.get("global_step", 0))
                best_val = float(ckpt.get("best_val", float("inf")))
                print(f"Resumed optimizer & scheduler from {args.resume} (step {global_step}, best_val {best_val:.4f})")
            except Exception as e:
                print(f"  [resume] Optimizer state skipped ({e}) — starting fresh optimizer & LR warmup for fine-tuning.")
                global_step = 0
                best_val = float("inf")
        else:
            global_step = 0
            best_val = float("inf")
            print(f"Resumed model weights from {args.resume} (starting fresh fine-tuning at step 0)")

    # ── Prepare Validation Batches ──────────────────────────────────────
    val_path = config["data"].get("val_path")
    val_batches = []
    val_max_batches = int(config["data"].get("val_max_batches", 50))
    if val_path and os.path.exists(val_path):
        for b in batch_stream(val_path, batch_size):
            val_batches.append(b)
            if len(val_batches) >= val_max_batches:
                break
        print(f"Loaded {len(val_batches)} validation batches from {val_path}")
    else:
        train_file = config["data"]["train_path"]
        if os.path.exists(train_file):
            held_lines = []
            with open(train_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        held_lines.append(line)
                        if len(held_lines) >= batch_size * val_max_batches:
                            break
            if held_lines:
                val_batches = [held_lines[i:i + batch_size] for i in range(0, len(held_lines), batch_size)]
                print(f"Held out {len(val_batches)} fixed validation batches from {train_file}")

    while global_step < total:
        for batch in batch_stream(config["data"]["train_path"], batch_size,
                                  buffer_size=config["data"].get("shuffle_buffer", 100000)):
            if global_step >= total:
                break
            clean_ids_cpu, attn_cpu, prompt_lens = batch_clean_ids_from_texts(embedder, batch,
                                                                               tokenizer.pad_token_id)
            clean_ids = clean_ids_cpu.to(device)
            attn = attn_cpu.to(device)
            noisy_ids, tag_labels, gen_labels = build_batch_labels(
                clean_ids, attn, prompt_lens, corruptor, scheme, mask_prob, mask_ratio,
                noise_pool, mask_id, pad_id, S, device, stutter_prob=stutter_prob,
            )

            optimizer.zero_grad()

            # Embed the noisy (DP1) and clean (DP2) canvases.
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                dp1 = embedder.embed_ids(noisy_ids, attn)
                dp2 = embedder.embed_ids(clean_ids, attn).detach()
                # Condition the score on ground-truth tags when using the
                # edit-conditioned score net (phase 2).
                cond = tag_labels.clamp(0, 5) if mcfg.get("edit_conditioned_score", False) else None
                # Scheduled sampling: anneal expose_ratio from 0 → max_expose
                # over training so GenHead gradually learns to decode from the
                # SDE's own imperfect reconstructions (closing the train-test gap).
                max_expose = tcfg.get("max_expose_ratio", 0.5)
                expose_warmup = tcfg.get("expose_warmup_steps", total // 4)
                expose_ratio = min(max_expose, max_expose * global_step / max(1, expose_warmup))
                if scheme == "full":
                    loss, loss_dict = hybrid.loss_edit(dp1, dp2, tag_labels, gen_labels,
                                                       condition_tags=cond,
                                                       attention_mask=attn,
                                                       expose_ratio=expose_ratio)
                else:
                    loss, loss_dict = hybrid.loss(dp1, dp2, clean_ids, noisy_ids, attn,
                                                  t=torch.rand(dp1.shape[0], device=device),
                                                  expose_ratio=expose_ratio)
            loss.backward()
            nn.utils.clip_grad_norm_(score_params, tcfg["grad_clip"])
            if head_params:
                nn.utils.clip_grad_norm_(head_params, tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % tcfg["log_every"] == 0:
                rout_str = f" rout {loss_dict['router']:.3f}" if loss_dict.get("router", 0.0) > 0 else ""
                print(f"step {global_step}/{total}  total {loss.item():.4f}  "
                      f"[sm {loss_dict['score_matching']:.3f}{rout_str} tag {loss_dict['tag']:.3f} "
                      f"gen {loss_dict['gen']:.3f}]  lr {scheduler.get_last_lr()[0]:.2e}")

                # Expensive interpretability diagnostics (baseline / signal /
                # full reverse-SDE reconstruction + discrete head accuracy) — run every `diag_every`
                # steps. Uses the hybrid's compute_diagnostics.
                if global_step % diag_every == 0:
                    with torch.no_grad():
                        diag = hybrid.compute_diagnostics(
                            dp1, dp2, tag_labels, gen_labels,
                            recon_steps=recon_steps,
                            embed_weight=embed_weight,
                            lm_head_fn=embedder.decode_logits if hasattr(embedder, "decode_logits") else None,
                            attention_mask=attn,
                        )
                    print(f"  [diag] SDE: signal {diag['signal']:.1f}%, cos_sim {diag['cos_sim']:.3f}, "
                          f"recon_err {diag['recon_err']:.4f} (ident {diag['identity']:.4f}) | "
                          f"rep_recon {diag.get('rep_recon', 0.0):.4f} (rep_ident {diag.get('rep_ident', 0.0):.4f})")
                    print(f"  [diag] Heads: Top-1 Acc {diag['top1_acc']:.1f}%, Top-5 Acc {diag['top5_acc']:.1f}%, "
                          f"Replace-F1 {diag['rep_f1']:.1f}% (prec {diag['rep_prec']:.1f}%, rec {diag['rep_rec']:.1f}%), "
                          f"Keep-Acc {diag['keep_acc']:.1f}%")
                    print(f"  [diag] LM-Head Decode: Top-1 {diag['lm_top1_acc']:.1f}%, Top-5 {diag['lm_top5_acc']:.1f}% | "
                          f"NN Decode: Top-1 {diag['nn_top1_acc']:.1f}%, Top-5 {diag['nn_top5_acc']:.1f}%")

            if global_step % tcfg["eval_every"] == 0:
                val_metrics = None
                if val_batches:
                    val_metrics = evaluate(
                        hybrid, embedder, tokenizer, corruptor, scheme,
                        val_batches, mask_prob, mask_ratio, noise_pool, mask_id, pad_id, S,
                        device, amp_dtype, mcfg,
                    )
                if val_metrics is not None:
                    val_loss = val_metrics["total"]
                    print(f"  [eval] step {global_step} val_loss {val_loss:.4f} "
                          f"[sm {val_metrics['score_matching']:.3f} tag {val_metrics['tag']:.3f} "
                          f"gen {val_metrics['gen']:.3f}] (best {best_val:.4f})")
                else:
                    val_loss = loss.item()

                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({
                        "hybrid": hybrid.state_dict(),
                        "embedder": embedder.state_dict(),
                        "config": config,
                        "dim": dim,
                        "embedder_name": mcfg["embedder"],
                        "best_val": best_val,
                        "global_step": global_step,
                    }, os.path.join(args.save_dir, "best.pt"))
                    print(f"  [eval] saved best (val_loss {val_loss:.4f})")

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