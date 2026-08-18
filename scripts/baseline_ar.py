#!/usr/bin/env python3
"""
Autoregressive baseline for DLLM.

Same RoBERTa backbone (causal-masked RobertaForCausalLM, reusing the
pretrained lm_head), same tiny_shakespeare dialogue pairs, same optimizer /
schedule / batch config, but a standard teacher-forced next-token objective.

Usage:
    python scripts/baseline_ar.py --config configs/default.yaml --save_dir ./baseline_ar --max_steps 17000
    python scripts/baseline_ar.py --checkpoint ./baseline_ar/best_model.pt --prompt "MENENIUS:"
"""

import argparse
import os
import sys
import random

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import RobertaTokenizer, RobertaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dllm import DLLMTokenizer
from dllm.dataset import DLLMDataset
from dllm.utils import set_seed


class ARDialogueDataset(Dataset):
    """Teacher-forced (prompt + response) sequences extracted from dialogue pairs."""

    def __init__(self, tokenizer, pairs, max_length: int = 128):
        self.seqs = []
        for prompt_turn, response_turn in pairs:
            ids = tokenizer.encode(
                f"{prompt_turn} {response_turn}".strip(),
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
            )
            if len(ids) >= 8:
                self.seqs.append(ids)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        return torch.tensor(self.seqs[idx], dtype=torch.long)


def make_collate(pad_id: int):
    def collate(batch):
        max_len = max(len(b) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            input_ids[i, : len(b)] = b
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return collate


def build_labels(input_ids, attention_mask, pad_id: int):
    """Shifted next-token labels; padding and the final position are ignored."""
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels = labels.masked_fill(attention_mask == 0, -100)
    labels[:, -1] = -100
    return labels


@torch.no_grad()
def evaluate(model, val_loader, device, pad_id: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in val_loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = build_labels(ids, attn, pad_id).to(device)
        loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
        total += loss.item() * ids.size(0)
        n += ids.size(0)
    model.train()
    return total / max(1, n)


@torch.no_grad()
def generate(model, tokenizer, prompt, device, max_new=64,
             temperature=1.0, top_k=50, top_p=0.9, greedy=False) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    tokens = torch.tensor([ids], device=device)
    for _ in range(max_new):
        logits = model(tokens).logits[:, -1, :] / max(temperature, 1e-8)
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                min_topk = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        tokens = torch.cat([tokens, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(tokens[0], skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Autoregressive baseline for DLLM")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--save_dir", type=str, default="./baseline_ar")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Training steps (default: config training.max_steps; use 17000 for the DLLM comparison)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="If set, skip training and generate instead")
    parser.add_argument("--prompt", type=str, default="MENENIUS:")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max_new", type=int, default=64)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = RobertaTokenizer.from_pretrained(config["tokenizer"]["base"])
    max_length = config["tokenizer"]["max_length"]

    # ── Generation mode ────────────────────────────────────────────────
    if args.checkpoint:
        model = RobertaForCausalLM.from_pretrained(config["model"]["backbone"])
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model_state_dict"])
        model.to(device).eval()
        print(f"Loaded baseline checkpoint: {args.checkpoint}")
        text = generate(model, tok, args.prompt, device, max_new=args.max_new, greedy=args.greedy)
        print(f"\nPrompt : {args.prompt}\nOutput : {text}")
        return

    # ── Data: reuse the DLLM dialogue-pair extraction ─────────────────
    dllm_tok = DLLMTokenizer(
        base_model=config["tokenizer"]["base"], max_length=max_length
    )
    pairs_ds = DLLMDataset(
        tokenizer=dllm_tok, corruptor=None,
        dataset_name=config["data"]["dataset_name"],
        mode="prompt_response",
        max_prompt_length=config["data"].get("max_prompt_length", 48),
        max_response_length=config["data"].get("max_response_length", 48),
    )
    pairs = list(pairs_ds._text_lines)
    print(f"Dialogue pairs: {len(pairs)}")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(0.05 * len(pairs)))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    train_ds = ARDialogueDataset(tok, train_pairs, max_length)
    val_ds = ARDialogueDataset(tok, val_pairs, max_length)
    print(f"Train sequences: {len(train_ds)}, val: {len(val_ds)}")

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=make_collate(tok.pad_token_id), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=make_collate(tok.pad_token_id), num_workers=0)

    # ── Model: causal RoBERTa with the pretrained LM head ──────────────
    model = RobertaForCausalLM.from_pretrained(config["model"]["backbone"])
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Optimizer / schedule (identical to DLLM trainer) ──────────────
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": float(config["training"]["weight_decay"])},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=float(config["training"]["learning_rate"]))

    warmup_steps = int(config["training"]["warmup_steps"])
    max_steps = args.max_steps or int(config["training"]["max_steps"])
    accum = int(config["training"]["gradient_accumulation_steps"])
    log_every = int(config["training"]["log_every"])
    eval_every = int(config["training"]["eval_every"])

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    global_step = 0
    accum_count = 0
    running_loss = 0.0
    best_val = float("inf")

    print(f"\nTraining to {max_steps} steps (batch {batch_size} x accum {accum})...\n")
    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = build_labels(ids, attn, tok.pad_token_id).to(device)

            loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            loss = loss / accum
            loss.backward()
            running_loss += loss.item() * accum
            accum_count += 1

            if accum_count >= accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

                if global_step % log_every == 0:
                    avg = running_loss / log_every
                    print(f"step {global_step:6d} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                    running_loss = 0.0

                if global_step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, device, tok.pad_token_id)
                    print(f"  >> val loss {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        torch.save({"model_state_dict": model.state_dict(),
                                    "global_step": global_step, "best_val_loss": best_val},
                                   os.path.join(args.save_dir, "best_model.pt"))
                        print(f"  >> saved best_model.pt (step {global_step})")

    torch.save({"model_state_dict": model.state_dict(), "global_step": global_step,
                "best_val_loss": best_val},
               os.path.join(args.save_dir, "final_model.pt"))
    print(f"\nDone. Final model: {args.save_dir}/final_model.pt (best val {best_val:.4f})")

    # ── Sample generations from the final model ───────────────────────
    model.eval()
    for prompt in ["Second Citizen: One word, good citizens.",
                   "First Citizen: You are all resolved rather to die than to famish?",
                   "MENENIUS: Now the good gods forbid"]:
        text = generate(model, tok, prompt, device, max_new=64, greedy=False)
        print(f"\nPrompt : {prompt}\nOutput : {text}")


if __name__ == "__main__":
    main()
