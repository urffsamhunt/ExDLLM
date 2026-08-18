#!/usr/bin/env python3
"""
Training script for DLLM — Discrete Diffusion Language Model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --device cuda --seed 42
"""

import argparse
import os
import sys
import yaml

# Reduce CUDA fragmentation (large-vocab gen logits churn allocations)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dllm import DLLMTokenizer, ForwardCorruptor, DLLMDataset, DLLM, DLLMTrainer
from dllm.dataset import collate_fn
from dllm.utils import set_seed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train DLLM")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override config training.max_steps")
    parser.add_argument("--n_gpu", type=int, default=None,
                        help="Number of GPUs to use (default: all available)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # ── Initialize Tokenizer ──────────────────────────────────────────
    print("\n[1/5] Initializing tokenizer...")
    tokenizer = DLLMTokenizer(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )

    # ── Initialize Corruptor ──────────────────────────────────────────
    print("\n[2/5] Initializing forward corruptor...")
    corruptor = ForwardCorruptor(
        tokenizer=tokenizer,
        replace_ratio=config["data"]["corruption"]["replace_ratio"],
        delete_ratio=config["data"]["corruption"]["delete_ratio"],
        insert_ratio=config["data"]["corruption"]["insert_ratio"],
        expand_ratio=config["data"]["corruption"]["expand_ratio"],
        mask_ratio=config["data"]["corruption"]["mask_ratio"],
        noise_vocab_size=config["data"]["corruption"]["noise_vocab_size"],
        expand_prob=config["data"]["corruption"].get("expand_prob", 0.10),
        insert_prob=config["data"]["corruption"].get("insert_prob", 0.15),
        t_skew=config["data"]["corruption"].get("t_skew", 2.0),
        shortage_prob=config["data"]["corruption"].get("shortage_prob", 0.35),
    )

    # ── Load Datasets ──────────────────────────────────────────────────
    print("\n[3/5] Loading datasets...")
    data_mode = config["data"].get("mode", "prompt_response")
    max_p_len = config["data"].get("max_prompt_length", 48)
    max_r_len = config["data"].get("max_response_length", 48)
    # Sub-iteration trajectory training: read stages from config (None = disabled)
    sub_iterations = config["training"].get("sub_iterations", None)

    train_dataset = DLLMDataset(
        tokenizer=tokenizer,
        corruptor=corruptor,
        dataset_name=config["data"]["dataset_name"],
        split="train",
        max_length=config["tokenizer"]["max_length"],
        mode=data_mode,
        max_prompt_length=max_p_len,
        max_response_length=max_r_len,
        sub_iterations=sub_iterations,  # multi-stage trajectory or single-stage
    )

    # Validation dataset: always single-stage (sub_iterations=None) for speed.
    # The val loader is used only for checkpointing; 5× extra forward passes add
    # no useful signal and make eval ~5× slower.
    try:
        val_dataset = DLLMDataset(
            tokenizer=tokenizer,
            corruptor=corruptor,
            dataset_name=config["data"].get("val_dataset_name", config["data"]["dataset_name"]),
            split="validation",
            max_length=config["tokenizer"]["max_length"],
            mode=data_mode,
            max_prompt_length=max_p_len,
            max_response_length=max_r_len,
            sub_iterations=None,  # always single-stage for fast validation
        )
    except Exception:
        print("No validation split found, using training data for validation.")
        val_dataset = DLLMDataset(
            tokenizer=tokenizer,
            corruptor=corruptor,
            dataset_name=config["data"]["dataset_name"],
            split="train",
            max_length=config["tokenizer"]["max_length"],
            mode=data_mode,
            max_prompt_length=max_p_len,
            max_response_length=max_r_len,
            sub_iterations=None,  # always single-stage for fast validation
        )

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Streaming datasets don't support multiprocessing well
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Quick sanity check: fetch one batch
    print("Fetching a sample batch...")
    sample_batch = next(iter(train_loader))
    noisy_shape = sample_batch["noisy_ids"].shape
    print(f"  noisy_ids shape:  {noisy_shape}  {'(B, K, S) multi-stage' if len(noisy_shape) == 3 else '(B, S) single-stage'}")
    print(f"  tag_labels shape: {sample_batch['tag_labels'].shape}")
    print(f"  gen_labels shape: {sample_batch['gen_labels'].shape}")
    print(f"  gen_mask sum:     {sample_batch['gen_mask'].sum().item()}")

    # ── Initialize Model ─────────────────────────────────────────────
    print("\n[4/5] Initializing model...")
    model = DLLM(
        tokenizer=tokenizer,
        backbone_name=config["model"]["backbone"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        tag_weights=config["model"].get("tag_weights"),
        length_head_max=config["data"].get("max_response_length", 48),
        length_weights=train_dataset.length_weights(),
        len_smoothing=config["model"].get("len_smoothing", 0.15),
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Initialize Trainer and Train ──────────────────────────────────
    print("\n[5/5] Starting training...")
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps

    # Multi-GPU: wrap in DataParallel (batch is split across devices).
    n_gpu = args.n_gpu or torch.cuda.device_count()
    print(f"Detected {torch.cuda.device_count()} GPU(s)")
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
        print(f"Using DataParallel across {n_gpu} GPUs (batch split {config['training']['batch_size'] // n_gpu} per GPU)")

    trainer = DLLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
    )

    # BLEU tracking on a held-out parallel test set (translation runs)
    bleu_every = int(config["training"].get("bleu_every", 0) or 0)
    bleu_test_path = config["data"].get("bleu_test_path")
    bleu_eval_fn = None
    if bleu_every > 0 and bleu_test_path:
        from dllm.bleu_eval import make_bleu_evaluator
        bleu_n = int(config["data"].get("bleu_n", 50))
        bleu_eval_fn = make_bleu_evaluator(
            model=model,
            tokenizer=tokenizer,
            device=trainer.device,
            test_path=bleu_test_path,
            n=bleu_n,
            max_length=config["tokenizer"]["max_length"],
        )
        print(f"BLEU evaluation every {bleu_every} steps on {bleu_test_path} (n={bleu_n})")

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
        bleu_eval_fn=bleu_eval_fn,
        bleu_every=bleu_every,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
