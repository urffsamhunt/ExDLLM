#!/usr/bin/env python3
"""
Training script for ARLM — a standard autoregressive language model.

This is the autoregressive counterpart to `scripts/train.py` (DLLM). It trains
a causal LM (default: causal-masked RoBERTa) with a teacher-forced next-token
objective on the same dialogue-pair data as the DLLM, so the two can be
benchmarked against each other.

Usage:
    python scripts/train_arlm.py --config configs/arlm.yaml --save_dir ./checkpoints_ar
    python scripts/train_arlm.py --config configs/arlm.yaml --device cuda --seed 42
"""

import argparse
import os
import sys
import yaml

# Reduce CUDA fragmentation (large-vocab gen logits churn allocations)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arlm import ARLMTokenizer, ARLMDataset, ARLM, ARLMTrainer
from arlm.dataset import make_collate
from dllm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARLM (autoregressive baseline)")
    parser.add_argument("--config", type=str, default="configs/arlm.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./models_ar",
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
    print("\n[1/4] Initializing tokenizer...")
    tokenizer = ARLMTokenizer(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )

    # ── Load Datasets ──────────────────────────────────────────────────
    print("\n[2/4] Loading datasets...")
    max_p_len = config["data"].get("max_prompt_length", 48)
    max_r_len = config["data"].get("max_response_length", 48)

    train_dataset = ARLMDataset(
        tokenizer=tokenizer,
        dataset_name=config["data"]["dataset_name"],
        split="train",
        max_length=config["tokenizer"]["max_length"],
        max_prompt_length=max_p_len,
        max_response_length=max_r_len,
    )

    try:
        val_dataset = ARLMDataset(
            tokenizer=tokenizer,
            dataset_name=config["data"].get("val_dataset_name", config["data"]["dataset_name"]),
            split="validation",
            max_length=config["tokenizer"]["max_length"],
            max_prompt_length=max_p_len,
            max_response_length=max_r_len,
        )
    except Exception:
        print("No validation split found, using training data for validation.")
        val_dataset = ARLMDataset(
            tokenizer=tokenizer,
            dataset_name=config["data"]["dataset_name"],
            split="train",
            max_length=config["tokenizer"]["max_length"],
            max_prompt_length=max_p_len,
            max_response_length=max_r_len,
        )

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_id),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer.pad_id),
        num_workers=0,
    )

    print(f"Train sequences: {len(train_dataset)}, val: {len(val_dataset)}")

    # ── Initialize Model ─────────────────────────────────────────────
    print("\n[3/4] Initializing model...")
    model = ARLM(
        tokenizer=tokenizer,
        backbone_name=config["model"]["backbone"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Initialize Trainer and Train ──────────────────────────────────
    print("\n[4/4] Starting training...")
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps

    # Multi-GPU: wrap in DataParallel (batch is split across devices).
    n_gpu = args.n_gpu or torch.cuda.device_count()
    print(f"Detected {torch.cuda.device_count()} GPU(s)")
    if n_gpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(n_gpu)))
        print(f"Using DataParallel across {n_gpu} GPUs (batch split {config['training']['batch_size'] // n_gpu} per GPU)")

    trainer = ARLMTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=args.device,
    )

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=args.save_dir,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    main()