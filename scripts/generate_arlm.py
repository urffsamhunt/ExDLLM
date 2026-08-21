#!/usr/bin/env python3
"""
Generation script for ARLM — autoregressive text generation.

This is the autoregressive counterpart to `scripts/generate.py` (DLLM). It
loads a trained ARLM checkpoint and generates a continuation of a prompt via
standard left-to-right decoding.

Usage:
    python scripts/generate_arlm.py --checkpoint ./models_ar/best_model.pt --prompt "MENENIUS:"
    python scripts/generate_arlm.py --checkpoint ./models_ar/best_model.pt --prompt "To be" --max_new 64 --greedy
"""

import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from arlm import ARLMTokenizer, ARLM, ARLMInference
from arlm.model import load_arlm_state
from dllm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with ARLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/arlm.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--prompt", type=str, default="The",
                        help="Starting text prompt")
    parser.add_argument("--max_new", type=int, default=None,
                        help="Maximum new tokens to generate (default: config inference.max_new_tokens)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy (argmax) decoding instead of sampling")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ── Initialize Tokenizer ──────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = ARLMTokenizer(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )

    # ── Initialize Model ─────────────────────────────────────────────
    print("Loading model...")
    model = ARLM(
        tokenizer=tokenizer,
        backbone_name=config["model"]["backbone"],
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    msg = load_arlm_state(model, checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} [{msg}]")
    print(f"  Trained steps: {checkpoint.get('global_step', 'unknown')}")

    # ── Initialize Inference ──────────────────────────────────────────
    inference = ARLMInference(
        model=model,
        tokenizer=tokenizer,
        max_length=config["tokenizer"]["max_length"],
        device=args.device,
    )

    max_new = args.max_new or config["inference"].get("max_new_tokens", 64)

    # ── Generate ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")
    print(f"Max new tokens: {max_new}  |  greedy: {args.greedy}")
    print(f"{'=' * 60}\n")

    text = inference.generate(
        prompt=args.prompt,
        max_new_tokens=max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
    )
    print(f"Generated Text:\n  {text}")


if __name__ == "__main__":
    main()