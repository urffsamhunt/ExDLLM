#!/usr/bin/env python3
"""
Generation script for DLLM — generate text via iterative denoising.

Usage:
    python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "The king"
    python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "To be" --max_iter 30 --target_length 64
"""

import argparse
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dllm import DLLMTokenizer, DLLM, DLLMInference
from dllm.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with DLLM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--prompt", type=str, default="The",
                        help="Starting text prompt")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Maximum refinement iterations")
    parser.add_argument("--target_length", type=int, default=None,
                        help="Override the prompt-predicted response length (default: predicted from the prompt by the model)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Token confidence threshold to force REPLACE refinement (default: 0.0)")

    parser.add_argument("--show_trajectory", action="store_true",
                        help="Show intermediate denoising steps")
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
    tokenizer = DLLMTokenizer(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )

    # ── Initialize Model ─────────────────────────────────────────────
    print("Loading model...")
    model = DLLM(
        tokenizer=tokenizer,
        backbone_name=config["model"]["backbone"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        tag_weights=config["model"].get("tag_weights"),
        length_head_max=config["data"].get("max_response_length", 48),
        len_smoothing=config["model"].get("len_smoothing", 0.15),
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    from dllm.model import load_dllm_state
    msg = load_dllm_state(model, checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} [{msg}]")
    print(f"  Trained steps: {checkpoint.get('global_step', 'unknown')}")

    # ── Initialize Inference ──────────────────────────────────────────
    inference = DLLMInference(
        model=model,
        tokenizer=tokenizer,
        max_length=config["tokenizer"]["max_length"],
        device=args.device,
    )

    # Use defaults from config if not overridden
    max_iter    = args.max_iterations or config["inference"]["max_iterations"]
    target_len  = args.target_length  # None -> predicted from the prompt by the length head

    # ── Generate ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")
    len_desc = f"Response slots: {target_len} (overridden)" if target_len else "Response length: predicted from prompt"
    print(f"Max iterations: {max_iter}  |  {len_desc}")
    print(f"{'=' * 60}\n")

    result = inference.generate(
        prompt=args.prompt,
        max_iterations=max_iter,
        target_length=target_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        return_trajectory=args.show_trajectory,
    )


    if args.show_trajectory:
        info = result
        print("Denoising Trajectory (State After Each Iteration):\n")
        for step in info["trajectory"]:
            step_num = step["step"]
            counts = step["tag_counts"]
            counts_str = f"KEEP={counts['KEEP']} REP={counts['REPLACE']} DEL={counts['DELETE']} INS={counts['INSERT']} EXP={counts['EXPAND']}"

            print(f"--- [ Step {step_num:02d} ] ({counts_str}) ---")
            print(f"  Response Only : {step['response_only']}")
            print(f"  Full Clean    : {step['full_clean']}")
            print(f"  Raw Canvas    : {step['raw_canvas'][:120]}...\n")

        print(f"{'='*60}")
        print(f"Final Clean Text:\n  {info['full_clean']}")
        print(f"\nFinal Response Only:\n  {info['response_only']}")
        print(f"{'='*60}")
    else:
        print(f"Generated Text:\n  {result}")




if __name__ == "__main__":
    main()
