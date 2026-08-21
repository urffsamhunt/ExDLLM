#!/usr/bin/env python3
"""
Benchmark DLLM vs ARLM (autoregressive baseline) on the same held-out prompts.

Loads one DLLM checkpoint and one ARLM checkpoint, then generates responses
for a shared set of held-out dialogue prompts and reports:
  - per-model generation time (tokens/sec)
  - per-model parameter count
  - BLEU / chrF against the reference responses (if sacrebleu is installed)
  - a few side-by-side sample generations for qualitative inspection

The prompts are drawn from the same dialogue-pair corpus used for training
(held-out split), so both models see identical inputs.

Usage:
    python scripts/benchmark.py \
        --dllm_checkpoint ./checkpoints_v2/best_model.pt \
        --arlm_checkpoint ./models_ar/best_model.pt \
        --config configs/default.yaml \
        --arlm_config configs/arlm.yaml \
        --n 50
"""

import argparse
import os
import random
import sys
import time
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dllm import DLLMTokenizer, DLLM, DLLMInference
from dllm.model import load_dllm_state
from arlm import ARLMTokenizer, ARLM, ARLMInference
from arlm.model import load_arlm_state
from dllm.utils import set_seed


def load_dllm(config, checkpoint, device):
    tokenizer = DLLMTokenizer(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )
    model = DLLM(
        tokenizer=tokenizer,
        backbone_name=config["model"]["backbone"],
        hidden_dropout_prob=config["model"]["hidden_dropout_prob"],
        attention_probs_dropout_prob=config["model"]["attention_probs_dropout_prob"],
        tag_weights=config["model"].get("tag_weights"),
        length_head_max=config["data"].get("max_response_length", 48),
        len_smoothing=config["model"].get("len_smoothing", 0.15),
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    msg = load_dllm_state(model, ckpt["model_state_dict"])
    inference = DLLMInference(model=model, tokenizer=tokenizer,
                              max_length=config["tokenizer"]["max_length"],
                              device=device)
    return tokenizer, model, inference, msg


def load_arlm(config, checkpoint, device):
    tokenizer = ARLMTokenizer(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )
    model = ARLM(
        tokenizer=tokenizer,
        backbone_name=config["model"]["backbone"],
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    msg = load_arlm_state(model, ckpt["model_state_dict"])
    inference = ARLMInference(model=model, tokenizer=tokenizer,
                              max_length=config["tokenizer"]["max_length"],
                              device=device)
    return tokenizer, model, inference, msg


def load_heldout_prompts(config, n, seed):
    """Load a held-out set of (prompt, reference) dialogue pairs."""
    from dllm import DLLMTokenizer as _DLLMTok
    from dllm.dataset import DLLMDataset

    dllm_tok = _DLLMTok(
        base_model=config["tokenizer"]["base"],
        max_length=config["tokenizer"]["max_length"],
    )
    pairs_ds = DLLMDataset(
        tokenizer=dllm_tok, corruptor=None,
        dataset_name=config["data"]["dataset_name"],
        mode="prompt_response",
        max_prompt_length=config["data"].get("max_prompt_length", 48),
        max_response_length=config["data"].get("max_response_length", 48),
    )
    pairs = list(pairs_ds._text_lines)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    # Take a held-out slice (the same 5% split the ARLM trainer uses for val).
    n_val = max(1, int(0.05 * len(pairs)))
    held_out = pairs[:n_val]
    rng2 = random.Random(seed)
    rng2.shuffle(held_out)
    return held_out[:n]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Benchmark DLLM vs ARLM")
    parser.add_argument("--dllm_checkpoint", type=str, required=True,
                        help="Path to DLLM checkpoint")
    parser.add_argument("--arlm_checkpoint", type=str, required=True,
                        help="Path to ARLM checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="DLLM config")
    parser.add_argument("--arlm_config", type=str, default="configs/arlm.yaml",
                        help="ARLM config")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of held-out prompts to evaluate")
    parser.add_argument("--max_new", type=int, default=64,
                        help="Max new tokens for ARLM generation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        dllm_config = yaml.safe_load(f)
    with open(args.arlm_config) as f:
        arlm_config = yaml.safe_load(f)

    # ── Load both models ─────────────────────────────────────────────
    print("Loading DLLM...")
    _, dllm_model, dllm_inf, dllm_msg = load_dllm(dllm_config, args.dllm_checkpoint, device)
    print(f"  DLLM checkpoint [{dllm_msg}]")

    print("Loading ARLM...")
    _, arlm_model, arlm_inf, arlm_msg = load_arlm(arlm_config, args.arlm_checkpoint, device)
    print(f"  ARLM checkpoint [{arlm_msg}]")

    # ── Held-out prompts ─────────────────────────────────────────────
    held = load_heldout_prompts(dllm_config, args.n, args.seed)
    print(f"\nEvaluating {len(held)} held-out prompts...\n")

    # ── Generate with both models ────────────────────────────────────
    dllm_outputs, arlm_outputs, refs = [], [], []
    prompts = []

    # DLLM
    dllm_start = time.time()
    for i, (prompt, ref) in enumerate(held):
        out = dllm_inf.generate(prompt, max_iterations=20).strip()
        dllm_outputs.append(out)
        refs.append(ref)
        prompts.append(prompt)
        if (i + 1) % 25 == 0:
            print(f"  DLLM {i + 1}/{len(held)}")
    dllm_time = time.time() - dllm_start

    # ARLM
    arlm_start = time.time()
    for i, (prompt, ref) in enumerate(held):
        full = arlm_inf.generate(prompt, max_new_tokens=args.max_new)
        out = arlm_inf.extract_response(full, prompt)
        arlm_outputs.append(out)
        if (i + 1) % 25 == 0:
            print(f"  ARLM {i + 1}/{len(held)}")
    arlm_time = time.time() - arlm_start

    # ── Metrics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    dllm_params = count_params(dllm_model)
    arlm_params = count_params(arlm_model)
    print(f"\nParameters:")
    print(f"  DLLM : {dllm_params:,}")
    print(f"  ARLM : {arlm_params:,}")

    print(f"\nGeneration time ({len(held)} prompts):")
    print(f"  DLLM : {dllm_time:.2f}s ({dllm_time / max(1, len(held)):.3f}s/prompt)")
    print(f"  ARLM : {arlm_time:.2f}s ({arlm_time / max(1, len(held)):.3f}s/prompt)")

    # BLEU / chrF (optional)
    try:
        import sacrebleu
        dllm_bleu = sacrebleu.corpus_bleu(dllm_outputs, [refs]).score
        arlm_bleu = sacrebleu.corpus_bleu(arlm_outputs, [refs]).score
        dllm_chrf = sacrebleu.corpus_chrf(dllm_outputs, [refs]).score
        arlm_chrf = sacrebleu.corpus_chrf(arlm_outputs, [refs]).score
        print(f"\nBLEU (vs reference):")
        print(f"  DLLM : {dllm_bleu:.2f}")
        print(f"  ARLM : {arlm_bleu:.2f}")
        print(f"\nchrF (vs reference):")
        print(f"  DLLM : {dllm_chrf:.2f}")
        print(f"  ARLM : {arlm_chrf:.2f}")
    except ImportError:
        print("\n(sacrebleu not installed; skipping BLEU/chrF)")

    # ── Sample outputs ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUTS")
    print("=" * 60)
    for i in range(min(5, len(held))):
        print(f"\nPrompt : {prompts[i]}")
        print(f"Reference : {refs[i]}")
        print(f"DLLM : {dllm_outputs[i]}")
        print(f"ARLM : {arlm_outputs[i]}")


if __name__ == "__main__":
    main()