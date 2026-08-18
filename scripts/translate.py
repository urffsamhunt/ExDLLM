#!/usr/bin/env python3
"""
Translate with a trained DLLM checkpoint (English <-> Hindi).

Usage:
    python scripts/translate.py --checkpoint ./translation_hi/best_model.pt \
        --src "How are you?" --config configs/translation.yaml
    python scripts/translate.py --checkpoint ... --eval_fwd --n 200   # BLEU on en->hi
    python scripts/translate.py --checkpoint ... --eval_rev --n 200   # BLEU on hi->en
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dllm import DLLMTokenizer, DLLM, DLLMInference
from dllm.utils import set_seed


def load_model(config, checkpoint, device):
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
    from dllm.model import load_dllm_state
    msg = load_dllm_state(model, ckpt["model_state_dict"])
    print(f"Loaded checkpoint [{msg}]")
    inference = DLLMInference(model=model, tokenizer=tokenizer,
                              max_length=config["tokenizer"]["max_length"],
                              device=device)
    return tokenizer, model, inference


def translate(inference, text):
    return inference.generate(text, max_iterations=20).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/translation.yaml")
    parser.add_argument("--src", type=str, default=None, help="Translate this string")
    parser.add_argument("--src_file", type=str, default=None, help="Translate lines from a file")
    parser.add_argument("--eval_fwd", action="store_true", help="BLEU on data/en_hi_test.tsv")
    parser.add_argument("--eval_rev", action="store_true", help="BLEU on data/en_hi_rev_test.tsv")
    parser.add_argument("--n", type=int, default=200, help="Evaluation samples")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    _, _, inference = load_model(config, args.checkpoint, device)

    if args.eval_fwd or args.eval_rev:
        path = "data/en_hi_test.tsv" if args.eval_fwd else "data/en_hi_rev_test.tsv"
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        pairs = pairs[: args.n]
        print(f"Evaluating {len(pairs)} pairs from {path} ...")
        try:
            import sacrebleu
            refs, hyps = [], []
            for i, (src, ref) in enumerate(pairs):
                hyp = translate(inference, src)
                refs.append(ref)
                hyps.append(hyp)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(pairs)}")
            score = sacrebleu.corpus_bleu(hyps, [refs])
            print(f"\nBLEU: {score.score:.2f}")
            for i in range(min(3, len(pairs))):
                print(f"\nSRC : {pairs[i][0]}\nREF : {refs[i]}\nOUT : {hyps[i]}")
        except ImportError:
            print("sacrebleu not installed; showing samples instead:")
            for src, ref in pairs[:5]:
                print(f"\nSRC : {src}\nREF : {ref}\nOUT : {translate(inference, src)}")
        return

    texts = []
    if args.src:
        texts.append(args.src)
    if args.src_file:
        with open(args.src_file, encoding="utf-8") as f:
            texts += [ln.strip() for ln in f if ln.strip()]

    for text in texts:
        print(f"\nSRC : {text}\nOUT : {translate(inference, text)}")


if __name__ == "__main__":
    main()
