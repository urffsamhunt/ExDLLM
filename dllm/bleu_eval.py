"""
BLEU evaluation callback for the training loop.

Translates a deterministic subset of the held-out parallel test set with the
current model (DLLMInference) and returns the corpus BLEU (sacrebleu).
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from .inference import DLLMInference


def make_bleu_evaluator(
    model,
    tokenizer,
    device: str,
    test_path: str,
    n: int = 50,
    max_length: int = 192,
) -> Callable[[], Dict[str, Optional[float]]]:
    """
    Build a callable that returns {'bleu': score} on a fixed subset of a test
    TSV (source<TAB>reference per line). The model is restored to its previous
    training/eval mode after each call.
    """
    if not os.path.isfile(test_path):
        raise FileNotFoundError(f"bleu_test_path not found: {test_path}")

    # Unwrap DataParallel for inference (batch size 1 cannot be scattered).
    inf_model = model.module if hasattr(model, "module") else model
    inference = DLLMInference(model=inf_model, tokenizer=tokenizer,
                              max_length=max_length, device=device)

    pairs = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                pairs.append((parts[0].strip(), parts[1].strip()))
            if len(pairs) >= n:
                break
    if not pairs:
        raise ValueError(f"no usable pairs in {test_path}")

    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        def _no_bleu() -> Dict[str, Optional[float]]:
            print("BLEU eval skipped: install sacrebleu (pip install sacrebleu)")
            return {"bleu": None}
        return _no_bleu

    def evaluate() -> Dict[str, Optional[float]]:
        was_training = inf_model.training
        inf_model.eval()
        refs, hyps = [], []
        for src, ref in pairs:
            hyp = inference.generate(src, max_iterations=20)
            hyps.append(hyp.strip())
            refs.append(ref)
        if was_training:
            inf_model.train()
        score = sacrebleu.corpus_bleu(hyps, [refs])
        return {"bleu": score.score}

    return evaluate
