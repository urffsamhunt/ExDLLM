"""
DLLM Utilities: Helper functions for tensor manipulation, logging, etc.

Includes:
- Sequence length manipulation utilities (expand/contract)
- Levenshtein visualization tools for debugging
- Random seed setting
"""

from __future__ import annotations

import json
import random
import torch
import numpy as np
from typing import List, Tuple


class JSONMetricsLogger:
    """
    Append training/validation metrics as newline-delimited JSON (JSONL).

    Each line is a flat dict of one logged step, e.g.:
        {"step": 10, "total_loss": 3.21, "lr": 5e-05, ...}

    JSONL is chosen over a single JSON array so the file can be written
    incrementally (safe to tail / partial-read / resume) and plotted by any
    consumer without loading the whole history into memory.

    Usage:
        logger = JSONMetricsLogger("checkpoints/training_metrics.jsonl")
        logger.log({"step": 10, "total_loss": 3.21})
        logger.close()
    """

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", encoding="utf-8")

    def log(self, metrics: dict) -> None:
        self._f.write(json.dumps(metrics) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def set_seed(seed: int):
    """Set random seed across torch, numpy, and Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def expand_sequence(
    ids: torch.Tensor,
    positions: torch.Tensor,  # boolean mask: True = EXPAND this position
    mask_id: int,
) -> torch.Tensor:
    """
    Deterministically expand a sequence by replacing each EXPAND-tagged
    position with two [MASK] tokens.

    Args:
        ids: (seq_len,) or (batch, seq_len) tensor of token IDs.
        positions: Boolean tensor same shape as ids, True where to expand.
        mask_id: Token ID for <MASK>.

    Returns:
        Expanded tensor with extra positions.
    """
    if ids.dim() == 1:
        return _expand_1d(ids, positions, mask_id)
    else:
        # Batch: process each row
        expanded = []
        max_len = 0
        for i in range(ids.size(0)):
            row = _expand_1d(ids[i], positions[i], mask_id)
            expanded.append(row)
            max_len = max(max_len, len(row))
        # Pad to max length
        padded = torch.full((ids.size(0), max_len), mask_id, dtype=ids.dtype, device=ids.device)
        for i, row in enumerate(expanded):
            padded[i, :len(row)] = torch.tensor(row, dtype=ids.dtype, device=ids.device)
        return padded


def _expand_1d(ids: torch.Tensor, positions: torch.Tensor, mask_id: int) -> List[int]:
    """Helper: expand a 1D sequence."""
    result = []
    for tok, expand in zip(ids.tolist(), positions.tolist()):
        if expand:
            result.append(mask_id)
            result.append(mask_id)
        else:
            result.append(tok)
    return result


def delete_positions(
    ids: torch.Tensor,
    positions: torch.Tensor,  # boolean mask: True = DELETE this position
    pad_id: int,
) -> torch.Tensor:
    """
    Remove positions marked for deletion. Pads to original length.

    Args:
        ids: (seq_len,) or (batch, seq_len) tensor.
        positions: Boolean tensor, True = delete.
        pad_id: Padding token ID.

    Returns:
        Tensor with deleted positions removed and padded.
    """
    if ids.dim() == 1:
        kept = ids[~positions]
        if len(kept) == 0:
            kept = ids[:1]  # Keep at least one token
        # Pad to original length
        padded = torch.full_like(ids, pad_id)
        padded[:len(kept)] = kept
        return padded
    else:
        result = ids.clone()
        for i in range(ids.size(0)):
            kept = ids[i][~positions[i]]
            if len(kept) == 0:
                kept = ids[i][:1]
            result[i, :len(kept)] = kept
            result[i, len(kept):] = pad_id
        return result


def visualize_edits(
    tokenizer,       # DLLMTokenizer
    noisy_ids: List[int],
    tag_labels: List[int],
    clean_ids: List[int] = None,
) -> str:
    """
    Create a human-readable visualization of the edit operations.

    Returns a multi-line string showing the noisy sequence, predicted
    edits, and (optionally) the clean target.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Noisy:  {tokenizer.decode(noisy_ids)}")

    # Show edit tags
    edit_chars = []
    for i, tid in enumerate(noisy_ids):
        if i < len(tag_labels):
            tag_id = tag_labels[i]
            if tag_id == tokenizer.keep_id:
                edit_chars.append("·")
            elif tag_id == tokenizer.delete_id:
                edit_chars.append("D")
            elif tag_id == tokenizer.replace_id:
                edit_chars.append("R")
            elif tag_id == tokenizer.insert_id:
                edit_chars.append("I")
            elif tag_id == tokenizer.expand_id:
                edit_chars.append("E")
            else:
                edit_chars.append("?")
        else:
            edit_chars.append(" ")
    lines.append(f"Edits:  {''.join(edit_chars)}")
    lines.append(f"        K=· D=DELETE R=REPLACE I=INSERT E=EXPAND")

    if clean_ids is not None:
        lines.append(f"Clean:  {tokenizer.decode(clean_ids)}")

    lines.append("=" * 60)
    return "\n".join(lines)


def compute_edit_accuracy(
    tag_logits: torch.Tensor,  # (B, S, num_tags)
    tag_labels: torch.Tensor,  # (B, S)
    ignore_index: int = -100,
) -> float:
    """
    Compute accuracy of the Tagger predictions.
    """
    mask = tag_labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    predictions = tag_logits.argmax(dim=-1)
    correct = (predictions == tag_labels) & mask
    return correct.sum().item() / mask.sum().item()
