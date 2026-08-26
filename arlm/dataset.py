"""
ARLM Dataset: teacher-forced (prompt + response) sequences for an
autoregressive language model.

The data source is identical to the DLLM training data: we reuse the DLLM
dialogue-pair extraction (consecutive turn pairs) so that benchmarking the
ARLM against the DLLM is apples-to-apples. Each pair is flattened into a
single token sequence `prompt <sep> response`, and the model is trained to
predict the next token at every position (teacher forcing).

The dataset is deterministic and stateless: it does not corrupt or augment
the text, because a standard autoregressive LM has no corruption process.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import ARLMTokenizer


class ARLMDataset(Dataset):
    """
    PyTorch Dataset of teacher-forced prompt+response sequences.

    Args:
        tokenizer: ARLMTokenizer instance.
        dataset_name: HuggingFace dataset name or path to a local text file.
        split: Dataset split ('train' or 'validation').
        max_length: Maximum total sequence length (prompt + response).
        max_prompt_length: Max tokens for the prompt context.
        max_response_length: Max tokens for the target response.
        cache_dir: Directory to cache downloaded datasets.
        text_file: Explicit path to a local text file (overrides dataset_name).
    """

    def __init__(
        self,
        tokenizer: ARLMTokenizer,
        dataset_name: str = "tiny_shakespeare",
        split: str = "train",
        max_length: int = 192,
        max_prompt_length: int = 48,
        max_response_length: int = 48,
        cache_dir: str = "./data",
        text_file: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

        # Reuse the DLLM dialogue-pair extraction so both models train on the
        # exact same (prompt, response) pairs.
        from dllm.dataset import DLLMDataset
        from dllm import DLLMTokenizer

        dllm_tok = DLLMTokenizer(
            base_model=tokenizer.base_model, max_length=max_length
        )
        pairs_ds = DLLMDataset(
            tokenizer=dllm_tok,
            corruptor=None,
            dataset_name=dataset_name,
            split=split,
            mode="prompt_response",
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            cache_dir=cache_dir,
            text_file=text_file,
        )
        self._pairs: List[Tuple[str, str]] = list(pairs_ds._text_lines)

        # Batch-encode all pairs in a single call (far faster than a per-item
        # Python loop of individual encode() calls). padding=False keeps each
        # sequence at its natural length; the collate fn pads per batch.
        texts = [
            f"{prompt_turn} {response_turn}".strip()
            for prompt_turn, response_turn in self._pairs
        ]
        encoded = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors=None,
        )
        self._seqs: List[List[int]] = [
            ids for ids in encoded["input_ids"] if len(ids) >= 8
        ]

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self._seqs[idx], dtype=torch.long)


def make_collate(pad_id: int):
    """Return a collate function that pads a batch to the longest sequence."""
    def collate(batch):
        max_len = max(len(b) for b in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            input_ids[i, : len(b)] = b
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return collate


def build_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Build shifted next-token labels for teacher forcing.

    The label at position i is the token at position i+1. Padding positions and
    the final position are ignored (set to -100), matching the standard causal
    LM objective.
    """
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels = labels.masked_fill(attention_mask == 0, -100)
    labels[:, -1] = -100
    return labels