"""
ARLM Tokenizer: thin wrapper around a HuggingFace causal LM tokenizer.

Unlike DLLMTokenizer, the autoregressive model needs no edit-operation tokens.
It uses the base tokenizer as-is, so the vocabulary is exactly the pretrained
one (no resizing of the embedding matrix is required).

The wrapper exposes a small, DLLM-compatible surface (encode / decode / pad /
bos / eos ids) so the rest of the ARLM module and the benchmark harness can
treat both tokenizers uniformly.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


class ARLMTokenizer:
    """Wrap a HuggingFace causal-LM tokenizer with a DLLM-compatible surface."""

    def __init__(self, base_model: str = "roberta-base", max_length: int = 192):
        """
        Args:
            base_model: HuggingFace model identifier for the base tokenizer.
            max_length: Maximum sequence length for tokenization.
        """
        self.base_model = base_model
        self.max_length = max_length

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(base_model)

        # Ensure padding / special tokens exist. RoBERTa has pad/bos/eos; GPT-2
        # has no pad token by default, so we add one.
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            else:
                self._tokenizer.add_special_tokens({"pad_token": "<pad>"})

        self.vocab_size = len(self._tokenizer)
        self.pad_id = self._tokenizer.pad_token_id
        self.bos_id = self._tokenizer.bos_token_id
        self.eos_id = self._tokenizer.eos_token_id
        self.unk_id = self._tokenizer.unk_token_id

    # ── Tokenization Methods ──────────────────────────────────────────

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode a text string into token IDs (no padding/truncation by default)."""
        params = {"add_special_tokens": True, "max_length": self.max_length}
        params.update(kwargs)
        return self._tokenizer.encode(text, **params)

    def decode(self, ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode token IDs back to a string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, **kwargs)

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        **kwargs,
    ) -> dict:
        """Tokenize text(s) with padding and truncation."""
        params = {
            "padding": padding,
            "truncation": truncation,
            "max_length": self.max_length,
            "return_tensors": return_tensors,
        }
        params.update(kwargs)
        return self._tokenizer(texts, **params)

    # ── Utility Methods ───────────────────────────────────────────────

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Access the underlying HuggingFace tokenizer."""
        return self._tokenizer

    def __repr__(self) -> str:
        return (f"ARLMTokenizer(base={self.base_model}, vocab_size={self.vocab_size}, "
                f"max_length={self.max_length})")