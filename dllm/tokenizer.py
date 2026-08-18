"""
DLLM Tokenizer: Extended RoBERTa tokenizer with edit operation tokens.

Adds the following special tokens to a standard RoBERTa tokenizer:
    <KEEP>    - Token is correct, no change needed
    <DELETE>  - Token should be removed
    <REPLACE> - Token should be substituted with new text
    <INSERT>  - New text should be inserted at this position
    <EXPAND>  - Token should be duplicated into two <MASK> tokens
    <MASK>    - Placeholder to be filled by the Generator head
"""

from __future__ import annotations

from typing import List, Dict, Optional, Union
import torch
from transformers import RobertaTokenizer, XLMRobertaTokenizer, PreTrainedTokenizer


# The six special tokens required for edit-based diffusion
EDIT_SPECIAL_TOKENS = [
    "<KEEP>",
    "<DELETE>",
    "<REPLACE>",
    "<INSERT>",
    "<EXPAND>",
    "<MASK>",
]


class DLLMTokenizer:
    """
    Wraps a RobertaTokenizer with additional edit-operation tokens.

    Usage:
        tok = DLLMTokenizer("roberta-base")
        tok.add_edit_tokens()  # called automatically in __init__

        # Access special token ids
        tok.keep_id       # <KEEP>
        tok.delete_id     # <DELETE>
        tok.replace_id    # <REPLACE>
        tok.insert_id     # <INSERT>
        tok.expand_id     # <EXPAND>
        tok.mask_id       # <MASK>
    """

    def __init__(self, base_model: str = "roberta-base", max_length: int = 128):
        """
        Args:
            base_model: HuggingFace model identifier for the base tokenizer.
            max_length: Maximum sequence length for tokenization.
        """
        self.base_model = base_model
        self.max_length = max_length

        # Load the underlying HuggingFace tokenizer.
        # Multilingual backbones (xlm-roberta) use the same API with a
        # SentencePiece vocab that includes Devanagari; roberta-base falls
        # back to byte-level tokens for non-Latin scripts.
        if base_model.startswith("xlm-roberta"):
            self._tokenizer: XLMRobertaTokenizer = XLMRobertaTokenizer.from_pretrained(base_model)
        else:
            self._tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(base_model)
        self.vocab_size = len(self._tokenizer)

        # Add edit special tokens and cache their IDs
        self._add_edit_tokens()

        # Updated vocabulary size after adding special tokens
        self.vocab_size = len(self._tokenizer)

    def _add_edit_tokens(self) -> None:
        """Add edit operation tokens to the tokenizer vocabulary."""
        num_added = self._tokenizer.add_special_tokens(
            {"additional_special_tokens": EDIT_SPECIAL_TOKENS}
        )
        print(f"Added {num_added} edit tokens to vocabulary.")
        print(f"New vocabulary size: {len(self._tokenizer)}")

        # Cache token IDs for fast access
        self.keep_id    = self._tokenizer.convert_tokens_to_ids("<KEEP>")
        self.delete_id  = self._tokenizer.convert_tokens_to_ids("<DELETE>")
        self.replace_id = self._tokenizer.convert_tokens_to_ids("<REPLACE>")
        self.insert_id  = self._tokenizer.convert_tokens_to_ids("<INSERT>")
        self.expand_id  = self._tokenizer.convert_tokens_to_ids("<EXPAND>")
        self.mask_id    = self._tokenizer.convert_tokens_to_ids("<MASK>")

        # Convenience mappings
        self.edit_token_ids: Dict[str, int] = {
            "<KEEP>":    self.keep_id,
            "<DELETE>":  self.delete_id,
            "<REPLACE>": self.replace_id,
            "<INSERT>":  self.insert_id,
            "<EXPAND>":  self.expand_id,
            "<MASK>":    self.mask_id,
        }

        self.id_to_edit: Dict[int, str] = {v: k for k, v in self.edit_token_ids.items()}

        # Store the original vocab size (before adding edit tokens)
        self.original_vocab_size = self.vocab_size  # saved before tokens were added

        # Also cache special token ids from the base tokenizer
        self.pad_id  = self._tokenizer.pad_token_id
        self.bos_id  = self._tokenizer.bos_token_id
        self.eos_id  = self._tokenizer.eos_token_id
        self.unk_id  = self._tokenizer.unk_token_id

        # The set of edit tag ids (for masking / filtering)
        self.edit_tag_ids = set(self.edit_token_ids.values())

    # ── Tokenization Methods ──────────────────────────────────────────

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode a text string into token IDs (no padding/truncation)."""
        params = {"add_special_tokens": True, "max_length": self.max_length}
        params.update(kwargs)
        return self._tokenizer.encode(text, **params)

    def decode(self, ids: List[int] or torch.Tensor, **kwargs) -> str:
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
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text(s) with padding and truncation.

        Returns a dict with 'input_ids' and 'attention_mask'.
        """
        params = {
            "padding": padding,
            "truncation": truncation,
            "max_length": self.max_length,
            "return_tensors": return_tensors,
        }
        params.update(kwargs)
        return self._tokenizer(texts, **params)

    # ── Utility Methods ───────────────────────────────────────────────

    def is_edit_token(self, token_id: int) -> bool:
        """Check if a token ID corresponds to one of the edit operation tokens."""
        return token_id in self.edit_tag_ids

    def is_vocab_token(self, token_id: int) -> bool:
        """Check if a token ID is a regular vocabulary token (not a special edit token)."""
        return token_id < self.original_vocab_size or token_id >= self.vocab_size

    def get_edit_tag_name(self, token_id: int) -> str:
        """Get the string name of an edit tag token."""
        return self.id_to_edit.get(token_id, "<UNKNOWN>")

    @property
    def tokenizer(self) -> RobertaTokenizer:
        """Access the underlying HuggingFace tokenizer."""
        return self._tokenizer

    @property
    def keep_token(self) -> str:
        return "<KEEP>"

    @property
    def delete_token(self) -> str:
        return "<DELETE>"

    @property
    def replace_token(self) -> str:
        return "<REPLACE>"

    @property
    def insert_token(self) -> str:
        return "<INSERT>"

    @property
    def expand_token(self) -> str:
        return "<EXPAND>"

    @property
    def mask_token(self) -> str:
        return "<MASK>"

    def __repr__(self) -> str:
        return (f"DLLMTokenizer(base={self.base_model}, vocab_size={self.vocab_size}, "
                f"max_length={self.max_length})")
