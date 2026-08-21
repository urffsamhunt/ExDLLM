"""
ARLM Model: standard autoregressive language model.

This is a thin wrapper around a HuggingFace causal LM (e.g. a causal-masked
RoBERTa via `RobertaForCausalLM`, or a GPT-2 via `GPT2LMHeadModel`). It
exposes a small, DLLM-compatible surface (`forward`, `compute_loss`,
`get_param_groups`) so the trainer and benchmark scripts can treat both
models uniformly, but the objective is the standard teacher-forced
next-token prediction.

The model is loaded with the pretrained LM head, so no new head is added:
the pretrained `lm_head` is reused directly.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class ARLM(nn.Module):
    """
    Autoregressive language model wrapper.

    Usage:
        model = ARLM("roberta-base")
        loss = model(input_ids, attention_mask, labels=labels).loss
    """

    def __init__(self, tokenizer, backbone_name: str = "roberta-base"):
        """
        Args:
            tokenizer: ARLMTokenizer instance.
            backbone_name: HuggingFace causal-LM model identifier.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_id

        # Load a causal LM with its pretrained LM head. `RobertaForCausalLM`
        # applies a causal mask to the RoBERTa backbone; `GPT2LMHeadModel` is
        # natively autoregressive. Both accept (input_ids, attention_mask,
        # labels) and return a `.loss`.
        self.backbone_name = backbone_name
        self.backbone = AutoModelForCausalLM.from_pretrained(backbone_name)

        # Resize embeddings to the tokenizer's vocab (a no-op for the standard
        # backbones, but keeps the wrapper robust to tokenizer mismatches).
        self.backbone.resize_token_embeddings(self.vocab_size)

    # ── Forward / Loss ────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,          # (batch, seq_len)
        attention_mask: torch.Tensor,     # (batch, seq_len)
        labels: Optional[torch.Tensor] = None,  # (batch, seq_len), -100 = ignore
    ) -> Dict[str, torch.Tensor]:
        """
        Standard causal-LM forward pass.

        Args:
            input_ids: Token IDs.
            attention_mask: Padding mask.
            labels: Optional next-token labels (shifted). If None, only logits
                    are returned (inference mode).

        Returns:
            Dict with 'logits' and, when labels are given, 'loss'.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        result = {"logits": outputs.logits}
        if labels is not None:
            result["loss"] = outputs.loss
        return result

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the next-token cross-entropy loss.

        Returns:
            (loss, loss_dict) where loss_dict contains 'lm_loss' and
            'total_loss' (identical here, kept for API symmetry with DLLM).
        """
        outputs = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs["loss"]
        loss_dict = {
            "lm_loss": loss.item(),
            "total_loss": loss.item(),
        }
        return loss, loss_dict

    # ── Utility: Get Model Parameters Grouped for Optimizer ────────────
    def get_param_groups(self, lr: float, weight_decay: float) -> list:
        """
        Return parameter groups with weight decay applied only to weights
        (not biases and layer norms), following the DLLM convention.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        seen = set()
        decay_params, no_decay_params = [], []
        for n, p in self.named_parameters():
            if id(p) in seen or not p.requires_grad:
                continue
            seen.add(id(p))
            if any(nd in n for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]


def load_arlm_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Load an ARLM state dict, unwrapping DataParallel if present.

    Returns a message describing what was done.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(state_dict)
    return "loaded (strict)"