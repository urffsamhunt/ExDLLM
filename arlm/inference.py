"""
ARLM Inference: standard autoregressive (left-to-right) generation.

The public API mirrors `DLLMInference.generate` (prompt -> response string,
with temperature / top-k / top-p sampling) so the benchmark harness can call
both models identically. Internally it is a plain greedy / sampled next-token
loop over the causal LM.

Note: unlike the DLLM (which returns only the generated response), the ARLM
generates the continuation of the prompt. The `generate` method returns the
full decoded text; use `extract_response` when you want just the newly
generated tokens.
"""

from __future__ import annotations

from typing import Optional

import torch

from .tokenizer import ARLMTokenizer


class ARLMInference:
    """
    Autoregressive sampling for the ARLM model.

    Usage:
        inference = ARLMInference(model, tokenizer)
        text = inference.generate("The king", max_new_tokens=64)
    """

    def __init__(
        self,
        model,
        tokenizer: ARLMTokenizer,
        max_length: int = 256,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ── Public API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        greedy: bool = False,
        stop_on_eos: bool = True,
    ) -> str:
        """
        Generate a continuation for the given prompt via autoregressive decoding.

        Args:
            prompt:             Input prompt string.
            max_new_tokens:     Maximum number of new tokens to generate.
            temperature:        Sampling temperature (1.0 = no scaling).
            top_k:              Top-k sampling (0 = disabled).
            top_p:              Nucleus sampling threshold (1.0 = disabled).
            greedy:             If True, use argmax decoding instead of sampling.
            stop_on_eos:        If True, stop when the EOS token is generated.

        Returns:
            The full decoded text (prompt + generated continuation).
        """
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        tokens = torch.tensor([ids], device=self.device)

        for _ in range(max_new_tokens):
            if tokens.size(1) >= self.max_length:
                break
            logits = self.model(tokens)["logits"][:, -1, :] / max(temperature, 1e-8)

            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = self._sample_filters(logits, top_k, top_p)
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            tokens = torch.cat([tokens, nxt], dim=1)
            if stop_on_eos and nxt.item() == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(tokens[0], skip_special_tokens=False)

    # ── Sampling Helpers ──────────────────────────────────────────────

    def _sample_filters(self, logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
        """Apply top-k and nucleus filtering to a (1, vocab) logits tensor."""
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            min_topk = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_topk, torch.full_like(logits, float("-inf")), logits
            )
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))
        return logits

    def _sample_tokens(self, logits: torch.Tensor, temperature: float = 1.0,
                       top_k: int = 50, top_p: float = 0.9,
                       chunk: int = 128) -> torch.Tensor:
        """Sample tokens from an (N, V) logits tensor (used by the benchmark)."""
        logits = logits / max(temperature, 1e-8)
        out = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
        for start in range(0, logits.size(0), chunk):
            part = logits[start:start + chunk]
            part = self._sample_filters(part, top_k, top_p)
            out[start:start + chunk] = torch.multinomial(
                torch.softmax(part, dim=-1), 1
            ).squeeze(-1)
        return out

    # ── Response Extraction ───────────────────────────────────────────

    def extract_response(self, full_text: str, prompt: str) -> str:
        """
        Extract the newly generated portion of `full_text` (the continuation
        after the prompt). Useful for benchmark metrics that compare only the
        generated response against a reference.
        """
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        # Fallback: strip a leading BOS token if present.
        return full_text.strip()