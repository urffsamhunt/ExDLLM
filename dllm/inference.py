"""
DLLM Inference: Iterative denoising with full 5-op canvas evolution.

Canvas positions carry a type label alongside their token ID so rules are
enforced correctly even as the sequence grows (INSERT/EXPAND) or shrinks
(DELETE) across iterations.

Position types:
  'bos'             – beginning-of-sequence (always KEEP)
  'eos'             – end-of-sequence (always KEEP)
  'prompt_imask'    – structural interleaving MASK in prompt section (always KEEP)
  'prompt_tok'      – visible prompt content token (cannot be pure KEEP → force REPLACE self)
  'response_slot'   – primary fillable MASK slot in response section
  'response_imask'  – structural interleaving MASK in response section (INSERT or KEEP)
  'response_filled' – response slot that has been filled by REPLACE
  'response_inserted'– token produced by an INSERT operation

Iteration loop:
  1. Model predicts (tag, gen_token) for every canvas position
  2. Convergence check: all un-filled response positions predict KEEP → done
  3. Execute edits, propagating types through INSERT / EXPAND / DELETE
  4. Repeat until converged or max_iterations reached

Edit semantics:
  KEEP    – position unchanged
  REPLACE – token at position → gen_token; type → 'response_filled'
  DELETE  – position removed from canvas
  INSERT  – insert gen_token BEFORE current position (with type 'response_inserted');
            current position (a MASK) stays so it can be filled later
  EXPAND  – MASK position → two new 'response_slot' MASKs (dynamic lengthening)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

# Internal tag indices (match DLLM model: 0=KEEP, 1=DELETE, 2=REPLACE, 3=INSERT, 4=EXPAND)
KEEP, DELETE, REPLACE, INSERT, EXPAND = 0, 1, 2, 3, 4


class DLLMInference:
    """
    Iterative denoising inference for the DLLM model.

    Usage:
        inference = DLLMInference(model, tokenizer)
        text = inference.generate("The king", target_length=16)
        text, traj = inference.generate("The king", return_trajectory=True)
    """

    def __init__(
        self,
        model,
        tokenizer,
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
        max_iterations: int = 20,
        target_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        return_trajectory: bool = False,
    ) -> str | Tuple[str, List[str]] | dict:


        """
        Generate a response for the given prompt using iterative canvas refinement.

        The response length M is predicted from the prompt by the model's length
        head: the canvas is initialized with exactly M response slot pairs (M is
        the initial mask count, not a cutoff -- INSERT/EXPAND can grow it).
        Pass target_length to override the predicted length.

        Args:
            prompt:               Input prompt string.
            max_iterations:       Maximum denoising steps.
            target_length:        Optional override for the predicted length M.
            temperature:          Sampling temperature for the generator head.
            top_k / top_p:        Nucleus / top-k sampling parameters.
            return_trajectory:    If True, return dict with trajectory history.
        """
        M   = self.tokenizer.mask_id
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id

        # Encode prompt, strip BOS/EOS/PAD
        raw_ids = self.tokenizer.encode(prompt)
        content_ids = [t for t in raw_ids
                       if t not in (bos, eos, self.tokenizer.pad_id)]

        # ── Stage 1: predict response length M from the prompt ──────────
        if target_length is None:
            prompt_ids   = [bos] + [tok for t in content_ids for tok in (M, t)] + [M, eos]
            out = self.model(
                noisy_ids=torch.tensor([prompt_ids], device=self.device),
                attention_mask=torch.ones(1, len(prompt_ids), device=self.device),
            )
            m_logits = out["length_logits"][0]
            # Expectation decoding: M = round(E[length]) — smoother than argmax,
            # which is mode-locked by the imbalanced length classes.
            probs = torch.softmax(m_logits, dim=-1)
            arange = torch.arange(1, self.model.length_head_max + 1, device=self.device)
            m_pred = int((probs * arange).sum().round())
            # Clamp to available canvas space (2*content + 2*M + 3 <= max_length)
            m_pred = max(1, min(m_pred, (self.max_length - 2 * len(content_ids) - 3) // 2))
            target_length = m_pred

        # ── Build initial canvas with type tags ───────────────────────
        canvas_ids:   List[int] = []
        canvas_types: List[str] = []

        canvas_ids.append(bos);  canvas_types.append('bos')

        for p_tok in content_ids:
            canvas_ids.append(M);     canvas_types.append('prompt_imask')
            canvas_ids.append(p_tok); canvas_types.append('prompt_tok')
        canvas_ids.append(M); canvas_types.append('prompt_imask')  # trailing structural

        for _ in range(target_length):
            canvas_ids.append(M); canvas_types.append('response_slot')   # fillable
            canvas_ids.append(M); canvas_types.append('response_imask')  # structural

        canvas_ids.append(eos); canvas_types.append('eos')

        trajectory = []
        tag_names = {0: "KEEP", 1: "DELETE", 2: "REPLACE", 3: "INSERT", 4: "EXPAND"}
        tag_names_list = ["KEEP", "DELETE", "REPLACE", "INSERT", "EXPAND"]

        if return_trajectory:
            # Step 0: initial canvas, no model predictions yet
            init_tokens = []
            for pos_i, (tok_id, tok_type) in enumerate(zip(canvas_ids, canvas_types)):
                tok_text = self.tokenizer.decode([tok_id]).strip() if tok_id not in (
                    bos, eos, self.tokenizer.pad_id, M,
                ) else {bos: "<BOS>", eos: "<EOS>", self.tokenizer.pad_id: "<PAD>", M: "<MASK>"}.get(tok_id, f"<{tok_id}>")
                init_tokens.append({
                    "pos": pos_i, "token": tok_text, "token_id": tok_id,
                    "type": tok_type, "tag": "KEEP", "tag_idx": 0,
                    "probs": {"KEEP": 1.0, "DELETE": 0.0, "REPLACE": 0.0, "INSERT": 0.0, "EXPAND": 0.0},
                    "confidence": 1.0,
                })
            trajectory.append({
                "step": 0,
                "raw_canvas": self._decode_canvas(canvas_ids, canvas_types, clean=False),
                "full_clean": self._decode_canvas(canvas_ids, canvas_types, clean=True),
                "response_only": self.extract_response(canvas_ids, canvas_types),
                "tag_counts": {"KEEP": len(canvas_ids), "DELETE": 0, "REPLACE": 0, "INSERT": 0, "EXPAND": 0},
                "tokens": init_tokens,
                "canvas_length": len(canvas_ids),
                "converged": False,
            })


        # ── Iterative refinement ──────────────────────────────────────
        for iteration in range(max_iterations):
            if len(canvas_ids) > self.max_length:
                canvas_ids   = canvas_ids[:self.max_length - 1]   + [eos]
                canvas_types = canvas_types[:self.max_length - 1] + ['eos']

            input_ids = torch.tensor([canvas_ids], device=self.device)
            attn_mask = torch.ones_like(input_ids)

            outputs = self.model(
                noisy_ids=input_ids,
                attention_mask=attn_mask,
                tag_labels=None,
            )

            tag_logits = outputs["tag_logits"][0]  # (seq_len, 5)
            gen_logits = outputs["gen_logits"][0]  # (seq_len, vocab)

            tag_probs       = F.softmax(tag_logits, dim=-1)
            tag_predictions = tag_probs.argmax(dim=-1).tolist()  # 0=KEEP, 1=DELETE, 2=REPLACE, 3=INSERT, 4=EXPAND

            # Learned Tagger Gating: The Tagger head's trained logits directly drive
            # edit decisions for every position (KEEP, REPLACE, DELETE, INSERT, EXPAND).
            generated_tokens = self._sample_tokens(gen_logits, temperature, top_k, top_p)



            # Count tags predicted in this step
            counts = {"KEEP": 0, "DELETE": 0, "REPLACE": 0, "INSERT": 0, "EXPAND": 0}
            for t in tag_predictions:
                counts[tag_names_list[t] if 0 <= t < len(tag_names_list) else "KEEP"] += 1

            # ── Execute edits with type propagation ────────────────────
            new_canvas_ids, new_canvas_types = self._execute_edits(
                canvas_ids, canvas_types, tag_predictions, generated_tokens
            )

            # ── Convergence check (after executing edits) ─────────────
            # Converge when:
            # 1. Canvas did not change at all from previous step (no edits made), OR
            # 2. Model predicts KEEP for all response positions (satisfied with text)
            response_positions = [
                i for i, typ in enumerate(canvas_types)
                if typ.startswith('response_') and i < len(tag_predictions)
            ]

            all_response_keep = (
                len(response_positions) > 0 and
                all(tag_predictions[i] == KEEP for i in response_positions)
            )
            canvas_unchanged = (new_canvas_ids == canvas_ids)

            canvas_ids, canvas_types = new_canvas_ids, new_canvas_types


            if return_trajectory:
                # Build per-token rich data for the visualizer.
                # We record state *before* executing edits (canvas_ids / canvas_types at
                # the start of this step) alongside the model's predictions so the UI can
                # show "what the model saw" and "what it decided" at every position.
                token_data = []
                tag_names_list = ["KEEP", "DELETE", "REPLACE", "INSERT", "EXPAND"]
                pre_canvas_ids   = new_canvas_ids   # state after edits (what we'll show next step)
                pre_canvas_types = new_canvas_types
                for pos_i, (tok_id, tok_type) in enumerate(zip(canvas_ids, canvas_types)):
                    tag_pred = tag_predictions[pos_i] if pos_i < len(tag_predictions) else 0
                    probs_list = tag_probs[pos_i].tolist() if pos_i < len(tag_predictions) else [1.0, 0, 0, 0, 0]
                    confidence = float(max(probs_list))
                    tok_text = self.tokenizer.decode([tok_id]).strip() if tok_id not in (
                        self.tokenizer.bos_id, self.tokenizer.eos_id,
                        self.tokenizer.pad_id, self.tokenizer.mask_id,
                    ) else {
                        self.tokenizer.bos_id: "<BOS>",
                        self.tokenizer.eos_id: "<EOS>",
                        self.tokenizer.pad_id: "<PAD>",
                        self.tokenizer.mask_id: "<MASK>",
                    }.get(tok_id, f"<{tok_id}>")
                    token_data.append({
                        "pos":        pos_i,
                        "token":      tok_text,
                        "token_id":   tok_id,
                        "type":       tok_type,
                        "tag":        tag_names_list[tag_pred],
                        "tag_idx":    tag_pred,
                        "probs":      {n: round(p, 4) for n, p in zip(tag_names_list, probs_list)},
                        "confidence": round(confidence, 4),
                    })
                trajectory.append({
                    "step": iteration + 1,
                    "raw_canvas": self._decode_canvas(canvas_ids, canvas_types, clean=False),
                    "full_clean": self._decode_canvas(canvas_ids, canvas_types, clean=True),
                    "response_only": self.extract_response(canvas_ids, canvas_types),
                    "tag_counts": counts,
                    "tokens": token_data,
                    "canvas_length": len(canvas_ids),
                    "converged": iteration > 0 and (all_response_keep or canvas_unchanged),
                })

            if iteration > 0 and (all_response_keep or canvas_unchanged):
                print(f"Converged after {iteration + 1} iterations.")
                break


        final_text = self._decode_canvas(canvas_ids, canvas_types, clean=True)
        response_text = self.extract_response(canvas_ids, canvas_types)

        if return_trajectory:
            return {
                "full_clean": final_text,
                "response_only": response_text,
                "trajectory": trajectory,
            }
        return final_text



    # ── Token Sampling ─────────────────────────────────────────────────

    def _sample_tokens(
        self,
        gen_logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[int]:
        """Top-k + nucleus sampling from generator logits. Returns one token per position."""
        logits = gen_logits / max(temperature, 1e-8)

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            min_topk = torch.topk(logits, k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0]  = False
            logits = logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))

        return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1).tolist()

    # ── Edit Execution with Type Propagation ───────────────────────────

    def _execute_edits(
        self,
        canvas_ids:   List[int],
        canvas_types: List[str],
        tag_predictions: List[int],
        generated_tokens: List[int],
    ) -> Tuple[List[int], List[str]]:
        """
        Apply predicted edit operations to the canvas, enforcing positional rules:

          BOS / EOS:         always KEEP (structural boundaries)
          prompt_imask:      always KEEP (Rule 3: structural spacing in prompt)
          prompt_tok:        KEEP → force REPLACE-self (Rule 2: cannot be pure-kept)
                             DELETE / REPLACE / INSERT / EXPAND allowed freely
          response_slot:     model predicts freely
          response_imask:    model predicts freely (INSERT or KEEP normally;
                             DELETE allowed to prune trailing structural masks)
          response_filled:   already contains a real token; KEEP unless model decides
                             to revise (DELETE / REPLACE allowed)
          response_inserted: same freedom as response_filled

        Type propagation after each op:
          KEEP              → type unchanged
          REPLACE on MASK   → 'response_filled'
          REPLACE on filled → type unchanged (in-place revision)
          DELETE            → position removed
          INSERT(gen_tok)   → [gen_tok:'response_inserted', original_tok:original_type]
          EXPAND            → [MASK:'response_slot', MASK:'response_slot']
        """
        M = self.tokenizer.mask_id

        new_ids:   List[int] = []
        new_types: List[str] = []

        for i, (tok_id, tok_type) in enumerate(zip(canvas_ids, canvas_types)):
            tag     = tag_predictions[i] if i < len(tag_predictions) else KEEP
            gen_tok = generated_tokens[i] if i < len(generated_tokens) else tok_id

            # ── Structural positions: always KEEP ──────────────────────
            if tok_type in ('bos', 'eos', 'prompt_imask'):
                new_ids.append(tok_id); new_types.append(tok_type)
                continue

            # ── Rule 2: prompt_tok cannot be pure KEEP ─────────────────
            # KEEP → replace with itself (no-op in effect, but proper tag)
            if tok_type == 'prompt_tok' and tag == KEEP:
                tag     = REPLACE
                gen_tok = tok_id

            # ── Apply tag ──────────────────────────────────────────────
            if tag == DELETE:
                continue  # position removed

            elif tag == REPLACE:
                new_ids.append(gen_tok)
                # After filling a MASK slot, mark as filled; otherwise preserve type
                if tok_type in ('response_slot', 'response_imask'):
                    new_types.append('response_filled')
                else:
                    new_types.append(tok_type)

            elif tag == INSERT:
                # Insert gen_tok BEFORE current position, keep MASK for next iteration
                new_ids.append(gen_tok);  new_types.append('response_inserted')
                new_ids.append(tok_id);   new_types.append(tok_type)  # MASK stays

            elif tag == EXPAND:
                # Single MASK → two fresh response_slot MASKs (dynamic lengthening)
                new_ids.append(M); new_types.append('response_slot')
                new_ids.append(M); new_types.append('response_slot')

            else:  # KEEP (or unknown)
                new_ids.append(tok_id); new_types.append(tok_type)

        # Safety: never return empty canvas
        if not new_ids:
            bos = self.tokenizer.bos_id
            eos = self.tokenizer.eos_id
            new_ids   = [bos, eos]
            new_types = ['bos', 'eos']

        return new_ids, new_types

    # ── Canvas Decoding ────────────────────────────────────────────────

    def _decode_canvas(self, canvas_ids: List[int], canvas_types: List[str], clean: bool = False) -> str:
        """
        Decode canvas IDs to string.
        If clean=True, strips structural <MASK> tokens and extra spaces for human reading.
        """
        if not clean:
            return self.tokenizer.decode(canvas_ids)

        # Filter out BOS, EOS, PAD and any remaining <MASK> tokens
        special_ids = {
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
            self.tokenizer.pad_id,
            self.tokenizer.mask_id,
        }
        clean_ids = [tid for tid in canvas_ids if tid not in special_ids]
        raw_text = self.tokenizer.decode(clean_ids)
        # Clean up whitespace artifacts around punctuation
        return " ".join(raw_text.split())

    def extract_response(self, canvas_ids: List[int], canvas_types: List[str]) -> str:
        """
        Extract only the generated response tokens from the canvas (excluding prompt).
        """
        response_types = {'response_filled', 'response_inserted'}
        resp_ids = [tok for tok, typ in zip(canvas_ids, canvas_types)
                    if typ in response_types and tok != self.tokenizer.mask_id]
        if not resp_ids:
            return ""
        text = self.tokenizer.decode(resp_ids)
        return " ".join(text.split())

