"""
Forward Corruptor: Corrupts clean text sequences and computes Levenshtein
edit paths for training the Tagger head.

Given a clean sequence of token IDs, the corruptor:
1. Applies random corruptions (replace, delete, insert noise, expand spans)
2. Computes the optimal Levenshtein alignment between corrupted and clean
3. Produces per-token edit tag labels (<KEEP>, <DELETE>, <REPLACE>, etc.)
"""

from __future__ import annotations

import random
from typing import List, Tuple, Optional
import Levenshtein


class ForwardCorruptor:
    """
    Applies synthetic edits to token sequences and computes ground-truth
    edit operation labels via Levenshtein alignment.

    The corruption simulates the forward diffusion process: we take clean
    text and degrade it so the model can learn to reverse the degradation.

    Usage:
        corruptor = ForwardCorruptor(tokenizer, replace_ratio=0.15, ...)
        noisy_ids, tag_labels = corruptor.corrupt(clean_ids)
    """

    def __init__(
        self,
        tokenizer,  # DLLMTokenizer instance
        replace_ratio: float = 0.15,
        delete_ratio: float = 0.05,
        insert_ratio: float = 0.05,
        expand_ratio: float = 0.05,
        mask_ratio: float = 0.10,
        noise_vocab_size: int = 100,
        expand_prob: float = 0.10,
        insert_prob: float = 0.15,
        t_skew: float = 2.0,
        shortage_prob: float = 0.35,
        seed: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: DLLMTokenizer instance for token ID access.
            replace_ratio: Fraction of tokens to randomly replace with noise.
            delete_ratio: Fraction of tokens to randomly delete.
            insert_ratio: Fraction of positions to insert noise tokens.
            expand_ratio: Fraction of spans to merge into a single [EXPAND].
            mask_ratio: Fraction of tokens to replace with <MASK>.
            noise_vocab_size: Number of top frequent tokens to use as noise source.
            expand_prob: Probability of EXPAND on any active response slot.
            insert_prob: Probability of INSERT on any structural iMask in response section.
            t_skew: Skews the noise level toward 1 via t = 1 - U^t_skew, so the
                model trains on near-fully-masked canvases like the inference start.
            shortage_prob: Probability of building a canvas with fewer slots than
                the answer (1-2 fewer), labeling the last iMask <INSERT> and/or the
                last slot <EXPAND> so the model learns to grow when content overflows.
            seed: Random seed for reproducibility.
        """
        self.tokenizer = tokenizer
        self.replace_ratio = replace_ratio
        self.delete_ratio = delete_ratio
        self.insert_ratio = insert_ratio
        self.expand_ratio = expand_ratio
        self.mask_ratio = mask_ratio
        self.noise_vocab_size = noise_vocab_size
        self.expand_prob = expand_prob
        self.insert_prob = insert_prob
        self.t_skew = t_skew
        self.shortage_prob = shortage_prob

        if seed is not None:
            random.seed(seed)

        # Pre-compute noise token pool: sample from ordinary vocab tokens
        # We use token IDs 4..noise_vocab_size+4 to avoid special tokens
        self.noise_pool = list(range(4, min(noise_vocab_size + 4, tokenizer.original_vocab_size)))

    def _get_random_noise_token(self, exclude_tok: int = -1) -> int:
        """Sample a random noise token from the noise pool, avoiding exclude_tok."""
        tok = random.choice(self.noise_pool)
        if tok == exclude_tok and len(self.noise_pool) > 1:
            tok = (tok + 1) % len(self.noise_pool)
            return self.noise_pool[tok]
        return tok


    # ── Public API ────────────────────────────────────────────────────

    def corrupt(self, clean_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Corrupt a clean token sequence and produce tag labels.

        Args:
            clean_ids: List of token IDs representing clean text.

        Returns:
            noisy_ids: Corrupted token sequence (includes edit tokens).
            tag_labels: Per-token edit operation label for each position
                        in the *corrupted* sequence, indicating what edit
                        is needed to move toward the clean target.
                        Labels are token IDs: keep_id, delete_id, replace_id,
                        insert_id, or expand_id.
        """
        # Step 1: Apply random corruptions to create the noisy sequence
        noisy_ids = list(clean_ids)  # copy
        noisy_ids = self._apply_replace(noisy_ids)
        noisy_ids = self._apply_delete(noisy_ids)
        noisy_ids = self._apply_insert(noisy_ids)
        noisy_ids = self._apply_expand(noisy_ids)
        noisy_ids = self._apply_mask(noisy_ids)

        # Step 2: Compute optimal Levenshtein alignment (noisy -> clean)
        tag_labels = self._align(noisy_ids, clean_ids)

        return noisy_ids, tag_labels

    def corrupt_prompt_response(
        self, prompt_ids: List[int], clean_response_ids: List[int]
    ) -> Tuple[List[int], List[int], List[bool], dict]:
        """
        Builds an interleaved canvas with a noise level t skewed toward 1:

        For each training sample, a noise level t ~ 1 - U(0,1)^t_skew is sampled:
          - t = 1.0 : Full masking (all response slots are <MASK> → REPLACE target)
          - t = 0.0 : Clean target (all response slots are clean tokens → KEEP target)
          - 0 < t < 1: Dynamic mixture:
              * fraction (1-t) of tokens are clean target tokens → KEEP
              * fraction (t * mask_ratio) are <MASK> tokens → REPLACE target
              * fraction (t * (1-mask_ratio)) are corrupted noise tokens → REPLACE target

        The response section has exactly M = answer-length slot pairs: the canvas
        is sized to the response, with no capacity padding and no pruning. M is
        the *initial* mask count only -- at inference it is predicted from the
        prompt by the model's length head, and INSERT/EXPAND can still grow the
        canvas beyond it.

        To teach growth, a fraction (shortage_prob) of samples use a canvas with
        1-2 *fewer* slots than the answer: the last iMask is labeled <INSERT>
        (target = the first overflow token) and/or the last slot <EXPAND>, so the
        model learns to grow when the answer overflows the initial capacity.

        This teaches the model:
          1. How to fill blank masks from scratch (step 1 generation)
          2. How to keep correct tokens (step 2 verification)
          3. How to fix corrupted/flawed words in-place (step 2+ refinement)
        """
        # Strip special tokens
        prompt_ids = [t for t in prompt_ids
                      if t not in (self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)]
        clean_response_ids = [t for t in clean_response_ids
                              if t not in (self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)]

        K  = self.tokenizer.keep_id
        R  = self.tokenizer.replace_id
        I  = self.tokenizer.insert_id
        E  = self.tokenizer.expand_id
        M  = self.tokenizer.mask_id

        noisy_ids:   List[int]  = []
        tag_labels:  List[int]  = []
        prompt_mask: List[bool] = []
        pos_to_clean: dict      = {}

        # Sample dynamic noise level t, skewed toward 1 (the all-<MASK> inference start)
        t_noise = 1.0 - random.random() ** self.t_skew

        # The response section has exactly M slots: usually M = n_resp, but with
        # prob shortage_prob the canvas is built 1-2 slots SHORT so the model
        # learns to grow (INSERT/EXPAND) when content overflows the capacity.
        n_resp = len(clean_response_ids)
        shortage_k = 0
        if n_resp >= 3 and random.random() < self.shortage_prob:
            shortage_k = random.randint(1, min(2, n_resp - 1))
        n_slots = n_resp - shortage_k

        # ── BOS ───────────────────────────────────────────────────────────
        noisy_ids.append(self.tokenizer.bos_id)
        tag_labels.append(K)
        prompt_mask.append(True)

        # ── Prompt section: iMask(KEEP) + token(REPLACE→self) ─────────────
        for p_tok in prompt_ids:
            noisy_ids.append(M);   tag_labels.append(K);  prompt_mask.append(True)   # iMask
            pos = len(noisy_ids)
            noisy_ids.append(p_tok); tag_labels.append(R); prompt_mask.append(True)  # token
            pos_to_clean[pos] = p_tok   # self-replacement target
        noisy_ids.append(M); tag_labels.append(K); prompt_mask.append(True)           # trailing iMask

        # ── Response section: n_slots pairs ───────────────────────────────
        for slot_idx in range(n_slots):

            r_tok      = clean_response_ids[slot_idx]
            next_r_tok = clean_response_ids[slot_idx + 1] if slot_idx + 1 < n_resp else r_tok
            is_last    = slot_idx == n_slots - 1

            if is_last and shortage_k == 2:
                # Two overflow tokens: expand the last slot into two fresh masks
                noisy_ids.append(M); tag_labels.append(E); prompt_mask.append(False)
            else:
                # Determine slot corruption state based on t_noise
                roll = random.random()

                if roll > t_noise:
                    # Clean target token present in input slot → target is KEEP
                    noisy_ids.append(r_tok)
                    tag_labels.append(K)
                    prompt_mask.append(False)
                elif roll < t_noise * self.mask_ratio:
                    # Masked slot → target is REPLACE or EXPAND
                    if random.random() < self.expand_prob:
                        slot_tag = E
                    else:
                        slot_tag = R
                        pos_to_clean[len(noisy_ids)] = r_tok
                    noisy_ids.append(M)
                    tag_labels.append(slot_tag)
                    prompt_mask.append(False)
                else:
                    # Corrupted noise token present in input slot → target is REPLACE
                    noise_tok = self._get_random_noise_token(r_tok)
                    pos_to_clean[len(noisy_ids)] = r_tok
                    noisy_ids.append(noise_tok)
                    tag_labels.append(R)
                    prompt_mask.append(False)

            # ── Structural interleaving mask: INSERT / KEEP ─────────────
            if is_last and shortage_k >= 1:
                # Overflow: insert the first extra answer token before the final mask
                pos_to_clean[len(noisy_ids)] = clean_response_ids[n_slots]
                noisy_ids.append(M); tag_labels.append(I); prompt_mask.append(False)
            elif random.random() < self.insert_prob and slot_idx < n_resp:
                pos_to_clean[len(noisy_ids)] = next_r_tok
                noisy_ids.append(M); tag_labels.append(I); prompt_mask.append(False)
            else:
                noisy_ids.append(M); tag_labels.append(K); prompt_mask.append(False)

        # ── EOS ───────────────────────────────────────────────────────────
        noisy_ids.append(self.tokenizer.eos_id)
        tag_labels.append(K)
        prompt_mask.append(False)

        return noisy_ids, tag_labels, prompt_mask, pos_to_clean

    def corrupt_prompt_response_trajectory(
        self,
        prompt_ids: List[int],
        clean_response_ids: List[int],
        stages: List[float],
    ) -> List[Tuple[List[int], List[int], List[bool], dict]]:
        """
        Build a progressive denoising trajectory for a single (prompt, response) pair.

        For each noise level t in `stages` (e.g. [1.0, 0.8, 0.6, 0.4, 0.2]), generates
        one corrupted canvas where exactly round((1-t)*n_slots) response slots contain
        clean tokens (tagged KEEP) and the rest are masked/noisy (tagged REPLACE/EXPAND).

        Critically different from calling corrupt_prompt_response() K times:
          • shortage_k is sampled ONCE and reused across all stages → all K canvases
            share the same n_slots, guaranteeing identical canvas lengths for stacking.
          • A single permutation π defines the cumulative revelation order: slots
            in π[:n_revealed] are clean at stage k, and π[:n_revealed+...] at stage k+1.
            This ensures the revealed set only grows monotonically, mirroring inference.
          • The prompt section and structural interleaving masks are built identically
            for all stages (they never change across denoising steps).

        Args:
            prompt_ids:          Token IDs of the prompt (no special tokens).
            clean_response_ids:  Token IDs of the clean response (no special tokens).
            stages:              Ordered list of noise levels, from high to low
                                 (e.g. [1.0, 0.8, 0.6, 0.4, 0.2]).

        Returns:
            List of K tuples, one per stage:
                (noisy_ids, tag_labels, prompt_mask, pos_to_clean)
            All tuples share the same canvas length (safe to stack into (K, S)).
        """
        # Strip special tokens (same as corrupt_prompt_response)
        prompt_ids = [t for t in prompt_ids
                      if t not in (self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)]
        clean_response_ids = [t for t in clean_response_ids
                               if t not in (self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id)]

        K_tok = self.tokenizer.keep_id
        R_tok = self.tokenizer.replace_id
        I_tok = self.tokenizer.insert_id
        E_tok = self.tokenizer.expand_id
        M_tok = self.tokenizer.mask_id

        n_resp = len(clean_response_ids)

        # ── Sample shortage_k ONCE and share across all stages ────────────────
        # This guarantees all K canvases have identical length (n_slots is fixed).
        shortage_k = 0
        if n_resp >= 3 and random.random() < self.shortage_prob:
            shortage_k = random.randint(1, min(2, n_resp - 1))
        n_slots = n_resp - shortage_k

        # ── Build a single permutation π over response slot indices ──────────
        # π defines the order in which slots are revealed across stages.
        # At stage k with n_revealed slots revealed, π[:n_revealed] are the
        # "already clean" slot indices — a growing cumulative prefix.
        pi = list(range(n_slots))
        random.shuffle(pi)

        results = []

        for t_noise in stages:
            # Number of slots to reveal as clean at this stage
            n_revealed = round((1.0 - t_noise) * n_slots)
            revealed_set = set(pi[:n_revealed])

            noisy_ids:    List[int]  = []
            tag_labels:   List[int]  = []
            prompt_mask:  List[bool] = []
            pos_to_clean: dict       = {}

            # ── BOS ──────────────────────────────────────────────────────────
            noisy_ids.append(self.tokenizer.bos_id)
            tag_labels.append(K_tok)
            prompt_mask.append(True)

            # ── Prompt section: iMask(KEEP) + token(REPLACE→self) ────────────
            for p_tok in prompt_ids:
                noisy_ids.append(M_tok);  tag_labels.append(K_tok);  prompt_mask.append(True)
                pos = len(noisy_ids)
                noisy_ids.append(p_tok);  tag_labels.append(R_tok);  prompt_mask.append(True)
                pos_to_clean[pos] = p_tok
            noisy_ids.append(M_tok); tag_labels.append(K_tok); prompt_mask.append(True)  # trailing iMask

            # ── Response section: n_slots pairs ──────────────────────────────
            for slot_idx in range(n_slots):
                r_tok      = clean_response_ids[slot_idx]
                next_r_tok = clean_response_ids[slot_idx + 1] if slot_idx + 1 < n_resp else r_tok
                is_last    = slot_idx == n_slots - 1

                if is_last and shortage_k == 2:
                    # Two overflow tokens: expand the last slot into two fresh masks
                    noisy_ids.append(M_tok); tag_labels.append(E_tok); prompt_mask.append(False)

                elif slot_idx in revealed_set:
                    # ── Slot is in the revealed prefix: place the clean token ──
                    noisy_ids.append(r_tok)
                    tag_labels.append(K_tok)
                    prompt_mask.append(False)

                else:
                    # ── Slot is still corrupted: mask or noise ────────────────
                    roll = random.random()
                    # Use t_noise as the effective corruption level for this slot.
                    # Since we already decided via revealed_set which slots are clean,
                    # for the remaining (corrupted) slots we split mask vs noise:
                    #   roll < mask_ratio  → <MASK>  (REPLACE or EXPAND)
                    #   roll >= mask_ratio → noise token (REPLACE)
                    if roll < self.mask_ratio:
                        if random.random() < self.expand_prob:
                            slot_tag = E_tok
                        else:
                            slot_tag = R_tok
                            pos_to_clean[len(noisy_ids)] = r_tok
                        noisy_ids.append(M_tok)
                        tag_labels.append(slot_tag)
                        prompt_mask.append(False)
                    else:
                        noise_tok = self._get_random_noise_token(r_tok)
                        pos_to_clean[len(noisy_ids)] = r_tok
                        noisy_ids.append(noise_tok)
                        tag_labels.append(R_tok)
                        prompt_mask.append(False)

                # ── Structural interleaving mask: INSERT / KEEP ───────────
                if is_last and shortage_k >= 1:
                    pos_to_clean[len(noisy_ids)] = clean_response_ids[n_slots]
                    noisy_ids.append(M_tok); tag_labels.append(I_tok); prompt_mask.append(False)
                elif random.random() < self.insert_prob and slot_idx < n_resp:
                    pos_to_clean[len(noisy_ids)] = next_r_tok
                    noisy_ids.append(M_tok); tag_labels.append(I_tok); prompt_mask.append(False)
                else:
                    noisy_ids.append(M_tok); tag_labels.append(K_tok); prompt_mask.append(False)

            # ── EOS ──────────────────────────────────────────────────────────
            noisy_ids.append(self.tokenizer.eos_id)
            tag_labels.append(K_tok)
            prompt_mask.append(False)

            results.append((noisy_ids, tag_labels, prompt_mask, pos_to_clean))

        return results


    # ── Corruption Steps ───────────────────────────────────────────────

    def _apply_replace(self, ids: List[int]) -> List[int]:
        """Randomly replace some tokens with noise tokens."""
        result = list(ids)
        for i in range(len(result)):
            if random.random() < self.replace_ratio:
                result[i] = random.choice(self.noise_pool)
        return result

    def _apply_delete(self, ids: List[int]) -> List[int]:
        """Randomly delete some tokens."""
        keep_mask = [random.random() >= self.delete_ratio for _ in ids]
        # Ensure we don't delete everything
        if not any(keep_mask) and len(ids) > 0:
            keep_mask[random.randint(0, len(ids) - 1)] = True
        return [tok for tok, keep in zip(ids, keep_mask) if keep]

    def _apply_insert(self, ids: List[int]) -> List[int]:
        """Insert random noise tokens at random positions."""
        result = []
        for tok in ids:
            if random.random() < self.insert_ratio:
                # Insert a noise token before this token
                result.append(random.choice(self.noise_pool))
            result.append(tok)
        return result

    def _apply_expand(self, ids: List[int]) -> List[int]:
        """
        Merge random consecutive spans into a single [EXPAND] token.
        A span of 2+ tokens is replaced by a single [EXPAND].
        During training this teaches the model that [EXPAND] means
        'duplicate me into two [MASK] tokens'.
        """
        if len(ids) < 3:
            return ids  # Need at least 3 tokens to merge a span meaningfully

        result = list(ids)
        i = 0
        while i < len(result) - 1:
            if random.random() < self.expand_ratio:
                # Choose a span length (2-4 tokens)
                span_len = random.randint(2, min(4, len(result) - i))
                # Replace the span with a single [EXPAND] token
                result = result[:i] + [self.tokenizer.expand_id] + result[i + span_len:]
            i += 1
        return result

    def _apply_mask(self, ids: List[int]) -> List[int]:
        """Randomly replace some tokens with <MASK> tokens.
        
        This is critical: it teaches the model that <MASK> means
        'predict text here' (tag as REPLACE, generate a word).
        Without this, the model never sees MASK during training and
        doesn't know what to do with MASK tokens at inference time.
        """
        result = list(ids)
        for i in range(len(result)):
            if random.random() < self.mask_ratio:
                result[i] = self.tokenizer.mask_id
        return result

    # ── Levenshtein Alignment ──────────────────────────────────────────

    def _align(self, noisy_ids: List[int], clean_ids: List[int]) -> List[int]:
        """
        Compute the optimal Levenshtein edit path from noisy -> clean
        and produce per-token tag labels for the noisy sequence.

        Uses character-level alignment by mapping each token ID to a
        unique unicode character, then translating the edit operations back.

        Returns:
            List of token IDs (keep_id, delete_id, replace_id, insert_id,
            expand_id) — one per position in `noisy_ids`.
        """
        # Map token IDs to unique characters for string alignment
        noisy_str, noisy_map = self._ids_to_string(noisy_ids)
        clean_str, clean_map = self._ids_to_string(clean_ids)

        # Get Levenshtein edit operations
        edit_ops = Levenshtein.editops(noisy_str, clean_str)
        # edit_ops is a list of (operation, pos_in_noisy, pos_in_clean)
        # operation is one of: 'insert', 'delete', 'replace'

        # Initialize all positions as KEEP
        labels = [self.tokenizer.keep_id] * len(noisy_ids)

        # Track which noisy positions have been accounted for
        matched_noisy = set()

        for op, noisy_pos, clean_pos in edit_ops:
            if op == "replace":
                labels[noisy_pos] = self.tokenizer.replace_id
                matched_noisy.add(noisy_pos)
            elif op == "delete":
                labels[noisy_pos] = self.tokenizer.delete_id
                matched_noisy.add(noisy_pos)
            elif op == "insert":
                # Insert means the clean sequence has an extra token at clean_pos
                # that doesn't exist in noisy. After processing, we treat this as:
                # the noisy position BEFORE the insertion point gets flagged.
                # Actually, in our tag scheme, INSERT means "insert new content here."
                # We handle this by marking the position where insertion is needed.
                # If insert happens at the end, mark the last position.
                if noisy_pos < len(labels):
                    labels[noisy_pos] = self.tokenizer.insert_id

        # Handle any EXPAND tokens in noisy that map to multiple clean tokens
        # The Levenshtein lib will see the EXPAND character as a single deletion
        # but we need to recognize it as an EXPAND operation
        for i, tok_id in enumerate(noisy_ids):
            if tok_id == self.tokenizer.expand_id:
                labels[i] = self.tokenizer.expand_id

        # Force MASK tokens to be tagged as REPLACE
        # (Levenshtein might not always detect MASK → REPLACE correctly
        #  since MASK doesn't match any clean token)
        for i, tok_id in enumerate(noisy_ids):
            if tok_id == self.tokenizer.mask_id:
                labels[i] = self.tokenizer.replace_id

        return labels

    def _ids_to_string(self, ids: List[int]) -> Tuple[str, dict]:
        """
        Map token IDs to unique unicode characters for Levenshtein alignment.

        Uses characters from the Private Use Area (U+E000+) to avoid collisions
        with actual text. Returns the string and a reverse mapping.
        """
        char_map = {}
        chars = []
        for tid in ids:
            if tid not in char_map:
                # Assign a new private-use character
                char_map[tid] = chr(0xE000 + len(char_map))
            chars.append(char_map[tid])
        return "".join(chars), char_map

    def compute_tag_labels(
        self, noisy_ids: List[int], clean_ids: List[int]
    ) -> List[int]:
        """
        Public method: compute edit tag labels for a (noisy, clean) pair.
        Wraps _align() for external use.
        """
        return self._align(noisy_ids, clean_ids)
