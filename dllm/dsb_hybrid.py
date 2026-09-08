"""
Auxiliary-head hybrid: Diffusion Schrödinger Bridge + discrete edit heads.

Keeps the continuous SDE (DSB) generative backbone from ``dsb.py`` and bolts on
the DLLM-style discrete edit supervision:

  * TaggerHead    — predicts the edit op (KEEP/DELETE/REPLACE/INSERT/EXPAND)
                    per position, exactly like the DLLM tagger.
  * GeneratorHead — predicts the clean token at REPLACE positions, like the
                    DLLM generator head.

The heads operate on the SDE *intermediate embedding* ``x_t`` (the point on the
bridge, not a discrete token canvas). The model therefore jointly learns:

  * the continuous transport DP1 -> DP2 (denoising score matching), and
  * the discrete edit structure needed to reverse token-level corruption
    (cross-entropy on edit tags and clean tokens).

Loss = score_matching + lambda_tag * tagger_ce + lambda_gen * generator_ce

Design note (phase 1): the token corruption here is FIXED-LENGTH and in-place
(mask/REPLACE only), so the noisy token sequence is guaranteed to be the same
length as the clean sequence. That makes the per-position edit/gen labels align
1:1 with the fixed ``S`` embedding canvas at no extra cost. The variable-length
grammar (INSERT/DELETE/EXPAND/Levenshtein) is intentionally deferred to phase 2
(see the docs/phase-2 section below), where it composes with the edit heads.

Phase-2 sketch (edit-aware SDE):
  1. Corrupt clean text with the full Levenshtein edit grammar (ForwardCorruptor.
     corrupt), producing variable-length noisy sequences and rich tag labels.
  2. Align each noisy position to an embedding slot (pad to S) and derive gen
     labels from the edit ops (as DLLM's dataset.py does).
  3. Defer INSERT/DELETE/EXPAND bookkeeping into the SDE: either grow/trim the
     canvas before the next refinement step, or penalize length-changing edits
     through the score network. Cross-condition the score on last-step edit-tag
     predictions so discrete structure informs the continuous drift.
"""

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.dsb import DiffSchrodingerBridge  # concrete type for annotations

# Edit tag indices, matching the DLLM tagger ordering.
KEEP, DELETE, REPLACE, INSERT, EXPAND = 0, 1, 2, 3, 4
NUM_TAGS = 5
TAG_NAMES = ["KEEP", "DELETE", "REPLACE", "INSERT", "EXPAND"]


# ── Edit-conditioned score network (Phase 2) ─────────────────────────────────

class EditConditionedScoreNet(nn.Module):
    """
    Score network that conditions the continuous drift on discrete edit tags.

    Unlike ``MLPScoreNet``, it takes an extra per-position ``tag_ids`` channel
    (the previous step's edit-op predictions, or ground-truth at train time) so
    the discrete structure informs the continuous reverse SDE. Tags are embedded
    to a vector and concatenated with the point ``x`` and time embedding.
    """

    def __init__(
        self,
        dim: int,
        num_tags: int = NUM_TAGS,
        hidden_dim: int = 512,
        num_layers: int = 3,
        time_embed_dim: int = 128,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_tags = num_tags
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.tag_emb = nn.Embedding(num_tags + 1, dim)  # +1 for a 'none' sentinel
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        in_dim = dim + cond_dim + dim + time_embed_dim  # cond + x + tag_emb + time
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                tag_ids: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, S, D) or (B, D); t: (B,); tag_ids: (B, S) int64 (default: sentinel);
        cond: optional conditioning with the same shape as x (e.g. DP1).
        """
        t_b = t.reshape(-1, 1)
        t_emb = self.time_mlp(t_b)  # (B, time_embed_dim)
        if x.dim() == 3:
            t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # (B, S, T)
            if tag_ids is None:
                tag_ids = torch.full((x.shape[0], x.shape[1]), self.num_tags,
                                     dtype=torch.long, device=x.device)
            tag_e = self.tag_emb(tag_ids)  # (B, S, D)
            h = torch.cat([x, tag_e, t_emb], dim=-1)
        else:
            t_emb = t_emb.expand(x.shape[0], -1)
            if tag_ids is None:
                tag_ids = torch.full((x.shape[0],), self.num_tags,
                                     dtype=torch.long, device=x.device)
            tag_e = self.tag_emb(tag_ids)  # (B, D)
            h = torch.cat([x, tag_e, t_emb], dim=-1)
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("score net built with cond_dim > 0 requires cond")
            h = torch.cat([cond, h], dim=-1)
        return self.net(h)


# ── Multi-Route Blob Corruption Engine ───────────────────────────────────────

QWERTY_NEIGHBORS = {
    'q': 'wa', 'w': 'qase', 'e': 'wsdr', 'r': 'edft', 't': 'rfgy',
    'y': 'tghu', 'u': 'yhji', 'i': 'ujko', 'o': 'iklp', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
    'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk'
}

def _perturb_word_typo(word: str, rng: random.Random) -> str:
    """Apply realistic keyboard neighbor, transposition, or omission typo."""
    if len(word) <= 3:
        return word
    roll = rng.random()
    if roll < 0.35 and len(word) >= 4:
        # Swap adjacent letters (transposition)
        idx = rng.randint(1, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:]
    elif roll < 0.70:
        # Substitute QWERTY adjacent key
        idx = rng.randint(0, len(word) - 1)
        c = word[idx].lower()
        if c in QWERTY_NEIGHBORS:
            n = rng.choice(QWERTY_NEIGHBORS[c])
            return word[:idx] + (n.upper() if word[idx].isupper() else n) + word[idx + 1:]
    else:
        # Drop character
        idx = rng.randint(1, len(word) - 1)
        return word[:idx] + word[idx + 1:]
    return word

def _perturb_word_morph(word: str) -> str:
    """Apply inflection / grammatical shift (tense, number, suffix)."""
    w_low = word.lower()
    if w_low.endswith('ing') and len(w_low) > 4:
        return word[:-3] + 'ed'
    elif w_low.endswith('ed') and len(w_low) > 3:
        return word[:-2] + 'ing'
    elif w_low.endswith('s') and len(w_low) > 3 and not w_low.endswith('ss'):
        return word[:-1]
    elif len(w_low) >= 3 and not w_low.endswith('s'):
        return word + 's'
    return word

def _is_syntax_token(tok: int, tokenizer: Optional[Any] = None) -> bool:
    """Check if token is structural syntax (indentation, brackets, separators) that should not be corrupted into stutters."""
    if tokenizer is None:
        return False
    try:
        w = tokenizer.decode([tok])
        if not w:
            return True
        # Pure whitespace or indentation (spaces, tabs, newlines,   )
        if w.isspace() or set(w).issubset({' ', '\t', '\n', '\r', ' '}):
            return True
        # Common structural delimiters and code symbols
        stripped = w.strip()
        if stripped in {'(', ')', '[', ']', '{', '}', ';', ',', ':', '=', '+', '-', '*', '/', '#', '<', '>', '.', '"', "'", '`', '|', '&', '!', '?', '\\', '%'}:
            return True
        # Comments / banners (repeated dashes, slashes, hashes)
        if len(stripped) >= 2 and len(set(stripped)) == 1 and stripped[0] in {'-', '=', '/', '#', '*'}:
            return True
    except Exception:
        pass
    return False


def apply_collapse_drift(
    dp2: torch.Tensor,
    del_mask: torch.Tensor,
    attn: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Interpolated Collapse Drift for continuous Diffusion Schrödinger Bridge.

    For positions marked for DELETE (del_mask == True), setting DP2 == DP1 causes
    zero drift in the continuous SDE (u_target = DP2 - DP1 = 0), which aliases
    DELETE with KEEP and freezes spurious stutters into the continuous trajectory.

    This function replaces DP2[b, i] at DELETE positions with the smooth geometric
    interpolation between its nearest valid non-deleted clean neighbors:
        DP2[b, i] = (1 - alpha_i) * DP2[b, left] + alpha_i * DP2[b, right]

    This gives the continuous bridge a non-zero, directional contraction vector
    pulling the spurious representation directly into the boundary seam between
    its valid neighbors.
    """
    if not del_mask.any():
        return dp2

    out_dp2 = dp2.clone()
    B, S, D = dp2.shape
    valid_mask = (attn == 1) if attn is not None else torch.ones((B, S), dtype=torch.bool, device=dp2.device)
    active_del = del_mask & valid_mask

    for b in range(B):
        row_del = active_del[b]
        if not row_del.any():
            continue

        indices = torch.where(row_del)[0].tolist()
        spans = []
        start = indices[0]
        prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                spans.append((start, prev))
                start = idx
                prev = idx
        spans.append((start, prev))

        for l, r in spans:
            left_idx = l - 1
            while left_idx >= 0 and (row_del[left_idx] or not valid_mask[b, left_idx]):
                left_idx -= 1

            right_idx = r + 1
            while right_idx < S and (row_del[right_idx] or not valid_mask[b, right_idx]):
                right_idx += 1

            has_left = (left_idx >= 0)
            has_right = (right_idx < S)

            if has_left and has_right:
                v_l = dp2[b, left_idx]
                v_r = dp2[b, right_idx]
                span_len = (right_idx - left_idx)
                for i in range(l, r + 1):
                    alpha = (i - left_idx) / span_len
                    out_dp2[b, i] = (1.0 - alpha) * v_l + alpha * v_r
            elif has_left:
                out_dp2[b, l:r+1] = dp2[b, left_idx]
            elif has_right:
                out_dp2[b, l:r+1] = dp2[b, right_idx]

    return out_dp2


def corrupt_multiroute(
    clean_ids: List[int],
    tokenizer: Optional[Any] = None,
    mask_prob: float = 0.35,
    mask_ratio: float = 0.25,
    replace_ratio: float = 0.15,
    delete_ratio: float = 0.08,
    insert_ratio: float = 0.08,
    expand_ratio: float = 0.05,
    stutter_prob: float = 0.30,
    burst_mask_prob: float = 0.20,
    burst_mask_max_len: int = 8,
    burst_insert_prob: float = 0.15,
    burst_insert_max_len: int = 6,
    burst_expand_prob: float = 0.10,
    burst_expand_max_len: int = 4,
    burst_stutter_prob: float = 0.25,
    burst_stutter_max_len: int = 20,
    noise_pool: Optional[List[int]] = None,
    mask_id: int = 250001,
    expand_id: int = 50269,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Non-destructive Multi-Route & Burst Blob Corruption Engine.
    Generates balanced distributions across all 5 edit operations with contiguous bursts:
      - BURST MASK DEMASKS: Multi-token mask spans (L in [2, burst_mask_max_len]) -> REPLACE
      - BURST INSERTION: Omitted clauses/tokens -> subsequent anchor token tagged INSERT with gen = omitted
      - BURST EXPANSION: Multi-word compressed spans (L in [2, burst_expand_max_len]) -> EXPAND
      - BURST STUTTERS: Degenerate repetition loops up to 20 tokens -> Dual-route:
          * 50% Route A (DELETE): spurious duplicate loops inserted without destroying clean words.
          * 50% Route B (REPLACE): degenerate loops covering semantic phrases.
      - SYNTAX / CODE GUARD: Structural syntax (indentation, brackets, separators) is strictly
        protected from artificial stutter corruption, preserving legitimate code syntax.
      - RESIDUAL TOKEN PERTURBATIONS: QWERTY typos, morphological inflections, and distractor noise.

    Returns (noisy_ids, clean_aligned_ids, tag_labels, gen_targets), all 1:1 aligned
    to the exact canvas length.
    """
    rng = rng if rng is not None else random.Random()
    n_tokens = len(clean_ids)
    if n_tokens <= 2:
        return list(clean_ids), list(clean_ids), [KEEP] * n_tokens, [-100] * n_tokens

    # Plan modifications per clean token index.
    # plan[i] = None (default residual), or special directive dict
    plan = [None] * n_tokens
    stutter_placed = False

    # ── Phase 1: Burst Stutters ──────────────────────────────────────────────
    if burst_stutter_prob > 0.0 and rng.random() < burst_stutter_prob and n_tokens > 6:
        candidates = [
            i for i in range(1, n_tokens - 2)
            if clean_ids[i] != mask_id and not _is_syntax_token(clean_ids[i], tokenizer)
        ]
        if candidates:
            idx = rng.choice(candidates)
            run_weights = [35, 25, 15, 10, 5, 4, 3, 2, 1]
            run_choices = [2, 3, 4, 5, 6, 8, 10, 15, 20]
            k = rng.choices(run_choices, weights=run_weights)[0]
            k = min(k, burst_stutter_max_len)
            route_a = (rng.random() < 0.5)

            if route_a:
                # Route A: Spurious duplicate loop -> insert extra duplicate tokens (DELETE)
                # Subsequent clean words are NOT destroyed!
                plan[idx] = {"op": "burst_stutter_a", "k": k, "tok": clean_ids[idx]}
                stutter_placed = True
            else:
                # Route B: In-place repetition covering clean words -> REPLACE
                max_b = min(k, n_tokens - 1 - idx)
                if max_b >= 2:
                    for j in range(idx + 1, idx + max_b):
                        plan[j] = {"op": "replace_stutter_b", "tok": clean_ids[idx]}
                    stutter_placed = True

    # ── Phase 2: Burst Mask Demask Spans ─────────────────────────────────────
    if burst_mask_prob > 0.0 and rng.random() < burst_mask_prob and n_tokens > 6:
        max_L = min(burst_mask_max_len, n_tokens - 2)
        if max_L >= 2:
            L = rng.randint(2, max_L)
            possible_starts = [
                i for i in range(1, n_tokens - L)
                if all(plan[i + off] is None and not _is_syntax_token(clean_ids[i + off], tokenizer) for off in range(L))
            ]
            if possible_starts:
                start_i = rng.choice(possible_starts)
                for j in range(start_i, start_i + L):
                    plan[j] = {"op": "mask_span"}

    # ── Phase 3: Burst Insertion Spans (Omitted clean tokens -> INSERT) ──────
    if burst_insert_prob > 0.0 and rng.random() < burst_insert_prob and n_tokens > 8:
        # A single omitted clean token taught as INSERT at the subsequent anchor token
        candidates = [
            i for i in range(1, n_tokens - 2)
            if plan[i] is None and plan[i + 1] is None
            and not _is_syntax_token(clean_ids[i], tokenizer)
        ]
        if candidates:
            idx = rng.choice(candidates)
            # Omit clean_ids[idx], mark clean_ids[idx+1] as INSERT with gen = clean_ids[idx]
            plan[idx] = {"op": "omitted_for_insert"}
            plan[idx + 1] = {"op": "anchor_insert", "omitted_tok": clean_ids[idx]}

    # ── Phase 4: Burst Expansion Spans ───────────────────────────────────────
    if burst_expand_prob > 0.0 and rng.random() < burst_expand_prob and n_tokens > 8:
        max_L = min(burst_expand_max_len, 4)
        candidates = [
            i for i in range(1, n_tokens - max_L)
            if all(plan[i + off] is None and not _is_syntax_token(clean_ids[i + off], tokenizer) for off in range(max_L))
        ]
        if candidates:
            idx = rng.choice(candidates)
            # Compress clean_ids[idx : idx + max_L] into a single expand_id slot
            plan[idx] = {"op": "expand_start", "span_len": max_L}
            for off in range(1, max_L):
                plan[idx + off] = {"op": "expand_skip"}

    # ── Phase 5: Fallback Pairwise Adjacent Stutter ──────────────────────────
    if not stutter_placed and stutter_prob > 0.0 and rng.random() < stutter_prob and n_tokens > 4:
        candidates = [
            i for i in range(1, n_tokens - 1)
            if plan[i] is None and plan[i - 1] is None
            and not _is_syntax_token(clean_ids[i - 1], tokenizer)
        ]
        if candidates:
            idx = rng.choice(candidates)
            plan[idx] = {"op": "adjacent_stutter", "tok": clean_ids[idx - 1]}

    # ── Assembly: Build 1:1 aligned outputs ──────────────────────────────────
    noisy: List[int] = []
    clean_aligned: List[int] = []
    tags: List[int] = []
    gen: List[int] = []

    typo_cache = {}

    i = 0
    while i < n_tokens:
        p = plan[i]
        clean_tok = clean_ids[i]

        if p is not None:
            op = p["op"]
            if op == "burst_stutter_a":
                # Clean token itself (KEEP)
                noisy.append(clean_tok)
                clean_aligned.append(clean_tok)
                tags.append(KEEP)
                gen.append(-100)
                # Extra duplicate tokens (DELETE)
                stutter_k = p["k"]
                stutter_tok = p["tok"]
                for _ in range(stutter_k - 1):
                    noisy.append(stutter_tok)
                    clean_aligned.append(stutter_tok)
                    tags.append(DELETE)
                    gen.append(-100)
                i += 1
                continue

            elif op == "replace_stutter_b":
                noisy.append(p["tok"])
                clean_aligned.append(clean_tok)
                tags.append(REPLACE)
                gen.append(clean_tok)
                i += 1
                continue

            elif op == "mask_span":
                noisy.append(mask_id)
                clean_aligned.append(clean_tok)
                tags.append(REPLACE)
                gen.append(clean_tok)
                i += 1
                continue

            elif op == "omitted_for_insert":
                # Omitted from noisy!
                i += 1
                continue

            elif op == "anchor_insert":
                # This token is present, and omitted_tok should be inserted before it
                noisy.append(clean_tok)
                clean_aligned.append(clean_tok)
                tags.append(INSERT)
                gen.append(p.get("omitted_tok", -100))
                i += 1
                continue

            elif op == "expand_start":
                # Compressed slot
                noisy.append(expand_id if expand_id is not None else mask_id)
                clean_aligned.append(clean_tok)
                tags.append(EXPAND)
                gen.append(clean_tok)
                i += 1
                continue

            elif op == "expand_skip":
                # Absorbed into expansion
                i += 1
                continue

            elif op == "adjacent_stutter":
                # Insert duplicate token before clean_tok
                dup_tok = p["tok"]
                noisy.append(dup_tok)
                clean_aligned.append(dup_tok)
                tags.append(DELETE)
                gen.append(-100)
                # Then the clean token itself
                noisy.append(clean_tok)
                clean_aligned.append(clean_tok)
                tags.append(KEEP)
                gen.append(-100)
                i += 1
                continue

        # Residual fine-grained perturbations
        if _is_syntax_token(clean_tok, tokenizer) or clean_tok == mask_id:
            noisy.append(clean_tok)
            clean_aligned.append(clean_tok)
            tags.append(KEEP)
            gen.append(-100)
            i += 1
            continue

        roll = rng.random()
        # 1. Single Mask -> REPLACE
        if roll < mask_prob * mask_ratio:
            noisy.append(mask_id)
            clean_aligned.append(clean_tok)
            tags.append(REPLACE)
            gen.append(clean_tok)

        # 2. Typos & Morphological Inflections -> REPLACE
        elif roll < mask_prob * mask_ratio + replace_ratio * 0.6:
            perturbed_tok = None
            if tokenizer is not None:
                if clean_tok not in typo_cache:
                    try:
                        w = tokenizer.decode([clean_tok]).strip()
                        if len(w) >= 3:
                            w_p = _perturb_word_typo(w, rng) if rng.random() < 0.6 else _perturb_word_morph(w)
                            enc = tokenizer.encode(w_p, add_special_tokens=False)
                            if enc:
                                typo_cache[clean_tok] = enc[0]
                    except Exception:
                        pass
                perturbed_tok = typo_cache.get(clean_tok)
            if perturbed_tok is None and noise_pool:
                perturbed_tok = rng.choice(noise_pool)
            noisy.append(perturbed_tok if perturbed_tok is not None else mask_id)
            clean_aligned.append(clean_tok)
            tags.append(REPLACE)
            gen.append(clean_tok)

        # 3. Real dictionary word distractor -> REPLACE
        elif roll < mask_prob * mask_ratio + replace_ratio:
            noisy.append(rng.choice(noise_pool) if noise_pool else mask_id)
            clean_aligned.append(clean_tok)
            tags.append(REPLACE)
            gen.append(clean_tok)

        # 4. Intrusive distractor token -> DELETE
        elif roll < mask_prob * mask_ratio + replace_ratio + delete_ratio:
            dist_tok = rng.choice(noise_pool) if noise_pool else mask_id
            noisy.append(dist_tok)
            clean_aligned.append(dist_tok)
            tags.append(DELETE)
            gen.append(-100)
            # The clean token remains present as KEEP
            noisy.append(clean_tok)
            clean_aligned.append(clean_tok)
            tags.append(KEEP)
            gen.append(-100)

        elif roll < mask_prob * mask_ratio + replace_ratio + delete_ratio + insert_ratio and (i + 1) < n_tokens:
            next_tok = clean_ids[i + 1]
            # Omit clean_tok, tag next_tok as INSERT (structural edit op) with target = omitted clean_tok
            noisy.append(next_tok)
            clean_aligned.append(next_tok)
            tags.append(INSERT)
            gen.append(clean_tok)
            i += 2  # Consumed both i and i+1
            continue

        # 6. Default: KEEP
        else:
            noisy.append(clean_tok)
            clean_aligned.append(clean_tok)
            tags.append(KEEP)
            gen.append(-100)

        i += 1

    return noisy, clean_aligned, tags, gen


# ── Fixed-length, in-place token corruption ──────────────────────────────────

def corrupt_fixed(
    clean_ids: List[int],
    mask_prob: float = 0.15,
    mask_ratio: float = 0.8,
    noise_pool: Optional[List[int]] = None,
    mask_id: int = 0,
    rng: Optional[random.Random] = None,
    stutter_prob: float = 0.0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Corrupt a clean token sequence in place (same length returned).

    Each real position is either kept (tag = KEEP), replaced with ``<MASK>``,
    or replaced with a random noise token. Both replacement kinds get
    tag = REPLACE and a generator target = the original clean token.

    ``stutter_prob`` controls the probability that ONE random non-special
    position is overwritten with the token immediately preceding it, creating
    an adjacent duplicate (e.g. "the the", "pack pack"). The position gets
    tag = REPLACE and gen = the original clean token. This trains the model
    to recognise and fix stuttering errors — a corruption type absent from
    mask/random-word corruption.

    Returns (noisy_ids, tag_labels, gen_targets), all aligned to ``clean_ids``.
    """
    rng = rng if rng is not None else random.Random()
    noisy = list(clean_ids)
    tags = [KEEP] * len(noisy)
    gen = [-1] * len(noisy)  # -1 -> ignored by CE (use -100 after loss setup)
    for i in range(len(noisy)):
        if noisy[i] == mask_id:
            # Already a mask token (e.g. from padding) — leave it.
            continue
        if rng.random() < mask_prob:
            if rng.random() < mask_ratio:
                noisy[i] = mask_id
            elif noise_pool:
                noisy[i] = rng.choice(noise_pool)
            tags[i] = REPLACE
            gen[i] = clean_ids[i]

    # Stutter injection: replace ONE position with the previous token so that
    # position i has token == token at position i-1 (adjacent duplicate).
    # Use MASK token as the noisy value so GenHead sees the same masked embedding
    # as in normal REPLACE training, rather than a clean-word embedding.
    if stutter_prob > 0.0 and rng.random() < stutter_prob:
        # Collect eligible positions: not special, not already corrupted,
        # has a real non-special left neighbour.
        candidates = [
            i for i in range(1, len(noisy) - 1)
            if clean_ids[i] != mask_id
            and tags[i] == KEEP          # don't double-corrupt
            and clean_ids[i - 1] != mask_id
        ]
        if candidates:
            idx = rng.choice(candidates)
            # Overwrite with MASK (keeps same distribution as normal REPLACE
            # training so GenHead learns to decode from a masked embedding).
            noisy[idx] = mask_id
            tags[idx] = REPLACE
            gen[idx] = clean_ids[idx]   # target = the word that SHOULD be here

    return noisy, tags, gen


# ── Discrete heads (operate per-position on the SDE embedding x_t) ───────────

class TaggerHead(nn.Module):
    """
    Predict the edit op (NUM_TAGS classes) at every position.

    Conditions on the noise level t and the source embedding DP1: without t
    the head cannot distinguish "x looks clean because t~0" from "this
    position was never corrupted", so its Bayes-optimal prediction collapses
    to the class prior (observed: tag loss stuck at ~0.4 for an entire run).
    With DP1 it can compare the current state against the source — which is
    also what makes the head usable at generation time (t=1, cond=DP1).
    """

    def __init__(
        self,
        dim: int,
        time_embed_dim: int = 128,
        cond_dim: int = 0,
        subspace_factorization: bool = False,
        macro_dim: int = 512,
        lexical_dim: int = 256,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.subspace_factorization = subspace_factorization
        self.macro_dim = macro_dim
        self.lexical_dim = lexical_dim
        if time_embed_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
            )
        else:
            self.time_mlp = None
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + time_embed_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, NUM_TAGS),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, D); t: (B,); cond: (B, S, D) e.g. DP1
        h = x
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("head built with cond_dim > 0 requires cond")
            h = torch.cat([cond, h], dim=-1)
        if self.time_mlp is not None:
            t_emb = self.time_mlp(t.reshape(-1, 1))          # (B, T)
            t_emb = t_emb.unsqueeze(1).expand(-1, h.shape[1], -1)
            h = torch.cat([h, t_emb], dim=-1)
        return self.net(h)


def select_blob_candidates(
    continuous_states: torch.Tensor,
    embed_weight: torch.Tensor,
    blob_size: int = 512,
    targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Selects the localized candidate semantic blob B_M(x) around the continuous states.
    Prunes the vocabulary down to the M most similar tokens in embedding space.
    When targets is provided, guarantees the ground-truth target is in the candidate set.

    Args:
        continuous_states: (N, D) continuous representations at active edit slots.
        embed_weight: (V, D) vocabulary token embedding / decoder matrix.
        blob_size: size M of candidate blob (e.g. 512).
        targets: (N,) optional ground truth token IDs for active-set guarantee.

    Returns:
        candidate_ids: (N, M) candidate token IDs.
    """
    N, D = continuous_states.shape
    V = embed_weight.shape[0]
    M = min(blob_size, V)

    norm_x = F.normalize(continuous_states.detach().to(embed_weight.device, dtype=torch.float32), dim=-1)
    norm_w = F.normalize(embed_weight.detach().to(dtype=torch.float32), dim=-1)
    cos_sims = norm_x @ norm_w.T  # (N, V)

    cand_ids = torch.topk(cos_sims, M, dim=-1).indices  # (N, M)

    if targets is not None:
        tgt = targets.view(-1, 1)
        in_blob = (cand_ids == tgt).any(dim=-1)  # (N,)
        if not in_blob.all():
            missing = (~in_blob).nonzero(as_tuple=True)[0]
            cand_ids[missing, -1] = targets[missing]

    return cand_ids


class GenHead(nn.Module):
    """Predict the clean token at REPLACE positions (sparse projection).

    Same t + DP1 conditioning rationale as ``TaggerHead``.
    When lm_head is provided (from AutoModelForMaskedLM), initializes weights from
    the pretrained LM head (dense, layer_norm, decoder weight, and vocab bias),
    giving instant zero-shot MLM prediction quality on Layer 12 contextual hidden states.
    When embed_weight is provided without lm_head, initializes/ties the final projection.

    Subspace Factorization:
    When subspace_factorization is enabled, the input embedding is partitioned into:
      - x_macro (0:macro_dim): broad syntactic and semantic basin
      - x_lexical (macro_dim:end): fine-grained lexical coordinates
    An angular margin projection (CosFace style) is applied to enforce wide separation
    between co-hyponyms in the lexical subspace, preventing adjacent Voronoi cell spillage.

    Blob-Restricted Diffusion:
    When candidate_ids is provided, restricts projection and probability distribution
    to the local semantic blob B_M(x) (e.g. M=512 tokens), reducing candidate space
    by ~500x and eliminating out-of-domain co-hyponym drift.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        time_embed_dim: int = 128,
        cond_dim: int = 0,
        embed_weight: Optional[torch.Tensor] = None,
        tie_weights: bool = True,
        lm_head: Optional[nn.Module] = None,
        subspace_factorization: bool = False,
        macro_dim: int = 512,
        lexical_dim: int = 256,
        angular_margin: float = 0.05,
        margin_scale: float = 64.0,
        op_embed_dim: int = 0,
        num_tags: int = NUM_TAGS,
        contextual_gen: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
        self.subspace_factorization = subspace_factorization
        self.macro_dim = macro_dim
        self.lexical_dim = lexical_dim
        self.angular_margin = angular_margin
        self.margin_scale = margin_scale
        self.op_embed_dim = op_embed_dim
        self.num_tags = num_tags
        self.contextual_gen = contextual_gen

        if time_embed_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.SiLU(),
            )
        else:
            self.time_mlp = None

        if op_embed_dim > 0:
            self.op_emb = nn.Embedding(num_tags + 1, op_embed_dim)  # +1 for none/unknown
            nn.init.zeros_(self.op_emb.weight)
        else:
            self.op_emb = None

        if contextual_gen:
            self.ctx_proj = nn.Linear(dim, dim)
            nn.init.zeros_(self.ctx_proj.weight)
            if self.ctx_proj.bias is not None:
                nn.init.zeros_(self.ctx_proj.bias)
        else:
            self.ctx_proj = None

        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + time_embed_dim + op_embed_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
        )

        if subspace_factorization:
            # Dedicated residual projection for fine-grained lexical coordinates
            self.lex_proj = nn.Linear(lexical_dim, dim, bias=False)
            nn.init.zeros_(self.lex_proj.weight)
        else:
            self.lex_proj = None

        if lm_head is not None:
            with torch.no_grad():
                # Zero out weights for cond, time, and op slices initially so GenHead
                # behaves identically to the pretrained lm_head on x at step 0
                self.net[0].weight.zero_()
                # When cond_dim > 0, cond occupies 0:cond_dim and x occupies cond_dim:cond_dim+dim
                x_start = self.cond_dim
                if hasattr(lm_head, "dense") and hasattr(lm_head.dense, "weight"):
                    self.net[0].weight[:, x_start : x_start + dim].copy_(lm_head.dense.weight)
                    if hasattr(lm_head.dense, "bias") and lm_head.dense.bias is not None:
                        self.net[0].bias.copy_(lm_head.dense.bias)
                if hasattr(lm_head, "layer_norm") and hasattr(lm_head.layer_norm, "weight"):
                    self.net[2].weight.copy_(lm_head.layer_norm.weight)
                    if hasattr(lm_head.layer_norm, "bias") and lm_head.layer_norm.bias is not None:
                        self.net[2].bias.copy_(lm_head.layer_norm.bias)
                if hasattr(lm_head, "decoder") and hasattr(lm_head.decoder, "weight"):
                    self.net[3].weight.copy_(lm_head.decoder.weight)
                elif embed_weight is not None:
                    self.net[3].weight.copy_(embed_weight)
                if hasattr(lm_head, "bias") and lm_head.bias is not None:
                    self.net[3].bias.copy_(lm_head.bias)
                elif hasattr(lm_head, "decoder") and hasattr(lm_head.decoder, "bias") and lm_head.decoder.bias is not None:
                    self.net[3].bias.copy_(lm_head.decoder.bias)
            if tie_weights:
                if hasattr(lm_head, "decoder") and hasattr(lm_head.decoder, "weight"):
                    self.net[3].weight = lm_head.decoder.weight
                elif embed_weight is not None:
                    self.net[3].weight = embed_weight
        elif embed_weight is not None:
            with torch.no_grad():
                self.net[3].weight.copy_(embed_weight)
            if tie_weights:
                self.net[3].weight = embed_weight

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        op_ids: Optional[torch.Tensor] = None,
        candidate_ids: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        return_features: bool = False,
        blob_size: Optional[int] = None,
        return_candidates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        # x: (N, D) selected positions; t: (N,); cond: (N, D); op_ids: (N,)
        h = x
        if context is not None:
            if getattr(self, "ctx_proj", None) is not None:
                h = h + self.ctx_proj(context)
            else:
                h = h + context

        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("head built with cond_dim > 0 requires cond")
            h = torch.cat([cond, h], dim=-1)
        if self.time_mlp is not None:
            t_emb = self.time_mlp(t.reshape(-1, 1))          # (N, T)
            h = torch.cat([h, t_emb], dim=-1)
        if self.op_emb is not None:
            if op_ids is None:
                op_ids = torch.full((x.shape[0],), self.num_tags, dtype=torch.long, device=x.device)
            else:
                op_ids = torch.where(op_ids < 0, torch.full_like(op_ids, self.num_tags), op_ids)
            op_e = self.op_emb(op_ids.clamp(0, self.num_tags))
            h = torch.cat([h, op_e], dim=-1)

        # Base hidden representation through dense + GELU + LayerNorm
        feat = self.net[2](self.net[1](self.net[0](h)))

        # Subspace enhancement: inject fine lexical coordinates
        if self.subspace_factorization and self.lex_proj is not None:
            x_lex = x[:, self.macro_dim:]
            feat = feat + self.lex_proj(x_lex)

        w_dec = self.net[3].weight  # (V, D)
        b_dec = self.net[3].bias    # (V,) or None
        raw_blob_hit = None
        selected_internal = False

        if candidate_ids is None and blob_size is not None and blob_size < self.vocab_size:
            # Candidate selection performed on the aligned semantic feature 'feat'
            with torch.no_grad():
                cand_scores = feat.detach() @ w_dec.T
                if b_dec is not None:
                    cand_scores = cand_scores + b_dec
                candidate_ids = torch.topk(cand_scores, min(blob_size, self.vocab_size), dim=-1).indices
                selected_internal = True
                if targets is not None:
                    in_blob = (candidate_ids == targets.view(-1, 1)).any(dim=-1)
                    raw_blob_hit = in_blob.float().mean().item()
                    if not in_blob.all():
                        candidate_ids = candidate_ids.clone()
                        candidate_ids[~in_blob, -1] = targets[~in_blob]

        if candidate_ids is not None:
            # Localized projection over candidate blob B_M(x):
            # candidate_ids: (N, M)
            w_cand = w_dec[candidate_ids]  # (N, M, D)
            logits = torch.bmm(w_cand, feat.unsqueeze(-1)).squeeze(-1)  # (N, M)
            if b_dec is not None:
                b_cand = b_dec[candidate_ids]  # (N, M)
                logits = logits + b_cand

            if self.subspace_factorization and self.angular_margin > 0 and self.training and targets is not None:
                target_mask = (candidate_ids == targets.view(-1, 1))
                local_target = target_mask.int().argmax(dim=-1)
                margin_penalty = float(self.angular_margin * self.margin_scale)
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, local_target.view(-1, 1), 1.0)
                logits = logits - one_hot * margin_penalty

            if selected_internal and not return_candidates and not self.training:
                # During inference, scatter localized candidate logits into full vocab shape with -1e9 for non-candidates
                full_logits = torch.full((x.shape[0], self.vocab_size), -1e9, device=x.device, dtype=logits.dtype)
                full_logits.scatter_(dim=-1, index=candidate_ids, src=logits)
                logits = full_logits
        else:
            # Full vocabulary projection:
            logits = self.net[3](feat)

            if self.subspace_factorization and self.angular_margin > 0 and self.training and targets is not None:
                margin_penalty = float(self.angular_margin * self.margin_scale)
                one_hot = torch.zeros_like(logits)
                one_hot.scatter_(1, targets.view(-1, 1), 1.0)
                logits = logits - one_hot * margin_penalty

        out = [logits]
        if return_features:
            out.append(feat)
        if return_candidates:
            out.append(candidate_ids)
            if raw_blob_hit is not None:
                out.append(raw_blob_hit)
        if len(out) == 1:
            return out[0]
        return tuple(out)


# ── The hybrid model ─────────────────────────────────────────────────────────

class DSBHybrid(nn.Module):
    """
    Wraps a ``DiffSchrodingerBridge`` with discrete tagger/generator heads and a
    joint loss. The bridge supplies DP1 -> DP2 SDE transport; the heads add
    discrete edit supervision on the intermediate embedding ``x_t``.
    """

    def __init__(
        self,
        bridge: DiffSchrodingerBridge,
        vocab_size: int,
        lambda_sm: float = 1.0,
        lambda_tag: float = 1.0,
        lambda_gen: float = 1.0,
        tag_weights: Optional[Tuple[float, ...]] = None,
        gen_ignore_index: int = -100,
        condition_heads: bool = False,
        time_embed_dim: int = 128,
        embed_weight: Optional[torch.Tensor] = None,
        tie_weights: bool = True,
        lm_head: Optional[nn.Module] = None,
        subspace_factorization: bool = False,
        macro_dim: int = 512,
        lexical_dim: int = 256,
        lexical_loss_weight: float = 1.5,
        angular_margin: float = 0.05,
        margin_scale: float = 64.0,
        op_embed_dim: int = 0,
        blob_diffusion: bool = True,
        blob_size: int = 512,
        contextual_gen: bool = True,
    ):
        super().__init__()
        self.bridge: DiffSchrodingerBridge = bridge
        dim = bridge.dim
        self.condition_heads = condition_heads
        self.subspace_factorization = subspace_factorization
        self.macro_dim = macro_dim
        self.lexical_dim = lexical_dim
        self.lexical_loss_weight = lexical_loss_weight
        self.angular_margin = angular_margin
        self.margin_scale = margin_scale
        self.op_embed_dim = op_embed_dim
        self.blob_diffusion = blob_diffusion
        self.blob_size = blob_size
        self.contextual_gen = contextual_gen

        head_cond_dim = dim if condition_heads else 0
        self.tagger = TaggerHead(
            dim, time_embed_dim=time_embed_dim, cond_dim=head_cond_dim,
            subspace_factorization=subspace_factorization,
            macro_dim=macro_dim, lexical_dim=lexical_dim,
        )
        self.generator = GenHead(
            dim, vocab_size, time_embed_dim=time_embed_dim,
            cond_dim=head_cond_dim, embed_weight=embed_weight,
            tie_weights=tie_weights, lm_head=lm_head,
            subspace_factorization=subspace_factorization,
            macro_dim=macro_dim, lexical_dim=lexical_dim,
            angular_margin=angular_margin, margin_scale=margin_scale,
            op_embed_dim=op_embed_dim,
            contextual_gen=contextual_gen,
        )
        self.lambda_sm = lambda_sm
        self.lambda_tag = lambda_tag
        self.lambda_gen = lambda_gen
        self.gen_ignore_index = gen_ignore_index
        self.embed_weight = embed_weight
        self.lm_head = lm_head
        if tag_weights is not None:
            self.register_buffer(
                "_tag_weights",
                torch.tensor(tag_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._tag_weights = None

    # Use bridge.score_matching_loss directly for the SDE part.

    def discrete_targets(
        self,
        clean_ids: torch.Tensor,  # (B, S), int64
        noisy_ids: torch.Tensor,  # (B, S), int64
        attention_mask: torch.Tensor,  # (B, S) 1=real, 0=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-position tag labels and generator labels from the token ids.

        A position is 'corrupted' if it is real AND noisy != clean. Corrupted
        positions get tag=REPLACE and target=clean token; real-but-kept
        positions get tag=KEEP; padding gets ignored (-100 style).
        """
        device = clean_ids.device
        B, S = clean_ids.shape
        real = attention_mask == 1
        corrupted = real & (noisy_ids != clean_ids)

        tag_labels = torch.full((B, S), -100, dtype=torch.long, device=device)
        tag_labels[real & ~corrupted] = KEEP
        tag_labels[corrupted] = REPLACE

        gen_labels = torch.full((B, S), self.gen_ignore_index, dtype=torch.long, device=device)
        gen_labels[corrupted] = clean_ids[corrupted]
        return tag_labels, gen_labels

    def loss(
        self,
        dp1: torch.Tensor,          # (B, D) or (B, S, D) corrupted embedding
        dp2: torch.Tensor,          # (B, D) or (B, S, D) clean embedding
        clean_ids: torch.Tensor,    # (B, S) clean token ids
        noisy_ids: torch.Tensor,    # (B, S) corrupted token ids
        attention_mask: torch.Tensor,  # (B, S)
        t: Optional[torch.Tensor] = None,
        expose_ratio: float = 0.0,  # scheduled sampling: probability of using SDE reconstruction for GenHead
        tag_labels: Optional[torch.Tensor] = None,
        gen_labels: Optional[torch.Tensor] = None,
        recon_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:  # type: ignore[type-arg]
        """
        Joint loss = score_matching + lambda_tag * tag_ce + lambda_gen * gen_ce.

        When ``expose_ratio > 0``, the generator head is occasionally trained
        on the SDE's own imperfect reconstruction (``bridge.sample(dp1)``) at
        ``t=1`` instead of the pristine bridge sample ``x_t``. This **scheduled
        sampling** technique closes the train-test distribution gap that causes
        GenHead to see near-clean embeddings during training but noisy
        reconstructions during inference.

        Returns (total_loss, dict) with per-term losses.
        """
        # 1) Discrete targets from tokens (if not supplied directly).
        if tag_labels is None or gen_labels is None:
            t_lbl, g_lbl = self.discrete_targets(clean_ids, noisy_ids, attention_mask)
            tag_labels = t_lbl if tag_labels is None else tag_labels
            gen_labels = g_lbl if gen_labels is None else gen_labels

        # 2) SDE: sample x_t and supervised score.
        B = dp1.shape[0]
        if t is None:
            t = torch.rand(B, device=dp1.device)
        x_t, u_target, mu = self.bridge.forward_sample(dp1, dp2, t, return_mu=True)

        has_router = getattr(self.bridge.score_net, "gated_drift", False)
        if has_router:
            u_pred, routing_logits = self.bridge.score_predict(
                x_t, t, dp1=dp1, attention_mask=attention_mask, return_routing=True
            )
        else:
            u_pred = self.bridge.score_predict(x_t, t, dp1=dp1, attention_mask=attention_mask)
            routing_logits = None

        if attention_mask is not None and u_pred.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            if self.subspace_factorization:
                m_dim = self.macro_dim
                u_pred_m = u_pred[..., :m_dim]
                u_tgt_m = u_target[..., :m_dim]
                loss_sm_macro = ((u_pred_m - u_tgt_m) ** 2 * mask).sum() / (mask.sum() * m_dim).clamp(min=1.0)

                u_pred_l = u_pred[..., m_dim:]
                u_tgt_l = u_target[..., m_dim:]
                l_dim = u_pred.shape[-1] - m_dim
                loss_sm_lex = ((u_pred_l - u_tgt_l) ** 2 * mask).sum() / (mask.sum() * l_dim).clamp(min=1.0)

                loss_sm = loss_sm_macro + self.lexical_loss_weight * loss_sm_lex
            else:
                loss_sm = ((u_pred - u_target) ** 2 * mask).sum() / (mask.sum() * u_pred.shape[-1]).clamp(min=1.0)
                loss_sm_macro = loss_sm
                loss_sm_lex = loss_sm
        else:
            loss_sm = F.mse_loss(u_pred, u_target)
            loss_sm_macro = loss_sm
            loss_sm_lex = loss_sm

        if x_t.dim() != 3:
            # Pooled (non-per-position) embeddings carry no per-token structure,
            # so the discrete heads cannot be applied. Return score matching only.
            return loss_sm, {
                "total": loss_sm.item(),
                "score_matching": loss_sm.item(),
                "tag": 0.0,
                "gen": 0.0,
                "note": "pooled embeddings: discrete heads skipped",
            }

        loss_router = torch.tensor(0.0, device=x_t.device)
        if routing_logits is not None and tag_labels is not None:
            if self._tag_weights is not None:
                rw = self._tag_weights.to(routing_logits.device)
            else:
                rw = None
            loss_router = F.cross_entropy(
                routing_logits.permute(0, 2, 1).float(), tag_labels, weight=rw, ignore_index=-100
            )

        # 3) Tagger head on the SDE intermediate embedding x_t (t + DP1
        #    conditioned so corruption stays identifiable at low t).
        tag_logits = self.tagger(x_t, t, cond=dp1)     # (B, S, NUM_TAGS)
        if self._tag_weights is not None:
            w = self._tag_weights.to(tag_logits.device)
        else:
            w = None
        loss_tag = F.cross_entropy(tag_logits.permute(0, 2, 1).float(), tag_labels,
                                   weight=w, ignore_index=-100)

        # 4) Generator head, evaluated at REPLACE/INSERT/EXPAND positions where gen_labels is set.
        use_exposed = expose_ratio > 0.0 and torch.rand(1).item() < expose_ratio
        if use_exposed:
            with torch.no_grad():
                steps_exp = recon_steps if recon_steps is not None else 50
                x_gen = self.bridge.sample(dp1, steps=steps_exp, attention_mask=attention_mask, return_clean=True)    # (B, S, D) — SDE reconstruction
            t_gen = torch.ones(B, device=dp1.device)
        else:
            # Train GenHead along the smooth continuous bridge mean trajectory (mu),
            # protecting token cosine discrimination from raw Gaussian noise (sigma*z).
            x_gen = mu
            t_gen = t

        select = (gen_labels.reshape(-1) != self.gen_ignore_index) & (gen_labels.reshape(-1) >= 0)
        blob_hit = 0.0
        blob_acc = 0.0
        if select.any():
            xp = x_gen.reshape(-1, x_gen.shape[-1])[select]
            lb = gen_labels.reshape(-1)[select]
            t_sel = t_gen.repeat_interleave(x_gen.shape[1], dim=0)[select]
            c_sel = dp1.reshape(-1, x_gen.shape[-1])[select]
            target_ids = lb.clamp(0, self.generator.vocab_size - 1)
            op_sel = tag_labels.reshape(-1)[select] if tag_labels is not None else None
            ctx_sel = dp1.reshape(-1, dp1.shape[-1])[select] if self.contextual_gen else None

            if self.blob_diffusion:
                try:
                    gl, cand_ids, raw_hit = self.generator(
                        xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel,
                        blob_size=self.blob_size, context=ctx_sel, return_candidates=True
                    )
                    target_local = (cand_ids == target_ids.view(-1, 1)).int().argmax(dim=-1)
                    loss_gen = F.cross_entropy(gl.float(), target_local)
                    blob_hit = raw_hit
                    blob_acc = (gl.argmax(-1) == target_local).float().mean().item()
                except Exception:
                    gl = self.generator(xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel, context=ctx_sel)
                    loss_gen = F.cross_entropy(gl.float(), target_ids)
            else:
                gl = self.generator(xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel, context=ctx_sel)
                loss_gen = F.cross_entropy(gl.float(), target_ids)
        else:
            loss_gen = torch.tensor(0.0, device=x_t.device)

        total = self.lambda_sm * loss_sm + self.lambda_tag * (loss_tag + loss_router) + self.lambda_gen * loss_gen
        res_dict = {
            "total": total.item(),
            "score_matching": loss_sm.item(),
            "tag": loss_tag.item(),
            "router": loss_router.item(),
            "gen": loss_gen.item(),
        }
        if self.blob_diffusion:
            res_dict["blob_hit"] = blob_hit * 100.0
            res_dict["blob_acc"] = blob_acc * 100.0
        if self.subspace_factorization:
            res_dict["sm_macro"] = loss_sm_macro.item() if isinstance(loss_sm_macro, torch.Tensor) else float(loss_sm_macro)
            res_dict["sm_lex"] = loss_sm_lex.item() if isinstance(loss_sm_lex, torch.Tensor) else float(loss_sm_lex)
        return total, res_dict

    @torch.no_grad()
    def sample_embeddings(
        self,
        dp1: torch.Tensor,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate an output embedding via the reverse SDE, then run the discrete
        heads on the final embedding to read off tag probabilities.

        Returns (final_embedding, tag_probs).
        """
        x = self.bridge.sample(dp1, steps=steps)
        t_full = torch.ones(x.shape[0], device=x.device)
        tag_logits = self.tagger(x, t_full, cond=dp1)   # (B, S, NUM_TAGS)
        return x, tag_logits.softmax(-1)

    @torch.no_grad()
    def compute_diagnostics(
        self,
        dp1: torch.Tensor,
        dp2: torch.Tensor,
        tag_labels: torch.Tensor,
        gen_labels: torch.Tensor,
        recon_steps: Optional[int] = None,
        num_eval: int = 16,
        embed_weight: Optional[torch.Tensor] = None,
        lm_head_fn: Optional[Callable] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Comprehensive interpretability diagnostics:
          1. Continuous SDE: baseline, signal %, reconstruction error vs identity, and cosine similarity.
          2. Discrete Tagger: KEEP accuracy, REPLACE precision/recall/F1.
          3. Discrete Generator: Top-1 and Top-5 token exact match accuracy on REPLACE/INSERT slots.
          4. Pretrained LM-Head Decode: Top-1 and Top-5 via backbone's pretrained MLM head on Layer 12 states.
          5. Nearest-Neighbor Cosine Decode: Top-1 and Top-5 via cosine similarity to embed_weight.
        """
        n = min(num_eval, dp1.shape[0])
        dp1_e = dp1[:n]
        dp2_e = dp2[:n]
        tag_e = tag_labels[:n]
        gen_e = gen_labels[:n]
        attn_e = attention_mask[:n] if attention_mask is not None else None

        # 1) Continuous SDE Diagnostics
        bl = self.bridge.baseline_loss(dp1_e, dp2=dp2_e, attention_mask=attn_e).item()
        _, _, sig = self.bridge.signal_captured(dp1_e, dp2_e, num_eval=n, attention_mask=attn_e)

        sampled_x = self.bridge.sample(dp1_e, steps=recon_steps, attention_mask=attn_e, return_clean=True)
        if attn_e is not None and sampled_x.dim() == 3:
            denom_m = attn_e.float().sum().clamp(min=1.0)
            recon = ((torch.norm(sampled_x - dp2_e, dim=-1) * attn_e.float()).sum() / denom_m).item()
            ident = ((torch.norm(dp2_e - dp1_e, dim=-1) * attn_e.float()).sum() / denom_m).item()
        else:
            recon = torch.norm(sampled_x - dp2_e, dim=-1).mean().item()
            ident = torch.norm(dp2_e - dp1_e, dim=-1).mean().item()

        # Denoising performance on actual corrupted (REPLACE) slots:
        rep_mask_sde = (tag_e == REPLACE) & (attn_e == 1) if attn_e is not None else (tag_e == REPLACE)
        if rep_mask_sde.any():
            rep_recon = torch.norm(sampled_x[rep_mask_sde] - dp2_e[rep_mask_sde], dim=-1).mean().item()
            rep_ident = torch.norm(dp2_e[rep_mask_sde] - dp1_e[rep_mask_sde], dim=-1).mean().item()
        else:
            rep_recon = recon
            rep_ident = ident

        # Denoising performance across ALL non-KEEP corrupted slots (REPLACE, DELETE, INSERT, EXPAND):
        corr_mask_sde = (tag_e != KEEP) & (tag_e != -100) & (attn_e == 1) if attn_e is not None else ((tag_e != KEEP) & (tag_e != -100))
        if corr_mask_sde.any():
            corr_recon = torch.norm(sampled_x[corr_mask_sde] - dp2_e[corr_mask_sde], dim=-1).mean().item()
            corr_ident = torch.norm(dp2_e[corr_mask_sde] - dp1_e[corr_mask_sde], dim=-1).mean().item()
        else:
            corr_recon = rep_recon
            corr_ident = rep_ident

        cos_sim = F.cosine_similarity(
            sampled_x.reshape(-1, sampled_x.shape[-1]),
            dp2_e.reshape(-1, dp2_e.shape[-1]),
            dim=-1,
        ).mean().item()

        # 2) Discrete Head Diagnostics on the Sampled State (evaluated at t=1, cond=dp1)
        t_ones = torch.ones(n, device=dp1.device)
        tag_logits = self.tagger(sampled_x, t_ones, cond=dp1_e)  # (n, S, NUM_TAGS)
        pred_tags = tag_logits.argmax(-1)                        # (n, S)

        valid_tag_mask = (tag_e != -100)
        keep_mask = valid_tag_mask & (tag_e == KEEP)
        replace_mask = valid_tag_mask & (tag_e == REPLACE)

        keep_acc = (pred_tags[keep_mask] == KEEP).float().mean().item() if keep_mask.any() else 0.0

        tp_rep = ((pred_tags == REPLACE) & replace_mask).sum().item()
        pred_rep = (pred_tags[valid_tag_mask] == REPLACE).sum().item()
        true_rep = replace_mask.sum().item()

        prec_rep = (tp_rep / pred_rep) if pred_rep > 0 else 0.0
        rec_rep = (tp_rep / true_rep) if true_rep > 0 else 0.0
        f1_rep = (2 * prec_rep * rec_rep / (prec_rep + rec_rep)) if (prec_rep + rec_rep) > 0 else 0.0

        del_mask = valid_tag_mask & (tag_e == DELETE)
        ins_mask = valid_tag_mask & (tag_e == INSERT)
        exp_mask = valid_tag_mask & (tag_e == EXPAND)

        del_rec = (pred_tags[del_mask] == DELETE).float().mean().item() if del_mask.any() else 0.0
        ins_rec = (pred_tags[ins_mask] == INSERT).float().mean().item() if ins_mask.any() else 0.0
        exp_rec = (pred_tags[exp_mask] == EXPAND).float().mean().item() if exp_mask.any() else 0.0

        # 3) Generator Top-1 and Top-5 Token Match on Corrupted Slots
        gen_select = (gen_e != self.gen_ignore_index) & (gen_e >= 0)
        top1_acc, top5_acc = 0.0, 0.0
        nn_top1_acc, nn_top5_acc = 0.0, 0.0
        lm_top1_acc, lm_top5_acc = 0.0, 0.0
        blob_hit_rate = 0.0
        blob_acc = 0.0
        if gen_select.any():
            xp = sampled_x.reshape(-1, sampled_x.shape[-1])[gen_select.reshape(-1)]
            lb = gen_e.reshape(-1)[gen_select.reshape(-1)]
            t_sel = t_ones.repeat_interleave(sampled_x.shape[1], dim=0)[gen_select.reshape(-1)]
            c_sel = dp1_e.reshape(-1, sampled_x.shape[-1])[gen_select.reshape(-1)]
            op_sel = tag_e.reshape(-1)[gen_select.reshape(-1)]
            target_ids = lb.clamp(0, self.generator.vocab_size - 1)
            ctx_sel = dp1_e.reshape(-1, dp1_e.shape[-1])[gen_select.reshape(-1)] if self.contextual_gen else None

            if self.blob_diffusion:
                try:
                    gl, cand_ids, raw_hit = self.generator(
                        xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel,
                        blob_size=self.blob_size, context=ctx_sel, return_candidates=True
                    )
                    blob_hit_rate = raw_hit
                    target_local = (cand_ids == target_ids.view(-1, 1)).int().argmax(dim=-1)
                    top1_acc = (gl.argmax(-1) == target_local).float().mean().item()
                    k = min(5, gl.shape[-1])
                    top5_acc = gl.topk(k, dim=-1).indices.eq(target_local.unsqueeze(1)).any(1).float().mean().item()
                    blob_acc = top1_acc
                except Exception:
                    gl = self.generator(xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel, context=ctx_sel)
                    top1_acc = (gl.argmax(-1) == target_ids).float().mean().item()
                    k = min(5, gl.shape[-1])
                    top5_acc = gl.topk(k, dim=-1).indices.eq(target_ids.unsqueeze(1)).any(1).float().mean().item()
            else:
                gl = self.generator(xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel, context=ctx_sel)  # (N_gen, V)
                top1_acc = (gl.argmax(-1) == target_ids).float().mean().item()
                k = min(5, gl.shape[-1])
                top5_acc = gl.topk(k, dim=-1).indices.eq(target_ids.unsqueeze(1)).any(1).float().mean().item()

            # 4) Pretrained LM-Head Decode Baseline:
            #    Evaluates the backbone's pretrained MLM head directly on sampled_x (Layer 12).
            #    Reveals the true semantic recovery quality of the SDE without depending on GenHead learning.
            fn = lm_head_fn if lm_head_fn is not None else self.lm_head
            if fn is not None:
                with torch.no_grad():
                    lm_logits = fn(xp)
                    lm_top1_acc = (lm_logits.argmax(-1) == lb).float().mean().item()
                    lm_k = min(5, lm_logits.shape[-1])
                    lm_top5_acc = lm_logits.topk(lm_k, dim=-1).indices.eq(lb.unsqueeze(1)).any(1).float().mean().item()

            # 5) Nearest-Neighbor Cosine Decode (Layer 0 baseline):
            if embed_weight is not None:
                normed_xp = F.normalize(xp, dim=-1)                    # (N, D)
                normed_w = F.normalize(embed_weight, dim=-1)           # (V, D)
                cos_sims = normed_xp @ normed_w.T                     # (N, V)
                nn_top1_acc = (cos_sims.argmax(-1) == lb).float().mean().item()
                nn_k = min(5, cos_sims.shape[-1])
                nn_top5_acc = cos_sims.topk(nn_k, dim=-1).indices.eq(lb.unsqueeze(1)).any(1).float().mean().item()

        return {
            "baseline": bl,
            "signal": sig * 100.0,
            "recon_err": recon,
            "identity": ident,
            "rep_recon": rep_recon,
            "rep_ident": rep_ident,
            "corr_recon": corr_recon,
            "corr_ident": corr_ident,
            "cos_sim": cos_sim,
            "keep_acc": keep_acc * 100.0,
            "rep_f1": f1_rep * 100.0,
            "rep_prec": prec_rep * 100.0,
            "rep_rec": rec_rep * 100.0,
            "del_rec": del_rec * 100.0,
            "ins_rec": ins_rec * 100.0,
            "exp_rec": exp_rec * 100.0,
            "top1_acc": top1_acc * 100.0,
            "top5_acc": top5_acc * 100.0,
            "blob_hit": blob_hit_rate * 100.0,
            "blob_acc": blob_acc * 100.0,
            "lm_top1_acc": lm_top1_acc * 100.0,
            "lm_top5_acc": lm_top5_acc * 100.0,
            "nn_top1_acc": nn_top1_acc * 100.0,
            "nn_top5_acc": nn_top5_acc * 100.0,
        }

    @torch.no_grad()
    def decode_nearest(
        self,
        x: torch.Tensor,
        embed_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode continuous embeddings x to token IDs by finding the nearest neighbor
        in the pretrained embedding table (via cosine similarity).

        Args:
            x: (B, S, D) or (N, D) continuous embeddings.
            embed_weight: Optional (V, D) token embedding matrix (defaults to self.embed_weight).

        Returns:
            token_ids: (B, S) or (N,) nearest token IDs.
        """
        w = embed_weight if embed_weight is not None else self.embed_weight
        if w is None:
            raise ValueError("decode_nearest requires embed_weight (either passed or in DSBHybrid)")
        orig_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        normed_x = F.normalize(x_flat, dim=-1)
        normed_w = F.normalize(w.to(x.device), dim=-1)
        sims = normed_x @ normed_w.T  # (N, V)
        return sims.argmax(-1).reshape(orig_shape)

    # ── Phase 2: full edit-aware joint loss & two-phase sampling ──────────────

    def build_edit_loss(
        self,
        x_t: torch.Tensor,          # (B, S, D) SDE intermediate embedding
        tag_labels: torch.Tensor,   # (B, S) int64 in 0..4, -100 = ignore
        gen_labels: torch.Tensor,   # (B, S) int64, -100 = ignore
        t: Optional[torch.Tensor] = None,   # (B,) noise levels
        dp1: Optional[torch.Tensor] = None,  # (B, S, D) source embedding
        x_gen: Optional[torch.Tensor] = None,
        t_gen: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tagger + generator cross-entropy on the SDE embedding x_t."""
        tag_logits = self.tagger(x_t, t, cond=dp1)         # (B, S, NUM_TAGS)
        w = self._tag_weights.to(tag_logits.device) if self._tag_weights is not None else None
        loss_tag = F.cross_entropy(tag_logits.permute(0, 2, 1).float(), tag_labels,
                                   weight=w, ignore_index=-100)
        select = (gen_labels.reshape(-1) != self.gen_ignore_index) & (gen_labels.reshape(-1) >= 0)
        if select.any():
            x_eval = x_gen if x_gen is not None else x_t
            t_eval = t_gen if t_gen is not None else t
            xp = x_eval.reshape(-1, x_eval.shape[-1])[select]
            lb = gen_labels.reshape(-1)[select]
            t_sel = t_eval.repeat_interleave(x_eval.shape[1], dim=0)[select]
            c_sel = dp1.reshape(-1, x_eval.shape[-1])[select] if dp1 is not None else None
            op_sel = tag_labels.reshape(-1)[select] if tag_labels is not None else None
            target_ids = lb.clamp(0, self.generator.vocab_size - 1)
            ctx_sel = dp1.reshape(-1, dp1.shape[-1])[select] if (self.contextual_gen and dp1 is not None) else None

            if self.blob_diffusion:
                try:
                    gl, cand_ids, _ = self.generator(
                        xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel,
                        blob_size=self.blob_size, context=ctx_sel, return_candidates=True
                    )
                    target_local = (cand_ids == target_ids.view(-1, 1)).int().argmax(dim=-1)
                    loss_gen = F.cross_entropy(gl.float(), target_local)
                except Exception:
                    gl = self.generator(xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel, context=ctx_sel)
                    loss_gen = F.cross_entropy(gl.float(), target_ids)
            else:
                gl = self.generator(xp, t_sel, cond=c_sel, targets=target_ids, op_ids=op_sel, context=ctx_sel)
                loss_gen = F.cross_entropy(gl.float(), target_ids)
        else:
            loss_gen = torch.tensor(0.0, device=x_t.device)
        return loss_tag, loss_gen

    def loss_edit(
        self,
        dp1: torch.Tensor,          # (B, S, D) corrupted embedding
        dp2: torch.Tensor,          # (B, S, D) clean embedding
        tag_labels: torch.Tensor,   # (B, S) full edit tags (0..4, -100 ignore)
        gen_labels: torch.Tensor,   # (B, S) gen targets (-100 ignore)
        t: Optional[torch.Tensor] = None,
        condition_tags: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        expose_ratio: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:  # type: ignore[type-arg]
        """
        Phase-2 joint loss using the FULL edit-grammar labels and an optional
        edit-conditioned score net (detected via ``num_tags`` on the score net).

        L = score_matching(conditioned on condition_tags)
          + lambda_tag * tag_ce + lambda_gen * gen_ce
        """
        B = dp1.shape[0]
        if t is None:
            t = torch.rand(B, device=dp1.device)
        x_t, u_target = self.bridge.forward_sample(dp1, dp2, t)
        has_router = getattr(self.bridge.score_net, "gated_drift", False)
        if has_router:
            u_pred, routing_logits = self.bridge.score_predict(
                x_t, t, dp1=dp1, attention_mask=attention_mask, return_routing=True
            )
        elif hasattr(self.bridge.score_net, "num_tags"):
            u_pred = self.bridge.score_predict(x_t, t, dp1=dp1, tag_ids=condition_tags, attention_mask=attention_mask)
            routing_logits = None
        else:
            u_pred = self.bridge.score_predict(x_t, t, dp1=dp1, attention_mask=attention_mask)
            routing_logits = None
        
        if attention_mask is not None and u_pred.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            loss_sm = ((u_pred - u_target) ** 2 * mask).sum() / (mask.sum() * u_pred.shape[-1]).clamp(min=1.0)
        else:
            loss_sm = F.mse_loss(u_pred, u_target)

        loss_router = torch.tensor(0.0, device=x_t.device)
        if routing_logits is not None and tag_labels is not None:
            if self._tag_weights is not None:
                rw = self._tag_weights.to(routing_logits.device)
            else:
                rw = None
            loss_router = F.cross_entropy(
                routing_logits.permute(0, 2, 1).float(), tag_labels, weight=rw, ignore_index=-100
            )

        use_exposed = expose_ratio > 0.0 and torch.rand(1).item() < expose_ratio
        if use_exposed:
            with torch.no_grad():
                x_gen = self.bridge.sample(dp1, attention_mask=attention_mask, return_clean=True)
            t_gen = torch.ones(B, device=dp1.device)
        else:
            x_gen = x_t
            t_gen = t

        loss_tag, loss_gen = self.build_edit_loss(x_t, tag_labels, gen_labels,
                                                  t=t, dp1=dp1, x_gen=x_gen, t_gen=t_gen)
        total = self.lambda_sm * loss_sm + self.lambda_tag * (loss_tag + loss_router) + self.lambda_gen * loss_gen
        return total, {
            "total": total.item(),
            "score_matching": loss_sm.item(),
            "tag": loss_tag.item(),
            "router": loss_router.item(),
            "gen": loss_gen.item(),
        }

    @torch.no_grad()
    def sample_text(
        self,
        dp1: torch.Tensor,          # (B, S, D) corrupted embedding (start canvas)
        sde_steps: int = 100,
        refine_iters: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Two-phase generation: reverse the SDE to an output embedding, then read
        off the edit operations.

        Phase A — continuous: SDE sample dp1 -> x_approx (= final embedding).
        Phase B — discrete: run the tagger head on the final embedding to
        produce per-position edit ops, and the generator head to fill tokens.

        Returns (final_embedding, edit_tags_argmax).
        """
        x = self.bridge.sample(dp1, steps=sde_steps)
        # Guard refine_iters==0 so tag_logits is always bound.
        B = x.shape[0]
        t_full = torch.ones(B, device=x.device)
        tag_logits = self.tagger(x, t_full, cond=dp1)
        for _ in range(max(1, refine_iters)):
            tag_logits = self.tagger(x, t_full, cond=dp1)   # (B, S, NUM_TAGS)
        return x, tag_logits.argmax(-1)                 # (B, S)

    @torch.no_grad()
    def decode_to_text(
        self,
        x: torch.Tensor,          # (B, S, D) SDE output embedding
        tokenizer,                # HF tokenizer (decode + special ids)
        noisy_canvas_ids: Optional[torch.Tensor] = None,  # (B, S) starting canvas
        pad_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        dp1: Optional[torch.Tensor] = None,  # (B, S, D) source embedding (head cond)
    ) -> List[str]:
        """
        Turn the SDE output embedding into literal text by porting the DLLM
        iterative edit-decode loop into the hybrid's discrete heads.

        The tagger head predicts an edit op per position; the generator head
        proposes vocab tokens at REPLACE/INSERT positions (sampled top-k/top-p).
        We then apply the exact same canvas-mutation rules as ``DLLMInference``:
        KEEP / DELETE / REPLACE / INSERT / EXPAND, with BOS/EOS/prompt anchors
        locked to KEEP, and decode the resulting token canvas to a string.

        Because the generator head emits real vocab logits, decoding never needs
        an embedding->text invert: the output IS discrete tokens.

        Returns one decoded string per batch element.
        """
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if pad_id is None else pad_id
        M = tokenizer.mask_token_id
        special = {bos, eos, pad, M}

        B, S = x.shape[0], x.shape[1]
        results = []

        for b in range(B):
            # Phase A: discrete edit tags + generator logits on the SDE embedding
            # (evaluated at t=1 against the source DP1, matching training-time
            # conditioning).
            t_row = torch.ones(1, device=x.device)
            tag_logits = self.tagger(x[b].unsqueeze(0), t_row,
                                     cond=dp1[b:b+1])[0]            # (S, NUM_TAGS)
            gen_logits = self.generator(x[b], t_row.expand(S),
                                        cond=dp1[b])                # (S, V)
            tags = tag_logits.argmax(-1).tolist()
            gen_toks = self._sample_topk(gen_logits, temperature, top_k, top_p)

            # Starting canvas: the noisy ids, or all-mask if none given.
            if noisy_canvas_ids is not None:
                canvas_ids = noisy_canvas_ids[b].tolist()
            else:
                canvas_ids = [M] * S

            # Phase B: apply edits (port of DLLM._execute_edits), then decode.
            new_ids = self._apply_edits(canvas_ids, tags, gen_toks, bos, eos, pad, M)
            clean = [t for t in new_ids if t not in special]
            results.append(" ".join(tokenizer.decode(clean).split()))
        return results

    @staticmethod
    def _sample_topk(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        generated_ids: Optional[Union[List[int], List[List[int]]]] = None,
        repetition_penalty: float = 1.0,
        x_states: Optional[torch.Tensor] = None,
        embed_weight: Optional[torch.Tensor] = None,
        distance_threshold: Optional[float] = None,
        feat_states: Optional[torch.Tensor] = None,
        decoder_weight: Optional[torch.Tensor] = None,
        exempt_tokens: Optional[Set[int]] = None,
    ) -> List[int]:
        """Vectorized distance-gated top-k / top-p sampling over vocab logits -> token ids (per position).

        When distance_threshold is provided, prunes candidate tokens whose continuous embedding is
        outside a minimum cosine similarity from the continuous state (preferring aligned feat_states
        and decoder_weight), strictly bounding selection within the local attractor blob.

        Args:
            temperature: <=1e-4 uses greedy argmax (mode-locking). >0 scales probabilities.
            repetition_penalty: >1.0 divides logits of already-generated tokens (HuggingFace convention).
            distance_threshold: Optional minimum cosine similarity to continuous SDE state (e.g. 0.25).
            feat_states: Projected feature vectors directly feeding decoder weights (shape: N, D).
            decoder_weight: Vocabulary decoder projection matrix (shape: V, D).
            exempt_tokens: Set of tokens (e.g. stopwords, syntax) to exempt from repetition penalty.
        """
        if logits.numel() == 0:
            return []

        logits = logits.detach()

        # Distance-gated candidate selection within learned/bounded macro blob around SDE state.
        # Uses aligned feat_states (projected features from GenHead) and decoder_weight when available.
        if distance_threshold is not None:
            f_ref = feat_states if feat_states is not None else x_states
            w_ref = decoder_weight if decoder_weight is not None else embed_weight
            if f_ref is not None and w_ref is not None:
                try:
                    norm_feat = F.normalize(f_ref.detach().to(logits.device, dtype=torch.float32), dim=-1)
                    norm_w = F.normalize(w_ref.detach().to(logits.device, dtype=torch.float32), dim=-1)
                    cos_sims = norm_feat @ norm_w.T  # (N, V)
                    dist_mask = (cos_sims < distance_threshold)
                    # Safeguard: if ALL tokens fail the threshold for a row, preserve the closest token (argmax cosine)
                    all_pruned = dist_mask.all(dim=-1, keepdim=True)
                    if all_pruned.any():
                        closest = cos_sims.argmax(dim=-1, keepdim=True)
                        dist_mask = dist_mask.scatter(-1, closest, False)
                    logits = torch.where(dist_mask, torch.full_like(logits, -1e9), logits)
                except Exception:
                    pass

        if temperature <= 1e-4:
            # Mode-locking greedy decode: pick highest density mode in candidate pool
            return logits.argmax(dim=-1).tolist()

        # Move projection to CPU for instantaneous AVX quickselect
        logits = logits.to("cpu", dtype=torch.float32) / max(temperature, 1e-8)

        # Repetition penalty: down-weight tokens that have already been generated in this canvas / window.
        if repetition_penalty != 1.0 and generated_ids:
            if isinstance(generated_ids[0], list) if len(generated_ids) > 0 else False:
                # Per-position list of penalized tokens
                for i_pos, pos_ids in enumerate(generated_ids):
                    if pos_ids and i_pos < logits.shape[0]:
                        filt = [t for t in pos_ids if exempt_tokens is None or t not in exempt_tokens]
                        if filt:
                            prev = torch.tensor(list(set(filt)), dtype=torch.long)
                            prev = prev[prev < logits.shape[-1]]
                            if len(prev) > 0:
                                scores = logits[i_pos, prev]
                                logits[i_pos, prev] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)
            else:
                filt = [t for t in generated_ids if exempt_tokens is None or t not in exempt_tokens]
                if filt:
                    prev = torch.tensor(list(set(filt)), dtype=torch.long)
                    prev = prev[prev < logits.shape[-1]]
                    if len(prev) > 0:
                        scores = logits[:, prev]
                        logits[:, prev] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)

        k = min(top_k, logits.shape[-1])
        topk_vals, topk_idx = logits.topk(k, dim=-1)   # (N, K) on CPU
        topk_probs = F.softmax(topk_vals, dim=-1)      # (N, K)

        if top_p < 1.0:
            cum = topk_probs.cumsum(dim=-1)
            keep = cum <= top_p
            keep[:, 0] = True  # Always preserve at least top-1
            topk_probs = torch.where(keep, topk_probs, torch.zeros_like(topk_probs))
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        sampled_indices = torch.multinomial(topk_probs, num_samples=1)  # (N, 1)
        chosen_tokens = topk_idx.gather(1, sampled_indices).squeeze(-1)  # (N,)
        return chosen_tokens.tolist()

    @staticmethod
    def _apply_edits(canvas_ids, tags, gen_toks, bos, eos, pad, M):
        """Apply KEEP/DELETE/REPLACE/INSERT/EXPAND, ported from DLLM._execute_edits."""
        out = []
        for i, tok in enumerate(canvas_ids):
            if i >= len(tags):
                break
            tag = tags[i]
            gen = gen_toks[i] if i < len(gen_toks) else tok
            # Only protect true boundaries: leading BOS and trailing EOS
            if i == 0 and tok == bos:
                out.append(tok); continue
            if i == len(canvas_ids) - 1 and tok == eos:
                if tag == INSERT:
                    out.append(gen if gen != M else tok)
                out.append(tok)
                continue
            if tag == DELETE:
                continue
            elif tag == REPLACE:
                out.append(gen if gen != M else tok)
            elif tag == INSERT:
                out.append(gen if gen != M else tok)
                out.append(tok)
            elif tag == EXPAND:
                out.append(M); out.append(M)
            else:  # KEEP
                out.append(tok)
        return out

    @torch.no_grad()
    def generate_text(
        self,
        x: torch.Tensor,          # (B, S, D) SDE output embedding (seed state)
        tokenizer,                # HF tokenizer (decode + special ids)
        embedder,                 # TextEmbedder: token_ids -> per-token embeddings
        pad_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        max_iterations: int = 8,
        min_iterations: int = 1,
        max_len: Optional[int] = None,
        dp1: Optional[torch.Tensor] = None,  # (B, S, D) source embedding (head cond)
        seed_ids: Optional[List[List[int]]] = None,  # per-row starting canvas token ids
        repetition_penalty: float = 1.3,      # >1.0 suppresses repeated tokens; 1.0 = off
        decode_mode: str = "genhead",         # "genhead" (use learned GenHead) or "nearest" (cosine similarity to embed_weight)
        embed_weight: Optional[torch.Tensor] = None,  # token embedding matrix (defaults to self.embed_weight)
        log_operations: bool = False,         # Whether to print per-iteration edit operation logs
        keep_threshold: float = 0.0,          # Confidence gate: minimum KEEP probability to allow KEEP (e.g. 0.85); below demotes
        fluency_threshold: float = 0.0,       # Rolling n-gram / MLM fluency gate: minimum likelihood to allow KEEP (e.g. 0.05)
        refine_cond_mode: str = "self",       # "self" (legacy self-conditioning, default) or "initial" (anchor to initial DP1 via smooth interp)
        distance_threshold: Optional[float] = None,  # Distance-gated candidate selection threshold (e.g. 0.25 min cosine)
        t_eval: float = 0.0,                  # Evaluation time level for tagger and generator (0.0 = static/encoder time, 1.0 = terminal SDE state)
        lm_blend_weight: float = 0.0,         # Logit blend factor with pretrained MLM head on REPLACE slots (0.0 = disabled, e.g. 0.35)
        progressive_fill: bool = False,       # Confidence-ranked sequential slot infilling on multi-token spans
        repetition_window: Optional[int] = 5, # Context window radius for repetition penalty (None/0 = whole canvas)
        exempt_stopwords: bool = True,        # Exempt common stopwords and syntax tokens from repetition penalty
        blob_diffusion: Optional[bool] = None,# Restrict generation candidates to continuous SDE localized blob
        blob_size: Optional[int] = None,      # Size of candidate blob (e.g. 512)
        return_trajectory: bool = False,      # If True, returns (results, trajectories) with step details
    ) -> Union[List[str], Tuple[List[str], List[List[dict]]]]:
        """
        True variable-length iterative refinement decode (DLLM-style, ported).

        Each round:
          Phase A — run tagger + generator heads on the current canvas embedding.
          Phase B — apply KEEP/DELETE/REPLACE/INSERT/EXPAND edits to a REAL,
                    variable-length token canvas (DLLM._execute_edits rules).
                    INSERT and EXPAND physically LENGTHEN the canvas; DELETE
                    shortens it — matching DLLM exactly, not a fixed-S canvas.
          Round-trip: re-embed each row at ITS CURRENT length (token -> embedding
                     via the embedder) so the next round sees the new state.

        Stops when no canvas changes across an iteration, or after
        ``max_iterations``. Growth is capped at ``max_len`` (defaults to the
        seed embedding width ``S``). Returns one decoded string per batch row.

        Requires an ``embedder`` whose ``embed_ids(input_ids, attention_mask)``
        accepts variable-length rows (batched with padding to the longest row in
        the batch) and returns per-position embeddings of that length.
        """
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if pad_id is None else pad_id
        M = tokenizer.mask_token_id
        special = {bos, eos, pad, M}

        # Build set of exempted tokens (special tokens, punctuation, and common stopwords)
        exempt_tokens: Set[int] = set(special)
        if exempt_stopwords and tokenizer is not None:
            common_stopwords = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
                "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
                "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
                "did", "will", "would", "shall", "should", "may", "might", "must", "can",
                "could", "that", "which", "who", "whom", "this", "these", "those", "it",
                "its", "as", "if", "than", "so", "no", "not", "only", "such", "there"
            }
            try:
                for word in common_stopwords:
                    for w_var in (word, " " + word, word.capitalize(), " " + word.capitalize()):
                        tok_enc = tokenizer.encode(w_var, add_special_tokens=False)
                        if len(tok_enc) == 1:
                            exempt_tokens.add(tok_enc[0])
            except Exception:
                pass
            try:
                punc_chars = ".,;:!?-—\"'()[]{}/<>*&^%$#@~`"
                for p in punc_chars:
                    tok_enc = tokenizer.encode(p, add_special_tokens=False)
                    if len(tok_enc) == 1:
                        exempt_tokens.add(tok_enc[0])
            except Exception:
                pass

        B = x.shape[0]
        S = x.shape[1]
        device = x.device
        max_len = max_len if max_len is not None else getattr(embedder, "max_length", 128)
        if getattr(self.tagger, "cond_dim", 0) > 0 and dp1 is None:
            raise ValueError("conditioned heads require dp1 (the source embedding)")

        # Per-row variable-length canvases: start from seed ids if given (e.g.
        # the prompt/corrupted tokens — KEEP then preserves REAL tokens, matching
        # training where uncorrupted positions hold real tokens), else all-mask.
        if seed_ids is not None:
            canvases: List[List[int]] = [list(row) for row in seed_ids]
        else:
            canvases = [[M] * S for _ in range(B)]

        # Start from the SDE's denoised continuous representation x (at t=1),
        # where the diffusion bridge has transported corrupted tokens toward their
        # clean attractor basins. Subsequent refinement rounds (iteration 1+) will
        # re-embed discrete canvas edits through the encoder.
        cur: List[torch.Tensor] = [x[b] for b in range(B)]  # (S, D) per row
        dp1_cur = dp1

        trajectories: List[List[dict]] = [[] for _ in range(B)]
        # Track previously visited tokens per position to eradicate ping-pong oscillation loops
        visited_tokens: List[Dict[int, Set[int]]] = [{} for _ in range(B)]

        for iteration in range(max_iterations):
            changed = False
            next_canvases: List[List[int]] = []
            next_emb: List[torch.Tensor] = []

            for b in range(B):
                for p_idx, tok_val in enumerate(canvases[b]):
                    if p_idx not in visited_tokens[b]:
                        visited_tokens[b][p_idx] = set()
                    visited_tokens[b][p_idx].add(tok_val)

                emb = cur[b].unsqueeze(0)                 # (1, L, D)
                L = emb.shape[1]
                # Evaluate heads at t_eval (0.0 for static encoder embeddings; 1.0 when reading SDE terminal states)
                t_row = torch.full((1,), t_eval, device=emb.device)
                c_row = None
                if dp1_cur is not None:
                    c = dp1_cur[b:b+1]
                    if c.shape[1] == L:
                        c_row = c
                    elif c.shape[1] > 1:
                        # Smooth 1D interpolation across sequence length dimension
                        c_row = F.interpolate(c.permute(0, 2, 1).float(), size=L, mode='linear', align_corners=True).permute(0, 2, 1).to(c.dtype)
                    else:
                        c_pad = torch.zeros(1, L - c.shape[1], c.shape[2],
                                            device=c.device, dtype=c.dtype)
                        c_row = torch.cat([c, c_pad], dim=1)
                tag_logits = self.tagger(emb, t_row, cond=c_row)[0]     # (L, T)
                # If score net has an edit router, ensemble its routing prediction with the tagger
                if getattr(self.bridge.score_net, "gated_drift", False) and getattr(self.bridge.score_net, "router", None) is not None:
                    try:
                        _, r_logits = self.bridge.score_net(emb, t_row, cond=c_row, return_routing=True)
                        if r_logits is not None:
                            tag_logits = 0.5 * tag_logits + 0.5 * r_logits[0]
                    except Exception:
                        pass

                # Calibration: debias inference tag logits by subtracting log(tag_weights)
                # so training upweighting of INSERT/EXPAND does not skew argmax away from REPLACE.
                if hasattr(self, "_tag_weights") and self._tag_weights is not None:
                    tag_logits = tag_logits - torch.log(self._tag_weights.to(tag_logits.device).clamp(min=1e-5))

                tag_probs = F.softmax(tag_logits, dim=-1)
                raw_tags = tag_logits.argmax(-1).tolist()
                raw_probs = tag_probs.tolist()
                tags = list(raw_tags)

                # Boundary token guard: only protect true leading <s> and trailing </s>
                cur_tokens = canvases[b]
                override_notes = ["" for _ in range(len(cur_tokens))]
                if len(cur_tokens) > 0 and cur_tokens[0] == bos:
                    if raw_tags[0] != KEEP:
                        override_notes[0] = "boundary token guard"
                    tags[0] = KEEP
                if len(cur_tokens) > 1 and cur_tokens[-1] == eos:
                    # Trailing </s> can only be KEEP or INSERT (inserting words before </s>).
                    # It must never be DELETE or REPLACE.
                    if tags[-1] in (DELETE, REPLACE):
                        override_notes[-1] = "boundary token guard"
                        tags[-1] = KEEP

                # Stutter / Repetition symmetry breaking:
                # If adjacent non-special tokens are duplicate (e.g. "this this", "lol lol"),
                # collapse the duplicate loop by unconditionally forcing DELETE on duplicate slots
                for idx in range(1, len(cur_tokens)):
                    if cur_tokens[idx] == cur_tokens[idx - 1] and cur_tokens[idx] not in special:
                        if idx < len(tags):
                            if tags[idx] != DELETE:
                                override_notes[idx] = "repetition stutter collapse"
                            tags[idx] = DELETE

                # Whitespace-punctuation delimiter guard:
                # If a standalone whitespace token (e.g. id 6 in sentencepiece) immediately precedes
                # punctuation or EOS, never replace or insert into it (collapses dangling space to DELETE)
                punc_ids = {4, 5, 20, 46, 1104}
                for idx in range(len(cur_tokens) - 1):
                    if cur_tokens[idx] == 6 and (cur_tokens[idx + 1] in punc_ids or cur_tokens[idx + 1] == eos):
                        if tags[idx] in (REPLACE, INSERT):
                            override_notes[idx] = "whitespace-punctuation guard"
                            tags[idx] = DELETE

                # Confidence gate: Reject weak KEEP predictions (demote to next-best edit op)
                if keep_threshold > 0.0:
                    for idx in range(len(cur_tokens)):
                        if cur_tokens[idx] not in special and tags[idx] == KEEP:
                            k_prob = raw_probs[idx][KEEP]
                            if k_prob < keep_threshold:
                                sub_logits = tag_logits[idx].clone()
                                sub_logits[KEEP] = -1e9
                                tags[idx] = sub_logits.argmax().item()
                                override_notes[idx] = f"confidence gate (p={k_prob:.3f} < {keep_threshold:.2f})"

                # Rolling N-Gram / MLM Fluency Gate: Check if current token / n-gram makes syntactic sense
                if fluency_threshold > 0.0:
                    lm_all_logits = None
                    if hasattr(embedder, "decode_logits"):
                        lm_all_logits = embedder.decode_logits(emb)[0]  # (L, V)
                    elif self.lm_head is not None:
                        lm_all_logits = self.lm_head(emb)[0]
                    if lm_all_logits is not None:
                        lm_all_probs = F.softmax(lm_all_logits, dim=-1)
                        for idx in range(len(cur_tokens)):
                            if cur_tokens[idx] not in special and tags[idx] == KEEP:
                                win_start = max(0, idx - 1)
                                win_end = min(len(cur_tokens), idx + 2)
                                win_probs = [lm_all_probs[j, cur_tokens[j]].clamp(min=1e-9).item() for j in range(win_start, win_end)]
                                import math
                                win_fluency = math.exp(sum(math.log(p) for p in win_probs) / max(1, len(win_probs)))
                                tok_prob = lm_all_probs[idx, cur_tokens[idx]].item()
                                # Demote only if the token itself is improbable in context (avoids innocent neighbors getting infected by adjacent corrupt tokens)
                                if tok_prob < fluency_threshold:
                                    sub_logits = tag_logits[idx].clone()
                                    sub_logits[KEEP] = -1e9
                                    tags[idx] = sub_logits.argmax().item()
                                    override_notes[idx] = f"fluency gate (tok_p={tok_prob:.4f} < {fluency_threshold:.2f})"

                # Sparse generator evaluation: only evaluate the 250k-vocab GenHead
                # and top-k sampling at positions that actually need token generation.
                gen_positions = [i for i, tg in enumerate(tags) if tg in (REPLACE, INSERT)]
                gen_toks = list(canvases[b])
                if gen_positions:
                    pos_tensor = torch.tensor(gen_positions, device=cur[b].device)
                    cur_sel = cur[b][pos_tensor]
                    t_sel = t_row.expand(len(gen_positions))
                    c_sel = c_row[0][pos_tensor] if c_row is not None else None
                    op_tensor = torch.tensor([tags[p] for p in gen_positions], device=cur[b].device)

                    gen_feat = None
                    # Decoder projection weights for blob candidate selection and distance gating:
                    w_dec = None
                    if hasattr(self.generator, "net") and len(self.generator.net) > 3:
                        w_dec = getattr(self.generator.net[3], "weight", None)
                    if w_dec is None:
                        w_dec = embed_weight if embed_weight is not None else self.embed_weight

                    if decode_mode == "lm_head":
                        if hasattr(embedder, "decode_logits"):
                            gen_logits = embedder.decode_logits(cur_sel)
                        elif self.lm_head is not None:
                            gen_logits = self.lm_head(cur_sel)
                        else:
                            gen_logits = self.generator(cur_sel, t_sel, cond=c_sel, op_ids=op_tensor)
                        # Extract aligned feature representation for distance gating from lm_head if available
                        target_lm = getattr(embedder, "lm_head", self.lm_head)
                        if target_lm is not None and hasattr(target_lm, "dense") and hasattr(target_lm, "layer_norm"):
                            try:
                                gen_feat = target_lm.layer_norm(F.gelu(target_lm.dense(cur_sel)))
                            except Exception:
                                gen_feat = None
                    elif decode_mode == "nearest":
                        if w_dec is None:
                            raise ValueError("decode_mode='nearest' requires embed_weight (either passed or in DSBHybrid)")
                        normed_cur = F.normalize(cur_sel, dim=-1)
                        normed_w = F.normalize(w_dec.to(cur_sel.device), dim=-1)
                        gen_logits = (normed_cur @ normed_w.T) * 20.0
                    else:
                        use_blob = (blob_diffusion if blob_diffusion is not None else getattr(self, "blob_diffusion", False)) and w_dec is not None
                        b_sz = blob_size if blob_size is not None else (self.blob_size if getattr(self, "blob_diffusion", False) else None)
                        ctx_sel = c_sel if getattr(self, "contextual_gen", False) else None
                        try:
                            gen_out = self.generator(
                                cur_sel, t_sel, cond=c_sel, op_ids=op_tensor,
                                blob_size=(b_sz if use_blob else None), context=ctx_sel, return_features=True
                            )
                        except TypeError:
                            try:
                                gen_out = self.generator(
                                    cur_sel, t_sel, cond=c_sel, op_ids=op_tensor,
                                    context=ctx_sel, return_features=True
                                )
                            except TypeError:
                                gen_out = self.generator(
                                    cur_sel, t_sel, cond=c_sel, op_ids=op_tensor, return_features=True
                                )
                        if isinstance(gen_out, tuple):
                            gen_logits, gen_feat = gen_out
                        else:
                            gen_logits, gen_feat = gen_out, None

                    # Dual-head logit blending: blend pretrained MLM head logits on REPLACE slots for factual exactness
                    if lm_blend_weight > 0.0 and decode_mode != "lm_head":
                        try:
                            if hasattr(embedder, "decode_logits"):
                                lm_logits = embedder.decode_logits(cur_sel)
                            elif self.lm_head is not None:
                                lm_logits = self.lm_head(cur_sel)
                            else:
                                lm_logits = None
                            if lm_logits is not None:
                                rep_indices = [idx for idx, p in enumerate(gen_positions) if tags[p] == REPLACE]
                                if rep_indices:
                                    rep_t = torch.tensor(rep_indices, device=gen_logits.device)
                                    gen_logits[rep_t] = (
                                        (1.0 - lm_blend_weight) * gen_logits[rep_t]
                                        + lm_blend_weight * lm_logits[rep_t]
                                    )
                        except Exception:
                            pass

                    # Suppress structural special tokens from being generated into content slots
                    for sp_id in (bos, eos, pad, M):
                        if sp_id is not None and sp_id < gen_logits.shape[-1]:
                            gen_logits[:, sp_id] = -1e9

                    # Prevent self-replacement loops, ping-pong oscillation, and spurious delimiter collapse:
                    punc_tokens = {46, 20, 1104}  # en-dash, hyphen, em-dash
                    # Dangling syntax tokens that should not end a clause immediately before punctuation or EOS:
                    dangling_syntax_tokens = {23, 10, 70, 450, 903, 136, 678, 1295, 3688}  # in, a, the, that, this, and, with, from, here
                    for i_sel, pos in enumerate(gen_positions):
                        tg = tags[pos]
                        cur_t = canvases[b][pos]
                        if tg in (REPLACE, INSERT):
                            if cur_t < gen_logits.shape[-1]:
                                gen_logits[i_sel, cur_t] = -1e9
                            # Ping-pong loop break: penalize any token previously seen at this position
                            if pos in visited_tokens[b]:
                                for prev_t in visited_tokens[b][pos]:
                                    if prev_t not in special and prev_t < gen_logits.shape[-1]:
                                        gen_logits[i_sel, prev_t] = -1e9
                            if cur_t not in punc_tokens and cur_t not in special:
                                for p_id in punc_tokens:
                                    if p_id < gen_logits.shape[-1]:
                                        gen_logits[i_sel, p_id] = -1e9
                            # Dangling syntax suppression: if replacing immediately before punctuation or EOS, suppress prepositions/connectors
                            if pos + 1 < len(canvases[b]) and canvases[b][pos + 1] in (eos, 4, 5, 20, 46):
                                for d_id in dangling_syntax_tokens:
                                    if d_id < gen_logits.shape[-1]:
                                        gen_logits[i_sel, d_id] = -1e9

                    # Build per-position penalized candidate list based on repetition_window
                    penalized_per_pos = []
                    for pos in gen_positions:
                        if repetition_window is not None and repetition_window > 0:
                            w_start = max(0, pos - repetition_window)
                            w_end = min(len(canvases[b]), pos + repetition_window + 1)
                            tok_window = [canvases[b][j] for j in range(w_start, w_end) if canvases[b][j] not in exempt_tokens]
                        else:
                            tok_window = [t for t in canvases[b] if t not in exempt_tokens]
                        penalized_per_pos.append(tok_window)

                    # Sampling
                    if progressive_fill and len(gen_positions) > 1:
                        # Confidence-ranked progressive in-filling for multi-token spans:
                        # Sort positions by highest max probability, sample, and sequentially resolve
                        conf_probs = F.softmax(gen_logits.detach(), dim=-1).max(dim=-1).values
                        sorted_order = torch.argsort(conf_probs, descending=True).tolist()
                        accum_chosen = []

                        for sel_idx in sorted_order:
                            pos = gen_positions[sel_idx]
                            sub_logits = gen_logits[sel_idx:sel_idx + 1]
                            sub_feat = gen_feat[sel_idx:sel_idx + 1] if gen_feat is not None else None
                            sub_cur = cur_sel[sel_idx:sel_idx + 1]
                            sub_penalized = [penalized_per_pos[sel_idx] + accum_chosen]

                            sub_sample = self._sample_topk(
                                sub_logits, temperature, top_k, top_p,
                                generated_ids=sub_penalized,
                                repetition_penalty=repetition_penalty,
                                x_states=sub_cur,
                                embed_weight=w_dec,
                                distance_threshold=distance_threshold,
                                feat_states=sub_feat,
                                decoder_weight=w_dec,
                                exempt_tokens=exempt_tokens,
                            )
                            if sub_sample:
                                gen_toks[pos] = sub_sample[0]
                                accum_chosen.append(sub_sample[0])
                    else:
                        sampled_tokens = self._sample_topk(
                            gen_logits, temperature, top_k, top_p,
                            generated_ids=penalized_per_pos,
                            repetition_penalty=repetition_penalty,
                            x_states=cur_sel,
                            embed_weight=w_dec,
                            distance_threshold=distance_threshold,
                            feat_states=gen_feat,
                            decoder_weight=w_dec,
                            exempt_tokens=exempt_tokens,
                        )
                        for pos, tok_id in zip(gen_positions, sampled_tokens):
                            if pos < len(gen_toks):
                                gen_toks[pos] = tok_id

                # Apply edits -> genuinely variable-length output (INSERT grows,
                # DELETE trims, EXPAND splits), capped at max_len.
                new_ids = self._apply_edits(
                    canvases[b], tags, gen_toks, bos, eos, pad, M
                )
                truncated = False
                if len(new_ids) > max_len:
                    if eos is not None and (eos in new_ids or (len(cur_tokens) > 0 and cur_tokens[-1] == eos)):
                        new_ids = new_ids[:max_len - 1] + [eos]
                    else:
                        new_ids = new_ids[:max_len]
                    truncated = True
                elif eos is not None and (len(cur_tokens) > 0 and cur_tokens[-1] == eos):
                    if len(new_ids) == 0 or new_ids[-1] != eos:
                        new_ids.append(eos)

                if log_operations:
                    cur_clean_text = " ".join(tokenizer.decode([t for t in cur_tokens if t not in special]).split())
                    print(f"\n{'─' * 70}")
                    batch_str = f" [Batch {b+1}/{B}]" if B > 1 else ""
                    print(f"🔄 Iteration {iteration + 1}/{max_iterations}{batch_str} (Canvas: {len(cur_tokens)} tokens)")
                    print(f"{'─' * 70}")
                    print(f"Canvas Before: \"{cur_clean_text}\"")
                    print(f"Tokens: {[tokenizer.decode([t]) for t in cur_tokens]}")
                    print(f"\n  {'Pos':>3} | {'Token':<14} | {'Tag':<7} | {'Conf':>6} | {'Probs (K / D / R / I / E)':<29} | {'Action / Detail'}")
                    print(f"  {'-'*3}-+-{'-'*14}-+-{'-'*7}-+-{'-'*6}-+-{'-'*29}-+-{'-'*30}")
                    for pos in range(len(cur_tokens)):
                        tok_id = cur_tokens[pos]
                        if tok_id == bos:
                            tok_disp = "<s>"
                        elif tok_id == eos:
                            tok_disp = "</s>"
                        elif tok_id == pad:
                            tok_disp = "<pad>"
                        elif tok_id == M:
                            tok_disp = "<mask>"
                        else:
                            tok_disp = tokenizer.decode([tok_id])
                            tok_disp = repr(tok_disp)[1:-1]
                        if len(tok_disp) > 14:
                            tok_disp = tok_disp[:11] + "..."

                        tg = tags[pos]
                        tag_name = TAG_NAMES[tg]
                        conf = raw_probs[pos][tg]
                        probs_str = f"{raw_probs[pos][0]:.2f}/{raw_probs[pos][1]:.2f}/{raw_probs[pos][2]:.2f}/{raw_probs[pos][3]:.2f}/{raw_probs[pos][4]:.2f}"

                        detail = ""
                        if override_notes[pos]:
                            detail += f"[{override_notes[pos]}] "
                        if tg == KEEP:
                            detail += "-"
                        elif tg == DELETE:
                            detail += "[DELETE]"
                        elif tg == REPLACE:
                            rep_tok_id = gen_toks[pos]
                            rep_str = repr(tokenizer.decode([rep_tok_id]))[1:-1]
                            detail += f"-> {rep_str} (id={rep_tok_id})"
                        elif tg == INSERT:
                            ins_tok_id = gen_toks[pos]
                            ins_str = repr(tokenizer.decode([ins_tok_id]))[1:-1]
                            detail += f"+ins {ins_str} (id={ins_tok_id})"
                        elif tg == EXPAND:
                            detail += "[EXPAND 2x<mask/mask>]"

                        print(f"  {pos:>3} | {tok_disp:<14} | {tag_name:<7} | {conf:>6.3f} | {probs_str:<29} | {detail}")

                    tag_counts = {name: tags.count(i) for i, name in enumerate(TAG_NAMES)}
                    print(f"\n  Tag Distribution: " + " | ".join(f"{k}: {v}" for k, v in tag_counts.items()))

                    edits = []
                    for pos in range(len(cur_tokens)):
                        if tags[pos] != KEEP:
                            orig_repr = repr(tokenizer.decode([cur_tokens[pos]]))
                            if tags[pos] == REPLACE:
                                new_repr = repr(tokenizer.decode([gen_toks[pos]]))
                                note = f" [{override_notes[pos]}]" if override_notes[pos] else ""
                                edits.append(f"Pos {pos} {orig_repr} -> REPLACE -> {new_repr}{note}")
                            elif tags[pos] == DELETE:
                                edits.append(f"Pos {pos} {orig_repr} -> DELETE")
                            elif tags[pos] == INSERT:
                                ins_repr = repr(tokenizer.decode([gen_toks[pos]]))
                                edits.append(f"Pos {pos} {orig_repr} -> INSERT {ins_repr}")
                            elif tags[pos] == EXPAND:
                                edits.append(f"Pos {pos} {orig_repr} -> EXPAND")

                    if edits:
                        print(f"  Executed Edits ({len(edits)}):")
                        for e in edits:
                            print(f"    • {e}")
                    else:
                        print("  Executed Edits: None (All tokens KEEP)")

                    new_clean_text = " ".join(tokenizer.decode([t for t in new_ids if t not in special]).split())
                    print(f"Canvas After:  \"{new_clean_text}\" ({len(new_ids)} tokens)" + (" [Truncated]" if truncated else ""))
                    status_str = "MODIFIED" if new_ids != cur_tokens else "UNCHANGED (Converged)"
                    print(f"Iteration Result: {status_str}")

                if return_trajectory:
                    token_details = []
                    for pos in range(len(cur_tokens)):
                        tg = tags[pos]
                        conf = raw_probs[pos][tg] if (raw_probs and pos < len(raw_probs)) else 1.0
                        tok_id = cur_tokens[pos]
                        act_tok = gen_toks[pos] if (tg in (REPLACE, INSERT) and pos < len(gen_toks)) else None
                        act_str = tokenizer.decode([act_tok]) if act_tok is not None else ""
                        token_details.append({
                            "pos": pos,
                            "token_id": tok_id,
                            "token_str": tokenizer.decode([tok_id]),
                            "tag": tg,
                            "tag_name": TAG_NAMES[tg],
                            "conf": float(conf),
                            "probs": [float(p) for p in raw_probs[pos]] if (raw_probs and pos < len(raw_probs)) else [],
                            "action_tok": act_tok,
                            "action_str": act_str,
                            "note": override_notes[pos] if pos < len(override_notes) else "",
                        })
                    with torch.no_grad():
                        pooled_emb = cur[b].mean(dim=0).detach().cpu().numpy()

                    trajectories[b].append({
                        "iteration": iteration + 1,
                        "tokens_before": list(cur_tokens),
                        "tokens_after": list(new_ids),
                        "text_before": " ".join(tokenizer.decode([t for t in cur_tokens if t not in special]).split()),
                        "text_after": " ".join(tokenizer.decode([t for t in new_ids if t not in special]).split()),
                        "token_details": token_details,
                        "embedding": pooled_emb,
                        "changed": (new_ids != cur_tokens),
                    })

                if new_ids != canvases[b]:
                    changed = True
                next_canvases.append(new_ids)

            canvases = next_canvases
            # Check convergence with min_iterations requirement
            if not changed and iteration >= max(1, min_iterations):
                if log_operations:
                    print(f"\n⏹️ Refinement completed early at iteration {iteration + 1}: Canvas unchanged across iteration.")
                break
            if iteration == max_iterations - 1 and changed and log_operations:
                print(f"\n⏹️ Refinement completed: reached maximum iterations ({max_iterations}).")

            # Re-embed: batch all rows padded to the longest current length.
            max_L = max((len(c) for c in canvases), default=0)
            batch_ids = torch.full((B, max_L), pad, dtype=torch.long, device=device)
            batch_attn = torch.zeros(B, max_L, dtype=torch.long, device=device)
            for b in range(B):
                row = torch.tensor(canvases[b], dtype=torch.long, device=device)
                batch_ids[b, :len(row)] = row
                batch_attn[b, :len(row)] = 1
            embedded = embedder.embed_ids(batch_ids, batch_attn)  # (B, max_L, D)
            cur = [embedded[b, :len(canvases[b])] for b in range(B)]
            if refine_cond_mode == "initial" and dp1 is not None:
                dp1_cur = dp1  # Maintain original prompt reference condition!
            else:
                dp1_cur = embedded  # Legacy self-conditioning

        # Final decode.
        results = []
        for b in range(B):
            clean = [t for t in canvases[b] if t not in special]
            results.append(" ".join(tokenizer.decode(clean).split()))
        if return_trajectory:
            return results, trajectories
        return results




# ════════════════════════ Phase 2: full edit-aware SDE ════════════════════════
#
# These helpers/callables let the full Levenshtein edit grammar (INSERT/DELETE/
# EXPAND) compose with the SDE, and enable edit-conditioned scoring + two-phase
# sampling. The core model (DSBHybrid) gains:
#   * loss_edit()       — joint loss that accepts full tag labels (any of
#                         KEEP/REPLACE/INSERT/DELETE/EXPAND) + per-position gen
#                         targets, and uses an edit-conditioned score net.
#   * sample_text()     — two-phase generation: reverse SDE to embeddings, then
#                         run tagger/generator heads to morph a noisy canvas into
#                         clean text (INSERT grows, DELETE trims, EXPAND splits).


def corrupt_full(clean_ids, corruptor):
    """
    Corrupt a clean token sequence with the FULL edit grammar (via
    ``ForwardCorruptor``) and return aligned ``(noisy_ids, tag_labels,
    gen_targets)``.

    ``noisy_ids`` may be a different length than ``clean_ids`` because of
    INSERT/DELETE/EXPAND. ``tag_labels[i]`` is the edit needed at noisy
    position ``i`` to approach the clean sequence; ``gen_targets[i]`` is the
    clean token to produce when the tag is REPLACE/INSERT.
    """
    clean_list = list(clean_ids)
    noisy = list(clean_ids)
    # Apply length-changing corruptions first, then mask.
    noisy = corruptor._apply_replace(noisy)
    noisy = corruptor._apply_delete(noisy)
    noisy = corruptor._apply_insert(noisy)
    noisy = corruptor._apply_expand(noisy)
    noisy = corruptor._apply_mask(noisy)

    # String encoding for Levenshtein alignment using a single shared char map
    char_map: dict = {}
    noisy_str, char_map = corruptor._ids_to_string(noisy, char_map=char_map)
    clean_str, char_map = corruptor._ids_to_string(clean_list, char_map=char_map)
    try:
        import Levenshtein
        edit_ops = Levenshtein.editops(noisy_str, clean_str)
    except Exception:
        edit_ops = []

    # Rich Levenshtein alignment noisy -> clean.
    tag_token_ids = corruptor.compute_tag_labels(noisy, clean_list)
    # Translate tokenizer tag IDs to our 0..4 indices.
    tid2idx = {
        getattr(corruptor.tokenizer, "keep_id", getattr(corruptor, "keep_id", 50265)): KEEP,
        getattr(corruptor.tokenizer, "delete_id", getattr(corruptor, "delete_id", 50266)): DELETE,
        getattr(corruptor.tokenizer, "replace_id", getattr(corruptor, "replace_id", 50267)): REPLACE,
        getattr(corruptor.tokenizer, "insert_id", getattr(corruptor, "insert_id", 50268)): INSERT,
        getattr(corruptor.tokenizer, "expand_id", getattr(corruptor, "expand_id", 50269)): EXPAND,
    }
    tags = [tid2idx.get(tok, KEEP) for tok in tag_token_ids]

    # Map noisy position -> clean token using exact Levenshtein alignment
    noisy_to_clean = {}
    noisy_to_clean_idx = {}
    for op, noisy_pos, clean_pos in edit_ops:
        if clean_pos < len(clean_list):
            noisy_to_clean[noisy_pos] = clean_list[clean_pos]
            noisy_to_clean_idx[noisy_pos] = clean_pos

    gen = [-1] * len(noisy)
    last_clean_idx = 0
    for i, tag in enumerate(tags):
        if i in noisy_to_clean:
            clean_tok = noisy_to_clean[i]
            last_clean_idx = noisy_to_clean_idx.get(i, last_clean_idx)
            if tag in (REPLACE, INSERT):
                gen[i] = clean_tok
        elif tag in (REPLACE, INSERT):
            # Fallback for positions without direct editop (e.g. forced mask or expand)
            fallback_idx = min(last_clean_idx, len(clean_list) - 1)
            gen[i] = clean_list[fallback_idx]
    return noisy, tags, gen


def align_to_fixed(noisy_ids, tags, gen, S, pad_id, gen_ignore=-100):
    """
    Align a variable-length (noisy, tags, gen) triple (from ``corrupt_full``)
    to a fixed canvas of length ``S`` (pad with ``pad_id`` / ignore labels).
    Returns padded torch tensors (S,).
    """
    noisy = noisy_ids[:S] + [pad_id] * max(0, S - len(noisy_ids))
    tags = tags[:S] + [-100] * max(0, S - len(tags))
    gen = gen[:S] + [gen_ignore] * max(0, S - len(gen))
    return (torch.tensor(noisy), torch.tensor(tags), torch.tensor(gen))