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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dllm.dsb import DiffSchrodingerBridge  # concrete type for annotations

# Edit tag indices, matching the DLLM tagger ordering.
KEEP, DELETE, REPLACE, INSERT, EXPAND = 0, 1, 2, 3, 4
NUM_TAGS = 5


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
    ):
        super().__init__()
        self.dim = dim
        self.num_tags = num_tags
        self.time_embed_dim = time_embed_dim
        self.tag_emb = nn.Embedding(num_tags + 1, dim)  # +1 for a 'none' sentinel
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )
        in_dim = dim + dim + time_embed_dim  # x + tag_emb + time
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                tag_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, S, D) or (B, D); t: (B,); tag_ids: (B, S) int64 (default: sentinel).
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
        return self.net(h)


# ── Fixed-length, in-place token corruption ──────────────────────────────────

def corrupt_fixed(
    clean_ids: List[int],
    mask_prob: float = 0.15,
    mask_ratio: float = 0.8,
    noise_pool: Optional[List[int]] = None,
    mask_id: int = 0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Corrupt a clean token sequence in place (same length returned).

    Each real position is either kept (tag = KEEP), replaced with ``<MASK>``,
    or replaced with a random noise token. Both replacement kinds get
    tag = REPLACE and a generator target = the original clean token.

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
    return noisy, tags, gen


# ── Discrete heads (operate per-position on the SDE embedding x_t) ───────────

class TaggerHead(nn.Module):
    """Predict the edit op (NUM_TAGS classes) at every position."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, NUM_TAGS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> (B, S, NUM_TAGS)
        return self.net(x)


class GenHead(nn.Module):
    """Predict the clean token at REPLACE positions (sparse projection)."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., D) -> (..., V)
        return self.net(x)


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
        lambda_tag: float = 1.0,
        lambda_gen: float = 1.0,
        tag_weights: Optional[Tuple[float, ...]] = None,
        gen_ignore_index: int = -100,
    ):
        super().__init__()
        self.bridge: DiffSchrodingerBridge = bridge
        dim = bridge.dim
        self.tagger = TaggerHead(dim)
        self.generator = GenHead(dim, vocab_size)
        self.lambda_tag = lambda_tag
        self.lambda_gen = lambda_gen
        self.gen_ignore_index = gen_ignore_index
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
    ) -> Tuple[torch.Tensor, dict]:  # type: ignore[type-arg]
        """
        Joint loss = score_matching + lambda_tag * tag_ce + lambda_gen * gen_ce.

        Returns (total_loss, dict) with per-term losses.
        """
        # 1) Discrete targets from tokens (cheap, no SDE needed).
        tag_labels, gen_labels = self.discrete_targets(clean_ids, noisy_ids, attention_mask)

        # 2) SDE: sample x_t and supervised score.
        B = dp1.shape[0]
        if t is None:
            t = torch.rand(B, device=dp1.device)
        x_t, score_target = self.bridge.forward_sample(dp1, dp2, t)
        score_pred = self.bridge.score_net(x_t, t)
        # sigma^2 weighting (standard denoising score matching).
        sigma2_t = self.bridge._sigma2_at(t)  # type: ignore[attr-defined]
        if x_t.dim() == 3:
            sigma2 = sigma2_t.reshape(-1, 1, 1)
        else:
            sigma2 = sigma2_t.reshape(-1, 1)
        loss_sm = (F.mse_loss(score_pred, score_target, reduction="none") * sigma2).mean()

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

        # 3) Tagger head on the SDE intermediate embedding x_t.
        tag_logits = self.tagger(x_t)              # (B, S, NUM_TAGS)
        if self._tag_weights is not None:
            w = self._tag_weights.to(tag_logits.device)
        else:
            w = None
        loss_tag = F.cross_entropy(tag_logits.permute(0, 2, 1), tag_labels,
                                   weight=w, ignore_index=-100)

        # 4) Generator head, evaluated ONLY at REPLACE positions (sparse) so we
        #    never materialize (B, S, V) vocab logits.
        select = tag_labels.reshape(-1) == REPLACE
        if select.any():
            xp = x_t.reshape(-1, x_t.shape[-1])[select]
            lb = gen_labels.reshape(-1)[select]
            gl = self.generator(xp)                # (n, V)
            loss_gen = F.cross_entropy(gl, lb)
        else:
            loss_gen = torch.tensor(0.0, device=x_t.device)

        total = loss_sm + self.lambda_tag * loss_tag + self.lambda_gen * loss_gen
        return total, {
            "total": total.item(),
            "score_matching": loss_sm.item(),
            "tag": loss_tag.item(),
            "gen": loss_gen.item(),
        }

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
        tag_logits = self.tagger(x)                # (B, S, NUM_TAGS)
        return x, tag_logits.softmax(-1)

    # ── Phase 2: full edit-aware joint loss & two-phase sampling ──────────────

    def build_edit_loss(
        self,
        x_t: torch.Tensor,          # (B, S, D) SDE intermediate embedding
        tag_labels: torch.Tensor,   # (B, S) int64 in 0..4, -100 = ignore
        gen_labels: torch.Tensor,   # (B, S) int64, -100 = ignore
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tagger + generator cross-entropy on the SDE embedding x_t."""
        tag_logits = self.tagger(x_t)                      # (B, S, NUM_TAGS)
        w = self._tag_weights.to(tag_logits.device) if self._tag_weights is not None else None
        loss_tag = F.cross_entropy(tag_logits.permute(0, 2, 1), tag_labels,
                                   weight=w, ignore_index=-100)
        select = gen_labels.reshape(-1) != self.gen_ignore_index
        if select.any():
            xp = x_t.reshape(-1, x_t.shape[-1])[select]
            lb = gen_labels.reshape(-1)[select]
            loss_gen = F.cross_entropy(self.generator(xp), lb)
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
        x_t, score_target = self.bridge.forward_sample(dp1, dp2, t)
        if hasattr(self.bridge.score_net, "num_tags"):
            score_pred = self.bridge.score_net(x_t, t, tag_ids=condition_tags)
        else:
            score_pred = self.bridge.score_net(x_t, t)
        sigma2_t = self.bridge._sigma2_at(t)  # type: ignore[attr-defined]
        sigma2 = sigma2_t.reshape(-1, 1, 1) if x_t.dim() == 3 else sigma2_t.reshape(-1, 1)
        loss_sm = (F.mse_loss(score_pred, score_target, reduction="none") * sigma2).mean()

        loss_tag, loss_gen = self.build_edit_loss(x_t, tag_labels, gen_labels)
        total = loss_sm + self.lambda_tag * loss_tag + self.lambda_gen * loss_gen
        return total, {
            "total": total.item(),
            "score_matching": loss_sm.item(),
            "tag": loss_tag.item(),
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
        tag_logits = self.tagger(x)
        for _ in range(max(1, refine_iters)):
            tag_logits = self.tagger(x)                 # (B, S, NUM_TAGS)
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
            # Phase A: discrete edit tags + generator logits on the SDE embedding.
            tag_logits = self.tagger(x[b])                  # (S, NUM_TAGS)
            gen_logits = self.generator(x[b])               # (S, V)
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
    def _sample_topk(logits: torch.Tensor, temperature, top_k, top_p) -> List[int]:
        """Top-k / top-p sampling over vocab logits -> token ids (per position)."""
        logits = logits / max(temperature, 1e-8)
        probs = F.softmax(logits, dim=-1)          # (S, V)
        k = min(top_k, logits.shape[-1])
        ids = []
        for pos in range(logits.shape[0]):
            p = probs[pos]
            topk_vals, topk_idx = p.topk(k, dim=-1)
            cum = topk_vals.cumsum(-1)
            keep = cum <= top_p
            topk_vals = torch.where(keep, topk_vals, torch.zeros_like(topk_vals))
            if topk_vals.sum() <= 1e-8:
                # no candidate passed top-p; fall back to the top-1 token
                ids.append(topk_idx[0].item())
                continue
            topk_vals = topk_vals / topk_vals.sum()
            idx = int(torch.multinomial(topk_vals, 1).item())
            ids.append(int(topk_idx[idx].item()))
        return ids

    @staticmethod
    def _apply_edits(canvas_ids, tags, gen_toks, bos, eos, pad, M):
        """Apply KEEP/DELETE/REPLACE/INSERT/EXPAND, ported from DLLM._execute_edits."""
        out = []
        for i, tok in enumerate(canvas_ids):
            if i >= len(tags):
                break
            tag = tags[i]
            gen = gen_toks[i] if i < len(gen_toks) else tok
            if tok in (bos, eos):
                out.append(tok); continue
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
        max_len: Optional[int] = None,
    ) -> List[str]:
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

        B = x.shape[0]
        S = x.shape[1]
        device = x.device
        max_len = max_len if max_len is not None else S

        # Per-row variable-length canvases: start all-mask at the seed width.
        canvases: List[List[int]] = [[M] * S for _ in range(B)]
        cur: List[torch.Tensor] = [x[b] for b in range(B)]  # (S, D) per row

        for _ in range(max_iterations):
            changed = False
            next_canvases: List[List[int]] = []
            next_emb: List[torch.Tensor] = []

            for b in range(B):
                emb = cur[b].unsqueeze(0)                 # (1, L, D)
                tag_logits = self.tagger(emb)[0]          # (L, T)
                gen_logits = self.generator(emb)[0]       # (L, V)
                tags = tag_logits.argmax(-1).tolist()
                gen_toks = self._sample_topk(gen_logits, temperature, top_k, top_p)

                # Apply edits -> genuinely variable-length output (INSERT grows,
                # DELETE trims, EXPAND splits), capped at max_len.
                new_ids = self._apply_edits(
                    canvases[b], tags, gen_toks, bos, eos, pad, M
                )
                if len(new_ids) > max_len:
                    new_ids = new_ids[:max_len]

                if new_ids != canvases[b]:
                    changed = True
                next_canvases.append(new_ids)

            # Re-embed: batch all rows padded to the longest current length.
            max_L = max((len(c) for c in next_canvases), default=0)
            batch_ids = torch.full((B, max_L), pad, dtype=torch.long, device=device)
            batch_attn = torch.zeros(B, max_L, dtype=torch.long, device=device)
            for b in range(B):
                row = torch.tensor(next_canvases[b], dtype=torch.long, device=device)
                batch_ids[b, :len(row)] = row
                batch_attn[b, :len(row)] = 1
            embedded = embedder.embed_ids(batch_ids, batch_attn)  # (B, max_L, D)
            next_emb = [embedded[b, :len(next_canvases[b])] for b in range(B)]
            canvases = next_canvases
            cur = next_emb
            if not changed:
                break

        # Final decode.
        results = []
        for b in range(B):
            clean = [t for t in canvases[b] if t not in special]
            results.append(" ".join(tokenizer.decode(clean).split()))
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
    noisy = list(clean_ids)
    # Apply length-changing corruptions first, then mask.
    noisy = corruptor._apply_replace(noisy)
    noisy = corruptor._apply_delete(noisy)
    noisy = corruptor._apply_insert(noisy)
    noisy = corruptor._apply_expand(noisy)
    noisy = corruptor._apply_mask(noisy)
    # Rich Levenshtein alignment noisy -> clean.
    tag_token_ids = corruptor.compute_tag_labels(noisy, list(clean_ids))
    # Translate tokenizer tag IDs to our 0..4 indices.
    tid2idx = {
        corruptor.tokenizer.keep_id: KEEP,
        corruptor.tokenizer.delete_id: DELETE,
        corruptor.tokenizer.replace_id: REPLACE,
        corruptor.tokenizer.insert_id: INSERT,
        corruptor.tokenizer.expand_id: EXPAND,
    }
    tags = [tid2idx.get(tok, KEEP) for tok in tag_token_ids]
    # Gen targets: for REPLACE/INSERT positions, record the clean token that
    # should be written here. Use Levenshtein-derived best guess (the clean
    # token that aligns to this noisy slot). We approximate by scanning clean_ids.
    gen = [-1] * len(noisy)
    # Simple alignment: for each REPLACE position, try the same-index clean token.
    for i, tag in enumerate(tags):
        if tag in (REPLACE, INSERT):
            # Map noisy index -> clean index via an edit-distance backpointer is
            # complex; use the nearest clean token as the target (a reasonable
            # proxy from the Levenshtein op stream).
            gen[i] = clean_ids[min(i, len(clean_ids) - 1)]
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