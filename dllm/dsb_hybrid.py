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
    """
    Predict the edit op (NUM_TAGS classes) at every position.

    Conditions on the noise level t and the source embedding DP1: without t
    the head cannot distinguish "x looks clean because t~0" from "this
    position was never corrupted", so its Bayes-optimal prediction collapses
    to the class prior (observed: tag loss stuck at ~0.4 for an entire run).
    With DP1 it can compare the current state against the source — which is
    also what makes the head usable at generation time (t=1, cond=DP1).
    """

    def __init__(self, dim: int, time_embed_dim: int = 128, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
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


class GenHead(nn.Module):
    """Predict the clean token at REPLACE positions (sparse projection).

    Same t + DP1 conditioning rationale as ``TaggerHead``.
    When lm_head is provided (from AutoModelForMaskedLM), initializes weights from
    the pretrained LM head (dense, layer_norm, decoder weight, and vocab bias),
    giving instant zero-shot MLM prediction quality on Layer 12 contextual hidden states.
    When embed_weight is provided without lm_head, initializes/ties the final projection.
    """

    def __init__(self, dim: int, vocab_size: int,
                 time_embed_dim: int = 128, cond_dim: int = 0,
                 embed_weight: Optional[torch.Tensor] = None,
                 tie_weights: bool = True,
                 lm_head: Optional[nn.Module] = None):
        super().__init__()
        self.cond_dim = cond_dim
        self.time_embed_dim = time_embed_dim
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
            nn.Linear(dim, vocab_size),
        )
        if lm_head is not None:
            with torch.no_grad():
                # Zero out weights for cond and time slices initially so GenHead
                # behaves identically to the pretrained lm_head on x at step 0
                self.net[0].weight.zero_()
                if hasattr(lm_head, "dense") and hasattr(lm_head.dense, "weight"):
                    self.net[0].weight[:, :dim].copy_(lm_head.dense.weight)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (N, D) selected positions; t: (N,); cond: (N, D)
        h = x
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("head built with cond_dim > 0 requires cond")
            h = torch.cat([cond, h], dim=-1)
        if self.time_mlp is not None:
            t_emb = self.time_mlp(t.reshape(-1, 1))          # (N, T)
            h = torch.cat([h, t_emb], dim=-1)
        return self.net(h)


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
        condition_heads: bool = False,
        time_embed_dim: int = 128,
        embed_weight: Optional[torch.Tensor] = None,
        tie_weights: bool = True,
        lm_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.bridge: DiffSchrodingerBridge = bridge
        dim = bridge.dim
        self.condition_heads = condition_heads
        head_cond_dim = dim if condition_heads else 0
        self.tagger = TaggerHead(dim, time_embed_dim=time_embed_dim,
                                 cond_dim=head_cond_dim)
        self.generator = GenHead(dim, vocab_size, time_embed_dim=time_embed_dim,
                                 cond_dim=head_cond_dim,
                                 embed_weight=embed_weight,
                                 tie_weights=tie_weights,
                                 lm_head=lm_head)
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
        # 1) Discrete targets from tokens (cheap, no SDE needed).
        tag_labels, gen_labels = self.discrete_targets(clean_ids, noisy_ids, attention_mask)

        # 2) SDE: sample x_t and supervised score.
        B = dp1.shape[0]
        if t is None:
            t = torch.rand(B, device=dp1.device)
        x_t, u_target = self.bridge.forward_sample(dp1, dp2, t)
        u_pred = self.bridge.score_predict(x_t, t, dp1=dp1, attention_mask=attention_mask)
        if attention_mask is not None and u_pred.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            loss_sm = ((u_pred - u_target) ** 2 * mask).sum() / (mask.sum() * u_pred.shape[-1]).clamp(min=1.0)
        else:
            loss_sm = F.mse_loss(u_pred, u_target)

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

        # 3) Tagger head on the SDE intermediate embedding x_t (t + DP1
        #    conditioned so corruption stays identifiable at low t).
        tag_logits = self.tagger(x_t, t, cond=dp1)     # (B, S, NUM_TAGS)
        if self._tag_weights is not None:
            w = self._tag_weights.to(tag_logits.device)
        else:
            w = None
        loss_tag = F.cross_entropy(tag_logits.permute(0, 2, 1).float(), tag_labels,
                                   weight=w, ignore_index=-100)

        # 4) Generator head, evaluated ONLY at REPLACE positions (sparse) so we
        #    never materialize (B, S, V) vocab logits.
        #
        #    Scheduled Sampling: with probability `expose_ratio`, evaluate GenHead
        #    on the SDE's own reconstruction (bridge.sample) at t=1 instead of
        #    the pristine bridge sample x_t. This trains GenHead on realistic
        #    inference-time inputs with ~31% L2 error, not just sigma=0.009 noise.
        use_exposed = expose_ratio > 0.0 and torch.rand(1).item() < expose_ratio
        if use_exposed:
            with torch.no_grad():
                x_gen = self.bridge.sample(dp1, attention_mask=attention_mask, return_clean=True)    # (B, S, D) — SDE reconstruction
            t_gen = torch.ones(B, device=dp1.device)
        else:
            x_gen = x_t
            t_gen = t

        select = tag_labels.reshape(-1) == REPLACE
        if select.any():
            xp = x_gen.reshape(-1, x_gen.shape[-1])[select]
            lb = gen_labels.reshape(-1)[select]
            t_sel = t_gen.repeat_interleave(x_gen.shape[1], dim=0)[select]
            c_sel = dp1.reshape(-1, x_gen.shape[-1])[select]
            gl = self.generator(xp, t_sel, cond=c_sel)         # (n, V)
            loss_gen = F.cross_entropy(gl.float(), lb.clamp(0, gl.shape[-1] - 1))
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
        bl = self.bridge.baseline_loss(dp1_e).item()
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

        # 3) Generator Top-1 and Top-5 Token Match on Corrupted Slots
        gen_select = (gen_e != self.gen_ignore_index)
        top1_acc, top5_acc = 0.0, 0.0
        nn_top1_acc, nn_top5_acc = 0.0, 0.0
        lm_top1_acc, lm_top5_acc = 0.0, 0.0
        if gen_select.any():
            xp = sampled_x.reshape(-1, sampled_x.shape[-1])[gen_select.reshape(-1)]
            lb = gen_e.reshape(-1)[gen_select.reshape(-1)]
            t_sel = t_ones.repeat_interleave(sampled_x.shape[1], dim=0)[gen_select.reshape(-1)]
            c_sel = dp1_e.reshape(-1, sampled_x.shape[-1])[gen_select.reshape(-1)]

            gl = self.generator(xp, t_sel, cond=c_sel)  # (N_gen, V)

            top1_acc = (gl.argmax(-1) == lb).float().mean().item()
            k = min(5, gl.shape[-1])
            top5_acc = gl.topk(k, dim=-1).indices.eq(lb.unsqueeze(1)).any(1).float().mean().item()

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
            "cos_sim": cos_sim,
            "keep_acc": keep_acc * 100.0,
            "rep_f1": f1_rep * 100.0,
            "rep_prec": prec_rep * 100.0,
            "rep_rec": rec_rep * 100.0,
            "top1_acc": top1_acc * 100.0,
            "top5_acc": top5_acc * 100.0,
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
        select = gen_labels.reshape(-1) != self.gen_ignore_index
        if select.any():
            x_eval = x_gen if x_gen is not None else x_t
            t_eval = t_gen if t_gen is not None else t
            xp = x_eval.reshape(-1, x_eval.shape[-1])[select]
            lb = gen_labels.reshape(-1)[select]
            t_sel = t_eval.repeat_interleave(x_eval.shape[1], dim=0)[select]
            c_sel = dp1.reshape(-1, x_eval.shape[-1])[select] if dp1 is not None else None
            gl = self.generator(xp, t_sel, cond=c_sel)
            loss_gen = F.cross_entropy(gl.float(), lb.clamp(0, gl.shape[-1] - 1))
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
        if hasattr(self.bridge.score_net, "num_tags"):
            u_pred = self.bridge.score_predict(x_t, t, dp1=dp1, tag_ids=condition_tags, attention_mask=attention_mask)
        else:
            u_pred = self.bridge.score_predict(x_t, t, dp1=dp1, attention_mask=attention_mask)
        
        if attention_mask is not None and u_pred.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            loss_sm = ((u_pred - u_target) ** 2 * mask).sum() / (mask.sum() * u_pred.shape[-1]).clamp(min=1.0)
        else:
            loss_sm = F.mse_loss(u_pred, u_target)

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
    def _sample_topk(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                     generated_ids: Optional[List[int]] = None, repetition_penalty: float = 1.0) -> List[int]:
        """Vectorized top-k / top-p sampling over vocab logits -> token ids (per position).

        Args:
            repetition_penalty: >1.0 divides logits of already-generated tokens (HuggingFace convention).
                                 1.0 = disabled (no penalty). Recommended: 1.2–1.5 for early training.
        """
        if logits.numel() == 0:
            return []
        # Move the small (N_replace, 250002) projection to CPU for instantaneous AVX quickselect
        # rather than triggering a slow 80s Level-Zero JIT compilation stall on Intel XPU.
        logits = logits.detach().to("cpu", dtype=torch.float32) / max(temperature, 1e-8)

        # Repetition penalty: down-weight tokens that have already been generated in this canvas.
        if repetition_penalty != 1.0 and generated_ids:
            prev = torch.tensor(list(set(generated_ids)), dtype=torch.long)
            prev = prev[prev < logits.shape[-1]]
            if len(prev) > 0:
                # Positive logits are divided; negative logits are multiplied (HF convention).
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
        dp1: Optional[torch.Tensor] = None,  # (B, S, D) source embedding (head cond)
        seed_ids: Optional[List[List[int]]] = None,  # per-row starting canvas token ids
        repetition_penalty: float = 1.3,      # >1.0 suppresses repeated tokens; 1.0 = off
        decode_mode: str = "genhead",         # "genhead" (use learned GenHead) or "nearest" (cosine similarity to embed_weight)
        embed_weight: Optional[torch.Tensor] = None,  # token embedding matrix (defaults to self.embed_weight)
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
        if self.tagger.cond_dim > 0 and dp1 is None:
            raise ValueError("conditioned heads require dp1 (the source embedding)")

        # Per-row variable-length canvases: start from seed ids if given (e.g.
        # the prompt/corrupted tokens — KEEP then preserves REAL tokens, matching
        # training where uncorrupted positions hold real tokens), else all-mask.
        if seed_ids is not None:
            canvases: List[List[int]] = [list(row) for row in seed_ids]
        else:
            canvases = [[M] * S for _ in range(B)]
        cur: List[torch.Tensor] = [x[b] for b in range(B)]  # (S, D) per row

        for _ in range(max_iterations):
            changed = False
            next_canvases: List[List[int]] = []
            next_emb: List[torch.Tensor] = []

            for b in range(B):
                emb = cur[b].unsqueeze(0)                 # (1, L, D)
                L = emb.shape[1]
                # Heads are evaluated at t=1 (the SDE output state) against the
                # source DP1, so "REPLACE" means "this position still differs
                # from the source". DP1 is aligned to the current canvas length
                # (INSERT/DELETE change L across refinement rounds).
                t_row = torch.ones(1, device=emb.device)
                c_row = None
                if dp1 is not None:
                    c = dp1[b:b+1]
                    if c.shape[1] >= L:
                        c_row = c[:, :L]
                    else:
                        pad = torch.zeros(1, L - c.shape[1], c.shape[2],
                                          device=c.device, dtype=c.dtype)
                        c_row = torch.cat([c, pad], dim=1)
                tag_logits = self.tagger(emb, t_row, cond=c_row)[0]     # (L, T)
                tags = tag_logits.argmax(-1).tolist()

                # Sparse generator evaluation: only evaluate the 250k-vocab GenHead
                # and top-k sampling at positions that actually need token generation.
                gen_positions = [i for i, tg in enumerate(tags) if tg in (REPLACE, INSERT)]
                gen_toks = list(canvases[b])
                if gen_positions:
                    pos_tensor = torch.tensor(gen_positions, device=cur[b].device)
                    cur_sel = cur[b][pos_tensor]
                    t_sel = t_row.expand(len(gen_positions))
                    c_sel = c_row[0][pos_tensor] if c_row is not None else None

                    if decode_mode == "lm_head":
                        if hasattr(embedder, "decode_logits"):
                            gen_logits = embedder.decode_logits(cur_sel)
                        elif self.lm_head is not None:
                            gen_logits = self.lm_head(cur_sel)
                        else:
                            gen_logits = self.generator(cur_sel, t_sel, cond=c_sel)
                    elif decode_mode == "nearest":
                        w = embed_weight if embed_weight is not None else self.embed_weight
                        if w is None:
                            raise ValueError("decode_mode='nearest' requires embed_weight (either passed or in DSBHybrid)")
                        normed_cur = F.normalize(cur_sel, dim=-1)
                        normed_w = F.normalize(w.to(cur_sel.device), dim=-1)
                        gen_logits = (normed_cur @ normed_w.T) * 20.0
                    else:
                        gen_logits = self.generator(cur_sel, t_sel, cond=c_sel)  # (N_gen, V)

                    # Pass currently committed non-special canvas tokens as context
                    # for repetition penalty so the generator avoids repeating them.
                    committed = [t for t in canvases[b] if t not in special]
                    sampled_tokens = self._sample_topk(
                        gen_logits, temperature, top_k, top_p,
                        generated_ids=committed,
                        repetition_penalty=repetition_penalty,
                    )
                    for pos, tok_id in zip(gen_positions, sampled_tokens):
                        if pos < len(gen_toks):
                            gen_toks[pos] = tok_id

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

            canvases = next_canvases
            if not changed:
                break

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
    clean_list = list(clean_ids)
    noisy = list(clean_ids)
    # Apply length-changing corruptions first, then mask.
    noisy = corruptor._apply_replace(noisy)
    noisy = corruptor._apply_delete(noisy)
    noisy = corruptor._apply_insert(noisy)
    noisy = corruptor._apply_expand(noisy)
    noisy = corruptor._apply_mask(noisy)

    # String encoding for Levenshtein alignment
    noisy_str, _ = corruptor._ids_to_string(noisy)
    clean_str, _ = corruptor._ids_to_string(clean_list)
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
    for op, noisy_pos, clean_pos in edit_ops:
        if clean_pos < len(clean_list):
            noisy_to_clean[noisy_pos] = clean_list[clean_pos]

    gen = [-1] * len(noisy)
    last_clean_idx = 0
    for i, tag in enumerate(tags):
        if i in noisy_to_clean:
            clean_tok = noisy_to_clean[i]
            if clean_tok in clean_list:
                last_clean_idx = clean_list.index(clean_tok)
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