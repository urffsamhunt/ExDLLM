"""
DLLM Model: A two-headed bidirectional Transformer for edit-based diffusion.

Architecture:
    Backbone:  RoBERTa (bidirectional, no causal mask)
    Head 1:    Tagger — classifies each token into edit operations
               (<KEEP>, <DELETE>, <REPLACE>, <INSERT>, <EXPAND>)
    Head 2:    Generator — predicts vocabulary tokens at positions
               tagged for REPLACE or INSERT, conditioned on tag embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, XLMRobertaModel, XLMRobertaConfig
from typing import Optional, Dict, Tuple, List


class DLLM(nn.Module):
    """
    Discrete Diffusion Language Model with a bidirectional Transformer
    backbone and two specialized heads.

    Usage:
        model = DLLM(tokenizer, hidden_dropout_prob=0.1)
        outputs = model(noisy_ids, attention_mask)
        # outputs: {tag_logits, gen_logits}
    """

    def __init__(
        self,
        tokenizer,  # DLLMTokenizer
        backbone_name: str = "roberta-base",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        tag_weights: Optional[List[float]] = None,
        length_head_max: int = 48,
        length_weights: Optional[List[float]] = None,
        len_smoothing: float = 0.15,
    ):
        """
        Args:
            tokenizer: DLLMTokenizer instance.
            backbone_name: HuggingFace model identifier for the backbone.
            hidden_dropout_prob: Dropout probability in the backbone.
            attention_probs_dropout_prob: Attention dropout in the backbone.
            tag_weights: Optional class weights for the tag loss, ordered
                [KEEP, DELETE, REPLACE, INSERT, EXPAND]. Used to counter the
                majority-class bias (e.g. downweight KEEP/DELETE, upweight
                the rare INSERT/EXPAND).
            length_head_max: Maximum answer length (in response tokens) that the
                length head can predict. The canvas is built with exactly the
                predicted number M of response slot pairs.
            length_weights: Optional per-class weights (1..length_head_max) for
                the length loss, countering the answer-length class imbalance.
            len_smoothing: Fraction of probability mass spread to the adjacent
                length classes (length is ordinal: a near miss should be less
                wrong than a far miss). 0 = hard targets.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.num_edit_tags = 5  # KEEP, DELETE, REPLACE, INSERT, EXPAND
        self.pad_id = tokenizer.pad_id

        # ── Backbone: Bidirectional RoBERTa (or XLM-R for multilingual) ──
        if backbone_name.startswith("xlm-roberta"):
            config = XLMRobertaConfig.from_pretrained(
                backbone_name,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            self.backbone = XLMRobertaModel.from_pretrained(backbone_name, config=config)
        else:
            config = RobertaConfig.from_pretrained(
                backbone_name,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            self.backbone = RobertaModel.from_pretrained(backbone_name, config=config)
        self.hidden_dim = config.hidden_size
        self.original_vocab_size = tokenizer.original_vocab_size

        # Resize token embeddings to accommodate our added edit tokens
        self.backbone.resize_token_embeddings(self.vocab_size)

        # ── Head 1: Tagger ─────────────────────────────────────────────
        # Predicts edit operations: KEEP, DELETE, REPLACE, INSERT, EXPAND
        self.tagger_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.num_edit_tags),
        )

        # ── Head 2: Generator ──────────────────────────────────────────
        # For positions needing text (REPLACE, INSERT), predicts vocab tokens.
        # Conditioned on the edit tag via a learned tag embedding that is
        # summed with the hidden state before the LM projection.

        # Embedding for the 5 edit tags (used to condition the generator)
        self.tag_embedding = nn.Embedding(self.num_edit_tags, self.hidden_dim)

        # Language modeling head: project hidden state -> vocabulary.
        # The final projection shares the backbone's input embeddings (true
        # parameter tying: one weight matrix, not a copy), projecting over the
        # full vocab (edit tokens included; gen labels never target them).
        self.generator_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.vocab_size),
        )

        # Weight tying: share generator output weights with input embeddings
        # (only for the original vocab part)
        self._tie_generator_weights()

        # ── Tag ID Mapping (eager init for device safety) ─────────────
        self._init_tag_mapping()

        # ── Head 3: Length (auxiliary) ───────────────────────────────────
        # Predicts the response length M from the prompt (prompt-only pooled
        # hidden state). The canvas is built with exactly M response slots,
        # so the tagger never needs to prune capacity.
        self.length_head_max = length_head_max
        self.length_head = nn.Linear(self.hidden_dim, length_head_max)
        self.len_smoothing = len_smoothing
        if length_weights is not None:
            self.register_buffer(
                "_len_weights",
                torch.tensor(length_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._len_weights = None

        # ── Loss Functions ──────────────────────────────────────────────
        # Tag loss uses optional class weights to counter the majority-class
        # bias (KEEP/DELETE dominate; INSERT/EXPAND are rare).
        if tag_weights is not None:
            self.register_buffer(
                "_tag_weights",
                torch.tensor(tag_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self._tag_weights = None
        self.gen_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _tie_generator_weights(self):
        """
        Tie the generator's final linear layer to the backbone's input token
        embeddings as ONE shared parameter (saves a full vocab-size weight
        matrix, its gradient, and its Adam states).
        """
        input_embeddings = self.backbone.get_input_embeddings()
        gen_linear = self.generator_head[-1]  # nn.Linear
        gen_linear.weight = input_embeddings.weight

    # ── Mapping Between Tag IDs and Internal Indices ──────────────────

    def _init_tag_mapping(self):
        """Build and register the tag ID mapping buffer eagerly."""
        tag_map = {
            self.tokenizer.keep_id: 0,
            self.tokenizer.delete_id: 1,
            self.tokenizer.replace_id: 2,
            self.tokenizer.insert_id: 3,
            self.tokenizer.expand_id: 4,
        }
        max_id = max(tag_map.keys())
        mapping = torch.full((max_id + 1,), -100, dtype=torch.long)
        for k, v in tag_map.items():
            mapping[k] = v
        self.register_buffer("_tag_mapping_tensor", mapping, persistent=False)

    def _tag_id_to_index(self, tag_id: torch.Tensor) -> torch.Tensor:
        """
        Convert tokenizer tag IDs (keep_id, delete_id, etc.) to internal
        0-based indices (0=KEEP, 1=DELETE, 2=REPLACE, 3=INSERT, 4=EXPAND).
        Preserves -100 (ignore_index) values.
        """
        # Ensure mapping is on the same device as input
        mapping = self._tag_mapping_tensor
        if mapping.device != tag_id.device:
            mapping = mapping.to(tag_id.device)

        # Handle -100 (ignore_index): return -100 directly
        result = torch.full_like(tag_id, -100)
        valid_mask = (tag_id >= 0) & (tag_id < len(mapping))
        result[valid_mask] = mapping[tag_id[valid_mask]]
        return result

    def _tag_index_to_id(self, tag_index: int) -> int:
        """Convert internal tag index (0-4) to tokenizer tag ID."""
        mapping = [
            self.tokenizer.keep_id,
            self.tokenizer.delete_id,
            self.tokenizer.replace_id,
            self.tokenizer.insert_id,
            self.tokenizer.expand_id,
        ]
        return mapping[tag_index]

    # ── Forward Pass ──────────────────────────────────────────────────

    def forward(
        self,
        noisy_ids: torch.Tensor,          # (batch, seq_len)
        attention_mask: torch.Tensor,      # (batch, seq_len)
        tag_labels: Optional[torch.Tensor] = None,  # (batch, seq_len) — for teacher forcing
        prompt_mask: Optional[torch.Tensor] = None, # (batch, seq_len) — True on prompt positions
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the two-headed model.

        Args:
            noisy_ids: Corrupted token IDs.
            attention_mask: Mask for padding positions.
            tag_labels: (Optional) Ground-truth tag labels for conditioning
                        the generator during training. If None, uses predicted
                        tags (inference mode).
            prompt_mask: (Optional) Boolean mask of prompt positions. Used to
                        pool the prompt representation for the length head.
                        If None, pools over all non-padding positions.

        Returns:
            Dict with:
                tag_logits: (batch, seq_len, num_edit_tags) — Tagger output
                gen_logits: (batch, seq_len, vocab_size) — Generator output
                length_logits: (batch, length_head_max) — Length head output
                hidden_states: (batch, seq_len, hidden_dim) — For analysis
        """
        batch_size, seq_len = noisy_ids.shape

        # ── Backbone ───────────────────────────────────────────────────
        backbone_outputs = self.backbone(
            input_ids=noisy_ids,
            attention_mask=attention_mask,
        )
        hidden_states = backbone_outputs.last_hidden_state  # (B, S, H)

        # ── Head 3: Length — pool hidden states over prompt positions ──
        if prompt_mask is not None and prompt_mask.sum() > 0:
            counts = prompt_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = (hidden_states * prompt_mask.unsqueeze(-1)).sum(dim=1) / counts
        else:
            counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / counts
        length_logits = self.length_head(pooled)  # (B, length_head_max)

        # ── Head 1: Tagger ─────────────────────────────────────────────
        tag_logits = self.tagger_head(hidden_states)  # (B, S, num_edit_tags)

        # ── Head 2: Generator ──────────────────────────────────────────
        # Determine which tags to use for conditioning
        if tag_labels is not None and self.training:
            # Teacher forcing: use ground-truth tags
            tag_indices = self._tag_id_to_index(tag_labels)  # (B, S)
        else:
            # Inference: use predicted tags
            tag_indices = tag_logits.argmax(dim=-1)  # (B, S)

        # Clamp tag_indices to valid range for embedding lookup
        # (positions with -100 will be clamped to 0, but their contribution
        #  doesn't matter since gen_loss ignores them)
        safe_indices = tag_indices.clamp(0, self.num_edit_tags - 1)

        # Get tag embeddings and add to hidden states
        tag_embeds = self.tag_embedding(safe_indices)  # (B, S, H)
        conditioned_hidden = hidden_states + tag_embeds     # (B, S, H)

        # Project to vocabulary
        gen_logits = self.generator_head(conditioned_hidden)  # (B, S, V_orig)

        return {
            "tag_logits": tag_logits,
            "gen_logits": gen_logits,
            "length_logits": length_logits,
            "hidden_states": hidden_states,
        }

    # ── Loss Computation ──────────────────────────────────────────────

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        tag_labels: torch.Tensor,
        gen_labels: torch.Tensor,
        gen_mask: torch.Tensor,
        resp_length: torch.Tensor,  # (B,) target response length in tokens (-100 = ignore)
        prompt_mask: Optional[torch.Tensor] = None,  # for response-only gen loss logging
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined tag + generator + length loss.

        Args:
            outputs: Dict from forward() with 'tag_logits', 'gen_logits',
                     and 'length_logits'.
            tag_labels: (B, S) — per-token edit tag labels.
            gen_labels: (B, S) — target vocabulary tokens.
            gen_mask: (B, S) — bool mask for positions needing generation.
            resp_length: (B,) — target response length for the length head.
            prompt_mask: (B, S) — optional; used to log a response-only
                     generator loss (excluding the trivially easy prompt
                     self-reproduction positions).

        Returns:
            total_loss: Scalar loss (tag_loss + gen_loss + len_loss).
            loss_dict: Individual losses for logging (includes
                     'gen_resp_loss' when prompt_mask is provided).
        """
        # Tag loss: predict the correct edit operation at every position
        tag_logits = outputs["tag_logits"].permute(0, 2, 1)  # (B, C, S) for CrossEntropy
        # Convert tokenizer tag IDs to internal 0-4 indices
        tag_indices = self._tag_id_to_index(tag_labels)

        weight = None
        if self._tag_weights is not None:
            weight = self._tag_weights.to(tag_logits.device)
        tag_loss = F.cross_entropy(tag_logits, tag_indices, weight=weight)

        # Generator loss: only at positions flagged by gen_mask
        gen_logits = outputs["gen_logits"]  # (B, S, V_orig)

        # Mask out non-generation positions
        active_gen_mask = gen_mask & (gen_labels != -100)
        if active_gen_mask.sum() > 0:
            # Flatten for CrossEntropy
            gen_logits_flat = gen_logits[active_gen_mask]  # (N, V_orig)
            gen_labels_flat = gen_labels[active_gen_mask]  # (N,)

            # Clamp gen_labels to be within original vocab (safety)
            gen_labels_flat = gen_labels_flat.clamp(0, self.original_vocab_size - 1)

            gen_loss = self.gen_loss_fn(gen_logits_flat, gen_labels_flat)
        else:
            gen_loss = torch.tensor(0.0, device=tag_loss.device)

        # Length loss: predict the answer length M from the prompt.
        # The canvas is built with exactly M response slots, so M is the
        # model's length decision (not a cutoff).
        #
        # Counter the length-class imbalance in two ways:
        #   1. inverse-frequency class weights (short answers are rare), and
        #   2. neighbor smoothing: length is ordinal, so a near miss should
        #      be penalized less than a far miss.
        length_logits = outputs["length_logits"]
        valid = (resp_length >= 1) & (resp_length <= self.length_head_max)
        if valid.sum() > 0:
            logits_v = length_logits[valid]
            t = (resp_length.clamp(1, self.length_head_max) - 1).long()[valid]
            B, C = logits_v.shape
            weight = self._len_weights.to(logits_v.device) if self._len_weights is not None else None

            alpha = 1.0 - self.len_smoothing
            if alpha < 1.0:
                soft = torch.zeros(B, C, device=logits_v.device)
                soft.scatter_(1, t.unsqueeze(1), alpha)
                side = (1 - alpha) / 2
                side_v = torch.full((B, 1), side, device=logits_v.device)
                soft.scatter_add_(1, (t - 1).clamp(0, C - 1).unsqueeze(1), side_v)
                soft.scatter_add_(1, (t + 1).clamp(0, C - 1).unsqueeze(1), side_v)
                soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
                per_sample = -(F.log_softmax(logits_v, dim=-1) * soft).sum(dim=1)
                if weight is not None:
                    per_sample = per_sample * weight[t]
                len_loss = per_sample.mean()
            else:
                len_loss = F.cross_entropy(logits_v, t, weight=weight)
        else:
            len_loss = torch.tensor(0.0, device=tag_loss.device)

        # Response-only generator loss (informational; not added to total).
        # The aggregate gen loss includes prompt self-reproduction positions,
        # which are trivially easy and mask the content-quality signal.
        gen_resp_loss = None
        if prompt_mask is not None:
            active_resp = active_gen_mask & ~prompt_mask
            if active_resp.sum() > 0:
                with torch.no_grad():
                    gl = gen_logits[active_resp]
                    lb = gen_labels[active_resp].clamp(0, self.original_vocab_size - 1)
                    gen_resp_loss = F.cross_entropy(gl, lb).item()

        total_loss = tag_loss + gen_loss + len_loss

        loss_dict = {
            "tag_loss": tag_loss.item(),
            "gen_loss": gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss,
            "gen_resp_loss": gen_resp_loss,
            "len_loss": len_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    # ── Utility: Get Model Parameters Grouped for Optimizer ────────────
    def get_param_groups(self, lr: float, weight_decay: float) -> list:
        """
        Return parameter groups with weight decay applied only to
        weights (not biases and layer norms), following standard practice.
        Parameters are deduplicated by identity so the generator's tied
        projection is optimized exactly once.
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


def load_dllm_state(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Load a DLLM state dict, tolerating older checkpoints:
    - checkpoints from the pre-tying era contain a separate
      'generator_head.4.weight' key (now tied to the embeddings),
    - checkpoints from before the length head / tagging changes may lack
      or carry extra keys, and
    - DataParallel-wrapped models are unwrapped first.

    Returns a message describing what was done.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    try:
        model.load_state_dict(state_dict)
        return "loaded (strict)"
    except RuntimeError:
        state_dict = {k: v for k, v in state_dict.items() if k != "generator_head.4.weight"}
        model.load_state_dict(state_dict, strict=False)
        return "loaded (non-strict: old-format checkpoint)"
