"""
DLLM Trainer: Training loop with combined Tagger + Generator loss.

Handles:
- Optimizer setup with weight decay
- Learning rate scheduling with warmup
- Training loop with gradient accumulation
- Logging and checkpointing
"""

from __future__ import annotations

import os
import math
import random
from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .utils import JSONMetricsLogger


class DLLMTrainer:
    """
    Trainer for the Discrete Diffusion Language Model.

    Usage:
        trainer = DLLMTrainer(model, tokenizer, config)
        trainer.train(train_loader, val_loader=None)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,  # DLLMTokenizer
        config: dict,
        device: Optional[str] = None,
    ):
        self.model = model
        # raw model (unwrapped) for loss computation, param groups, and saves;
        # self.model may be a DataParallel wrapper for forward passes.
        self.raw_model = model.module if isinstance(model, nn.DataParallel) else model
        self.tokenizer = tokenizer
        self.config = config

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

        # Training hyperparameters
        self.batch_size = int(config["training"]["batch_size"])
        self.lr = float(config["training"]["learning_rate"])
        self.weight_decay = float(config["training"]["weight_decay"])
        self.warmup_steps = int(config["training"]["warmup_steps"])
        self.max_steps = int(config["training"]["max_steps"])
        self.gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
        self.log_every = int(config["training"]["log_every"])
        self.eval_every = int(config["training"]["eval_every"])
        self.scheduled_sampling_prob = float(
            config["training"].get("scheduled_sampling_prob", 0.3)
        )

        # Sub-iteration trajectory training config
        sub_iters_cfg = config["training"].get("sub_iterations", None)
        self.sub_iterations = sub_iters_cfg  # List[float] or None
        if self.sub_iterations is not None:
            raw_weights = config["training"].get(
                "sub_iteration_weights",
                [1.0] * len(self.sub_iterations),  # uniform if not specified
            )
            # Normalize weights so they sum to 1.0
            w_sum = sum(raw_weights)
            self._stage_weights = [w / w_sum for w in raw_weights]
        else:
            self._stage_weights = [1.0]

        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.metrics_logger = None

    def _setup_optimizer(self):
        """Initialize AdamW optimizer with proper weight decay grouping."""
        param_groups = self.raw_model.get_param_groups(self.lr, self.weight_decay)
        self.optimizer = AdamW(param_groups, lr=self.lr)

    def _setup_scheduler(self):
        """Linear warmup followed by linear decay."""

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            # Linear decay from 1.0 to 0.0
            progress = float(step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return max(0.0, 1.0 - progress)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    # ── Training Loop ─────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_dir: str = "./checkpoints",
        log_fn: Optional[Callable] = None,
        bleu_eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
        bleu_every: int = 0,
        metrics_path: Optional[str] = "training_metrics.jsonl",
    ):
        """
        Main training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation.
            save_dir: Directory for saving checkpoints.
            log_fn: Optional callback for logging metrics (e.g., wandb.log).
            bleu_eval_fn: Optional callback returning {'bleu': ...} — called
                every bleu_every steps (e.g., translation quality tracking).
            bleu_every: Evaluate BLEU every N optimizer steps (0 = disabled).
            metrics_path: Filename (or None to disable) for a JSONL metrics log
                written into `save_dir`. Defaults to 'training_metrics.jsonl'.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.train()

        if metrics_path is not None:
            self.metrics_logger = JSONMetricsLogger(os.path.join(save_dir, metrics_path))

        running_tag_loss = 0.0
        running_gen_loss = 0.0
        running_gen_resp = 0.0
        running_len_loss = 0.0
        accumulated_steps = 0  # Track batches since last optimizer step

        progress_bar = tqdm(total=self.max_steps, initial=self.global_step, desc="Training")

        while self.global_step < self.max_steps:
            self.current_epoch += 1

            for batch in train_loader:
                if self.global_step >= self.max_steps:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if not batch:
                    continue

                # ── Detect multi-stage vs single-stage batch ─────────────────
                # Multi-stage: noisy_ids is (B, K, S); single-stage: (B, S).
                is_multistage = batch["noisy_ids"].dim() == 3
                K = batch["noisy_ids"].shape[1] if is_multistage else 1
                stage_weights = self._stage_weights  # normalized, len K

                # ── Scheduled sampling decision: made ONCE per batch ─────────
                # Applying self-draft separately for each stage would cost K extra
                # forward passes; we check once and apply per stage inside the loop.
                do_self_draft = (
                    self.scheduled_sampling_prob > 0.0
                    and random.random() < self.scheduled_sampling_prob
                )

                # ── Sub-iteration inner loop ─────────────────────────────────
                for k in range(K):
                    # Extract sub-batch for stage k: always (B, S)
                    if is_multistage:
                        sub_batch = {
                            "noisy_ids":      batch["noisy_ids"][:, k, :],
                            "tag_labels":     batch["tag_labels"][:, k, :],
                            "gen_labels":     batch["gen_labels"][:, k, :],
                            "gen_mask":       batch["gen_mask"][:, k, :],
                            "prompt_mask":    batch["prompt_mask"][:, k, :],
                            # attention_mask and resp_length are shared (not stage-indexed)
                            "attention_mask": batch["attention_mask"],
                            "resp_length":    batch["resp_length"],
                        }
                    else:
                        sub_batch = batch

                    # Scheduled sampling: replace REPLACE-position tokens with model's
                    # own predictions so it trains on its own decode errors.
                    if do_self_draft:
                        noisy_ids = self._self_draft(sub_batch)
                    else:
                        noisy_ids = sub_batch["noisy_ids"]

                    # Forward pass
                    outputs = self.model(
                        noisy_ids=noisy_ids,
                        attention_mask=sub_batch["attention_mask"],
                        tag_labels=sub_batch["tag_labels"],
                        prompt_mask=sub_batch["prompt_mask"],
                    )

                    # Compute loss
                    loss, loss_dict = self.raw_model.compute_loss(
                        outputs=outputs,
                        tag_labels=sub_batch["tag_labels"],
                        gen_labels=sub_batch["gen_labels"],
                        gen_mask=sub_batch["gen_mask"],
                        resp_length=sub_batch["resp_length"],
                        prompt_mask=sub_batch["prompt_mask"],
                    )

                    # Scale loss: by stage weight and gradient accumulation steps.
                    # stage_weights are normalized (sum to 1), so the total gradient
                    # magnitude across all K stages equals one unweighted backward pass.
                    stage_w = stage_weights[k] if is_multistage else 1.0
                    scaled_loss = loss * stage_w / self.gradient_accumulation_steps
                    scaled_loss.backward()
                    # No retain_graph needed: each stage slices a fresh sub-batch,
                    # producing an independent computation graph freed after backward.

                    running_tag_loss += loss_dict["tag_loss"] * stage_w
                    running_gen_loss += loss_dict["gen_loss"] * stage_w
                    running_len_loss += loss_dict["len_loss"] * stage_w
                    if loss_dict.get("gen_resp_loss") is not None:
                        running_gen_resp += loss_dict["gen_resp_loss"] * stage_w

                # One outer batch = one accumulation step (across all K sub-iters)
                accumulated_steps += 1

                # Gradient accumulation: step optimizer every N batches
                if accumulated_steps >= self.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    accumulated_steps = 0
                    progress_bar.update(1)

                    # Logging
                    if self.global_step % self.log_every == 0:
                        n = max(1, self.log_every)
                        avg_tag = running_tag_loss / n
                        avg_gen = running_gen_loss / n
                        avg_gen_resp = running_gen_resp / n if running_gen_resp > 0 else None
                        avg_len = running_len_loss / n
                        lr = self.scheduler.get_last_lr()[0]

                        metrics = {
                            "step": self.global_step,
                            "tag_loss": avg_tag,
                            "gen_loss": avg_gen,
                            "gen_resp_loss": avg_gen_resp,
                            "len_loss": avg_len,
                            "total_loss": avg_tag + avg_gen + avg_len,
                            "lr": lr,
                        }

                        if log_fn:
                            log_fn(metrics)

                        if self.metrics_logger is not None:
                            self.metrics_logger.log(metrics)

                        postfix = {
                            "tag": f"{avg_tag:.3f}",
                            "gen": f"{avg_gen:.3f}",
                            "len": f"{avg_len:.3f}",
                            "lr": f"{lr:.2e}",
                        }
                        if avg_gen_resp is not None:
                            postfix["gen_resp"] = f"{avg_gen_resp:.3f}"
                        progress_bar.set_postfix(postfix)

                        running_tag_loss = 0.0
                        running_gen_loss = 0.0
                        running_gen_resp = 0.0
                        running_len_loss = 0.0

                    # Validation
                    if val_loader is not None and self.global_step % self.eval_every == 0:
                        val_metrics = self.evaluate(val_loader)
                        if log_fn:
                            log_fn({f"val_{k}": v for k, v in val_metrics.items()})
                            log_fn({"step": self.global_step})

                        if self.metrics_logger is not None:
                            self.metrics_logger.log({"step": self.global_step, **{f"val_{k}": v for k, v in val_metrics.items()}})

                        # Save best model (only checkpoint kept — no periodic saves)
                        val_loss = val_metrics["total_loss"]
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(
                                os.path.join(save_dir, "best_model.pt"),
                                metrics=val_metrics,
                            )

                    # Overwriteable resume checkpoint (for session-limited runs
                    # like Kaggle's 12h cap): enables --resume continuation.
                    if self.global_step % self.eval_every == 0:
                        self.save_checkpoint(os.path.join(save_dir, "resume.pt"))

                    # BLEU tracking (translation quality), every bleu_every steps
                    if bleu_eval_fn is not None and bleu_every > 0 and self.global_step % bleu_every == 0:
                        bleu_metrics = bleu_eval_fn()
                        print(f"  >> BLEU @ step {self.global_step}: {bleu_metrics}")
                        if log_fn:
                            log_fn({"step": self.global_step, **bleu_metrics})

        progress_bar.close()

        if self.metrics_logger is not None:
            self.metrics_logger.close()
            self.metrics_logger = None

    # ── Scheduled Sampling ─────────────────────────────────────────────

    def _self_draft(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Build a "self-draft" input: forward pass in eval mode (no grad, like
        inference), sample tokens from the generator at REPLACE positions, and
        put the model's own samples where the corruption had noise tokens or
        masks. Prompt positions are never touched.

        The training labels stay the same, so the model learns to apply the
        REPLACE edit (toward the clean token) to inputs that look like its own
        decoding output rather than uniform random noise.
        """
        if "prompt_mask" not in batch:
            return batch["noisy_ids"]

        replace_mask = (
            (batch["tag_labels"] == self.tokenizer.replace_id)
            & ~batch["prompt_mask"]
        )
        if replace_mask.sum() == 0:
            return batch["noisy_ids"]

        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(
                    noisy_ids=batch["noisy_ids"],
                    attention_mask=batch["attention_mask"],
                    tag_labels=None,
                    prompt_mask=batch["prompt_mask"],
                )
        finally:
            if was_training:
                self.model.train()

        # Gather logits ONLY at the replace positions (small (N, V) tensor
        # instead of materializing the full (B, S, V) for sampling).
        positions = replace_mask.nonzero()
        logits = out["gen_logits"][positions[:, 0], positions[:, 1]]  # (N, V)
        sampled = self._sample_tokens(logits)  # (N,)
        new_noisy = batch["noisy_ids"].clone()
        new_noisy[replace_mask] = sampled
        return new_noisy

    @torch.no_grad()
    def _sample_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        chunk: int = 128,
    ) -> torch.Tensor:
        """
        Top-k + nucleus sampling over an (N, V) logits tensor, processed in
        chunks to bound peak memory with large vocabularies.
        """
        logits = logits / max(temperature, 1e-8)
        out = torch.empty(logits.size(0), dtype=torch.long, device=logits.device)
        for start in range(0, logits.size(0), chunk):
            part = logits[start:start + chunk]
            if top_k > 0:
                k = min(top_k, part.size(-1))
                min_topk = torch.topk(part, k, dim=-1).values[:, -1].unsqueeze(-1)
                part = part.masked_fill(part < min_topk, float("-inf"))
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(part, descending=True, dim=-1)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                part = part.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))
            out[start:start + chunk] = torch.multinomial(
                torch.softmax(part, dim=-1), 1
            ).squeeze(-1)
        return out

    # ── Evaluation ────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run evaluation on validation set.
        Returns averaged metrics.
        """
        self.model.eval()

        total_tag_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if not batch:
                continue

            outputs = self.model(
                noisy_ids=batch["noisy_ids"],
                attention_mask=batch["attention_mask"],
                tag_labels=batch["tag_labels"],
                prompt_mask=batch["prompt_mask"],
            )

            _, loss_dict = self.raw_model.compute_loss(
                outputs=outputs,
                tag_labels=batch["tag_labels"],
                gen_labels=batch["gen_labels"],
                gen_mask=batch["gen_mask"],
                resp_length=batch["resp_length"],
                prompt_mask=batch["prompt_mask"],
            )

            total_tag_loss += loss_dict["tag_loss"]
            total_gen_loss += loss_dict["gen_loss"]
            num_batches += 1

        self.model.train()

        if num_batches == 0:
            return {"tag_loss": 0.0, "gen_loss": 0.0, "total_loss": 0.0}

        avg_tag = total_tag_loss / num_batches
        avg_gen = total_gen_loss / num_batches

        return {
            "tag_loss": avg_tag,
            "gen_loss": avg_gen,
            "total_loss": avg_tag + avg_gen,
        }

    # ── Checkpointing ─────────────────────────────────────────────────

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None):
        """Save model, optimizer, and training state."""
        # The generator projection shares the embedding parameter; drop the
        # duplicate state-dict key to halve the checkpoint size.
        model_state = self.raw_model.state_dict()
        model_state.pop("generator_head.4.weight", None)
        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model, optimizer, and training state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        from .model import load_dllm_state
        msg = load_dllm_state(self.model, checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {path} (step {checkpoint['global_step']}) [{msg}]")
