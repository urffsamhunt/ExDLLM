"""
ARLM Trainer: standard teacher-forced autoregressive training loop.

The optimizer, learning-rate schedule, gradient accumulation, and checkpoint
format mirror the DLLM trainer so that training the ARLM is directly
comparable (same batch config, same warmup/decay, same save layout). The only
difference is the objective: next-token prediction instead of edit-based
diffusion.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .dataset import build_labels
from dllm.utils import JSONMetricsLogger, to_cpu, resolve_device


class ARLMTrainer:
    """
    Trainer for the autoregressive language model.

    Usage:
        trainer = ARLMTrainer(model, tokenizer, config)
        trainer.train(train_loader, val_loader, save_dir)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,  # ARLMTokenizer
        config: dict,
        device: Optional[str] = None,
    ):
        self.model = model
        self.raw_model = model.module if isinstance(model, nn.DataParallel) else model
        self.tokenizer = tokenizer
        self.config = config

        # Determine device
        if device is None:
            self.device = resolve_device()
        else:
            self.device = device
        self.model.to(self.device)
        # bf16 autocast halves the dominant transient memory (activations and
        # the large-vocab logits) which can overflow XPU's shared host DRAM on
        # backward(). Safe to enable on cuda/xpu; no grad scaler needed for bf16.
        self.use_amp = self.device in ("cuda", "xpu")

        # Training hyperparameters (same keys as the DLLM config)
        self.batch_size = int(config["training"]["batch_size"])
        self.lr = float(config["training"]["learning_rate"])
        self.weight_decay = float(config["training"]["weight_decay"])
        self.warmup_steps = int(config["training"]["warmup_steps"])
        self.max_steps = int(config["training"]["max_steps"])
        self.gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
        self.log_every = int(config["training"]["log_every"])
        self.eval_every = int(config["training"]["eval_every"])
        self.pad_id = tokenizer.pad_id
        # Cap the number of batches evaluated per `eval_every` step (None = full
        # validation set). See the DLLM trainer for rationale.
        self.val_max_batches = config["training"].get("val_max_batches", None)

        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.metrics_logger = None
        self._save_threads = []  # in-flight async checkpoint writes

    def _setup_optimizer(self):
        """Initialize AdamW optimizer with proper weight-decay grouping."""
        param_groups = self.raw_model.get_param_groups(self.lr, self.weight_decay)
        self.optimizer = AdamW(param_groups, lr=self.lr)

    def _setup_scheduler(self):
        """Linear warmup followed by linear decay (identical to DLLM)."""

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            return max(0.0, 1.0 - progress)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    # ── Training Loop ─────────────────────────────────────────────────

    def train(
        self,
        train_loader,
        val_loader=None,
        save_dir: str = "./checkpoints_ar",
        log_fn: Optional[Callable] = None,
        metrics_path: Optional[str] = "training_metrics.jsonl",
    ):
        """
        Main training loop.

        Args:
            train_loader: DataLoader yielding {'input_ids', 'attention_mask'}.
            val_loader: Optional DataLoader for validation.
            save_dir: Directory for saving checkpoints.
            log_fn: Optional callback for logging metrics (e.g., wandb.log).
            metrics_path: Filename (or None to disable) for a JSONL metrics log
                written into `save_dir`. Defaults to 'training_metrics.jsonl'.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.train()

        if metrics_path is not None:
            self.metrics_logger = JSONMetricsLogger(os.path.join(save_dir, metrics_path))

        running_loss = 0.0
        accumulated_steps = 0

        progress_bar = tqdm(total=self.max_steps, initial=self.global_step, desc="Training")

        while self.global_step < self.max_steps:
            self.current_epoch += 1

            for batch in train_loader:
                if self.global_step >= self.max_steps:
                    break

                ids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                labels = build_labels(ids, attn, self.pad_id).to(self.device)

                autocast_ctx = (
                    torch.autocast(device_type=self.device, dtype=torch.bfloat16)
                    if self.use_amp
                    else torch.nullcontext()
                )
                with autocast_ctx:
                    loss = self.model(input_ids=ids, attention_mask=attn, labels=labels)["loss"]
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                running_loss += loss.item() * self.gradient_accumulation_steps
                accumulated_steps += 1

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
                        avg_loss = running_loss / n
                        lr = self.scheduler.get_last_lr()[0]
                        metrics = {
                            "step": self.global_step,
                            "lm_loss": avg_loss,
                            "total_loss": avg_loss,
                            "lr": lr,
                        }
                        if log_fn:
                            log_fn(metrics)
                        if self.metrics_logger is not None:
                            self.metrics_logger.log(metrics)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})
                        running_loss = 0.0

                    # Validation
                    if val_loader is not None and self.global_step % self.eval_every == 0:
                        val_metrics = self.evaluate(val_loader)
                        if log_fn:
                            log_fn({f"val_{k}": v for k, v in val_metrics.items()})
                            log_fn({"step": self.global_step})
                        if self.metrics_logger is not None:
                            self.metrics_logger.log({"step": self.global_step, **{f"val_{k}": v for k, v in val_metrics.items()}})

                        val_loss = val_metrics["total_loss"]
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(
                                os.path.join(save_dir, "best_model.pt"),
                                metrics=val_metrics,
                            )

                    # Overwriteable resume checkpoint (for session-limited runs)
                    if self.global_step % self.eval_every == 0:
                        self.save_checkpoint(os.path.join(save_dir, "resume.pt"))

        progress_bar.close()

        # Wait for any in-flight checkpoint writes to finish before returning.
        for t in self._save_threads:
            t.join()

        if self.metrics_logger is not None:
            self.metrics_logger.close()
            self.metrics_logger = None

    # ── Evaluation ────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        """
        Run evaluation on the validation set. Returns averaged metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(val_loader):
            if self.val_max_batches is not None and i >= self.val_max_batches:
                break
            ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            labels = build_labels(ids, attn, self.pad_id).to(self.device)
            loss = self.model(input_ids=ids, attention_mask=attn, labels=labels)["loss"]
            total_loss += loss.item()
            num_batches += 1

        self.model.train()

        if num_batches == 0:
            return {"lm_loss": 0.0, "total_loss": 0.0}

        avg_loss = total_loss / num_batches
        return {"lm_loss": avg_loss, "total_loss": avg_loss}

    # ── Checkpointing ─────────────────────────────────────────────────

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None):
        """
        Save model, optimizer, and training state.

        State dicts are snapshotted to CPU on the calling thread (fast, and
        immutable once detached), then serialized + written to disk on a
        background thread so the training loop is not blocked by the slow
        checkpoint write.
        """
        checkpoint = {
            "model_state_dict": to_cpu(self.raw_model.state_dict()),
            "optimizer_state_dict": to_cpu(self.optimizer.state_dict()),
            "scheduler_state_dict": to_cpu(self.scheduler.state_dict()),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        if metrics:
            checkpoint["metrics"] = metrics

        writer = threading.Thread(
            target=self._write_checkpoint, args=(checkpoint, path), daemon=True
        )
        self._save_threads.append(writer)
        writer.start()
        print(f"Checkpoint save started for {path}")

    def _write_checkpoint(self, checkpoint: dict, path: str):
        """Serialize and write a checkpoint on a background thread."""
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model, optimizer, and training state from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        from .model import load_arlm_state
        msg = load_arlm_state(self.model, checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {path} (step {checkpoint['global_step']}) [{msg}]")