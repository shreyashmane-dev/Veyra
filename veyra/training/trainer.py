from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Iterator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..checkpoints.manager import CheckpointManager
from ..config.model_config import ModelConfig
from ..config.training_config import TrainingConfig
from ..data.pipeline import ShardedTokenDataset
from ..model.model import VeyraLM


class Trainer:
    """Production training engine for VEYRA-LM with safeguards, mixed precision, and scheduling."""

    def __init__(
        self,
        model: VeyraLM,
        training_config: TrainingConfig,
        train_dataset: ShardedTokenDataset,
        val_dataset: ShardedTokenDataset | None = None,
    ) -> None:
        self.model = model
        self.config = training_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # 1. Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        self.model.to(self.device)

        # 2. Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(self.config.output_dir)

        # 3. Setup optimizer with weight decay separation
        self.optimizer = self._create_optimizer()

        # 4. Data loaders with small dataset protection
        if len(self.train_dataset) == 0:
            raise ValueError(
                "Training dataset contains 0 sequences! "
                "Ensure sufficient raw text is placed under data/raw/ and sharded before training."
            )

        effective_batch_size = min(self.config.batch_size, len(self.train_dataset))
        if effective_batch_size < self.config.batch_size:
            print(
                f"[WARN] Train dataset only has {len(self.train_dataset)} sequences. "
                f"Automatically adjusting batch_size from {self.config.batch_size} to {effective_batch_size}."
            )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=False if len(self.train_dataset) < self.config.batch_size else True,
        )

        # 5. Tracking state
        self.current_step = 0
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> torch.optim.AdamW:
        """Separates 2D parameters (weights) for weight decay from 1D parameters (norms, biases)."""
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

    def get_lr(self, step: int) -> float:
        """Calculates learning rate via linear warmup and cosine decay."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * (step + 1) / max(1, self.config.warmup_steps)
        if step > self.config.max_steps:
            return self.config.min_learning_rate
        decay_ratio = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.config.min_learning_rate + coeff * (self.config.learning_rate - self.config.min_learning_rate)

    @torch.no_grad()
    def evaluate(self, max_batches: int = 20) -> dict[str, float]:
        """Evaluates model on validation dataset."""
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return {}

        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        total_loss = 0.0
        batches = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            output = self.model(input_ids, labels=labels)
            if output.loss is not None:
                total_loss += output.loss.item()
                batches += 1
            if batches >= max_batches:
                break

        self.model.train()
        avg_loss = total_loss / max(1, batches)
        ppl = math.exp(min(avg_loss, 50.0))
        return {"val_loss": avg_loss, "val_perplexity": ppl}

    def train(self) -> dict[str, Any]:
        """Executes full training loop."""
        self.model.train()
        print("\n" + "=" * 70)
        print(f"  VEYRA-LM TRAINING INITIALIZED: {self.config.run_name}")
        print("=" * 70)
        print(f"  Model Architecture : {self.model.config.model_name}")
        print(f"  Parameters         : {self.model.get_num_params():,}")
        print(f"  Compute Device     : {self.device}")
        print(f"  Precision          : {self.config.precision}")
        print(f"  Batch Size         : {self.config.batch_size}")
        print(f"  Grad Accumulation  : {self.config.gradient_accumulation_steps}")
        print(f"  Max Steps          : {self.config.max_steps}")
        print(f"  Dataset Sequences  : {len(self.train_dataset):,}")
        print("=" * 70 + "\n")

        train_iter: Iterator[dict[str, torch.Tensor]] = iter(self.train_loader)
        accum_loss = 0.0
        start_time = time.time()
        step_start_time = time.time()
        tokens_processed = 0

        tokens_per_batch = self.config.batch_size * self.model.config.context_length

        while self.current_step < self.config.max_steps:
            self.optimizer.zero_grad()
            step_loss = 0.0

            # Gradient accumulation loop
            for micro_step in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        raise RuntimeError(
                            f"DataLoader has 0 batches to yield! Total dataset sequences: {len(self.train_dataset)}, "
                            f"configured batch_size: {self.config.batch_size}. Please provide more training data."
                        )

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                output = self.model(input_ids, labels=labels)
                loss = output.loss

                if loss is None:
                    raise RuntimeError("Model did not return loss during training forward pass.")

                # Safeguard: Check for NaN
                if self.config.detect_nan and (torch.isnan(loss) or torch.isinf(loss)):
                    raise FloatingPointError(
                        f"CRITICAL: NaN/Inf detected at step {self.current_step}. Aborting to preserve state."
                    )

                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                step_loss += loss.item() / self.config.gradient_accumulation_steps
                tokens_processed += input_ids.numel()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
            if self.config.max_grad_norm_warn > 0 and grad_norm > self.config.max_grad_norm_warn:
                print(f"[WARN] Large grad norm at step {self.current_step}: {grad_norm:.2f}")

            # Update learning rate
            lr = self.get_lr(self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.step()
            self.current_step += 1
            accum_loss += step_loss

            # Periodic Logging
            if self.current_step % self.config.log_interval == 0:
                elapsed = time.time() - step_start_time
                tok_sec = tokens_processed / max(1e-5, elapsed)
                avg_train_loss = accum_loss / self.config.log_interval
                ppl = math.exp(min(avg_train_loss, 50.0))
                remaining_steps = self.config.max_steps - self.current_step
                eta_sec = remaining_steps * (elapsed / self.config.log_interval)

                print(
                    f"Step {self.current_step:05d}/{self.config.max_steps:05d} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"PPL: {ppl:.2f} | "
                    f"LR: {lr:.2e} | "
                    f"Tok/s: {tok_sec:,.0f} | "
                    f"ETA: {int(eta_sec)}s"
                )

                accum_loss = 0.0
                tokens_processed = 0
                step_start_time = time.time()

            # Periodic Evaluation
            if self.current_step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    val_loss = eval_metrics["val_loss"]
                    val_ppl = eval_metrics["val_perplexity"]
                    print(f"--> [EVAL] Step {self.current_step} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.checkpoint_mgr.save_checkpoint(
                            self.model,
                            self.optimizer,
                            self.current_step,
                            val_loss,
                            self.model.config,
                            self.config,
                            tag="best.pt",
                        )

            # Periodic Checkpoint Saving
            if self.current_step % self.config.save_interval == 0:
                self.checkpoint_mgr.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.current_step,
                    step_loss,
                    self.model.config,
                    self.config,
                )

        # Final save
        latest_path = self.checkpoint_mgr.save_checkpoint(
            self.model,
            self.optimizer,
            self.current_step,
            step_loss,
            self.model.config,
            self.config,
            tag="latest.pt",
        )
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"  TRAINING COMPLETED: {self.current_step} steps in {total_time:.1f}s")
        print(f"  Latest Checkpoint Saved: {latest_path}")
        print("=" * 70 + "\n")

        return {
            "final_step": self.current_step,
            "final_loss": step_loss,
            "best_val_loss": self.best_val_loss,
            "latest_checkpoint": str(latest_path),
        }
