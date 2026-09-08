from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any
import torch
from torch.utils.data import DataLoader

from ..checkpoints.manager import CheckpointManager
from ..config.training_config import TrainingConfig
from ..data.instruction import InstructionDataset
from ..model.model import VeyraLM


class SFTTrainer:
    """Supervised Fine-Tuning (SFT) Trainer to turn VEYRA-LM into an advanced instruction-following model."""

    def __init__(
        self,
        model: VeyraLM,
        train_dataset: InstructionDataset,
        val_dataset: InstructionDataset | None = None,
        output_dir: str | Path = "checkpoints/veyra_instruct_125m",
        learning_rate: float = 3e-5,
        min_learning_rate: float = 3e-6,
        max_steps: int = 1500,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        device: str = "auto",
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.max_steps = max_steps
        self.batch_size = min(batch_size, len(train_dataset))
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps

        # Device
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device))
        self.model.to(self.device)

        # Checkpointer
        self.checkpoint_mgr = CheckpointManager(self.output_dir)

        # Optimizer: low learning rate for SFT
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )

        # DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.current_step = 0

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / max(1, self.warmup_steps)
        if step > self.max_steps:
            return self.min_learning_rate
        ratio = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return self.min_learning_rate + coeff * (self.learning_rate - self.min_learning_rate)

    def train(self) -> dict[str, Any]:
        self.model.train()
        print("\n" + "=" * 70)
        print("  SUPERVISED FINE-TUNING (SFT) INITIALIZED: VEYRA-INSTRUCT")
        print("=" * 70)
        print(f"  Base Architecture : {self.model.config.model_name}")
        print(f"  Instruction Pairs : {len(self.train_dataset):,}")
        print(f"  Target Steps      : {self.max_steps:,}")
        print(f"  Learning Rate     : {self.learning_rate}")
        print(f"  Device            : {self.device}")
        print("=" * 70 + "\n")

        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        step_start_time = time.time()

        while self.current_step < self.max_steps:
            self.optimizer.zero_grad()
            step_loss = 0.0

            for _ in range(self.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                output = self.model(input_ids, labels=labels)
                loss = output.loss

                if loss is None or torch.isnan(loss):
                    continue

                scaled_loss = loss / self.gradient_accumulation_steps
                scaled_loss.backward()
                step_loss += loss.item() / self.gradient_accumulation_steps

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.get_lr(self.current_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            self.optimizer.step()
            self.current_step += 1
            accum_loss += step_loss

            if self.current_step % 10 == 0:
                elapsed = time.time() - step_start_time
                avg_loss = accum_loss / 10
                ppl = math.exp(min(avg_loss, 50.0))
                print(f"[SFT] Step {self.current_step:04d}/{self.max_steps:04d} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")
                accum_loss = 0.0
                step_start_time = time.time()

            if self.current_step % 200 == 0:
                self.checkpoint_mgr.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.current_step,
                    step_loss,
                    self.model.config,
                    tag=f"instruct_step_{self.current_step:05d}.pt",
                )

        final_path = self.checkpoint_mgr.save_checkpoint(
            self.model,
            self.optimizer,
            self.current_step,
            step_loss,
            self.model.config,
            tag="latest.pt",
        )
        print("\n" + "=" * 70)
        print(f"  SFT COMPLETE: VEYRA-INSTRUCT SAVED TO {final_path}")
        print("=" * 70 + "\n")
        return {"final_step": self.current_step, "checkpoint": str(final_path)}
