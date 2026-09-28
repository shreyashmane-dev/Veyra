from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader

from ..config.model_config import ModelConfig
from ..data.pipeline import ShardedTokenDataset
from ..model.generation import GenerationEngine
from ..model.model import VeyraLM
from ..tokenizer.bpe import VeyraTokenizer


class ModelEvaluator:
    """Evaluates VeyraLM checkpoints on loss, perplexity, and qualitative benchmark prompts."""

    def __init__(
        self,
        model: VeyraLM,
        tokenizer: VeyraTokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.generator = GenerationEngine(self.model, device=self.device)

    @torch.no_grad()
    def evaluate_loss_and_perplexity(
        self,
        dataset: ShardedTokenDataset,
        batch_size: int = 4,
        max_batches: int = 50,
    ) -> dict[str, float]:
        """Calculates exact cross-entropy loss and perplexity across dataset sequences."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_tokens = 0
        batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            output = self.model(input_ids, labels=labels)
            if output.loss is not None:
                total_loss += output.loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
                batches += 1

            if batches >= max_batches:
                break

        if total_tokens == 0:
            return {"loss": 0.0, "perplexity": 1.0}

        avg_loss = total_loss / total_tokens
        ppl = math.exp(min(avg_loss, 50.0))
        return {
            "loss": avg_loss,
            "perplexity": ppl,
            "tokens_evaluated": total_tokens,
            "batches_evaluated": batches,
        }

    @torch.no_grad()
    def evaluate_prompts(
        self,
        prompts: list[str],
        max_new_tokens: int = 40,
        temperature: float = 0.0,  # greedy deterministic
    ) -> list[dict[str, str]]:
        """Generates completions for a list of evaluation prompts."""
        results: list[dict[str, str]] = []
        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
            output_ids = self.generator.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )
            # Only decode newly generated tokens
            gen_ids = output_ids[len(prompt_ids) :]
            completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "completion": completion.strip(),
            })
        return results
