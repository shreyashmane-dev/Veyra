from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

from ..tokenizer.bpe import VeyraTokenizer


INSTRUCTION_DATASETS = {
    "openhermes": {
        "repo": "teknium/OpenHermes-2.5",
        "description": "General reasoning, coding, math, and conversational instruction following.",
    },
    "ultrachat": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "description": "High-quality multi-turn conversational dialogs.",
    },
    "code_alpaca": {
        "repo": "sahil2801/CodeAlpaca-20k",
        "description": "Programming instructions, algorithm implementations, and code explanations.",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "description": "Step-by-step mathematical reasoning and word problem solving.",
    },
}


class InstructionDataset(Dataset):
    """PyTorch Dataset for Supervised Fine-Tuning (SFT) with prompt loss masking.

    Prompt tokens are masked with -100 in labels, ensuring the model only computes loss
    and updates weights on the assistant's response tokens (the standard technique
    used to create models like Llama 3 Instruct and Mistral Instruct).
    """

    def __init__(
        self,
        samples: list[dict[str, str]],
        tokenizer: VeyraTokenizer,
        max_length: int = 512,
        system_prompt: str = "You are VEYRA, an advanced, intelligent personal artificial intelligence assistant.",
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.encoded_items: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._process_samples()

    def _process_samples(self) -> None:
        """Tokenizes each sample and generates prompt-masked target labels."""
        for item in self.samples:
            prompt_text = f"<SYSTEM> {self.system_prompt} <USER> {item['prompt']} <ASSISTANT> "
            response_text = f"{item['response']} <EOS>"

            prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True)
            response_ids = self.tokenizer.encode(response_text, add_bos=False)

            full_ids = prompt_ids + response_ids
            if len(full_ids) > self.max_length:
                # Truncate while keeping EOS
                full_ids = full_ids[: self.max_length]
                full_ids[-1] = self.tokenizer.eos_token_id

            # Create labels: -100 on prompt tokens (ignored by loss), target IDs on response tokens
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            # Pad to max_length
            pad_amount = max(0, self.max_length - len(full_ids))
            padded_input_ids = full_ids + [self.tokenizer.pad_token_id] * pad_amount
            padded_labels = labels + [-100] * pad_amount

            self.encoded_items.append((
                torch.tensor(padded_input_ids[:-1], dtype=torch.long),
                torch.tensor(padded_labels[1:], dtype=torch.long),
            ))

    def __len__(self) -> int:
        return len(self.encoded_items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.encoded_items[idx]
        return {"input_ids": input_ids, "labels": labels}


class InstructionDataIngester:
    """Streams and prepares instruction tuning datasets for VEYRA-INSTRUCT."""

    def __init__(self, raw_dir: str | Path = "data/raw/instruction") -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def stream_instruction_dataset(
        self,
        dataset_key: str = "openhermes",
        max_samples: int = 5000,
    ) -> list[dict[str, str]]:
        """Streams instruction pairs from Hugging Face."""
        if dataset_key not in INSTRUCTION_DATASETS:
            raise ValueError(f"Unknown instruction dataset: {dataset_key}")

        repo = INSTRUCTION_DATASETS[dataset_key]["repo"]
        print(f"--> Streaming instruction dataset from {repo} (max: {max_samples:,})...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets via: pip install datasets")

        samples: list[dict[str, str]] = []

        if dataset_key == "openhermes":
            ds = load_dataset(repo, split="train", streaming=True)
            for item in ds:
                # OpenHermes has 'conversations' list
                convs = item.get("conversations", [])
                if len(convs) >= 2:
                    prompt = convs[0].get("value", "")
                    response = convs[1].get("value", "")
                    if prompt and response:
                        samples.append({"prompt": prompt, "response": response})
                if len(samples) >= max_samples:
                    break

        elif dataset_key == "code_alpaca":
            ds = load_dataset(repo, split="train", streaming=True)
            for item in ds:
                instruction = item.get("instruction", "")
                inp = item.get("input", "")
                output = item.get("output", "")
                prompt = f"{instruction}\n{inp}".strip()
                if prompt and output:
                    samples.append({"prompt": prompt, "response": output})
                if len(samples) >= max_samples:
                    break

        elif dataset_key == "gsm8k":
            ds = load_dataset(repo, "main", split="train", streaming=True)
            for item in ds:
                prompt = item.get("question", "")
                response = item.get("answer", "")
                if prompt and response:
                    samples.append({"prompt": prompt, "response": response})
                if len(samples) >= max_samples:
                    break

        print(f"    Loaded {len(samples):,} instruction-response pairs.")
        return samples
