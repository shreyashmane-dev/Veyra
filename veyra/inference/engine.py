from __future__ import annotations

from pathlib import Path
from typing import Callable
import torch

from ..checkpoints.manager import CheckpointManager
from ..model.generation import GenerationEngine
from ..model.interface import LanguageEngine
from ..model.model import VeyraLM
from ..tokenizer.bpe import VeyraTokenizer


class VeyraInferenceEngine(LanguageEngine):
    """Production inference engine bridging trained VeyraLM checkpoints into the cognitive core."""

    def __init__(
        self,
        model: VeyraLM,
        tokenizer: VeyraTokenizer,
        device: torch.device | str = "cpu",
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 100,
        system_prompt: str = "You are VEYRA, an advanced, intelligent personal artificial intelligence assistant.",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.generator = GenerationEngine(self.model, device=self.device)
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = "auto",
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 100,
    ) -> VeyraInferenceEngine:
        """Loads inference engine directly from checkpoint file and tokenizer directory."""
        dev = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        model, _ = CheckpointManager.load_checkpoint(checkpoint_path, device=dev)
        tokenizer = VeyraTokenizer.load(tokenizer_path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=dev,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

    def format_prompt(self, text: str, context: list[tuple[str, str]]) -> str:
        """Formats conversational history into prompt sequence."""
        # Use base Q&A format only if explicitly marked as a raw base model
        is_base = getattr(self.model.config, "is_base", False) or "base" in str(getattr(self.model.config, "model_name", "")).lower()

        if not is_base:
            prompt_parts: list[str] = [f"<SYSTEM> {self.system_prompt}"]
            for role, msg in context[-4:]:
                if role.lower() == "user":
                    prompt_parts.append(f"<USER> {msg}")
                elif role.lower() == "assistant":
                    prompt_parts.append(f"<ASSISTANT> {msg}")
            prompt_parts.append(f"<USER> {text} <ASSISTANT>")
            return " ".join(prompt_parts) + " "
        else:
            # Base pretrained model: use natural Q&A format without noisy untrained special tokens
            parts = []
            for role, msg in context[-3:]:
                r_name = "User" if role.lower() == "user" else "Assistant"
                parts.append(f"{r_name}: {msg}")
            parts.append(f"User: {text}\nAssistant:")
            return "\n".join(parts)

    def generate(
        self,
        text: str,
        context: list[tuple[str, str]],
        callback: Callable[[str], None] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generates response for input text and conversational context."""
        formatted_prompt = self.format_prompt(text, context)
        prompt_ids = self.tokenizer.encode(formatted_prompt, add_bos=True)

        stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.user_token_id,
            self.tokenizer.system_token_id,
        ]

        def token_stream_cb(token_id: int) -> None:
            if callback is not None and token_id not in stop_token_ids:
                piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
                callback(piece)

        output_ids = self.generator.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            stop_token_ids=stop_token_ids,
            callback=token_stream_cb if callback else None,
        )

        gen_ids = output_ids[len(prompt_ids) :]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return response if response else "..."
