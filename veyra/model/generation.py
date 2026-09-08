from __future__ import annotations

from typing import Callable, Generator
import torch
import torch.nn.functional as F

from .model import VeyraLM


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_tokens: list[int] | None = None,
) -> int:
    """Samples next token ID from logits with temperature, top-k, top-p, and repetition penalty."""
    # Squeeze to 1D: [vocab_size]
    logits = logits.squeeze().clone()

    # Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        for prev_token in set(generated_tokens):
            if logits[prev_token] > 0:
                logits[prev_token] /= repetition_penalty
            else:
                logits[prev_token] *= repetition_penalty

    # Greedy if temperature is effectively zero
    if temperature <= 1e-5:
        return int(torch.argmax(logits).item())

    # Apply temperature
    logits = logits / temperature

    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")

    # Top-P (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # Compute probabilities and sample
    probs = F.softmax(logits, dim=-1)
    # Check for NaNs
    if torch.isnan(probs).any():
        return int(torch.argmax(logits).item())

    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


class GenerationEngine:
    """Autoregressive text generation engine for VeyraLM."""

    def __init__(self, model: VeyraLM, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_token_ids: list[int] | None = None,
        callback: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Generates tokens autoregressively given prompt token IDs."""
        self.model.eval()
        stop_ids = set(stop_token_ids or [])
        generated: list[int] = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Enforce context length limit
            curr_input = generated[-self.model.config.context_length :]
            input_tensor = torch.tensor([curr_input], dtype=torch.long, device=self.device)

            output = self.model(input_tensor)
            # Logits for the last token position
            next_logits = output.logits[:, -1, :]

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=generated,
            )

            generated.append(next_token)
            if callback is not None:
                callback(next_token)

            if next_token in stop_ids:
                break

        return generated
