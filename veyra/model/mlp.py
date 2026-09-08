from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Gated Linear Unit Feed-Forward Network.

    Formula:
        SwiGLU(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    SwiGLU provides superior representational capacity and smoother gradient dynamics
    compared to standard ReLU or GELU feed-forward layers.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(gated)
        return self.dropout(out)


class StandardFeedForward(nn.Module):
    """Standard GELU Feed-Forward Network fallback."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_size)
        self.fc2 = nn.Linear(config.ffn_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.gelu(self.fc1(x)))
        return self.dropout(self.fc2(x))


def build_feed_forward(config: ModelConfig) -> nn.Module:
    if config.activation.lower() == "swiglu":
        return SwiGLUFeedForward(config)
    return StandardFeedForward(config)
