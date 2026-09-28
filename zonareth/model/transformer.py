from __future__ import annotations

import torch
import torch.nn as nn

from ..config.model_config import ModelConfig
from .attention import CausalSelfAttention
from .mlp import build_feed_forward
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer decoder block with residual connections."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = build_feed_forward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-norm self-attention with residual
        norm_x = self.input_norm(x)
        attn_out, new_past_kv = self.self_attn(
            norm_x,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Pre-norm MLP with residual
        norm_x = self.post_attention_norm(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out

        return x, new_past_kv
