from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig
from .embeddings import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats Key or Value heads for Grouped-Query Attention (GQA).

    Args:
        x: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of repetitions (num_heads // num_kv_heads)
    """
    if n_rep == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """Causal Self-Attention supporting RoPE, Grouped-Query Attention (GQA), and KV caching."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Linear projections without bias (modern standard)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Reshape into [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Positional Embeddings
        q = apply_rotary_emb(q, cos, sin, position_ids)
        k = apply_rotary_emb(k, cos, sin, position_ids)

        # 4. KV Cache handling
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        new_past_kv = (k, v) if use_cache else None

        # 5. Expand KV heads for Grouped-Query Attention if needed
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # 6. Scaled Dot-Product Attention with causal mask
        kv_seq_len = k.shape[2]
        scale = 1.0 / math.sqrt(self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask (only when decoding without cache or during initial prompt processing)
        if seq_len > 1 or past_key_value is None:
            # Query positions: [kv_seq_len - seq_len, ..., kv_seq_len - 1]
            # Key positions: [0, ..., kv_seq_len - 1]
            q_idx = torch.arange(kv_seq_len - seq_len, kv_seq_len, device=x.device).unsqueeze(1)
            k_idx = torch.arange(kv_seq_len, device=x.device).unsqueeze(0)
            causal_mask = q_idx < k_idx  # True where key is in the future
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
        attn_weights = self.attn_dropout(attn_weights)

        # 7. Aggregate values and project output
        out = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        out = self.resid_dropout(out)

        return out, new_past_kv
