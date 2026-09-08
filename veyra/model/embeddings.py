from __future__ import annotations

import torch
import torch.nn as nn


def precompute_rope_frequencies(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cosine and sine frequency matrices for Rotary Positional Embeddings (RoPE).

    Args:
        dim: Head dimension (must be even).
        max_seq_len: Maximum context window length.
        theta: Base frequency scaling factor (default 10000.0).
        device: Target torch device.

    Returns:
        cos: [max_seq_len, dim // 2]
        sin: [max_seq_len, dim // 2]
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")

    # theta_i = 1.0 / (theta ** (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    # Outer product -> [max_seq_len, dim // 2]
    angles = torch.outer(positions, freqs)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Applies Rotary Positional Embeddings (RoPE) to Query or Key tensors.

    Args:
        x: [batch, num_heads, seq_len, head_dim]
        cos: [max_seq_len, head_dim // 2] or indexed slice
        sin: [max_seq_len, head_dim // 2] or indexed slice
        position_ids: [batch, seq_len] optional positions

    Returns:
        Tensor of same shape with rotary positional encodings applied.
    """
    # Duplicate cos and sin along last dimension: [seq_len, head_dim]
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    if position_ids is not None:
        # [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        # [1, 1, seq_len, head_dim]
        seq_len = x.shape[2]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    return (x * cos) + (rotate_half(x) * sin)
