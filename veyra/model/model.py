from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.model_config import ModelConfig
from .embeddings import precompute_rope_frequencies
from .normalization import RMSNorm
from .transformer import TransformerBlock


@dataclass
class VeyraLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class VeyraLM(nn.Module):
    """VEYRA-LM: Autoregressive Decoder-Only Transformer Language Model.

    Engineered from scratch for the VEYRA personal AI ecosystem.
    Features:
    - Rotary Positional Embeddings (RoPE)
    - RMSNorm pre-normalization
    - Grouped-Query Attention (GQA) option
    - SwiGLU feed-forward networks
    - Optional embedding and LM head weight tying
    - Numerical stabilization with scaled residual initialization
    - CrossEntropyLoss computation with target masking
    - KV-cache support for fast autoregressive generation
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_frequencies(
            dim=config.head_dim,
            max_seq_len=config.context_length,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Apply residual projection scaling: 1 / sqrt(2 * num_layers)
        residual_scale = 1.0 / math.sqrt(2.0 * config.num_layers)
        for layer in self.layers:
            nn.init.normal_(
                layer.self_attn.o_proj.weight,
                mean=0.0,
                std=config.initializer_range * residual_scale,
            )
            if hasattr(layer.mlp, "down_proj"):
                nn.init.normal_(
                    layer.mlp.down_proj.weight,
                    mean=0.0,
                    std=config.initializer_range * residual_scale,
                )

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights using standard normal distribution with configurable std."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Returns the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> VeyraLMOutput:
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds model context length ({self.config.context_length})"
            )

        # Token embeddings
        h = self.tok_embeddings(input_ids)

        # Retrieve RoPE tables
        cos = self.rope_cos
        sin = self.rope_sin
        if position_ids is not None:
            # Reindex if explicit position_ids provided (e.g. KV cache generation)
            pass

        # Forward through transformer blocks
        new_past_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if past_key_values is not None else None
            h, updated_kv = layer(
                h,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
                past_key_value=layer_past_kv,
                use_cache=use_cache,
            )
            if use_cache and updated_kv is not None:
                new_past_kvs.append(updated_kv)

        # Final RMSNorm
        h = self.norm(h)

        # Project to vocabulary logits
        logits = self.lm_head(h)

        # Loss calculation
        loss = None
        if labels is not None:
            # Flatten logits and labels for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return VeyraLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=new_past_kvs,
        )
