from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class ModelConfig:
    """Configuration for VEYRA-LM decoder-only Transformer architecture."""

    model_name: str = "veyra-tiny"
    version: str = "0.1.0"
    vocab_size: int = 4096
    context_length: int = 256
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8  # Set < num_heads for Grouped-Query Attention (GQA)
    ffn_size: int = 688    # 256 * 8/3 ≈ 682 -> 688 for SwiGLU standard ratio
    activation: str = "swiglu"  # "swiglu" or "gelu"
    normalization: str = "rmsnorm"  # "rmsnorm" or "layernorm"
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    precision: str = "float32"  # "float32", "bfloat16", "float16"

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def calculate_parameter_count(self) -> dict[str, int]:
        """Calculates theoretical parameter counts for this architecture."""
        # 1. Embedding: vocab_size * hidden_size
        embed_params = self.vocab_size * self.hidden_size

        # 2. Attention per layer:
        # Q: hidden_size * (num_heads * head_dim) = hidden_size^2
        # K: hidden_size * (num_kv_heads * head_dim)
        # V: hidden_size * (num_kv_heads * head_dim)
        # O: (num_heads * head_dim) * hidden_size = hidden_size^2
        q_proj = self.hidden_size * (self.num_heads * self.head_dim)
        k_proj = self.hidden_size * (self.num_kv_heads * self.head_dim)
        v_proj = self.hidden_size * (self.num_kv_heads * self.head_dim)
        o_proj = (self.num_heads * self.head_dim) * self.hidden_size
        attn_params_per_layer = q_proj + k_proj + v_proj + o_proj

        # 3. FFN per layer:
        # For SwiGLU: gate_proj (h->ffn), up_proj (h->ffn), down_proj (ffn->h)
        # 3 * hidden_size * ffn_size
        # For standard MLP: 2 * hidden_size * ffn_size
        if self.activation.lower() == "swiglu":
            ffn_params_per_layer = 3 * self.hidden_size * self.ffn_size
        else:
            ffn_params_per_layer = 2 * self.hidden_size * self.ffn_size

        # 4. Normalization per layer (pre-attn norm + pre-ffn norm):
        # RMSNorm has 1 scale parameter per dimension (no bias)
        norm_params_per_layer = 2 * self.hidden_size

        layer_params = (attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer) * self.num_layers

        # 5. Final norm:
        final_norm_params = self.hidden_size

        # 6. LM Head:
        # If tied, 0 additional unique parameters; if untied: vocab_size * hidden_size
        lm_head_params = 0 if self.tie_embeddings else (self.vocab_size * self.hidden_size)

        total_params = embed_params + layer_params + final_norm_params + lm_head_params

        return {
            "embedding": embed_params,
            "layers": layer_params,
            "final_norm": final_norm_params,
            "lm_head": lm_head_params,
            "total": total_params,
            "trainable": total_params,
        }

    def estimate_memory_mb(self, precision: str | None = None) -> dict[str, float]:
        """Estimate model weights memory footprint in megabytes."""
        prec = (precision or self.precision).lower()
        bytes_per_param = 4
        if prec in ("float16", "bfloat16", "fp16", "bf16"):
            bytes_per_param = 2
        elif prec in ("int8", "qint8"):
            bytes_per_param = 1

        params = self.calculate_parameter_count()["total"]
        weights_mb = (params * bytes_per_param) / (1024 * 1024)
        # Training memory with AdamW: weights (2/4B) + grads (4B) + adam m (4B) + adam v (4B)
        adam_train_mb = (params * (bytes_per_param + 4 + 4 + 4)) / (1024 * 1024)

        return {
            "weights_mb": round(weights_mb, 2),
            "training_optimizer_mb": round(adam_train_mb, 2),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
