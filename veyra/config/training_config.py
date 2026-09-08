from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class TrainingConfig:
    """Configuration for VEYRA-LM pretraining and fine-tuning runs."""

    run_name: str = "veyra-train-run"
    model_config_path: str = "configs/models/veyra_tiny.yaml"
    dataset_dir: str = "data/shards"
    output_dir: str = "checkpoints/veyra_tiny"
    device: str = "auto"  # "auto", "cuda", "cpu"
    precision: str = "float32"  # "float32", "bfloat16", "float16"

    # Batch and step parameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 200
    log_interval: int = 10

    # Optimization parameters
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Safeguards & tracking
    seed: int = 42
    detect_nan: bool = True
    max_grad_norm_warn: float = 5.0
    compile_model: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
