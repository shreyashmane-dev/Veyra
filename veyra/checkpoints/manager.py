from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import torch

from ..config.model_config import ModelConfig
from ..config.training_config import TrainingConfig
from ..model.model import VeyraLM


class CheckpointManager:
    """Manages atomic saving, loading, inspection, and resumption of VEYRA-LM checkpoints."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: VeyraLM,
        optimizer: torch.optim.Optimizer | None,
        step: int,
        loss: float,
        model_config: ModelConfig,
        training_config: TrainingConfig | None = None,
        tag: str | None = None,
    ) -> Path:
        """Atomically saves a training checkpoint."""
        checkpoint_name = tag or f"step_{step:07d}.pt"
        target_path = self.checkpoint_dir / checkpoint_name
        tmp_path = self.checkpoint_dir / f".tmp_{checkpoint_name}"

        payload: dict[str, Any] = {
            "step": step,
            "loss": loss,
            "perplexity": float(torch.exp(torch.tensor(loss)).item()) if loss < 50 else float("inf"),
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict() if training_config else None,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save to temp file and rename atomically
        torch.save(payload, tmp_path)
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)

        # Save human-readable metadata file
        meta_path = self.checkpoint_dir / f"{target_path.stem}_meta.json"
        meta = {
            "step": step,
            "loss": round(loss, 4),
            "perplexity": round(payload["perplexity"], 2),
            "model_name": model_config.model_name,
            "vocab_size": model_config.vocab_size,
            "context_length": model_config.context_length,
            "hidden_size": model_config.hidden_size,
            "num_layers": model_config.num_layers,
            "total_parameters": model.get_num_params(),
            "saved_at": payload["saved_at"],
            "checkpoint_file": str(target_path.name),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return target_path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> tuple[VeyraLM, dict[str, Any]]:
        """Loads a checkpoint into a newly constructed VeyraLM model."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

        payload = torch.load(p, map_location=device, weights_only=False)
        config = ModelConfig(**payload["model_config"])
        model = VeyraLM(config)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)

        return model, payload

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Lists all available checkpoints in the managed directory."""
        results: list[dict[str, Any]] = []
        for pt_file in sorted(self.checkpoint_dir.glob("*.pt")):
            meta_file = self.checkpoint_dir / f"{pt_file.stem}_meta.json"
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    results.append(meta)
            else:
                results.append({
                    "checkpoint_file": pt_file.name,
                    "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(pt_file.stat().st_mtime).isoformat(),
                })
        return results

    @staticmethod
    def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
        """Inspects checkpoint metadata and architecture without full model initialization."""
        p = Path(checkpoint_path)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        cfg_dict = payload.get("model_config", {})
        config = ModelConfig(**cfg_dict) if cfg_dict else None
        return {
            "path": str(p.resolve()),
            "step": payload.get("step", 0),
            "loss": payload.get("loss"),
            "perplexity": payload.get("perplexity"),
            "saved_at": payload.get("saved_at"),
            "model_config": cfg_dict,
            "calculated_parameters": config.calculate_parameter_count() if config else None,
        }
