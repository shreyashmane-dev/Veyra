from __future__ import annotations

from pathlib import Path
from veyra.core.cognitive import CognitiveCore
from veyra.core.state import CognitiveState
from veyra.memory.database import MemoryDB
from veyra.inference.engine import VeyraInferenceEngine
from veyra.model.interface import LanguageEngine
from veyra.model.starter import StarterLanguageEngine
from veyra.app import make_banner, main as veyra_main


def get_active_engine(memory: MemoryDB) -> tuple[LanguageEngine, str, str]:
    """Detects trained ZONARETH-LM checkpoint or falls back to StarterLanguageEngine."""
    candidates = [
        ("checkpoints/zonareth_instruct_125m/model.pt", "checkpoints/zonareth_instruct_125m/tokenizer"),
        ("checkpoints/zonareth_125m/best.pt", "data/tokenizer"),
        ("checkpoints/zonareth_125m/best.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/latest.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/latest.pt", "data/tokenizer"),
        ("checkpoints/zonareth_125m/model.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/model.pt", "data/tokenizer"),
        ("checkpoints/veyra_instruct_125m/model.pt", "checkpoints/veyra_instruct_125m/tokenizer"),
        ("checkpoints/zonareth_tiny/latest.pt", "data/tokenizer"),
        ("checkpoints/zonareth_tiny/best.pt", "data/tokenizer"),
    ]

    for ckpt_path, tok_path in candidates:
        if Path(ckpt_path).exists() and Path(tok_path).exists():
            try:
                engine = VeyraInferenceEngine.from_checkpoint(ckpt_path, tok_path)
                model_name = getattr(engine.model.config, "model_name", "125m")
                tag = "ZONARETH-INSTRUCT" if "instruct" in ckpt_path else "ZONARETH-LM"
                params = engine.model.get_num_params()
                return engine, f"{tag} ({model_name})", f"{ckpt_path} ({params:,} params)"
            except Exception:
                pass

    return StarterLanguageEngine(memory), "Starter Engine", "ZONARETH-LM slot ready"


def main() -> None:
    veyra_main()


if __name__ == "__main__":
    main()
