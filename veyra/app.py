from __future__ import annotations

from pathlib import Path

from .core.cognitive import CognitiveCore
from .core.state import CognitiveState
from .memory.database import MemoryDB
from .inference.engine import VeyraInferenceEngine
from .model.starter import StarterLanguageEngine


def get_active_engine(memory: MemoryDB) -> tuple[LanguageEngine, str, str]:
    """Detects trained VEYRA-LM checkpoint or falls back to StarterLanguageEngine."""
    candidates = [
        ("checkpoints/veyra_tiny/latest.pt", "data/tokenizer"),
        ("checkpoints/veyra_tiny/best.pt", "data/tokenizer"),
        ("checkpoints/veyra_125m/latest.pt", "data/tokenizer"),
    ]

    for ckpt_path, tok_path in candidates:
        if Path(ckpt_path).exists() and Path(tok_path).exists():
            try:
                engine = VeyraInferenceEngine.from_checkpoint(ckpt_path, tok_path)
                return engine, "VEYRA-LM (Self-Trained Active)", ckpt_path
            except Exception as e:
                # If checkpoint loading fails, continue checking or fallback
                pass

    return StarterLanguageEngine(memory), "Starter Engine", "VEYRA-LM slot ready"


def make_banner(engine_desc: str, model_desc: str) -> str:
    return f"""╭──────────────────────────────────────────────────╮
│                    VEYRA 0.1                     │
│          Personal Artificial Intelligence        │
├──────────────────────────────────────────────────┤
│ Language : {engine_desc:<38}│
│ Memory   : SQLite                                │
│ Voice    : Standby                               │
│ Model    : {model_desc:<38}│
╰──────────────────────────────────────────────────╯"""


def main() -> None:
    root = Path.home() / ".veyra"
    root.mkdir(parents=True, exist_ok=True)
    memory = MemoryDB(root / "memory.db")
    state = CognitiveState()

    engine, engine_desc, model_desc = get_active_engine(memory)
    core = CognitiveCore(engine=engine, memory=memory, state=state)

    print(make_banner(engine_desc, model_desc))
    print("Type /help for commands. Type /quit to exit.\n")

    while True:
        try:
            text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nVEYRA > Session closed.")
            break

        if not text:
            continue
        if text == "/quit":
            print("VEYRA > Goodbye, Sensei.")
            break
        print(f"VEYRA > {core.handle(text)}")
