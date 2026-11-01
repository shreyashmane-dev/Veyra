from __future__ import annotations

from pathlib import Path

from .core.cognitive import CognitiveCore
from .core.state import CognitiveState
from .memory.database import MemoryDB
from .inference.engine import ZonarethInferenceEngine, VeyraInferenceEngine
from .model.interface import LanguageEngine
from .model.starter import StarterLanguageEngine


def get_active_engine(memory: MemoryDB) -> tuple[LanguageEngine, str, str]:
    """Detects trained ZONARETH-LM checkpoint or falls back to StarterLanguageEngine."""
    candidates = [
        ("checkpoints/zonareth_instruct_125m/model.pt", "checkpoints/zonareth_instruct_125m/tokenizer"),
        ("checkpoints/veyra_instruct_125m/model.pt", "checkpoints/veyra_instruct_125m/tokenizer"),
        ("checkpoints/zonareth_125m/model.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/best.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/latest.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/latest.pt", "data/tokenizer"),
        ("checkpoints/veyra_125m/model.pt", "checkpoints/veyra_125m/tokenizer"),
        ("checkpoints/veyra_125m/best.pt", "checkpoints/veyra_125m/tokenizer"),
        ("checkpoints/veyra_125m/latest.pt", "checkpoints/veyra_125m/tokenizer"),
        ("checkpoints/veyra_125m/latest.pt", "data/tokenizer"),
        ("checkpoints/zonareth_tiny/latest.pt", "data/tokenizer"),
        ("checkpoints/zonareth_tiny/best.pt", "data/tokenizer"),
        ("checkpoints/veyra_tiny/latest.pt", "data/tokenizer"),
        ("checkpoints/veyra_tiny/best.pt", "data/tokenizer"),
    ]

    for ckpt_path, tok_path in candidates:
        if Path(ckpt_path).exists() and Path(tok_path).exists():
            try:
                engine = ZonarethInferenceEngine.from_checkpoint(ckpt_path, tok_path)
                model_name = getattr(engine.model.config, "model_name", "125m")
                tag = "ZONARETH-INSTRUCT" if "instruct" in ckpt_path else "ZONARETH-LM"
                return engine, f"{tag} ({model_name})", ckpt_path
            except Exception:
                pass

    return StarterLanguageEngine(memory), "Starter Engine", "ZONARETH-LM slot ready"


def make_banner(engine_desc: str, model_desc: str) -> str:
    return f"""╭──────────────────────────────────────────────────╮
│                  ZONARETH 0.1                    │
│         Personal Artificial Intelligence         │
├──────────────────────────────────────────────────┤
│ Language : {engine_desc:<38}│
│ Memory   : SQLite                                │
│ Voice    : Standby                               │
│ Model    : {model_desc:<38}│
╰──────────────────────────────────────────────────╯"""


def main() -> None:
    root = Path.home() / ".zonareth"
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
            print("\nZONARETH > Session closed.")
            break

        if not text:
            continue
        if text == "/quit":
            print("ZONARETH > Goodbye, Sensei.")
            break
        print(f"ZONARETH > {core.handle(text)}")
