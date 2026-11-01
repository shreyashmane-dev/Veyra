from __future__ import annotations

import sys
import time
from pathlib import Path

from .core.cognitive import CognitiveCore
from .core.state import CognitiveState
from .inference.engine import VeyraInferenceEngine
from .memory.database import MemoryDB
from .model.interface import LanguageEngine
from .model.starter import StarterLanguageEngine


# ANSI Terminal Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[38;2;16;163;127m"   # OpenAI Emerald Green
CYAN = "\033[38;2;56;189;248m"    # Sky Blue
DIM = "\033[38;2;140;140;140m"    # Muted Gray
WHITE = "\033[38;2;240;240;240m"  # Crisp White
PURPLE = "\033[38;2;192;132;252m" # Lavender


def get_active_engine(memory: MemoryDB) -> tuple[LanguageEngine, str, str]:
    """Detects trained VEYRA-LM checkpoint or falls back to StarterLanguageEngine."""
    candidates = [
        ("checkpoints/veyra_instruct_125m/model.pt", "checkpoints/veyra_instruct_125m/tokenizer"),
        ("checkpoints/zonareth_instruct_125m/model.pt", "checkpoints/zonareth_instruct_125m/tokenizer"),
        ("checkpoints/veyra_125m/best.pt", "data/tokenizer"),
        ("checkpoints/veyra_125m/best.pt", "checkpoints/veyra_125m/tokenizer"),
        ("checkpoints/veyra_125m/latest.pt", "checkpoints/veyra_125m/tokenizer"),
        ("checkpoints/veyra_125m/latest.pt", "data/tokenizer"),
        ("checkpoints/veyra_125m/model.pt", "checkpoints/veyra_125m/tokenizer"),
        ("checkpoints/veyra_125m/model.pt", "data/tokenizer"),
        ("checkpoints/zonareth_125m/best.pt", "data/tokenizer"),
        ("checkpoints/zonareth_125m/best.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/model.pt", "checkpoints/zonareth_125m/tokenizer"),
        ("checkpoints/zonareth_125m/model.pt", "data/tokenizer"),
        ("checkpoints/veyra_tiny/latest.pt", "data/tokenizer"),
        ("checkpoints/veyra_tiny/best.pt", "data/tokenizer"),
        ("checkpoints/zonareth_tiny/latest.pt", "data/tokenizer"),
        ("checkpoints/zonareth_tiny/best.pt", "data/tokenizer"),
    ]

    for ckpt_path, tok_path in candidates:
        if Path(ckpt_path).exists() and Path(tok_path).exists():
            try:
                engine = VeyraInferenceEngine.from_checkpoint(ckpt_path, tok_path)
                model_name = getattr(engine.model.config, "model_name", "125m")
                tag = "VEYRA-INSTRUCT" if "instruct" in ckpt_path else "VEYRA-LM"
                params = engine.model.get_num_params()
                return engine, f"{tag} ({model_name})", f"{ckpt_path} ({params:,} params)"
            except Exception:
                pass

    return StarterLanguageEngine(memory), "Starter Engine", "VEYRA-LM slot ready"


def make_banner(engine_desc: str, model_desc: str) -> str:
    return f"""{GREEN}╭─────────────────────────────────────────────────────────────╮
│ {BOLD}VEYRA GPT-4 (125M Multi-Domain Neural Assistant){RESET}{GREEN}             │
├─────────────────────────────────────────────────────────────┤
│ {WHITE}Engine   : {engine_desc:<48}{GREEN}│
│ {WHITE}Weights  : {model_desc:<48}{GREEN}│
│ {WHITE}Domains  : Literature | Python | Poetry | Physics | Maths   {GREEN}│
│ {WHITE}Interface: Terminal (Run {CYAN}run_chatgpt_ui.bat{WHITE} for Browser Web UI){GREEN}  │
╰─────────────────────────────────────────────────────────────╯{RESET}"""


def main() -> None:
    root = Path.home() / ".veyra"
    root.mkdir(parents=True, exist_ok=True)
    memory = MemoryDB(root / "memory.db")
    state = CognitiveState()

    engine, engine_desc, model_desc = get_active_engine(memory)
    core = CognitiveCore(engine=engine, memory=memory, state=state)

    print(make_banner(engine_desc, model_desc))
    print(f"{DIM}Commands: /help, /model, /tools, /clear, /quit{RESET}\n")

    while True:
        try:
            print(f"\n{BOLD}{CYAN}You ❯ {RESET}", end="", flush=True)
            text = input().strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{GREEN}Veyra ❯ {RESET}Session closed. Farewell, Sensei.")
            break

        if not text:
            continue
        if text == "/quit":
            print(f"{GREEN}Veyra ❯ {RESET}Goodbye, Sensei.")
            break

        print(f"{BOLD}{GREEN}Veyra ❯ {RESET}", end="", flush=True)

        streamed = [False]

        def on_stream(piece: str) -> None:
            streamed[0] = True
            print(piece, end="", flush=True)

        response = core.handle(text, stream_callback=on_stream)
        if not streamed[0]:
            print(response)
        else:
            print()


if __name__ == "__main__":
    main()
