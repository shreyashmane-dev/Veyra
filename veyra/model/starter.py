from __future__ import annotations

from .interface import LanguageEngine
from ..memory.database import MemoryDB


class StarterLanguageEngine(LanguageEngine):
    """Small deterministic starter engine.

    This is intentionally NOT presented as the final AI. It provides a stable
    interface that VEYRA-LM will replace in a later phase.
    """

    def __init__(self, memory: MemoryDB) -> None:
        self.memory = memory

    def generate(self, text: str, context: list[tuple[str, str]]) -> str:
        facts = self.memory.search_facts(text)
        if facts:
            remembered = "; ".join(f"{s} {p} {o}" for s, p, o, _ in facts[:3])
            return f"I found something related in memory: {remembered}."
        return (
            "I understand the message, but my self-trained language model is not connected yet. "
            "This is VEYRA 0.1; the next major component will be VEYRA-LM."
        )
