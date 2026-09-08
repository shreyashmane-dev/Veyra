from __future__ import annotations

from ..language.parser import LanguageParser
from ..memory.database import MemoryDB
from ..model.interface import LanguageEngine
from .state import CognitiveState


class CognitiveCore:
    def __init__(self, engine: LanguageEngine, memory: MemoryDB, state: CognitiveState) -> None:
        self.engine = engine
        self.memory = memory
        self.state = state
        self.parser = LanguageParser()

    def handle(self, text: str) -> str:
        parsed = self.parser.parse(text)
        self.state.last_user_message = text
        self.state.turn_count += 1
        self.memory.add_message("user", text)

        if parsed.intent == "greet":
            response = "Hello. VEYRA is online."
        elif parsed.intent == "identity":
            response = "I am VEYRA, a project being built into a small personal AI system."
        elif parsed.intent == "help":
            response = (
                "Commands: /help, /memory, /status, /model, /tools, /clear, /quit.\n"
                "You can also teach me with: teach <subject> is <fact>."
            )
        elif parsed.intent == "model":
            engine_name = self.engine.__class__.__name__
            if hasattr(self.engine, "model"):
                cfg = self.engine.model.config
                response = (
                    f"Model: {cfg.model_name} (Active Neural Model)\n"
                    f"Parameters: {self.engine.model.get_num_params():,}\n"
                    f"Layers: {cfg.num_layers}, Heads: {cfg.num_heads}, Context: {cfg.context_length}\n"
                    f"Device: {self.engine.device}"
                )
            else:
                response = (
                    f"Engine: {engine_name} (Deterministic Starter)\n"
                    f"Status: Awaiting trained VEYRA-LM checkpoint."
                )
        elif parsed.intent == "tools":
            response = (
                "Available tools in registry:\n"
                "  - read_file [SAFE]: Read text file content\n"
                "  - list_directory [SAFE]: List directory items\n"
                "  - calculate [SAFE]: Calculate math expression\n"
                "  - write_file [CAUTION]: Write content to file (requires approval)"
            )
        elif parsed.intent == "memory":
            facts = self.memory.search_facts()
            if not facts:
                response = "My long-term fact memory is empty."
            else:
                response = "\n".join(
                    f"- {s} {p} {o} (confidence {c:.2f})" for s, p, o, c in facts
                )
        elif parsed.intent == "status":
            response = (
                f"Turns: {self.state.turn_count}\n"
                f"Active goal: {self.state.active_goal or 'none'}\n"
                f"Current task: {self.state.current_task or 'none'}\n"
                f"Confidence: {self.state.confidence:.2f}"
            )
        elif parsed.intent == "clear":
            response = "The terminal session context is reset. Persistent memories are kept."
        elif parsed.intent == "teach":
            response = self._teach(parsed.argument or "")
        else:
            context = self.memory.recent_messages(8)
            response = self.engine.generate(text, context)

        self.state.last_response = response
        self.memory.add_message("assistant", response)
        return response

    def _teach(self, instruction: str) -> str:
        text = instruction.strip()
        if not text:
            return "Teaching format: teach <subject> is <fact>"
        lowered = text.lower()
        if " is " in lowered:
            idx = lowered.index(" is ")
            subject = text[:idx].strip()
            obj = text[idx + 4 :].strip()
            if subject and obj:
                self.memory.add_fact(subject, "is", obj)
                return f"Learned: {subject} is {obj}."
        if " means " in lowered:
            idx = lowered.index(" means ")
            subject = text[:idx].strip()
            obj = text[idx + 7 :].strip()
            if subject and obj:
                self.memory.add_fact(subject, "means", obj)
                return f"Learned: {subject} means {obj}."
        return "I could not parse that teaching example. Try: teach CN is Computer Networks."
