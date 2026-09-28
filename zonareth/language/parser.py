from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParsedInput:
    intent: str
    text: str
    argument: str | None = None


class LanguageParser:
    def parse(self, text: str) -> ParsedInput:
        lowered = text.lower().strip()
        if lowered in {"hi", "hello", "hey", "hii"}:
            return ParsedInput("greet", text)
        if lowered in {"who are you", "what are you"}:
            return ParsedInput("identity", text)
        if lowered.startswith("teach "):
            return ParsedInput("teach", text, text[6:].strip())
        if lowered in {"remember this", "remember that"}:
            return ParsedInput("teach", text, "")
        if lowered in {"/help", "/memory", "/status", "/clear", "/model", "/tools"}:
            return ParsedInput(lowered[1:], text)
        if lowered.startswith("what do you remember"):
            return ParsedInput("memory", text)
        return ParsedInput("chat", text)
