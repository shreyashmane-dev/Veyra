from __future__ import annotations

from abc import ABC, abstractmethod


class LanguageEngine(ABC):
    @abstractmethod
    def generate(self, text: str, context: list[tuple[str, str]]) -> str:
        raise NotImplementedError
