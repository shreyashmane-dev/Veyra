from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ParsedInput:
    intent: str
    text: str
    argument: str | None = None


WORD_NUMS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "fifty": "50", "hundred": "100",
}


class LanguageParser:
    def parse(self, text: str) -> ParsedInput:
        lowered = text.lower().strip()
        cleaned_no_punct = re.sub(r"[?!.,;]+$", "", lowered).strip()

        # 1. Greetings
        if cleaned_no_punct in {"hi", "hello", "hey", "hii", "greetings"}:
            return ParsedInput("greet", text)

        # 2. Identity
        if cleaned_no_punct in {"who are you", "what are you", "what is your name", "who made you"}:
            return ParsedInput("identity", text)

        # 3. Help & Capabilities
        if cleaned_no_punct in {
            "help", "/help", "what you can do", "what can you do", "what can u do",
            "capabilities", "what are your capabilities", "what are you capable of",
            "features", "commands"
        }:
            return ParsedInput("help", text)

        # 4. Session Context Reset
        if cleaned_no_punct in {"clear", "/clear", "reset", "clear context", "restart"}:
            return ParsedInput("clear", text)

        # 5. Teaching & Memory
        if lowered.startswith("teach "):
            return ParsedInput("teach", text, text[6:].strip())
        if lowered in {"remember this", "remember that"}:
            return ParsedInput("teach", text, "")
        if lowered in {"/memory", "/status", "/model", "/tools"}:
            return ParsedInput(lowered[1:], text)
        if lowered.startswith("what do you remember"):
            return ParsedInput("memory", text)

        # 6. Explicit Calculate Commands
        if lowered.startswith("calculate "):
            return ParsedInput("calculate", text, text[10:].strip())
        if lowered.startswith("calc "):
            return ParsedInput("calculate", text, text[5:].strip())
        if lowered.startswith("math "):
            return ParsedInput("calculate", text, text[5:].strip())

        # 7. Pure Symbol Math Expression (e.g. "2+2", "2 + 2", "15 * 4", "(100 / 5) + 3")
        symbol_clean = re.sub(r"\s+", "", cleaned_no_punct)
        if symbol_clean and all(c in "0123456789+-*/().%^" for c in symbol_clean):
            if any(c in "+-*/%^" for c in symbol_clean) and any(c.isdigit() for c in symbol_clean):
                return ParsedInput("calculate", text, symbol_clean)

        # 8. Natural Language Math Phrases (e.g. "two add two", "2 plus 2", "what is 5 times 8")
        tokens = cleaned_no_punct.split()
        if (cleaned_no_punct.startswith("what is ") or cleaned_no_punct.startswith("solve ")) and any(
            op in cleaned_no_punct for op in ["+", "-", "*", "/", "%", "^", "plus", "minus", "times", "divided"]
        ):
            prefix = "what is " if cleaned_no_punct.startswith("what is ") else "solve "
            expr = text[len(prefix) :].rstrip("?").strip()
            return ParsedInput("calculate", text, self._words_to_math_expr(expr))

        # Check for natural word math like "two add two", "five times six", "add 5 and 10"
        math_words = {"add", "plus", "sum", "minus", "subtract", "times", "multiplied", "divide", "divided"}
        if any(w in math_words for w in tokens) and len(tokens) <= 7:
            converted = self._words_to_math_expr(cleaned_no_punct)
            if converted:
                return ParsedInput("calculate", text, converted)

        return ParsedInput("chat", text)

    def _words_to_math_expr(self, text: str) -> str:
        """Translates word expressions into mathematical formulas."""
        s = text.lower()
        # Word numbers to digits
        for word, num in WORD_NUMS.items():
            s = re.sub(rf"\b{word}\b", num, s)

        # Word operations to symbols
        s = re.sub(r"\b(multiplied\s+by|times)\b", "*", s)
        s = re.sub(r"\b(divided\s+by|over)\b", "/", s)
        s = re.sub(r"\b(plus|add|added\s+to|sum\s+of)\b", "+", s)
        s = re.sub(r"\b(minus|subtract|subtracted\s+from)\b", "-", s)
        s = re.sub(r"\band\b", "+", s)

        # Keep safe math characters
        clean = "".join(c for c in s if c in "0123456789+-*/().%^ ").strip()
        if any(c in "+-*/%^" for c in clean) and any(c.isdigit() for c in clean):
            return clean
        return ""
