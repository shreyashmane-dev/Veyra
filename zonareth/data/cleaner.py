from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import NamedTuple


class CleaningStats(NamedTuple):
    total_docs: int
    kept_docs: int
    dropped_empty: int
    dropped_length: int
    dropped_repetitive: int
    dropped_duplicate: int


class TextCleaner:
    """Document cleaning, quality filtering, and deduplication for VEYRA training data."""

    def __init__(
        self,
        min_chars: int = 20,
        max_chars: int = 100_000,
        max_repetition_ratio: float = 0.3,
        filter_duplicates: bool = True,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_repetition_ratio = max_repetition_ratio
        self.filter_duplicates = filter_duplicates
        self.seen_hashes: set[str] = set()

        # Regex for excessive whitespace/control chars
        self.whitespace_regex = re.compile(r"[ \t]+")
        self.multinewline_regex = re.compile(r"\n{3,}")
        self.control_char_regex = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def normalize(self, text: str) -> str:
        """Applies Unicode NFC normalization and strips non-printable control characters."""
        if not text:
            return ""
        # Unicode Normalization Form C
        text = unicodedata.normalize("NFC", text)
        # Remove undesirable control chars (preserving tab and newline)
        text = self.control_char_regex.sub("", text)
        # Normalize redundant spaces
        text = self.whitespace_regex.sub(" ", text)
        # Limit excessive newlines
        text = self.multinewline_regex.sub("\n\n", text)
        return text.strip()

    def is_repetitive(self, text: str) -> bool:
        """Detects if text is dominated by repetitive characters, lines, or tokens."""
        # Check for long repeated character runs (e.g. "aaaaa..." or "========...")
        if re.search(r"(.)\1{10,}", text):
            return True

        if len(text) < 50:
            return False

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) >= 4:
            unique_lines = set(lines)
            if len(unique_lines) / len(lines) < (1.0 - self.max_repetition_ratio):
                return True

        # Check for long repeated character runs (e.g. "aaaaa..." or "========...")
        if re.search(r"(.)\1{15,}", text):
            return True

        return False

    def hash_text(self, text: str) -> str:
        """Generates SHA-256 fingerprint of normalized text."""
        compact = "".join(text.split()).lower()
        return hashlib.sha256(compact.encode("utf-8")).hexdigest()

    def clean_document(self, text: str) -> tuple[str | None, str | None]:
        """Cleans and validates a single document.

        Returns:
            (cleaned_text, drop_reason)
            If document is valid, drop_reason is None.
            If document is invalid, cleaned_text is None.
        """
        normalized = self.normalize(text)
        if not normalized:
            return None, "empty"

        if len(normalized) < self.min_chars:
            return None, "too_short"

        if len(normalized) > self.max_chars:
            return None, "too_long"

        if self.is_repetitive(normalized):
            return None, "repetitive"

        if self.filter_duplicates:
            doc_hash = self.hash_text(normalized)
            if doc_hash in self.seen_hashes:
                return None, "duplicate"
            self.seen_hashes.add(doc_hash)

        return normalized, None

    def clean_corpus(self, documents: list[str]) -> tuple[list[str], CleaningStats]:
        """Cleans a list of documents and returns kept documents with statistics."""
        kept: list[str] = []
        dropped_empty = 0
        dropped_length = 0
        dropped_repetitive = 0
        dropped_duplicate = 0

        for doc in documents:
            cleaned, reason = self.clean_document(doc)
            if cleaned is not None:
                kept.append(cleaned)
            else:
                if reason == "empty":
                    dropped_empty += 1
                elif reason in ("too_short", "too_long"):
                    dropped_length += 1
                elif reason == "repetitive":
                    dropped_repetitive += 1
                elif reason == "duplicate":
                    dropped_duplicate += 1

        stats = CleaningStats(
            total_docs=len(documents),
            kept_docs=len(kept),
            dropped_empty=dropped_empty,
            dropped_length=dropped_length,
            dropped_repetitive=dropped_repetitive,
            dropped_duplicate=dropped_duplicate,
        )
        return kept, stats

    def reset_deduplication(self) -> None:
        self.seen_hashes.clear()
