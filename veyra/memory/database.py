from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class MemoryDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            self.conn.commit()

    def add_message(self, role: str, content: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages(role, content) VALUES (?, ?)", (role, content)
            )
            self.conn.commit()

    def recent_messages(self, limit: int = 8) -> list[tuple[str, str]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return list(reversed(rows))

    def clear_messages(self) -> None:
        """Clears all conversation messages while preserving learned long-term facts."""
        with self._lock:
            self.conn.execute("DELETE FROM messages")
            self.conn.commit()

    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO facts(subject, predicate, object, confidence) VALUES (?, ?, ?, ?)",
                (subject, predicate, obj, confidence),
            )
            self.conn.commit()

    def search_facts(self, query: str = "") -> list[tuple[str, str, str, float]]:
        with self._lock:
            if not query:
                rows = self.conn.execute(
                    "SELECT subject, predicate, object, confidence FROM facts ORDER BY id DESC LIMIT 20"
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = self.conn.execute(
                    """SELECT subject, predicate, object, confidence
                       FROM facts
                       WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ?
                       ORDER BY id DESC LIMIT 20""",
                    (like, like, like),
                ).fetchall()
        return rows

    def close(self) -> None:
        """Closes the underlying SQLite connection."""
        with self._lock:
            self.conn.close()
