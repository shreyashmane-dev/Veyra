from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CognitiveState:
    active_goal: str | None = None
    current_task: str | None = None
    last_user_message: str | None = None
    last_response: str | None = None
    pending_actions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    turn_count: int = 0
    started_at: datetime = field(default_factory=datetime.now)
