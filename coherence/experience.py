from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Experience:
    query: str
    active_ids: list[str]
    outcome: float
    timestamp: float = field(default_factory=time.time)
    episode: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "active_ids": list(self.active_ids),
            "outcome": float(self.outcome),
            "timestamp": float(self.timestamp),
            "episode": int(self.episode),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        return cls(
            query=data["query"],
            active_ids=list(data.get("active_ids", [])),
            outcome=float(data.get("outcome", 0.0)),
            timestamp=float(data.get("timestamp", time.time())),
            episode=int(data.get("episode", 0)),
            metadata=dict(data.get("metadata", {})),
        )
