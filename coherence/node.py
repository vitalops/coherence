from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Node:
    id: str
    text: str
    weight: float = 0.0
    activation: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0
    failure_count: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "weight": self.weight,
            "activation": self.activation,
            "created_at": self.created_at,
            "last_reinforced_at": self.last_reinforced_at,
            "reinforcement_count": self.reinforcement_count,
            "failure_count": self.failure_count,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Node":
        return cls(
            id=data["id"],
            text=data["text"],
            weight=float(data.get("weight", 0.0)),
            activation=float(data.get("activation", 0.0)),
            created_at=float(data.get("created_at", time.time())),
            last_reinforced_at=float(data.get("last_reinforced_at", time.time())),
            reinforcement_count=int(data.get("reinforcement_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )
