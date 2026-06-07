from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..graph import Memory
from ..node import Node


class MemorySession:
    def __init__(self, memory: Memory, *, k: int | None = None) -> None:
        self.memory = memory
        self.k = k
        self._last_query: str | None = None
        self._last_active: list[str] = []

    def recall(self, query: str, k: int | None = None) -> list[Node]:
        self._last_query = query
        nodes = self.memory.recall(query, k=k if k is not None else self.k)
        self._last_active = [n.id for n in nodes]
        return nodes

    def report(
        self,
        outcome: float,
        *,
        query: str | None = None,
        node_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        q = query if query is not None else self._last_query
        ids = node_ids if node_ids is not None else self._last_active
        if q is None or not ids:
            return
        self.memory.reinforce(q, ids, outcome, metadata=metadata)

    def ingest(self, text: str, **kw: Any) -> str:
        return self.memory.ingest(text, **kw)

    @contextmanager
    def episode(self, query: str, k: int | None = None) -> Iterator["_EpisodeHandle"]:
        nodes = self.recall(query, k=k)
        handle = _EpisodeHandle(
            session=self,
            query=query,
            nodes=nodes,
            used_ids=list(self._last_active),
        )
        try:
            yield handle
        finally:
            if handle.outcome is not None:
                self.report(
                    handle.outcome,
                    query=query,
                    node_ids=handle.used_ids or self._last_active,
                    metadata=handle.metadata,
                )


@dataclass
class _EpisodeHandle:
    session: MemorySession
    query: str
    nodes: list[Node]
    used_ids: list[str] = field(default_factory=list)
    outcome: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def use(self, node_or_id: Node | str) -> None:
        nid = node_or_id if isinstance(node_or_id, str) else node_or_id.id
        if nid not in self.used_ids:
            self.used_ids.append(nid)

    def success(self, score: float = 1.0, **meta: Any) -> None:
        self.outcome = float(score)
        self.metadata.update(meta)

    def failure(self, score: float = -1.0, **meta: Any) -> None:
        self.outcome = float(score)
        self.metadata.update(meta)


__all__ = ["MemorySession"]
