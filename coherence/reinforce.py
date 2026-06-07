from __future__ import annotations

import time
from typing import Iterable


def reinforce_nodes(
    nodes,
    active_ids: Iterable[str],
    outcome: float,
    eligibility_map: dict[str, float],
    eta: float,
) -> None:
    now = time.time()
    for nid in active_ids:
        if nid not in nodes:
            continue
        elig = float(eligibility_map.get(nid, 0.0))
        delta = eta * float(outcome) * elig
        node = nodes[nid]
        node.weight += delta
        node.last_reinforced_at = now
        if outcome > 0:
            node.reinforcement_count += 1
        elif outcome < 0:
            node.failure_count += 1


def reinforce_edges(
    edges: dict[tuple[str, str], float],
    active_ids: Iterable[str],
    outcome: float,
    eligibility_map: dict[str, float],
    eta_edge: float,
) -> None:
    ids = list(active_ids)
    if eta_edge == 0.0 or len(ids) < 2:
        return
    o = float(outcome)
    for idx_i, i in enumerate(ids):
        ei = float(eligibility_map.get(i, 0.0))
        if ei <= 0.0:
            continue
        for j in ids[idx_i + 1:]:
            ej = float(eligibility_map.get(j, 0.0))
            if ej <= 0.0:
                continue
            key = (i, j) if i < j else (j, i)
            edges[key] = edges.get(key, 0.0) + eta_edge * o * ei * ej


__all__ = ["reinforce_nodes", "reinforce_edges"]
