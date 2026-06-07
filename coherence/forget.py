from __future__ import annotations

import time
from typing import Iterable


def decay(
    nodes,
    edges: dict[tuple[str, str], float],
    decay_node: float,
    decay_edge: float,
) -> None:
    keep_node = 1.0 - decay_node
    keep_edge = 1.0 - decay_edge
    for n in nodes.values():
        n.weight *= keep_node
    for k in list(edges.keys()):
        edges[k] *= keep_edge


def prune(
    nodes,
    edges: dict[tuple[str, str], float],
    index,
    prune_floor: float,
    edge_floor: float,
    *,
    min_age_seconds: float = 0.0,
    protect_ids: Iterable[str] = (),
) -> tuple[list[str], list[tuple[str, str]]]:
    now = time.time()
    protected = set(protect_ids)

    removed_nodes: list[str] = []
    for nid in list(nodes.keys()):
        if nid in protected:
            continue
        n = nodes[nid]
        if min_age_seconds > 0 and (now - n.created_at) < min_age_seconds:
            continue
        if abs(n.weight) < prune_floor:
            del nodes[nid]
            index.remove(nid)
            removed_nodes.append(nid)

    removed_edges: list[tuple[str, str]] = []
    for key in list(edges.keys()):
        i, j = key
        if i not in nodes or j not in nodes or abs(edges[key]) < edge_floor:
            del edges[key]
            removed_edges.append(key)

    return removed_nodes, removed_edges


__all__ = ["decay", "prune"]
