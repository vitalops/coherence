from __future__ import annotations

import math
from typing import Callable

from .matcher import LexicalIndex, tokenize


def _sigmoid(x: float) -> float:
    if x > 60.0:
        return 1.0
    if x < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _tanh(x: float) -> float:
    return math.tanh(x)


_SQUASH: dict[str, Callable[[float], float]] = {
    "tanh": _tanh,
    "sigmoid": _sigmoid,
    "identity": lambda x: x,
}


def forward_pass(
    query: str,
    nodes,
    edges,
    index: LexicalIndex,
    *,
    k: int = 5,
    gamma: float = 0.5,
    squash: str = "tanh",
    weight_boost: float = 1.0,
) -> tuple[dict[str, float], list[str]]:
    sq = _SQUASH[squash]

    raw_scores = index.score_all(query) if nodes else {}

    # Normalize match to [0, 1] by per-query max so it's commensurate with weight.
    if raw_scores:
        peak = max(raw_scores.values())
        if peak > 0:
            match = {nid: s / peak for nid, s in raw_scores.items()}
        else:
            match = {nid: 0.0 for nid in raw_scores}
    else:
        match = {}

    # Bound the weight contribution via tanh so a runaway-weight node can't
    # dominate retrieval regardless of the query.
    base: dict[str, float] = {}
    for nid, node in nodes.items():
        base[nid] = match.get(nid, 0.0) + math.tanh(weight_boost * node.weight)

    spread: dict[str, float] = {nid: 0.0 for nid in nodes}
    if gamma != 0.0:
        for (i, j), w in edges.items():
            if i in base and j in base:
                spread[i] += w * base[j]
                spread[j] += w * base[i]

    activations: dict[str, float] = {}
    for nid, node in nodes.items():
        a = sq(base[nid] + gamma * spread[nid])
        activations[nid] = a
        node.activation = a

    top = sorted(activations.items(), key=lambda kv: -kv[1])[:k]
    return activations, [nid for nid, _ in top]


__all__ = ["forward_pass"]
