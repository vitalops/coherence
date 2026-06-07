from __future__ import annotations

import math
from typing import Iterable


def proportional_eligibility(
    activations: dict[str, float],
    active_ids: Iterable[str],
) -> dict[str, float]:
    ids = list(active_ids)
    if not ids:
        return {}
    parts = {nid: max(0.0, float(activations.get(nid, 0.0))) for nid in ids}
    total = sum(parts.values())
    if total <= 0.0:
        share = 1.0 / len(ids)
        return {nid: share for nid in ids}
    return {nid: parts[nid] / total for nid in ids}


def softmax_eligibility(
    activations: dict[str, float],
    active_ids: Iterable[str],
    *,
    temperature: float = 0.5,
) -> dict[str, float]:
    ids = list(active_ids)
    if not ids:
        return {}
    if temperature <= 0:
        temperature = 1e-3
    vals = [float(activations.get(nid, 0.0)) / temperature for nid in ids]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    z = sum(exps) or 1.0
    return {nid: e / z for nid, e in zip(ids, exps)}


def eligibility(
    activations: dict[str, float],
    active_ids: Iterable[str],
    *,
    kind: str = "proportional",
    temperature: float = 0.5,
) -> dict[str, float]:
    if kind == "softmax":
        return softmax_eligibility(activations, active_ids, temperature=temperature)
    return proportional_eligibility(activations, active_ids)


__all__ = ["eligibility", "proportional_eligibility", "softmax_eligibility"]
