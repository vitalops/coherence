from __future__ import annotations

import json
import re
from typing import Any, Callable


_RECALL_SYSTEM_SIMPLE = (
    "You are a memory retrieval engine. You will be given a user query and a list of stored "
    "memories with short IDs. Pick the IDs of the memories most relevant to answering or "
    "informing the query. Order by relevance (most relevant first). Return ONLY a JSON "
    "array of the chosen IDs as strings. Do not add prose or fences."
)

_RECALL_SYSTEM_WITH_SALIENCE = (
    "You are a memory retrieval engine. You will be given a user query and a list of stored "
    "memories. Each memory line carries:\n"
    "  [id]  w=<salience>  ok=<times_helped> bad=<times_hurt>  <text>\n"
    "Optionally followed by a 'links' line listing the strongest co-activation "
    "associations and a 'recent' line summarising the most recent outcome.\n\n"
    "The salience weight reflects the framework's outcome-driven learning rule: a higher w means "
    "this memory has been reinforced as useful in past episodes; a negative w means it has been "
    "associated with failures. The 'ok' / 'bad' counters give the raw success / failure history. "
    "Use this learned signal as a tiebreaker when text relevance alone is ambiguous, and prefer "
    "memories whose history suggests they actually help. Do not blindly defer to weight when a "
    "lower-weighted memory is clearly more relevant to the query — the LLM's text judgement comes "
    "first; the salience signal arbitrates close calls.\n\n"
    "Pick the IDs of the memories most relevant to answering or informing the query. Order by "
    "relevance (most relevant first). Return ONLY a JSON array of the chosen IDs as strings. Do "
    "not add prose or fences."
)


def has_learning_signal(nodes) -> bool:
    """True when at least one node has been reinforced or has failed —
    i.e., the salience prefix carries information beyond cold-start noise."""
    for n in nodes:
        if n.reinforcement_count > 0 or n.failure_count > 0:
            return True
    return False


def _format_node_line_rich(node, *, edges, edge_floor: float, max_links: int = 3) -> list[str]:
    lines = [
        f"[{node.id}]  w={node.weight:+.3f}  ok={node.reinforcement_count} bad={node.failure_count}  {node.text}"
    ]
    if edges:
        link_pairs: list[tuple[str, float]] = []
        for (i, j), w in edges.items():
            if abs(w) < edge_floor:
                continue
            if i == node.id:
                link_pairs.append((j, w))
            elif j == node.id:
                link_pairs.append((i, w))
        link_pairs.sort(key=lambda kv: -abs(kv[1]))
        link_pairs = link_pairs[:max_links]
        if link_pairs:
            joined = ", ".join(f"{other}({w:+.2f})" for other, w in link_pairs)
            lines.append(f"     links: {joined}")
    return lines


def _format_recent_outcome(node, experiences) -> str | None:
    if not experiences:
        return None
    for exp in reversed(experiences):
        if node.id in exp.active_ids:
            return f"     recent: episode {exp.episode} outcome={exp.outcome:+.2f}"
    return None


def llm_recall(
    query: str,
    memory,
    *,
    chat_fn: Callable[..., dict],
    k: int = 5,
    model: str | None = None,
    max_memory_chars: int = 16000,
) -> list:
    """Use the LLM to pick the top-k relevant memories for ``query``.

    When the memory has accumulated reinforcement signal (any node has been
    reinforced or has failed), the listing surfaces each node's learned
    salience weight, success/failure history, strongest co-activation links,
    and most recent outcome — so the LLM can use the framework's full learning
    state, not just raw text. In the cold-start regime (no reinforcement yet),
    the listing falls back to the simple ``[id] text`` format to avoid
    cluttering the prompt with uninformative zero-prefixes.

    Returns a list of Node objects in relevance order. Returns an empty list
    if the memory dump would not fit within ``max_memory_chars`` or if the LLM
    call fails to produce parseable output.
    """
    if not memory.nodes:
        return []

    edges = getattr(memory, "edges", {}) or {}
    edge_floor = float(getattr(memory, "edge_floor", 0.01))
    experiences = getattr(memory, "experiences", []) or []

    nodes_list = list(memory.nodes.values())
    use_salience = has_learning_signal(nodes_list)

    if use_salience:
        # Sort by salience descending so high-priority items appear first.
        sorted_nodes = sorted(nodes_list, key=lambda n: -n.weight)
    else:
        # No reinforcement signal yet — preserve insertion order.
        sorted_nodes = nodes_list

    listing_parts: list[str] = []
    total_chars = 0
    for n in sorted_nodes:
        if use_salience:
            node_lines = _format_node_line_rich(n, edges=edges, edge_floor=edge_floor)
            recent_line = _format_recent_outcome(n, experiences)
            if recent_line is not None:
                node_lines.append(recent_line)
            block = "\n".join(node_lines)
        else:
            block = f"[{n.id}] {n.text}"
        total_chars += len(block) + 1
        if total_chars > max_memory_chars:
            return []  # too big — caller will fall back to BM25
        listing_parts.append(block)

    listing = "\n".join(listing_parts)
    if use_salience:
        listing_header = f"Memories ({len(listing_parts)} total, sorted by salience):"
    else:
        listing_header = f"Memories ({len(listing_parts)} total):"
    user_prompt = (
        f"Query: {query}\n\n"
        f"{listing_header}\n{listing}\n\n"
        f"Return the {k} most relevant memory IDs as a JSON array of strings."
    )

    system_prompt = _RECALL_SYSTEM_WITH_SALIENCE if use_salience else _RECALL_SYSTEM_SIMPLE
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kw: dict[str, Any] = {"temperature": 0.0}
    if model is not None:
        kw["model"] = model

    try:
        resp = chat_fn(messages=messages, **kw)
        content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        m = re.search(r"\[.*?\]", content, re.DOTALL)
        if not m:
            return []
        raw_ids = json.loads(m.group(0))
        results = []
        for rid in raw_ids:
            nid = str(rid).strip()
            if nid in memory.nodes:
                results.append(memory.nodes[nid])
            if len(results) >= k:
                break
        return results
    except Exception:
        return []


__all__ = ["llm_recall", "has_learning_signal"]
