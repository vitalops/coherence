from __future__ import annotations

import json
import re
from typing import Any, Callable


_RECALL_SYSTEM = (
    "You are a memory retrieval engine. You will be given a user query and a list of stored "
    "memories with short IDs. Pick the IDs of the memories most relevant to answering or "
    "informing the query. Order by relevance (most relevant first). Return ONLY a JSON "
    "array of the chosen IDs as strings. Do not add prose or fences."
)


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

    Returns a list of Node objects (in relevance order). Returns an empty list
    if the memory dump would not fit within ``max_memory_chars`` or if the
    LLM call fails to produce parseable output.
    """
    if not memory.nodes:
        return []

    listing_parts = []
    total_chars = 0
    for n in memory.nodes.values():
        line = f"[{n.id}] {n.text}"
        total_chars += len(line) + 1
        if total_chars > max_memory_chars:
            return []  # too big — caller will fall back to BM25
        listing_parts.append(line)

    listing = "\n".join(listing_parts)
    user_prompt = (
        f"Query: {query}\n\n"
        f"Memories ({len(listing_parts)} total):\n{listing}\n\n"
        f"Return the {k} most relevant memory IDs as a JSON array of strings."
    )

    messages = [
        {"role": "system", "content": _RECALL_SYSTEM},
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


__all__ = ["llm_recall"]
