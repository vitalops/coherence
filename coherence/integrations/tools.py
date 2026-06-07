from __future__ import annotations

import json
from typing import Any, Callable

from ..graph import Memory


# Schemas are written once, used by every adapter.

RECALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "The question or context to retrieve memories for. The "
                "framework matches the query lexically and biases the result "
                "by each node's learned salience and one-hop association "
                "spread."
            ),
        },
        "k": {
            "type": "integer",
            "description": "Number of memories to return (default 5).",
            "default": 5,
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

INGEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": (
                "The new memory chunk. Typically a paragraph or small section, "
                "self-contained enough to make sense in a future session, with "
                "entities named explicitly (no pronouns). Include surrounding "
                "context where it adds meaning."
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional free-form tags to group the memory.",
        },
    },
    "required": ["text"],
    "additionalProperties": False,
}

REINFORCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The query that was used in the recall step.",
        },
        "node_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "The ids of the memory nodes that were actually in the "
                "agent's context this episode."
            ),
        },
        "outcome": {
            "type": "number",
            "description": (
                "Outcome scalar in [-1, 1]. Use +1 for a clean success, -1 "
                "for a clean failure. Graded values (e.g. +0.4 for a "
                "partial success) are honored and propagate through the "
                "eligibility trace."
            ),
            "minimum": -1.0,
            "maximum": 1.0,
        },
    },
    "required": ["query", "node_ids", "outcome"],
    "additionalProperties": False,
}

MAINTENANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "consolidate": {
            "type": "boolean",
            "description": (
                "If true, run the near-duplicate merge pass before decay."
            ),
            "default": False,
        },
        "similarity_threshold": {
            "type": "number",
            "description": "Token-Jaccard threshold for consolidation.",
            "default": 0.82,
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
    "additionalProperties": False,
}


RECALL_DESCRIPTION = (
    "Retrieve the top-k most relevant memory nodes for a query. Use this "
    "before answering any question whose answer might depend on what the "
    "user has told you in past conversations. The returned list contains "
    "each memory's id and text; quote ids back in the reinforce step."
)

INGEST_DESCRIPTION = (
    "Persist a new memory. Use this when the current conversation surfaces a "
    "durable fact, preference, constraint, or goal that will be useful in a "
    "future session. A memory should be a self-contained chunk — typically a "
    "paragraph or a small section — that carries enough surrounding context to "
    "stand on its own when surfaced again later. Avoid storing single bare "
    "sentences when adjacent context makes the memory more meaningful, and "
    "avoid ephemeral state that only matters for the current turn."
)

REINFORCE_DESCRIPTION = (
    "Mark the outcome of the most recent recall. Call this once an answer "
    "has been judged as correct or incorrect (by the user, by an automated "
    "checker, or by a downstream system). The framework will move the "
    "salience of the active nodes and their pairwise edges accordingly."
)

MAINTENANCE_DESCRIPTION = (
    "Run a maintenance pass: decay unused weights, drop nodes that have "
    "fallen below the retention floor, and optionally merge near-duplicate "
    "memories. Call this periodically — once per session is a reasonable "
    "default — to keep the memory footprint bounded."
)


def make_tools(memory: Memory) -> dict[str, dict[str, Any]]:
    return {
        "memory_recall": {
            "description": RECALL_DESCRIPTION,
            "parameters": RECALL_SCHEMA,
            "handler": lambda args: _handle_recall(memory, args),
        },
        "memory_ingest": {
            "description": INGEST_DESCRIPTION,
            "parameters": INGEST_SCHEMA,
            "handler": lambda args: _handle_ingest(memory, args),
        },
        "memory_reinforce": {
            "description": REINFORCE_DESCRIPTION,
            "parameters": REINFORCE_SCHEMA,
            "handler": lambda args: _handle_reinforce(memory, args),
        },
        "memory_maintenance": {
            "description": MAINTENANCE_DESCRIPTION,
            "parameters": MAINTENANCE_SCHEMA,
            "handler": lambda args: _handle_maintenance(memory, args),
        },
    }


def dispatch(
    memory: Memory,
    tool_name: str,
    arguments: dict[str, Any] | str,
) -> dict[str, Any]:
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments.strip() else {}
    tools = make_tools(memory)
    if tool_name not in tools:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        return tools[tool_name]["handler"](arguments)
    except Exception as exc:  # noqa: BLE001 — boundary
        return {"error": f"{type(exc).__name__}: {exc}"}


def _handle_recall(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"])
    k = int(args.get("k", memory.k_default))
    nodes = memory.recall(query, k=k)
    return {
        "memories": [
            {
                "id": n.id,
                "text": n.text,
                "weight": round(n.weight, 4),
                "tags": list(n.tags),
            }
            for n in nodes
        ]
    }


def _handle_ingest(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    text = str(args["text"])
    tags = list(args.get("tags", []))
    nid = memory.ingest(text, tags=tags)
    return {"id": nid, "status": "ingested", "nodes": len(memory.nodes)}


def _handle_reinforce(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args["query"])
    node_ids = list(args["node_ids"])
    outcome = float(args["outcome"])
    exp = memory.reinforce(query, node_ids, outcome)
    return {
        "status": "reinforced" if exp else "skipped",
        "episode": exp.episode if exp else None,
        "stats": memory.stats(),
    }


def _handle_maintenance(memory: Memory, args: dict[str, Any]) -> dict[str, Any]:
    merges = []
    if bool(args.get("consolidate", False)):
        merges = memory.consolidate(
            similarity_threshold=float(args.get("similarity_threshold", 0.82))
        )
    forgotten = memory.forget()
    return {
        "status": "maintained",
        "merges": merges,
        "removed_nodes": forgotten["removed_nodes"],
        "removed_edges": [list(e) for e in forgotten["removed_edges"]],
        "stats": memory.stats(),
    }


__all__ = [
    "make_tools",
    "dispatch",
    "RECALL_SCHEMA",
    "INGEST_SCHEMA",
    "REINFORCE_SCHEMA",
    "MAINTENANCE_SCHEMA",
    "RECALL_DESCRIPTION",
    "INGEST_DESCRIPTION",
    "REINFORCE_DESCRIPTION",
    "MAINTENANCE_DESCRIPTION",
]
