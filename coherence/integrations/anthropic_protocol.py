from __future__ import annotations

from typing import Any

from ..graph import Memory
from .tools import make_tools, dispatch


def anthropic_tool_specs(memory: Memory) -> list[dict[str, Any]]:
    tools = make_tools(memory)
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": spec["parameters"],
        }
        for name, spec in tools.items()
    ]


def run_anthropic_tool_call(
    memory: Memory,
    tool_use_block: dict[str, Any],
) -> dict[str, Any]:
    name = tool_use_block.get("name", "")
    inputs = tool_use_block.get("input", {}) or {}
    result = dispatch(memory, name, inputs)
    import json
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_block.get("id", ""),
        "content": json.dumps(result),
    }


def run_anthropic_tool_calls(
    memory: Memory,
    content_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        run_anthropic_tool_call(memory, b)
        for b in content_blocks
        if b.get("type") == "tool_use"
    ]


__all__ = [
    "anthropic_tool_specs",
    "run_anthropic_tool_call",
    "run_anthropic_tool_calls",
]
