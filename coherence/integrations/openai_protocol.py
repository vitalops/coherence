from __future__ import annotations

import json
from typing import Any

from ..graph import Memory
from .tools import make_tools, dispatch


def openai_tool_specs(memory: Memory) -> list[dict[str, Any]]:
    tools = make_tools(memory)
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for name, spec in tools.items()
    ]


def run_openai_tool_call(memory: Memory, tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    raw_args = fn.get("arguments", "{}")
    result = dispatch(memory, name, raw_args)
    return {
        "tool_call_id": tool_call.get("id", ""),
        "name": name,
        "content": json.dumps(result),
    }


def run_openai_tool_calls(
    memory: Memory,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [run_openai_tool_call(memory, tc) for tc in tool_calls]


__all__ = [
    "openai_tool_specs",
    "run_openai_tool_call",
    "run_openai_tool_calls",
]
