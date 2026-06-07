from __future__ import annotations

from typing import Any

from ..graph import Memory
from .tools import make_tools


def mcp_tool_descriptors(memory: Memory) -> list[dict[str, Any]]:
    tools = make_tools(memory)
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": spec["parameters"],
            "handler": spec["handler"],
        }
        for name, spec in tools.items()
    ]


__all__ = ["mcp_tool_descriptors"]
