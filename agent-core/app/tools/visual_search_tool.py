"""
Visual search tool wrapper — optional Day 6 stretch tool.
"""
from __future__ import annotations

from typing import Any

from app.tools.registry import execute_tool


async def call_visual_search(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("visual_search", arguments or {})
