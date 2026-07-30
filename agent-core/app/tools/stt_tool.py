"""
STT / translation tool wrapper — optional Day 6 stretch tool.

STT itself is browser Web Speech on the frontend; this wraps the translate
microservice the PLAN node can select for non-English goals.
"""
from __future__ import annotations

from typing import Any

from app.tools.registry import execute_tool


async def call_translate(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("translate", arguments or {})
