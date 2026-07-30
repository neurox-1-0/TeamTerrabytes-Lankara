"""
Recommender tool wrapper — HTTP call via registry (no direct service import).
"""
from __future__ import annotations

from typing import Any

from app.tools.registry import execute_tool


async def call_recommend(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("recommend", arguments or {})
