"""
Analytics tool wrappers — thin typed helpers over registry HTTP calls.

Why separate files: plan Section 3 lists analytics_tool.py / recommender_tool.py
as agent-facing entry points; registry holds the shared manifest + execute_tool.
"""
from __future__ import annotations

from typing import Any

from app.tools.registry import execute_tool


async def call_segmentation(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("segmentation", arguments or {})


async def call_churn_scoring(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("churn_scoring", arguments or {})


async def call_forecasting(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("forecasting", arguments or {})


async def call_basket_mining(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return await execute_tool("basket_mining", arguments or {})
