"""Agent state + trail helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TypedDict
from uuid import uuid4


class AgentState(TypedDict, total=False):
    run_id: str
    goal: str
    perceived_events: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    assumptions: list[str]
    clarifying_question: str | None
    blocked_reason: str | None
    proposal: dict[str, Any] | None
    decision_trail: list[dict[str, Any]]
    overall_confidence: float


def new_run_id() -> str:
    return str(uuid4())


def trail_entry(
    run_id: str,
    step: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "step": step,
        "content": content,
        "tool_calls": tool_calls or [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
