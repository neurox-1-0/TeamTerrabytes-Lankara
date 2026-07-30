"""
REASON — execute planned tools, merge results, compute overall confidence.
"""
from __future__ import annotations

import asyncio

from app.state import AgentState, trail_entry
from app.tools.base import BlockedProposal
from app.tools.registry import execute_tool


def reason(state: AgentState) -> AgentState:
    run_id = state["run_id"]
    if state.get("clarifying_question"):
        trail = list(state.get("decision_trail") or [])
        trail.append(
            trail_entry(
                run_id,
                "REASON",
                "REASON: Skipped tooling — waiting on clarifying question from PLAN.",
            )
        )
        return {**state, "tool_results": [], "overall_confidence": 0.0, "decision_trail": trail}

    plan = state.get("plan") or []
    results = []
    blocked = None

    async def _run():
        nonlocal blocked
        out = []
        for step in plan:
            name = step.get("tool")
            args = step.get("arguments") or {}
            try:
                res = await execute_tool(name, args)
                out.append(res)
            except BlockedProposal as exc:
                blocked = exc.reason
                out.append(
                    {
                        "tool": name,
                        "status": "error",
                        "confidence": 0.0,
                        "data": {},
                        "data_slice": "",
                        "error_reason": exc.reason,
                    }
                )
                break
        return out

    results = asyncio.run(_run())

    confs = [float(r.get("confidence", 0)) for r in results if r.get("status") != "error"]
    if not confs:
        overall = 0.2
    else:
        overall = sum(confs) / len(confs)
        if any(r.get("status") == "degraded" for r in results):
            overall *= 0.7

    summary_bits = [
        f"{r.get('tool')}={r.get('status')}(c={r.get('confidence')})" for r in results
    ]
    content = "REASON: Executed tools -> " + ", ".join(summary_bits)
    content += f". Overall confidence={overall:.3f}."
    if blocked:
        content += f" BLOCKED: {blocked}"

    trail = list(state.get("decision_trail") or [])
    trail.append(
        trail_entry(
            run_id,
            "REASON",
            content,
            tool_calls=[
                {
                    "tool": r.get("tool"),
                    "input": {},
                    "output": {
                        "status": r.get("status"),
                        "confidence": r.get("confidence"),
                        "error_reason": r.get("error_reason"),
                        "data_slice": r.get("data_slice"),
                    },
                }
                for r in results
            ],
        )
    )
    return {
        **state,
        "tool_results": results,
        "overall_confidence": overall,
        "blocked_reason": blocked,
        "decision_trail": trail,
    }
