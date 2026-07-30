"""
PROPOSE — assemble Proposal object (Section 4.3) from tool evidence.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import AgentState, trail_entry


def _infer_action_type(goal: str) -> str:
    """Soft prior only — LLM draft may override via action_type field."""
    g = goal.lower()
    if any(w in g for w in ("churn", "retain", "retention", "at risk", "at-risk")):
        return "retention_campaign"
    if any(w in g for w in ("reorder", "stock", "forecast", "demand", "inventory")):
        return "reorder"
    if any(w in g for w in ("price", "discount", "markdown")):
        return "price_change"
    return "report"


def propose(state: AgentState) -> AgentState:
    run_id = state["run_id"]
    goal = state["goal"]

    if state.get("clarifying_question"):
        trail = list(state.get("decision_trail") or [])
        trail.append(
            trail_entry(
                run_id,
                "PROPOSE",
                f"PROPOSE: No proposal yet — need answer to: {state['clarifying_question']}",
            )
        )
        return {**state, "proposal": None, "decision_trail": trail}

    if state.get("blocked_reason"):
        proposal = {
            "proposal_id": str(uuid4()),
            "goal": goal,
            "action_type": "report",
            "target": {},
            "payload": {"blocked": True, "reason": state["blocked_reason"]},
            "evidence": state.get("tool_results") or [],
            "assumptions": state.get("assumptions") or [],
            "confidence": 0.0,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        trail = list(state.get("decision_trail") or [])
        trail.append(
            trail_entry(
                run_id,
                "PROPOSE",
                f"PROPOSE: Blocked proposal — {state['blocked_reason']}",
            )
        )
        return {**state, "proposal": proposal, "decision_trail": trail}

    action_type = _infer_action_type(goal)
    evidence = state.get("tool_results") or []
    assumptions = state.get("assumptions") or []
    confidence = float(state.get("overall_confidence") or 0.5)

    feedback: list[dict[str, Any]] = []
    try:
        from app.approval_queue import recent_feedback

        feedback = recent_feedback(action_type=action_type, limit=3)
    except Exception:
        feedback = []

    # Ask LLM to draft a concrete payload from evidence (still judge-explainable)
    llm = get_llm()
    prompt = {
        "goal": goal,
        "action_type": action_type,
        "assumptions": assumptions,
        "reviewer_feedback_fewshot": feedback,
        "evidence_summary": [
            {
                "tool": e.get("tool"),
                "status": e.get("status"),
                "confidence": e.get("confidence"),
                "data_slice": e.get("data_slice"),
                "data_preview": str(e.get("data"))[:800],
            }
            for e in evidence
        ],
    }
    resp = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Draft a B2B retail action as JSON only: "
                    '{"action_type":"price_change|reorder|retention_campaign|report",'
                    '"target":{...},"payload":{...},"summary":"one sentence"}. '
                    "Choose action_type from the evidence and goal — do not ignore tool results."
                )
            ),
            HumanMessage(content=json.dumps(prompt)),
        ]
    )
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        drafted = json.loads(text)
    except Exception:
        drafted = {
            "target": {"region": "unspecified"},
            "payload": {"notes": "Auto-draft from tool evidence", "raw": raw[:500]},
            "summary": "Proposal drafted from tool evidence with parse fallback.",
        }

    allowed = {"price_change", "reorder", "retention_campaign", "report"}
    llm_action = str(drafted.get("action_type") or "").strip()
    if llm_action in allowed:
        action_type = llm_action

    proposal: dict[str, Any] = {
        "proposal_id": str(uuid4()),
        "goal": goal,
        "action_type": action_type,
        "target": drafted.get("target") or {},
        "payload": drafted.get("payload") or drafted,
        "evidence": evidence,
        "assumptions": assumptions,
        "confidence": confidence,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    summary = drafted.get("summary") or f"{action_type} proposal @ confidence {confidence:.2f}"
    trail = list(state.get("decision_trail") or [])
    trail.append(trail_entry(run_id, "PROPOSE", f"PROPOSE: {summary}"))
    return {**state, "proposal": proposal, "decision_trail": trail}
