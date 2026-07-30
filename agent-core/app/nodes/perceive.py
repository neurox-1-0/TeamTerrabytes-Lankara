"""
PERCEIVE — pull a relevant event slice from event-store (or offline via analytics path).

Why first: grounds the plan in real behavioral data, not a hardcoded assumption.
"""
from __future__ import annotations

import os
import re
from typing import Any

import httpx

from app.state import AgentState, trail_entry

EVENT_STORE_URL = os.getenv("EVENT_STORE_URL", "http://localhost:8001")
REGIONS = ["Colombo", "Kandy", "Galle", "Jaffna"]


def _extract_region(goal: str) -> str | None:
    for r in REGIONS:
        if re.search(rf"\b{re.escape(r)}\b", goal, re.I):
            return r
    return None


def perceive(state: AgentState) -> AgentState:
    goal = state["goal"]
    run_id = state["run_id"]
    region = _extract_region(goal)
    events: list[dict[str, Any]] = []
    note = ""

    params: dict[str, Any] = {"limit": 50}
    if region:
        params["region"] = region

    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.get(f"{EVENT_STORE_URL}/events", params=params)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, list):
                events = payload
            elif isinstance(payload, dict):
                events = payload.get("events") or []
            # region sanity
            if region and events:
                matched = [e for e in events if str(e.get("region", "")).lower() == region.lower()]
                events = matched or events
            note = f"Fetched {len(events)} events from event-store"
            if region:
                note += f" (region={region})"
    except Exception as exc:
        note = f"event-store unavailable ({exc}); continuing with empty perception — plan will rely on tool services' own data access"
        events = []

    content = f"PERCEIVE: {note}. Goal: {goal}"
    trail = list(state.get("decision_trail") or [])
    trail.append(trail_entry(run_id, "PERCEIVE", content))
    return {
        **state,
        "perceived_events": events[:50],
        "decision_trail": trail,
        "assumptions": list(state.get("assumptions") or []),
    }
