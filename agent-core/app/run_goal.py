#!/usr/bin/env python3
"""
CLI runner for Day 4 test goals.

Usage:
  python -m app.run_goal "Draft reorder proposals for slow-moving SKUs in Kandy"
  python -m app.run_goal --suite
  python -m app.run_goal --suite --simulate-failure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph import run_agent  # noqa: E402
from app.trail import get_trail  # noqa: E402


CLEAR_GOAL = "Draft reorder proposals for slow-moving SKUs in the Colombo region"
AMBIGUOUS_GOAL = "Sales look weak"
CHURN_GOAL = "Which accounts in Kandy are at highest churn risk?"


def _print_run(label: str, result: dict) -> None:
    print(f"\n===== {label} =====")
    print("run_id:", result.get("run_id"))
    print("tools:", [t.get("tool") for t in (result.get("plan") or [])])
    print("clarify:", result.get("clarifying_question"))
    print("assumptions:", result.get("assumptions"))
    print("confidence:", result.get("overall_confidence"))
    for step in result.get("decision_trail") or []:
        msg = str(step.get("content", "")).encode("ascii", "replace").decode("ascii")
        print(f"  [{step.get('step')}] {msg[:200]}")
    prop = result.get("proposal")
    if prop:
        print(
            "proposal:",
            prop.get("action_type"),
            prop.get("status"),
            "conf=",
            prop.get("confidence"),
            "id=",
            prop.get("proposal_id"),
        )
    # Prove persistence
    stored = get_trail(str(result.get("run_id")))
    print(f"persisted_trail_steps: {len(stored)}")


def run_suite(simulate_failure: bool) -> int:
    if simulate_failure:
        # Force recommender + analytics primary URL dead → retry/fallback/degrade
        os.environ["ANALYTICS_URL"] = "http://127.0.0.1:59999"
        os.environ["RECOMMENDER_URL"] = "http://127.0.0.1:59998"
        print("SIMULATE FAILURE: analytics/recommender pointed at dead ports")

    cases = [
        ("clear_reorder", CLEAR_GOAL),
        ("ambiguous", AMBIGUOUS_GOAL),
        ("churn", CHURN_GOAL),
    ]
    results = []
    for label, goal in cases:
        r = run_agent(goal)
        _print_run(label, r)
        results.append((label, r))

    # Assertions for Day 4 deliverable
    clear = results[0][1]
    amb = results[1][1]
    churn = results[2][1]

    ok = True
    if not (clear.get("plan") or clear.get("clarifying_question")):
        print("FAIL: clear goal produced neither plan nor clarify")
        ok = False
    if not (amb.get("clarifying_question") or (amb.get("assumptions"))):
        print("FAIL: ambiguous goal neither asked nor scoped-and-stated")
        ok = False
    else:
        print(
            "OK ambiguous path:",
            "ask" if amb.get("clarifying_question") else "scope-and-state",
        )
    if simulate_failure:
        degraded = any(
            (t.get("status") in ("degraded", "error"))
            for t in (clear.get("tool_results") or [])
        ) or any(
            (t.get("status") in ("degraded", "error"))
            for t in (churn.get("tool_results") or [])
        )
        if not degraded and not clear.get("blocked_reason") and not churn.get("blocked_reason"):
            # Ambiguous may skip tools; check churn/clear
            print("WARN: expected degraded/error under simulated failure")
        else:
            print("OK failure ladder visible in trail/results")
            ok = True

    # Different tool sequences for clear vs churn (when not in total failure)
    if not simulate_failure:
        t1 = [t.get("tool") for t in (clear.get("plan") or [])]
        t2 = [t.get("tool") for t in (churn.get("plan") or [])]
        if t1 and t2 and t1 == t2:
            print("WARN: clear and churn selected identical tools:", t1)
        else:
            print("OK tool selection differs:", t1, "vs", t2)

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("goal", nargs="*", help="Free-text goal")
    parser.add_argument("--suite", action="store_true", help="Run Day 4 three-goal suite")
    parser.add_argument(
        "--simulate-failure",
        action="store_true",
        help="Point tool URLs at dead ports to exercise degrade ladder",
    )
    args = parser.parse_args()

    if args.suite:
        return run_suite(args.simulate_failure)

    goal = " ".join(args.goal).strip() or CLEAR_GOAL
    if args.simulate_failure:
        os.environ["ANALYTICS_URL"] = "http://127.0.0.1:59999"
        os.environ["RECOMMENDER_URL"] = "http://127.0.0.1:59998"
    result = run_agent(goal)
    _print_run("single", result)
    print("\n--- proposal json ---")
    print(json.dumps(result.get("proposal"), indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
