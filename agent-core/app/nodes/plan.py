"""
PLAN — LLM decides which tools to call at runtime (no hardcoded pipeline).

Uses native tool/function calling when the provider supports bind_tools
(Anthropic / OpenAI / AgentRouter). Falls back to JSON-schema prompting
for Gemini and other models without reliable tool_use.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import AgentState, trail_entry
from app.tools.registry import tool_manifest


PLAN_SCHEMA_HINT = """
Respond with ONLY valid JSON (no markdown):
{
  "ambiguous": true/false,
  "clarifying_question": "string or null",
  "assumptions": ["..."],
  "tool_calls": [{"tool": "<name>", "arguments": {...}}]
}

Allowed tool names: segmentation, churn_scoring, forecasting, basket_mining, recommend, visual_search, translate.

Rules:
- Prefer scope-and-state over asking when a region (Kandy/Colombo/Galle/Jaffna) is already named.
- Only set ambiguous=true when region AND category/time window are missing AND would change the answer (e.g. "sales look weak").
- You MUST populate tool_calls whenever ambiguous=false. Choose tools from the goal meaning — vary the set by goal.
- visual_search and translate are optional; use only when substitutes or non-English text matter.
- Never return an empty tool_calls array when ambiguous=false.
"""


class PlanDecision(BaseModel):
    """Structured PLAN decision for native tool-calling models."""

    ambiguous: bool = Field(description="True only if region/category/window would change the answer")
    clarifying_question: str | None = Field(default=None)
    assumptions: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(
        default_factory=list,
        description="Ordered tool names from the allowed set to execute",
    )
    tool_arguments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Parallel list of argument dicts for each tool_names entry",
    )


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _decision_tool() -> StructuredTool:
    def _record(
        ambiguous: bool,
        clarifying_question: str | None = None,
        assumptions: list[str] | None = None,
        tool_names: list[str] | None = None,
        tool_arguments: list[dict[str, Any]] | None = None,
    ) -> str:
        return "ok"

    return StructuredTool.from_function(
        func=_record,
        name="emit_plan",
        description=(
            "Emit the PLAN decision: whether the goal is ambiguous, optional "
            "clarifying question, assumptions, and ordered tool_names with tool_arguments."
        ),
        args_schema=PlanDecision,
    )


def _from_tool_message(msg: AIMessage) -> dict[str, Any] | None:
    calls = getattr(msg, "tool_calls", None) or []
    if not calls:
        # Older langchain shape
        kwargs = getattr(msg, "additional_kwargs", {}) or {}
        calls = kwargs.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
    if not isinstance(args, dict):
        return None
    names = list(args.get("tool_names") or [])
    arg_list = list(args.get("tool_arguments") or [])
    tool_calls = []
    for i, name in enumerate(names):
        tool_calls.append(
            {
                "tool": name,
                "arguments": arg_list[i] if i < len(arg_list) else {},
            }
        )
    return {
        "ambiguous": bool(args.get("ambiguous")),
        "clarifying_question": args.get("clarifying_question"),
        "assumptions": list(args.get("assumptions") or []),
        "tool_calls": tool_calls,
        "planner": "bind_tools",
    }


def _ask_llm_tools(goal: str, summary: dict[str, Any], extra: str | None = None) -> dict[str, Any] | None:
    """Native Anthropic/OpenAI-style tool calling via LangChain bind_tools."""
    llm = get_llm()
    if not hasattr(llm, "bind_tools"):
        return None
    try:
        tool = _decision_tool()
        bound = llm.bind_tools([tool], tool_choice="emit_plan")
    except Exception:
        try:
            bound = llm.bind_tools([_decision_tool()])
        except Exception:
            return None

    human = {"goal": goal, "perceived_summary": summary, "allowed_tools": tool_manifest()}
    if extra:
        human["correction"] = extra
    messages = [
        SystemMessage(
            content=(
                "You are the PLAN node of Lankara. Call emit_plan exactly once. "
                "Select tools at runtime from the allowed set — never invent a fixed pipeline. "
                "If a Sri Lankan region is already named, set ambiguous=false and pick tools."
            )
        ),
        HumanMessage(content=json.dumps(human, indent=2)),
    ]
    try:
        resp = bound.invoke(messages)
        parsed = _from_tool_message(resp)
        return parsed
    except Exception:
        return None


def _ask_llm_json(goal: str, summary: dict[str, Any], extra: str | None = None) -> dict[str, Any]:
    llm = get_llm()
    human = {"goal": goal, "perceived_summary": summary}
    if extra:
        human["correction"] = extra
    messages = [
        SystemMessage(
            content=(
                "You are the PLAN node of Lankara, a B2B retail autonomous agent. "
                "You choose tools at runtime. Available tools:\n"
                + json.dumps(tool_manifest(), indent=2)
                + "\n"
                + PLAN_SCHEMA_HINT
            )
        ),
        HumanMessage(content=json.dumps(human, indent=2)),
    ]
    resp = llm.invoke(messages)
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    parsed = _parse_json(raw)
    parsed["planner"] = "json_schema"
    return parsed


def _ask_llm(goal: str, summary: dict[str, Any], extra: str | None = None) -> dict[str, Any]:
    native = _ask_llm_tools(goal, summary, extra)
    if native is not None:
        return native
    return _ask_llm_json(goal, summary, extra)


def plan(state: AgentState) -> AgentState:
    run_id = state["run_id"]
    goal = state["goal"]
    events = state.get("perceived_events") or []
    summary = {
        "n_events": len(events),
        "sample_regions": list({e.get("region") for e in events[:20] if e.get("region")})[:5],
        "sample_categories": list({e.get("category") for e in events[:20] if e.get("category")})[:5],
    }

    parsed = _ask_llm(goal, summary)
    planner = parsed.get("planner", "json_schema")

    assumptions = list(state.get("assumptions") or []) + list(parsed.get("assumptions") or [])
    clarifying = parsed.get("clarifying_question")
    tool_calls = list(parsed.get("tool_calls") or [])

    region_present = bool(re.search(r"\b(Colombo|Kandy|Galle|Jaffna)\b", goal, re.I))
    if region_present and parsed.get("ambiguous"):
        if clarifying:
            assumptions.append(
                f"Deferred clarifying question (region already set): {clarifying}"
            )
        if not any("category" in a.lower() for a in assumptions):
            assumptions.append("Assumed category=Fashion when unspecified.")
        clarifying = None
        parsed["ambiguous"] = False

    if not parsed.get("ambiguous") and not tool_calls:
        parsed = _ask_llm(
            goal,
            summary,
            extra=(
                "Your previous answer had ambiguous=false but empty tool_calls. "
                "Return at least one tool_call appropriate for the goal."
            ),
        )
        planner = parsed.get("planner", planner)
        assumptions = assumptions + list(parsed.get("assumptions") or [])
        clarifying = parsed.get("clarifying_question")
        tool_calls = list(parsed.get("tool_calls") or [])
        if region_present:
            clarifying = None
            parsed["ambiguous"] = False

    if not tool_calls and not (parsed.get("ambiguous") and clarifying):
        clarifying = (
            clarifying
            or "Which analysis should I run — demand forecast/reorder, churn/retention, "
            "segmentation, basket rules, or product substitutes?"
        )
        parsed["ambiguous"] = True

    if parsed.get("ambiguous") and clarifying:
        content = (
            f"PLAN: Goal is ambiguous. Asking clarifying question before tooling: {clarifying}"
        )
        trail = list(state.get("decision_trail") or [])
        trail.append(trail_entry(run_id, "PLAN", content))
        return {
            **state,
            "plan": [],
            "clarifying_question": clarifying,
            "assumptions": assumptions,
            "decision_trail": trail,
        }

    if assumptions:
        content = (
            "PLAN: Scoped the goal with assumptions: "
            + "; ".join(assumptions)
            + f". Tool sequence (LLM/{planner}): {[t.get('tool') for t in tool_calls]}"
        )
    else:
        content = f"PLAN: Selected tools (runtime LLM/{planner}): {[t.get('tool') for t in tool_calls]}"

    trail = list(state.get("decision_trail") or [])
    trail.append(
        trail_entry(
            run_id,
            "PLAN",
            content,
            tool_calls=[
                {
                    "tool": t.get("tool"),
                    "input": t.get("arguments", {}),
                    "output": {},
                }
                for t in tool_calls
            ],
        )
    )
    return {
        **state,
        "plan": tool_calls,
        "clarifying_question": None,
        "assumptions": assumptions,
        "decision_trail": trail,
    }
