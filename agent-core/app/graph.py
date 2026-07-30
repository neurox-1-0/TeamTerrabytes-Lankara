"""
LangGraph StateGraph: PERCEIVE → PLAN → REASON → PROPOSE.

Persists decision_trail + proposal after each completed run (Day 4).
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.nodes.perceive import perceive
from app.nodes.plan import plan
from app.nodes.propose import propose
from app.nodes.reason import reason
from app.state import AgentState, new_run_id
from app.trail import persist_proposal, persist_trail_entries


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("perceive", perceive)
    g.add_node("plan", plan)
    g.add_node("reason", reason)
    g.add_node("propose", propose)
    g.set_entry_point("perceive")
    g.add_edge("perceive", "plan")
    g.add_edge("plan", "reason")
    g.add_edge("reason", "propose")
    g.add_edge("propose", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(goal: str) -> AgentState:
    graph = get_graph()
    initial: AgentState = {
        "run_id": new_run_id(),
        "goal": goal,
        "perceived_events": [],
        "plan": [],
        "tool_results": [],
        "assumptions": [],
        "clarifying_question": None,
        "blocked_reason": None,
        "proposal": None,
        "decision_trail": [],
        "overall_confidence": 0.0,
    }
    result = graph.invoke(initial)

    # Day 4: persist full trail + proposal for judge replay
    trail = result.get("decision_trail") or []
    persist_trail_entries(trail)
    proposal = result.get("proposal")
    if proposal:
        persist_proposal(proposal, str(result["run_id"]))

    return result
