"""
Agent-core FastAPI — run goals, approval queue, trail replay (Day 5).
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.approval_queue import router as approval_router
from app.db import get_database_url, get_engine
from app.graph import run_agent
from app.llm import active_provider, get_llm
from app.trail import get_proposal, get_trail, list_proposals

app = FastAPI(title="Lankara Agent Core", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(approval_router)


class RunRequest(BaseModel):
    goal: str


@app.on_event("startup")
def _startup() -> None:
    get_engine()


@app.get("/health")
def health() -> dict[str, Any]:
    llm_ok = False
    llm_err = None
    provider = None
    try:
        import os

        from app.llm import _load_dotenv

        _load_dotenv()
        provider = os.getenv("LLM_PROVIDER")
        get_llm()
        llm_ok = True
    except Exception as exc:
        llm_err = str(exc)
    return {
        "status": "ok",
        "service": "agent-core",
        "ready": llm_ok,
        "llm_provider": provider,
        "llm_active": active_provider(),
        "database": get_database_url(),
        "llm_error": llm_err,
    }


@app.post("/run")
def run(body: RunRequest) -> dict[str, Any]:
    result = run_agent(body.goal)
    return {
        "run_id": result.get("run_id"),
        "goal": result.get("goal"),
        "clarifying_question": result.get("clarifying_question"),
        "assumptions": result.get("assumptions"),
        "plan": result.get("plan"),
        "tool_results": result.get("tool_results"),
        "overall_confidence": result.get("overall_confidence"),
        "proposal": result.get("proposal"),
        "decision_trail": result.get("decision_trail"),
    }


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket) -> None:
    """Stream decision trail steps after a full agent run (progressive UX)."""
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        goal = str(payload.get("goal") or "").strip()
        if not goal:
            await websocket.send_json({"type": "error", "message": "goal required"})
            await websocket.close()
            return

        await websocket.send_json({"type": "status", "message": "Agent started…"})
        # Run sync graph in thread so event loop stays responsive
        import asyncio

        result = await asyncio.to_thread(run_agent, goal)
        trail = result.get("decision_trail") or []
        for step in trail:
            await websocket.send_json({"type": "trail", "entry": step})
            await asyncio.sleep(0.35)

        await websocket.send_json(
            {
                "type": "done",
                "run_id": result.get("run_id"),
                "clarifying_question": result.get("clarifying_question"),
                "assumptions": result.get("assumptions"),
                "plan": result.get("plan"),
                "proposal": result.get("proposal"),
                "overall_confidence": result.get("overall_confidence"),
            }
        )
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/runs/{run_id}/trail")
def trail(run_id: str) -> dict[str, Any]:
    rows = get_trail(run_id)
    if not rows:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "trail": rows}


# Keep REST list for clients that don't use approval router prefix collisions
@app.get("/proposals/{proposal_id}")
def proposal_detail(proposal_id: str) -> dict[str, Any]:
    row = get_proposal(proposal_id)
    if not row:
        raise HTTPException(status_code=404, detail="proposal not found")
    return row
