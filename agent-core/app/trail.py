"""
Decision trail + proposal persistence.

Every agent step is written so judges can replay any run_id from the DB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db import DecisionTrailRow, ProposalRow, session


def persist_trail_entries(entries: list[dict[str, Any]]) -> int:
    if not entries:
        return 0
    db = session()
    try:
        for e in entries:
            ts = e.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif ts is None:
                ts = datetime.now(timezone.utc)
            db.add(
                DecisionTrailRow(
                    run_id=str(e["run_id"]),
                    step=str(e["step"]),
                    content=str(e["content"]),
                    tool_calls=e.get("tool_calls") or [],
                    timestamp=ts,
                )
            )
        db.commit()
        return len(entries)
    finally:
        db.close()


def persist_proposal(proposal: dict[str, Any], run_id: str) -> str:
    db = session()
    try:
        created = proposal.get("created_at")
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        elif created is None:
            created = datetime.now(timezone.utc)

        row = ProposalRow(
            proposal_id=str(proposal["proposal_id"]),
            run_id=run_id,
            goal=str(proposal.get("goal") or ""),
            action_type=str(proposal.get("action_type") or "report"),
            target=proposal.get("target") or {},
            payload=proposal.get("payload") or {},
            evidence=proposal.get("evidence") or [],
            assumptions=proposal.get("assumptions") or [],
            confidence=float(proposal.get("confidence") or 0.0),
            status=str(proposal.get("status") or "pending"),
            created_at=created,
        )
        db.merge(row)
        db.commit()
        return row.proposal_id
    finally:
        db.close()


def get_trail(run_id: str) -> list[dict[str, Any]]:
    db = session()
    try:
        rows = (
            db.query(DecisionTrailRow)
            .filter(DecisionTrailRow.run_id == run_id)
            .order_by(DecisionTrailRow.id.asc())
            .all()
        )
        return [
            {
                "run_id": r.run_id,
                "step": r.step,
                "content": r.content,
                "tool_calls": r.tool_calls,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]
    finally:
        db.close()


def get_proposal(proposal_id: str) -> dict[str, Any] | None:
    db = session()
    try:
        r = db.get(ProposalRow, proposal_id)
        if not r:
            return None
        return {
            "proposal_id": r.proposal_id,
            "run_id": r.run_id,
            "goal": r.goal,
            "action_type": r.action_type,
            "target": r.target,
            "payload": r.payload,
            "evidence": r.evidence,
            "assumptions": r.assumptions,
            "confidence": r.confidence,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
    finally:
        db.close()


def list_proposals(status: str | None = "pending") -> list[dict[str, Any]]:
    db = session()
    try:
        q = db.query(ProposalRow)
        if status:
            q = q.filter(ProposalRow.status == status)
        rows = q.order_by(ProposalRow.created_at.desc()).limit(100).all()
        return [
            {
                "proposal_id": r.proposal_id,
                "run_id": r.run_id,
                "goal": r.goal,
                "action_type": r.action_type,
                "target": r.target,
                "payload": r.payload,
                "evidence": r.evidence,
                "assumptions": r.assumptions,
                "confidence": r.confidence,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()
