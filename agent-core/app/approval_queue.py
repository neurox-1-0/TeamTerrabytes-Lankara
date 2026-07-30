"""
Approval Queue — human-in-the-loop review of agent proposals.

Approve writes a *simulated* execution row (not a real price change/email).
Edit/reject capture reviewer_feedback for future few-shot compounding.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db import ActionsExecutedRow, ProposalRow, ReviewerFeedbackRow, session
from app.trail import get_proposal

router = APIRouter(tags=["approval-queue"])


class EditBody(BaseModel):
    payload: dict[str, Any]
    target: dict[str, Any] | None = None
    reason: str | None = None


class RejectBody(BaseModel):
    reason: str = Field(min_length=1)


def _row_to_dict(r: ProposalRow) -> dict[str, Any]:
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


@router.get("/proposals")
def list_queue(status: str | None = "pending") -> dict[str, Any]:
    db = session()
    try:
        q = db.query(ProposalRow)
        if status and status != "all":
            q = q.filter(ProposalRow.status == status)
        rows = q.order_by(ProposalRow.created_at.desc()).limit(100).all()
        return {"proposals": [_row_to_dict(r) for r in rows]}
    finally:
        db.close()


@router.post("/proposals/{proposal_id}/approve")
def approve(proposal_id: str) -> dict[str, Any]:
    db = session()
    try:
        row = db.get(ProposalRow, proposal_id)
        if not row:
            raise HTTPException(status_code=404, detail="proposal not found")
        row.status = "executed"
        db.add(
            ActionsExecutedRow(
                proposal_id=proposal_id,
                action_type=row.action_type,
                payload=row.payload or {},
                executed_at=datetime.now(timezone.utc),
                note="simulated execution — no real downstream mutation",
            )
        )
        db.commit()
        db.refresh(row)
        return {
            "proposal": _row_to_dict(row),
            "execution": "simulated",
            "message": "Approved and logged as simulated execution",
        }
    finally:
        db.close()


@router.post("/proposals/{proposal_id}/edit")
def edit(proposal_id: str, body: EditBody) -> dict[str, Any]:
    db = session()
    try:
        row = db.get(ProposalRow, proposal_id)
        if not row:
            raise HTTPException(status_code=404, detail="proposal not found")
        original_payload = dict(row.payload or {})
        row.payload = body.payload
        if body.target is not None:
            row.target = body.target
        row.status = "edited"
        # Light confidence bump/penalty if human edited
        row.confidence = min(0.95, float(row.confidence or 0.5) * 0.95)
        db.add(
            ReviewerFeedbackRow(
                proposal_id=proposal_id,
                segment=str((row.target or {}).get("segment") or (row.target or {}).get("region") or ""),
                action_type=row.action_type,
                original_payload=original_payload,
                human_payload=body.payload,
                reason=body.reason or "edited by reviewer",
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        db.refresh(row)
        return {"proposal": _row_to_dict(row)}
    finally:
        db.close()


@router.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str, body: RejectBody) -> dict[str, Any]:
    db = session()
    try:
        row = db.get(ProposalRow, proposal_id)
        if not row:
            raise HTTPException(status_code=404, detail="proposal not found")
        row.status = "rejected"
        db.add(
            ReviewerFeedbackRow(
                proposal_id=proposal_id,
                segment=str((row.target or {}).get("segment") or (row.target or {}).get("region") or ""),
                action_type=row.action_type,
                original_payload=row.payload or {},
                human_payload={},
                reason=body.reason,
                created_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
        db.refresh(row)
        return {"proposal": _row_to_dict(row)}
    finally:
        db.close()


def recent_feedback(action_type: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
    """Few-shot context for propose node (compounding accuracy)."""
    db = session()
    try:
        q = db.query(ReviewerFeedbackRow).order_by(ReviewerFeedbackRow.id.desc())
        if action_type:
            q = q.filter(ReviewerFeedbackRow.action_type == action_type)
        rows = q.limit(limit).all()
        return [
            {
                "action_type": r.action_type,
                "segment": r.segment,
                "original_payload": r.original_payload,
                "human_payload": r.human_payload,
                "reason": r.reason,
            }
            for r in rows
        ]
    finally:
        db.close()
