"""
Agent-core persistence — proposals + decision_trail.

Why SQLAlchemy here: same Postgres as event-store for judge replay; SQLite
fallback when Docker/Postgres is down so Day 4 demos still work locally.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import DateTime, Float, String, Text, create_engine
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import JSON

from app.llm import _load_dotenv, _repo_root


class Base(DeclarativeBase):
    pass


# Prefer generic JSON; works on Postgres + SQLite
JsonType = JSON().with_variant(SQLITE_JSON(), "sqlite")


class ProposalRow(Base):
    __tablename__ = "proposals"

    proposal_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    evidence: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    assumptions: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class DecisionTrailRow(Base):
    __tablename__ = "decision_trail"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    step: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ActionsExecutedRow(Base):
    """Simulated execution log (Day 5 Approval Queue)."""

    __tablename__ = "actions_executed"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    proposal_id: Mapped[str] = mapped_column(String(64), index=True)
    action_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    note: Mapped[str] = mapped_column(Text, default="simulated execution")


class ReviewerFeedbackRow(Base):
    """Human edits/rejects — few-shot context for future proposes."""

    __tablename__ = "reviewer_feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    proposal_id: Mapped[str] = mapped_column(String(64), index=True)
    segment: Mapped[str] = mapped_column(String(128), default="")
    action_type: Mapped[str] = mapped_column(String(64), default="")
    original_payload: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    human_payload: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    reason: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


_ENGINE = None
_SessionLocal = None


def _default_sqlite_url() -> str:
    path = _repo_root() / "data" / "processed" / "agent_core.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def get_database_url() -> str:
    _load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return _default_sqlite_url()
    # If Postgres URL but unreachable, callers may fall back — keep as configured
    return url


def get_engine():
    global _ENGINE, _SessionLocal
    if _ENGINE is not None:
        return _ENGINE

    url = get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    try:
        engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        # Probe
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        _ENGINE = engine
    except Exception:
        # Local Day 4 fallback when Postgres/Docker is down
        sqlite_url = _default_sqlite_url()
        os.environ["DATABASE_URL_ACTIVE"] = sqlite_url
        _ENGINE = create_engine(
            sqlite_url, pool_pre_ping=True, connect_args={"check_same_thread": False}
        )

    Base.metadata.create_all(bind=_ENGINE)
    _SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)
    return _ENGINE


def session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()
