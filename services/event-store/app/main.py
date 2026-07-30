"""
Event Store microservice — single source of truth for behavioral events.

Why FastAPI + Postgres: judges need a real seeded DB, not cached JSON.
Auto-create tables on startup keeps Day 1 simple without Alembic overhead.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Query
from sqlalchemy.orm import Session

from app.crud import bulk_upsert_events, get_account_events, query_events
from app.models import Base, get_engine, get_session_factory
from app.schemas import EventBulkCreate, EventCreate, EventResponse

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://lankara:lankara@localhost:5432/lankara"
)
SessionLocal = get_session_factory(DATABASE_URL)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = get_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Lankara Event Store", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "service": "event-store"}


@app.post("/events", response_model=dict)
def create_events(
    payload: Any = Body(...),
    db: Session = Depends(get_db),
):
    """Accept a single event object OR an array of events (plan Section Day 1)."""
    if isinstance(payload, list):
        events = [EventCreate.model_validate(item) for item in payload]
    elif isinstance(payload, dict) and "events" in payload:
        events = EventBulkCreate.model_validate(payload).events
    else:
        events = [EventCreate.model_validate(payload)]
    count = bulk_upsert_events(db, events)
    return {"inserted": count}


@app.post("/events/bulk", response_model=dict)
def create_events_bulk(
    payload: EventBulkCreate | list[EventCreate],
    db: Annotated[Session, Depends(get_db)],
):
    events = payload if isinstance(payload, list) else payload.events
    count = bulk_upsert_events(db, events)
    return {"inserted": count}


@app.get("/events", response_model=list[EventResponse])
def list_events(
    db: Annotated[Session, Depends(get_db)],
    account_id: str | None = Query(None),
    region: str | None = Query(None),
    event_type: str | None = Query(None),
    sku: str | None = Query(None),
    since: datetime | None = Query(None),
    limit: int = Query(100, ge=1, le=10000),
):
    rows = query_events(
        db,
        account_id=account_id,
        region=region,
        event_type=event_type,
        sku=sku,
        since=since,
        limit=limit,
    )
    return rows


@app.get("/accounts/{account_id}/events", response_model=list[EventResponse])
def account_history(
    account_id: str,
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(1000, ge=1, le=50000),
):
    return get_account_events(db, account_id, limit=limit)
