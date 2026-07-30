"""CRUD operations for events — bulk upsert and filtered queries."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import Event
from app.schemas import EventCreate


def _to_row(event: EventCreate) -> dict:
    return {
        "event_id": str(event.event_id) if event.event_id else str(uuid4()),
        "event_type": event.event_type,
        "account_id": event.account_id,
        "sku": event.sku,
        "category": event.category,
        "quantity": event.quantity,
        "price": event.price,
        "discount_applied": event.discount_applied,
        "channel": event.channel,
        "reason_code": event.reason_code,
        "sentiment": event.sentiment,
        "session_id": event.session_id,
        "device": event.device,
        "region": event.region,
        "timestamp": event.timestamp,
    }


def bulk_upsert_events(db: Session, events: list[EventCreate]) -> int:
    """Insert events; on conflict (event_id) update all fields."""
    if not events:
        return 0

    rows = [_to_row(e) for e in events]
    stmt = insert(Event).values(rows)
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in Event.__table__.columns
        if c.name != "event_id"
    }
    stmt = stmt.on_conflict_do_update(index_elements=["event_id"], set_=update_cols)
    db.execute(stmt)
    db.commit()
    return len(rows)


def query_events(
    db: Session,
    *,
    account_id: str | None = None,
    region: str | None = None,
    event_type: str | None = None,
    sku: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[Event]:
    stmt = select(Event).order_by(Event.timestamp.desc()).limit(limit)

    if account_id:
        stmt = stmt.where(Event.account_id == account_id)
    if region:
        stmt = stmt.where(Event.region == region)
    if event_type:
        stmt = stmt.where(Event.event_type == event_type)
    if sku:
        stmt = stmt.where(Event.sku == sku)
    if since:
        stmt = stmt.where(Event.timestamp >= since)

    return list(db.scalars(stmt).all())


def get_account_events(db: Session, account_id: str, limit: int = 1000) -> list[Event]:
    stmt = (
        select(Event)
        .where(Event.account_id == account_id)
        .order_by(Event.timestamp.asc())
        .limit(limit)
    )
    return list(db.scalars(stmt).all())
