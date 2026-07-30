"""Pydantic request/response schemas for event-store API."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventCreate(BaseModel):
    event_id: UUID | None = None
    event_type: str
    account_id: str
    sku: str | None = None
    category: str | None = None
    quantity: int | None = None
    price: float | None = None
    discount_applied: float | None = None
    channel: str | None = None
    reason_code: str | None = None
    sentiment: str | None = None
    session_id: str
    device: str | None = None
    region: str | None = None
    timestamp: datetime


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    event_type: str
    account_id: str
    sku: str | None
    category: str | None
    quantity: int | None
    price: float | None
    discount_applied: float | None
    channel: str | None
    reason_code: str | None
    sentiment: str | None
    session_id: str
    device: str | None
    region: str | None
    timestamp: datetime


class EventBulkCreate(BaseModel):
    events: list[EventCreate] = Field(min_length=1)


class EventQueryParams(BaseModel):
    account_id: str | None = None
    region: str | None = None
    event_type: str | None = None
    sku: str | None = None
    since: datetime | None = None
    limit: int = Field(default=100, ge=1, le=10000)
