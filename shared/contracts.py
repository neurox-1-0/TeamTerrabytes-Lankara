"""
Shared Pydantic contracts for Lankara microservices.
Define once here so event-store, analytics, agent-core stay aligned.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class EventType(str, Enum):
    VIEW = "view"
    CART_ADD = "cart_add"
    CART_REMOVE = "cart_remove"
    ORDER = "order"
    RETURN = "return"
    SUPPORT_TICKET = "support_ticket"
    VOICE_SEARCH = "voice_search"
    VISUAL_SEARCH = "visual_search"


class EventCreate(BaseModel):
    """Unified event schema (Section 4.1). event_id optional on insert."""

    event_id: UUID | None = None
    event_type: EventType
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


class EventResponse(EventCreate):
    event_id: UUID


class ToolStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class ToolResponseEnvelope(BaseModel):
    """Standard tool response (Section 4.2)."""

    tool: str
    status: ToolStatus
    confidence: float = Field(ge=0.0, le=1.0)
    data: dict[str, Any] | list[Any]
    data_slice: str
    error_reason: str | None = None


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    EXECUTED = "executed"


class ProposalObject(BaseModel):
    """Proposal emitted by PROPOSE node (Section 4.3)."""

    proposal_id: UUID
    goal: str
    action_type: Literal["price_change", "reorder", "retention_campaign", "report"]
    target: dict[str, Any]
    payload: dict[str, Any]
    evidence: list[ToolResponseEnvelope]
    assumptions: list[str]
    confidence: float
    status: ProposalStatus
    created_at: datetime


class ToolCallRecord(BaseModel):
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]


class DecisionTrailEntry(BaseModel):
    """Streamed to frontend and persisted (Section 4.4)."""

    run_id: UUID
    step: Literal["PERCEIVE", "PLAN", "REASON", "PROPOSE"]
    content: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    timestamp: datetime
