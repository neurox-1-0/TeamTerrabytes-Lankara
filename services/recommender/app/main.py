"""Hybrid recommender FastAPI service."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.hybrid import load_bundle, recommend

app = FastAPI(title="Lankara Recommender", version="1.0.0")


class RecommendRequest(BaseModel):
    account_id: str | None = None
    sku: str | None = None
    k: int = Field(default=10, ge=1, le=50)
    collab_weight: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/health")
def health():
    ready = load_bundle() is not None
    return {"status": "ok", "service": "recommender", "ready": ready}


@app.post("/recommend")
def recommend_endpoint(body: RecommendRequest):
    return recommend(
        account_id=body.account_id,
        sku=body.sku,
        k=body.k,
        collab_weight=body.collab_weight,
    )
