"""
Analytics Pipeline FastAPI — 4 trained ML endpoints, each returning ToolResponseEnvelope.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.basket_mining import load_basket_bundle, run_basket_mining
from app.churn import load_churn_bundle, run_churn_scoring
from app.forecasting import load_forecast_bundle, run_forecasting
from app.segmentation import load_segmentation_bundle, run_segmentation

app = FastAPI(title="Lankara Analytics Pipeline", version="1.0.0")


class SegmentationRequest(BaseModel):
    region: str | None = None
    category: str | None = None


class ChurnRequest(BaseModel):
    account_ids: list[str] | None = None
    region: str | None = None


class ForecastRequest(BaseModel):
    sku: str | None = None
    category: str | None = None
    horizon_days: int = Field(default=14, ge=1, le=90)


class BasketRequest(BaseModel):
    sku: str | None = None
    category: str | None = None
    min_support: float = Field(default=0.02, ge=0.001, le=1.0)
    min_confidence: float = Field(default=0.2, ge=0.01, le=1.0)


def _models_ready() -> dict[str, bool]:
    return {
        "segmentation": load_segmentation_bundle() is not None,
        "churn_scoring": load_churn_bundle() is not None,
        "forecasting": load_forecast_bundle() is not None,
        "basket_mining": load_basket_bundle() is not None,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    ready = _models_ready()
    return {
        "status": "ok",
        "service": "analytics-pipeline",
        "ready": all(ready.values()),
        "models": ready,
    }


@app.post("/segmentation")
def segmentation(body: SegmentationRequest) -> dict:
    return run_segmentation(region=body.region, category=body.category)


@app.post("/churn-scoring")
def churn_scoring(body: ChurnRequest) -> dict:
    return run_churn_scoring(account_ids=body.account_ids, region=body.region)


@app.post("/forecasting")
def forecasting(body: ForecastRequest) -> dict:
    return run_forecasting(
        sku=body.sku, category=body.category, horizon_days=body.horizon_days
    )


@app.post("/basket-mining")
def basket_mining(body: BasketRequest) -> dict:
    return run_basket_mining(
        sku=body.sku,
        category=body.category,
        min_support=body.min_support,
        min_confidence=body.min_confidence,
    )
