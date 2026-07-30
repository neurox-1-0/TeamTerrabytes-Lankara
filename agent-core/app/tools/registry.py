"""Tool manifests + HTTP executors for all microservices."""
from __future__ import annotations

import os
from typing import Any

from app.tools.base import call_tool


def _analytics_url() -> str:
    return os.getenv("ANALYTICS_URL", "http://localhost:8002")


def _recommender_url() -> str:
    return os.getenv("RECOMMENDER_URL", "http://localhost:8003")


def _visual_url() -> str:
    return os.getenv("VISUAL_SEARCH_URL", "http://localhost:8004")


def _stt_url() -> str:
    return os.getenv("STT_TRANSLATION_URL", "http://localhost:8005")


def tool_manifest() -> list[dict[str, Any]]:
    analytics = _analytics_url()
    recommender = _recommender_url()
    visual = _visual_url()
    stt = _stt_url()
    return [
        {
            "name": "segmentation",
            "description": "RFM customer segmentation by optional region/category. Use for audience targeting and retention.",
            "url": f"{analytics}/segmentation",
            "parameters": {"region": "string?", "category": "string?"},
        },
        {
            "name": "churn_scoring",
            "description": "Score churn risk for accounts in a region or explicit account_ids. Use for retention campaigns.",
            "url": f"{analytics}/churn-scoring",
            "parameters": {"region": "string?", "account_ids": "string[]?"},
        },
        {
            "name": "forecasting",
            "description": "Forecast SKU/category demand over horizon_days. Use for reorder / inventory proposals.",
            "url": f"{analytics}/forecasting",
            "parameters": {"sku": "string?", "category": "string?", "horizon_days": "int"},
        },
        {
            "name": "basket_mining",
            "description": "Association rules for frequently co-purchased SKUs. Use for bundles and cross-sell.",
            "url": f"{analytics}/basket-mining",
            "parameters": {"sku": "string?", "category": "string?", "min_support": "float?", "min_confidence": "float?"},
        },
        {
            "name": "recommend",
            "description": "Hybrid recommender for an account or seed SKU. Use for substitute SKUs and personalized offers.",
            "url": f"{recommender}/recommend",
            "parameters": {"account_id": "string?", "sku": "string?", "k": "int?"},
        },
        {
            "name": "visual_search",
            "description": "Find visually/attribute-similar substitute SKUs. Use when reorder needs a substitute or lookalike product. Optional — not for every goal.",
            "url": f"{visual}/visual-search",
            "parameters": {"sku": "string?", "text_query": "string?", "k": "int?"},
        },
        {
            "name": "translate",
            "description": "Translate text between English/Sinhala/Tamil. Use when the goal is non-English.",
            "url": f"{stt}/translate",
            "parameters": {"text": "string", "target_lang": "en|si|ta", "source_lang": "string?"},
        },
    ]


TOOL_MANIFEST = tool_manifest()


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    analytics = _analytics_url()
    recommender = _recommender_url()
    visual = _visual_url()
    stt = _stt_url()
    mapping = {
        "segmentation": (f"{analytics}/segmentation", {"region": None}, False),
        "churn_scoring": (f"{analytics}/churn-scoring", {"region": None}, False),
        "forecasting": (
            f"{analytics}/forecasting",
            {"horizon_days": 14, "category": "Fashion"},
            False,
        ),
        "basket_mining": (f"{analytics}/basket-mining", {"min_support": 0.01}, False),
        "recommend": (f"{recommender}/recommend", {"k": 5}, False),
        "visual_search": (f"{visual}/visual-search", {"k": 5, "text_query": "fashion"}, False),
        "translate": (
            f"{stt}/translate",
            {"text": "hello", "target_lang": "en"},
            False,
        ),
    }
    if name not in mapping:
        return {
            "tool": name,
            "status": "error",
            "confidence": 0.0,
            "data": {},
            "data_slice": "",
            "error_reason": f"unknown tool {name}",
        }
    url, fallback, essential = mapping[name]
    return await call_tool(
        name,
        url,
        arguments,
        essential=essential,
        fallback_payload=fallback,
    )
