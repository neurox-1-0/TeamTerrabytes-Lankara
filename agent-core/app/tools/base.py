"""
HTTP tool wrapper — retry → fallback → degrade → block ladder.

Why: every microservice call goes through this so failures show in the
decision trail instead of silent crashes (rubric: recover from failures).
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx


class BlockedProposal(Exception):
    """Essential tool totally unavailable — do not guess."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


async def call_tool(
    name: str,
    url: str,
    payload: dict[str, Any],
    *,
    essential: bool = False,
    fallback_payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """
    1) try once
    2) retry once
    3) coarser fallback payload if provided
    4) degrade (status=degraded) if non-essential
    5) block if essential and still failing
    """
    last_err: str | None = None

    async def _post(body: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "tool" not in data:
                # wrap raw responses
                return {
                    "tool": name,
                    "status": "ok",
                    "confidence": 0.7,
                    "data": data,
                    "data_slice": f"url={url}",
                    "error_reason": None,
                }
            return data

    for attempt in (1, 2):
        try:
            return await _post(payload)
        except Exception as exc:
            last_err = str(exc)
            if attempt == 1:
                await asyncio.sleep(0.4)

    if fallback_payload is not None:
        try:
            result = await _post(fallback_payload)
            result["status"] = "degraded"
            result["error_reason"] = f"primary failed ({last_err}); used fallback query"
            result["confidence"] = min(float(result.get("confidence", 0.5)), 0.45)
            return result
        except Exception as exc:
            last_err = f"{last_err}; fallback={exc}"

    if essential:
        raise BlockedProposal(
            f"Essential tool '{name}' unavailable after retry/fallback: {last_err}"
        )

    return {
        "tool": name,
        "status": "degraded",
        "confidence": 0.15,
        "data": {},
        "data_slice": f"url={url}",
        "error_reason": last_err,
    }


def call_tool_sync(
    name: str,
    url: str,
    payload: dict[str, Any],
    *,
    essential: bool = False,
    fallback_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return asyncio.get_event_loop().run_until_complete(
        call_tool(
            name,
            url,
            payload,
            essential=essential,
            fallback_payload=fallback_payload,
        )
    )
