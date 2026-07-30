#!/usr/bin/env python3
"""
Offline Recall@K for the hybrid recommender (Day 7 eval).

Hold out the last interacted SKU per account; score whether it appears in top-K
recommendations trained on the remaining history. Writes metrics suitable for
docs/EVAL_RESULTS.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "recommender"))

from app.hybrid import recommend, train_hybrid  # noqa: E402


def recall_at_k(df: pd.DataFrame, k: int = 10, min_history: int = 3) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    interact = df[df["event_type"].isin(["order", "cart_add", "view"])].dropna(
        subset=["sku", "account_id"]
    )
    interact = interact.sort_values("timestamp")

    holdouts: list[tuple[str, str]] = []
    train_rows: list[pd.DataFrame] = []
    for account_id, g in interact.groupby("account_id"):
        if len(g) < min_history:
            continue
        holdouts.append((str(account_id), str(g.iloc[-1]["sku"])))
        train_rows.append(g.iloc[:-1])

    if not holdouts:
        return {"recall_at_k": 0.0, "k": k, "n_accounts": 0, "error": "insufficient history"}

    train_df = pd.concat(train_rows, ignore_index=True)
    # Keep non-interaction columns from full df for content features
    other = df[~df.index.isin(interact.index)]
    train_full = pd.concat([train_df, other], ignore_index=True)
    train_hybrid(train_full, collab_weight=0.6)

    hits = 0
    for account_id, true_sku in holdouts:
        result = recommend(account_id=account_id, k=k)
        skus = [r["sku"] for r in (result.get("data") or [])]
        if true_sku in skus:
            hits += 1

    n = len(holdouts)
    return {
        "recall_at_k": round(hits / n, 4) if n else 0.0,
        "k": k,
        "n_accounts": n,
        "hits": hits,
    }


def main() -> int:
    parquet = ROOT / "data" / "processed" / "events_demo.parquet"
    if not parquet.exists():
        print("Missing events_demo.parquet — run train_analytics_models.py first")
        return 1
    df = pd.read_parquet(parquet)
    metrics = recall_at_k(df, k=10)
    out = ROOT / "docs" / "eval_recommender_recall.json"
    out.write_text(json.dumps(metrics, indent=2))
    print("[Recall@K]", metrics)
    print(f"[wrote] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
