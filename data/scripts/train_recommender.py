#!/usr/bin/env python3
"""Train hybrid recommender from demo parquet or analytics offline events."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "recommender"))

from app.hybrid import train_hybrid  # noqa: E402


def main() -> int:
    parquet = ROOT / "data" / "processed" / "events_demo.parquet"
    if not parquet.exists():
        print("Missing events_demo.parquet — run train_analytics_models.py first")
        return 1
    df = pd.read_parquet(parquet)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    metrics = train_hybrid(df, collab_weight=0.6)
    print("[done]", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
