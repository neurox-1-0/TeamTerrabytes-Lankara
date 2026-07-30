#!/usr/bin/env python3
"""
One-shot bootstrap for clean clone / Docker seed:
  1) train analytics + recommender (ALS)
  2) generate catalog images
  3) ETL demo events into Postgres when DATABASE_URL reachable
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    py = sys.executable
    run([py, str(ROOT / "data/scripts/train_analytics_models.py"), "--source", "demo"])
    run([py, str(ROOT / "data/scripts/train_recommender.py")])
    run([py, str(ROOT / "data/scripts/generate_catalog_images.py")])
    run([py, str(ROOT / "data/scripts/eval_recommender_recall.py")])

    db = os.getenv("DATABASE_URL")
    if db:
        try:
            run(
                [
                    py,
                    str(ROOT / "data/scripts/etl_to_event_schema.py"),
                    "--mode",
                    "demo",
                    "--database-url",
                    db,
                    "--truncate",
                ]
            )
        except subprocess.CalledProcessError as exc:
            print("[warn] ETL skipped:", exc)
    else:
        print("[skip] DATABASE_URL not set — demo parquet already trained")
    print("[bootstrap] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
