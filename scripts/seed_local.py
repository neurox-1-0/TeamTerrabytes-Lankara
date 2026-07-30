#!/usr/bin/env python3
"""One-command local seed: Postgres up + demo ETL."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    run(["docker", "compose", "up", "-d", "postgres"])
    print("Waiting for Postgres...")
    time.sleep(8)

    run([sys.executable, "-m", "pip", "install", "-q", "-r", "data/scripts/requirements.txt"])
    run(
        [
            sys.executable,
            "data/scripts/etl_to_event_schema.py",
            "--mode",
            "demo",
            "--truncate",
        ]
    )
    run(["docker", "compose", "up", "-d", "--build", "event-store"])
    print("\nSeed complete. Try: curl http://localhost:8001/events?region=Kandy&limit=3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
