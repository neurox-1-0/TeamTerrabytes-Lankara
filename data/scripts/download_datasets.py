#!/usr/bin/env python3
"""
Download public datasets for Lankara ETL.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\).
Run: python data/scripts/download_datasets.py [--subset demo]

Use --subset demo to skip large H&M download during local dev.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


DATASETS = {
    "retailrocket": {
        "slug": "retailrocket/ecommerce-dataset",
        "files": ["events.csv", "item_properties_part1.csv", "item_properties_part2.csv"],
    },
    "instacart": {
        "slug": "instacart-market-basket-analysis",
        "files": ["orders.csv", "order_products__prior.csv", "products.csv"],
    },
    "hm": {
        "slug": "h-and-m-personalized-fashion-recommendations",
        "files": ["articles.csv", "transactions_train.csv"],
        "optional": True,
    },
}


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None


def download_dataset(name: str, slug: str, skip: bool = False) -> bool:
    dest = RAW / name
    dest.mkdir(parents=True, exist_ok=True)

    if skip:
        print(f"[skip] {name} (--subset demo)")
        return False

    if not _kaggle_available():
        print("[warn] kaggle CLI not found. Install: pip install kaggle")
        print("       Place credentials in ~/.kaggle/kaggle.json")
        return False

    print(f"[download] {slug} -> {dest}")
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout)
        return False
    return True


def download_online_retail() -> bool:
    """Online Retail II from UCI — direct HTTP, no Kaggle."""
    dest = RAW / "online_retail"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".downloaded"
    if marker.exists():
        print("[skip] online_retail already present")
        return True

    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online%20retail.zip"
    )
    zip_path = dest / "online_retail.zip"
    try:
        import urllib.request

        print(f"[download] Online Retail II -> {dest}")
        urllib.request.urlretrieve(url, zip_path)
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        marker.touch()
        zip_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        print(f"[warn] Online Retail download failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Lankara datasets")
    parser.add_argument(
        "--subset",
        choices=["full", "demo"],
        default="full",
        help="demo skips heavy H&M download",
    )
    args = parser.parse_args()
    skip_hm = args.subset == "demo"

    RAW.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, meta in DATASETS.items():
        if not download_dataset(name, meta["slug"], skip=(name == "hm" and skip_hm)):
            if not meta.get("optional"):
                ok = False

    if not download_online_retail():
        ok = False

    if ok:
        print("\n[done] All required datasets downloaded.")
    else:
        print(
            "\n[partial] Some downloads failed. Run ETL with --mode demo for synthetic seed data."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
