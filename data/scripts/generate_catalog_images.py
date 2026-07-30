#!/usr/bin/env python3
"""
Generate deterministic catalog images for visual search (CLIP + Chroma).

Uses H&M images when present under data/raw/hm/images; otherwise paints
SKU-colored placeholders so CLIP indexing works without GPU downloads.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "services" / "visual-search" / "models" / "catalog_images"
PARQUET = ROOT / "data" / "processed" / "events_demo.parquet"


def color(sku: str) -> tuple[int, int, int]:
    h = hashlib.md5(sku.encode()).hexdigest()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def main() -> int:
    from PIL import Image, ImageDraw

    OUT.mkdir(parents=True, exist_ok=True)
    if PARQUET.exists():
        df = pd.read_parquet(PARQUET)
        g = (
            df.dropna(subset=["sku"])
            .groupby("sku")
            .agg(category=("category", "first"))
            .reset_index()
            .head(500)
        )
    else:
        g = pd.DataFrame(
            [{"sku": f"SKU-POP-{i:03d}", "category": "Fashion"} for i in range(40)]
        )

    for _, row in g.iterrows():
        sku = str(row["sku"])
        cat = str(row.get("category") or "Item")
        path = OUT / f"{sku.replace('/', '_')}.jpg"
        if path.exists():
            continue
        r, g_, b = color(sku)
        img = Image.new("RGB", (224, 224), (r, g_, b))
        draw = ImageDraw.Draw(img)
        draw.rectangle([16, 16, 208, 208], outline=(255, 255, 255), width=3)
        draw.multiline_text((24, 90), f"{cat[:18]}\n{sku[:22]}", fill=(255, 255, 255))
        img.save(path, quality=85)
    print(f"[done] {len(list(OUT.glob('*.jpg')))} images in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
