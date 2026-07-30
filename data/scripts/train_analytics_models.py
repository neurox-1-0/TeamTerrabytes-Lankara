#!/usr/bin/env python3
"""
Offline training for analytics-pipeline models.

Trains from:
  1) --source demo  (synthetic, always works)
  2) --source events.csv exported from Postgres
  3) --source raw   (RetailRocket / Instacart subsets if downloaded)

Saves artifacts under services/analytics-pipeline/models/
Churn label: no order in last 14 days (after prior activity).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "services" / "analytics-pipeline"))

from app.basket_mining import train_basket_mining  # noqa: E402
from app.churn import train_churn  # noqa: E402
from app.forecasting import train_forecasting  # noqa: E402
from app.segmentation import train_segmentation  # noqa: E402

RAW = ROOT / "data" / "raw"
REGIONS = ["Colombo", "Kandy", "Galle", "Jaffna"]
CATEGORIES = ["Fashion", "Groceries", "Electronics", "Home", "Beauty"]
CHANNELS = ["web", "mobile", "store"]
EVENT_TYPES = ["view", "cart_add", "order", "return"]


def assign_region(account_id: str) -> str:
    h = int(hashlib.md5(str(account_id).encode()).hexdigest(), 16)
    return REGIONS[h % len(REGIONS)]


def generate_demo_df(n_accounts: int = 800, events_per_account: int = 40) -> pd.DataFrame:
    """Richer demo set so FP-Growth and churn labels are meaningful."""
    rows = []
    base = datetime(2024, 6, 1, tzinfo=timezone.utc)
    # Shared popular SKUs so baskets have associations
    popular = [f"SKU-POP-{i:03d}" for i in range(25)]
    catalog = popular + [f"SKU-CAT-{i:04d}" for i in range(200)]

    for i in range(n_accounts):
        account_id = f"ACC-{i:05d}"
        category = CATEGORIES[i % len(CATEGORIES)]
        # ~30% of accounts go dormant early → churn positives
        dormant = i % 10 < 3
        active_days = 20 if dormant else events_per_account

        for j in range(active_days):
            day_offset = j if not dormant else j  # early window only for dormant
            if dormant:
                day_offset = j  # days 0..19 from base → inactive by "now" (base+90)
            else:
                day_offset = j * 2  # spread over ~80 days

            ts = base + timedelta(days=day_offset % 90, hours=(i + j) % 24)
            # Bundle popular items together often
            if j % 5 == 0:
                skus = [popular[i % 25], popular[(i + 1) % 25], popular[(i + 3) % 25]]
            else:
                skus = [catalog[(i * 7 + j) % len(catalog)]]

            et = EVENT_TYPES[j % len(EVENT_TYPES)]
            if dormant and et == "order" and j > 5:
                et = "view"

            for sku in skus:
                rows.append(
                    {
                        "event_id": str(uuid.uuid4()),
                        "event_type": et if len(skus) == 1 else ("order" if j % 3 == 0 else "cart_add"),
                        "account_id": account_id,
                        "sku": sku,
                        "category": category,
                        "quantity": (j % 3) + 1,
                        "price": float(500 + (i * 17 + j * 3) % 9500),
                        "region": assign_region(account_id),
                        "timestamp": ts,
                        "session_id": f"sess-{account_id}-{day_offset}",
                    }
                )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[demo] {len(df)} events, {df['account_id'].nunique()} accounts, {df['sku'].nunique()} skus")
    return df


def load_retailrocket() -> pd.DataFrame | None:
    path = RAW / "retailrocket" / "events.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path, nrows=80000)
    et_map = {"view": "view", "addtocart": "cart_add", "transaction": "order"}
    rows = []
    for _, r in raw.iterrows():
        event_type = et_map.get(str(r.get("event", "")).lower(), "view")
        account_id = str(r.get("visitorid"))
        ts = pd.to_datetime(r.get("timestamp"), unit="ms", utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "account_id": account_id,
                "sku": str(r.get("itemid")) if pd.notna(r.get("itemid")) else None,
                "category": "General",
                "quantity": 1 if event_type == "order" else None,
                "price": None,
                "region": assign_region(account_id),
                "timestamp": ts,
                "session_id": f"rr-{account_id}-{int(ts.timestamp()) // 3600}",
            }
        )
    df = pd.DataFrame(rows)
    print(f"[raw] RetailRocket: {len(df)} events")
    return df


def load_instacart_baskets() -> pd.DataFrame | None:
    """Map Instacart order lines into unified events for basket mining."""
    orders_p = RAW / "instacart" / "orders.csv"
    products_p = RAW / "instacart" / "products.csv"
    prior_p = RAW / "instacart" / "order_products__prior.csv"
    # kaggle unzip may nest
    for base in [RAW / "instacart", *RAW.glob("instacart/**/")]:
        if (base / "orders.csv").exists():
            orders_p = base / "orders.csv"
            products_p = base / "products.csv"
            prior_p = base / "order_products__prior.csv"
            break
    if not prior_p.exists() or not orders_p.exists():
        return None

    orders = pd.read_csv(orders_p, usecols=["order_id", "user_id"], nrows=50000)
    prior = pd.read_csv(prior_p, usecols=["order_id", "product_id"], nrows=200000)
    merged = prior.merge(orders, on="order_id")
    if products_p.exists():
        products = pd.read_csv(products_p, usecols=["product_id", "aisle_id"])
        merged = merged.merge(products, on="product_id", how="left")
        merged["category"] = merged["aisle_id"].astype(str)
    else:
        merged["category"] = "Grocery"

    # Cap to keep training fast
    merged = merged.head(150000)
    base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = []
    for _, r in merged.iterrows():
        account_id = str(r["user_id"])
        rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "order",
                "account_id": account_id,
                "sku": f"IC-{r['product_id']}",
                "category": str(r.get("category", "Grocery")),
                "quantity": 1,
                "price": 10.0,
                "region": assign_region(account_id),
                "timestamp": base_ts + timedelta(hours=int(r["order_id"]) % 8000),
                "session_id": f"ic-{r['order_id']}",
            }
        )
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    print(f"[raw] Instacart: {len(df)} events")
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["demo", "raw", "csv"], default="demo")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if args.source == "csv":
        if not args.csv or not args.csv.exists():
            print("--csv path required")
            return 1
        df = load_csv(args.csv)
    elif args.source == "raw":
        frames = []
        rr = load_retailrocket()
        if rr is not None:
            frames.append(rr)
        ic = load_instacart_baskets()
        if ic is not None:
            frames.append(ic)
        if not frames:
            print("[warn] No raw datasets found — falling back to demo")
            df = generate_demo_df()
        else:
            df = pd.concat(frames, ignore_index=True)
            # Blend demo for regions/price coverage if RR has no prices
            demo = generate_demo_df(n_accounts=400, events_per_account=30)
            df = pd.concat([df, demo], ignore_index=True)
    else:
        df = generate_demo_df()

    # Persist demo slice for offline inference when event-store is down
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    demo_path = processed / "events_demo.parquet"
    df.to_parquet(demo_path, index=False)
    print(f"[save] Offline events fallback -> {demo_path}")

    results = {}
    print("\n=== Training segmentation ===")
    results["segmentation"] = train_segmentation(df)

    print("\n=== Training churn (14-day inactivity) ===")
    results["churn"] = train_churn(df)

    print("\n=== Training forecasting ===")
    results["forecasting"] = train_forecasting(df)

    print("\n=== Training basket mining ===")
    # Prefer lower support for demo SKU cardinality
    try:
        results["basket"] = train_basket_mining(df, min_support=0.01, min_confidence=0.15)
    except ValueError as e:
        print(f"[retry] {e} — regenerating denser baskets")
        dense = generate_demo_df(n_accounts=600, events_per_account=50)
        results["basket"] = train_basket_mining(dense, min_support=0.01, min_confidence=0.1)

    print("\n[done] Training complete:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
