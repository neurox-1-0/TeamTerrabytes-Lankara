#!/usr/bin/env python3
"""
ETL: transform raw datasets into unified event schema and load Postgres.

Synthetic region tags (Colombo, Kandy, Galle, Jaffna) are assigned deterministically
from account_id — public datasets have no Sri Lankan regions; document this for judges.

Usage:
  python data/scripts/etl_to_event_schema.py --mode demo
  python data/scripts/etl_to_event_schema.py --mode full --database-url postgresql://...
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

REGIONS = ["Colombo", "Kandy", "Galle", "Jaffna"]
CATEGORIES = ["Fashion", "Groceries", "Electronics", "Home", "Beauty"]
CHANNELS = ["web", "mobile", "store"]
EVENT_TYPES = ["view", "cart_add", "order", "return"]


def assign_region(account_id: str) -> str:
    h = int(hashlib.md5(account_id.encode()).hexdigest(), 16)
    return REGIONS[h % len(REGIONS)]


def event_row(
    *,
    event_type: str,
    account_id: str,
    session_id: str,
    timestamp: datetime,
    sku: str | None = None,
    category: str | None = None,
    quantity: int | None = None,
    price: float | None = None,
    channel: str | None = "web",
) -> tuple:
    return (
        str(uuid.uuid4()),
        event_type,
        str(account_id),
        sku,
        category,
        quantity,
        price,
        None,
        channel,
        None,
        None,
        session_id,
        "desktop",
        assign_region(str(account_id)),
        timestamp,
    )


def generate_demo_events(n_accounts: int = 500, events_per_account: int = 20) -> list[tuple]:
    """Realistic demo seed when Kaggle data unavailable."""
    rows: list[tuple] = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(n_accounts):
        account_id = f"ACC-{i:05d}"
        session_id = f"sess-{uuid.uuid4().hex[:8]}"
        region = assign_region(account_id)
        category = CATEGORIES[i % len(CATEGORIES)]

        for j in range(events_per_account):
            ts = base + timedelta(days=j % 90, hours=j % 24)
            sku = f"SKU-{category[:3].upper()}-{1000 + (i + j) % 9000}"
            et = EVENT_TYPES[j % len(EVENT_TYPES)]

            qty = (j % 3) + 1 if et == "order" else None
            price = round(500 + (i * 17 + j * 3) % 9500, 2) if et in ("order", "cart_add") else None

            rows.append(
                event_row(
                    event_type=et,
                    account_id=account_id,
                    session_id=session_id,
                    timestamp=ts,
                    sku=sku,
                    category=category,
                    quantity=qty,
                    price=price,
                    channel=CHANNELS[j % len(CHANNELS)],
                )
            )

    print(f"[demo] Generated {len(rows)} synthetic events ({n_accounts} accounts)")
    return rows


def etl_retailrocket() -> list[tuple]:
    path = RAW / "retailrocket" / "events.csv"
    if not path.exists():
        print("[skip] retailrocket/events.csv not found")
        return []

    df = pd.read_csv(path, nrows=50000)
    # RetailRocket: visitorid, timestamp, event, itemid, transactionid
    rows = []
    for _, r in df.iterrows():
        et_map = {"view": "view", "addtocart": "cart_add", "transaction": "order"}
        event_type = et_map.get(str(r.get("event", "")).lower(), "view")
        account_id = str(r.get("visitorid", r.get("user_id", "unknown")))
        ts = pd.to_datetime(r.get("timestamp"), unit="ms", errors="coerce")
        if pd.isna(ts):
            ts = datetime.now(timezone.utc)
        else:
            ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)

        rows.append(
            event_row(
                event_type=event_type,
                account_id=account_id,
                session_id=f"rr-{account_id}-{int(ts.timestamp()) // 3600}",
                timestamp=ts,
                sku=str(r.get("itemid", "")) if pd.notna(r.get("itemid")) else None,
                category="General",
            )
        )
    print(f"[etl] RetailRocket: {len(rows)} events")
    return rows


def etl_online_retail() -> list[tuple]:
    candidates = list(RAW.glob("online_retail/**/*.xlsx")) + list(
        RAW.glob("online_retail/**/*.csv")
    )
    if not candidates:
        print("[skip] Online Retail II not found")
        return []

    path = candidates[0]
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_excel(path, nrows=30000)
    df = df.head(30000)

    rows = []
    for _, r in df.iterrows():
        account_id = str(r.get("Customer ID", r.get("CustomerID", "unknown")))
        if account_id in ("nan", "None", ""):
            continue
        ts = pd.to_datetime(r.get("InvoiceDate", r.get("Invoice Date")), errors="coerce")
        if pd.isna(ts):
            continue
        ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)

        qty = int(r["Quantity"]) if pd.notna(r.get("Quantity")) else 1
        price = float(r["Price"]) if pd.notna(r.get("Price")) else None
        sku = str(int(r["StockCode"])) if pd.notna(r.get("StockCode")) else None

        rows.append(
            event_row(
                event_type="order" if qty > 0 else "return",
                account_id=account_id,
                session_id=f"or-{account_id}-{ts.date()}",
                timestamp=ts,
                sku=sku,
                category=str(r.get("Description", "Retail"))[:64],
                quantity=abs(qty),
                price=price,
            )
        )
    print(f"[etl] Online Retail II: {len(rows)} events")
    return rows


def etl_instacart(max_orders: int = 20000) -> list[tuple]:
    """Basket events from Instacart when competition files are present."""
    products_path = RAW / "instacart" / "products.csv"
    orders_path = RAW / "instacart" / "orders.csv"
    opp_path = RAW / "instacart" / "order_products__prior.csv"
    # Also accept nested unzip folders
    if not products_path.exists():
        hits = list(RAW.glob("instacart/**/products.csv"))
        if hits:
            base = hits[0].parent
            products_path = base / "products.csv"
            orders_path = base / "orders.csv"
            opp_path = base / "order_products__prior.csv"
    if not (products_path.exists() and orders_path.exists() and opp_path.exists()):
        print("[skip] Instacart files not found (accept Kaggle competition terms to download)")
        return []

    products = pd.read_csv(products_path, usecols=["product_id", "product_name", "aisle_id"])
    products["category"] = products["product_name"].astype(str).str.slice(0, 48)
    orders = pd.read_csv(
        orders_path,
        usecols=["order_id", "user_id", "order_hour_of_day", "order_dow"],
        nrows=max_orders,
    )
    opp = pd.read_csv(opp_path, usecols=["order_id", "product_id", "add_to_cart_order"])
    opp = opp[opp["order_id"].isin(orders["order_id"])]
    merged = opp.merge(orders, on="order_id").merge(products, on="product_id", how="left")

    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows: list[tuple] = []
    for _, r in merged.iterrows():
        account_id = f"IC-{int(r['user_id'])}"
        day = int(r.get("order_dow", 0) or 0)
        hour = int(r.get("order_hour_of_day", 12) or 12)
        ts = base + timedelta(days=day + (int(r["order_id"]) % 60), hours=hour)
        rows.append(
            event_row(
                event_type="order",
                account_id=account_id,
                session_id=f"ic-{int(r['order_id'])}",
                timestamp=ts,
                sku=f"IC-SKU-{int(r['product_id'])}",
                category=str(r.get("category") or "Groceries")[:64],
                quantity=1,
                price=None,
                channel="mobile",
            )
        )
    print(f"[etl] Instacart: {len(rows)} events")
    return rows


def etl_hm(max_tx: int = 30000) -> list[tuple]:
    """H&M transactions → fashion order/view events when files are present."""
    articles_path = RAW / "hm" / "articles.csv"
    tx_path = RAW / "hm" / "transactions_train.csv"
    if not articles_path.exists():
        hits = list(RAW.glob("hm/**/articles.csv"))
        if hits:
            base = hits[0].parent
            articles_path = base / "articles.csv"
            tx_path = base / "transactions_train.csv"
    if not (articles_path.exists() and tx_path.exists()):
        print("[skip] H&M articles/transactions not found")
        return []

    articles = pd.read_csv(
        articles_path,
        usecols=["article_id", "product_type_name", "product_group_name"],
        nrows=50000,
    )
    articles["category"] = articles["product_type_name"].fillna(
        articles["product_group_name"]
    ).astype(str).str.slice(0, 64)
    tx = pd.read_csv(tx_path, nrows=max_tx)
    merged = tx.merge(articles[["article_id", "category"]], on="article_id", how="left")

    rows: list[tuple] = []
    for _, r in merged.iterrows():
        account_id = f"HM-{r['customer_id']}"
        ts = pd.to_datetime(r.get("t_dat"), errors="coerce")
        if pd.isna(ts):
            continue
        ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)
        rows.append(
            event_row(
                event_type="order",
                account_id=account_id,
                session_id=f"hm-{account_id}-{ts.date()}",
                timestamp=ts,
                sku=f"HM-{int(r['article_id'])}",
                category=str(r.get("category") or "Fashion")[:64],
                quantity=1,
                price=float(r["price"]) if pd.notna(r.get("price")) else None,
                channel="web",
            )
        )
    print(f"[etl] H&M: {len(rows)} events")
    return rows


def bulk_insert(conn, rows: list[tuple]) -> int:
    if not rows:
        return 0

    sql = """
        INSERT INTO events (
            event_id, event_type, account_id, sku, category, quantity, price,
            discount_applied, channel, reason_code, sentiment, session_id, device,
            region, timestamp
        ) VALUES %s
        ON CONFLICT (event_id) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def ensure_table(conn) -> None:
    ddl = """
    CREATE TABLE IF NOT EXISTS events (
        event_id UUID PRIMARY KEY,
        event_type VARCHAR(32) NOT NULL,
        account_id VARCHAR(64) NOT NULL,
        sku VARCHAR(64),
        category VARCHAR(128),
        quantity INTEGER,
        price DOUBLE PRECISION,
        discount_applied DOUBLE PRECISION,
        channel VARCHAR(32),
        reason_code VARCHAR(64),
        sentiment VARCHAR(32),
        session_id VARCHAR(64) NOT NULL,
        device VARCHAR(32),
        region VARCHAR(64),
        timestamp TIMESTAMPTZ NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_account ON events(account_id);
    CREATE INDEX IF NOT EXISTS idx_events_region ON events(region);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql://lankara:lankara@localhost:5432/lankara"),
    )
    parser.add_argument("--truncate", action="store_true", help="Clear events table first")
    args = parser.parse_args()

    if args.mode == "demo":
        rows = generate_demo_events()
    else:
        rows = []
        rows.extend(etl_retailrocket())
        rows.extend(etl_online_retail())
        rows.extend(etl_instacart())
        rows.extend(etl_hm())
        if not rows:
            print("[fallback] No raw data found — using demo generator")
            rows = generate_demo_events(n_accounts=1000, events_per_account=15)

    conn = psycopg2.connect(args.database_url)
    try:
        ensure_table(conn)
        if args.truncate:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE events")
            conn.commit()
            print("[etl] Truncated events table")

        inserted = bulk_insert(conn, rows)
        with conn.cursor() as cur:
            cur.execute("SELECT region, COUNT(*) FROM events GROUP BY region ORDER BY region")
            summary = cur.fetchall()
        print(f"[done] Loaded {inserted} events. Region breakdown:")
        for region, count in summary:
            print(f"  {region}: {count}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
