"""
Market-basket association rules via FP-Growth (mlxtend).

Why FP-Growth: classic interpretable co-purchase mining for reorder / bundle proposals.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from app.envelope import events_to_df, fetch_events, tool_envelope

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
BASKET_MODEL_PATH = MODELS_DIR / "basket_rules.joblib"


def _baskets_from_events(df: pd.DataFrame) -> list[list[str]]:
    """One basket = unique SKUs in a session (or account-day if session sparse)."""
    if df.empty:
        return []
    orders = df[df["event_type"].isin(["order", "cart_add"])].dropna(subset=["sku"])
    if orders.empty:
        return []

    baskets = []
    if "session_id" in orders.columns:
        for _, g in orders.groupby("session_id"):
            items = sorted(set(g["sku"].astype(str)))
            if len(items) >= 2:
                baskets.append(items)
    if len(baskets) < 20:
        orders = orders.copy()
        orders["day"] = orders["timestamp"].dt.floor("D")
        baskets = []
        for _, g in orders.groupby(["account_id", "day"]):
            items = sorted(set(g["sku"].astype(str)))
            if len(items) >= 2:
                baskets.append(items)
    return baskets


def train_basket_mining(
    df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.2,
) -> dict:
    baskets = _baskets_from_events(df)
    if len(baskets) < 30:
        # Relax: include single-item padded pairs from popular SKUs for demo stability
        raise ValueError(f"Need >=30 multi-item baskets, got {len(baskets)}")

    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)

    # Cap columns for memory
    if basket_df.shape[1] > 500:
        freqs = basket_df.sum().nlargest(500).index
        basket_df = basket_df[freqs]

    frequent = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
    if frequent.empty:
        frequent = fpgrowth(basket_df, min_support=max(0.005, min_support / 4), use_colnames=True)

    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    if rules.empty and not frequent.empty:
        rules = association_rules(frequent, metric="confidence", min_threshold=0.05)

    rules = rules.sort_values("lift", ascending=False).head(200)
    records = []
    for _, r in rules.iterrows():
        records.append(
            {
                "antecedents": sorted(list(r["antecedents"])),
                "consequents": sorted(list(r["consequents"])),
                "support": float(r["support"]),
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"]),
            }
        )

    metrics = {
        "n_baskets": len(baskets),
        "n_rules": len(records),
        "min_support": min_support,
        "min_confidence": min_confidence,
        "top_lift": float(records[0]["lift"]) if records else 0.0,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"rules": records, "metrics": metrics}, BASKET_MODEL_PATH)
    return metrics


def load_basket_bundle():
    if not BASKET_MODEL_PATH.exists():
        return None
    return joblib.load(BASKET_MODEL_PATH)


def run_basket_mining(
    sku: str | None = None,
    category: str | None = None,
    min_support: float = 0.02,
    min_confidence: float = 0.2,
) -> dict:
    bundle = load_basket_bundle()
    if bundle is None:
        return tool_envelope(
            "basket_mining",
            status="error",
            confidence=0.0,
            data=[],
            data_slice="no model",
            error_reason="basket rules not trained",
        )

    rules = bundle["rules"]
    filtered = rules
    if sku:
        filtered = [
            r
            for r in rules
            if sku in r["antecedents"] or sku in r["consequents"]
        ]
    # category filter needs live events for SKU→category map
    if category and not sku:
        events = fetch_events(limit=5000)
        df = events_to_df(events)
        cat_skus = set(df.loc[df["category"] == category, "sku"].dropna().astype(str))
        filtered = [
            r
            for r in rules
            if cat_skus.intersection(r["antecedents"]) or cat_skus.intersection(r["consequents"])
        ]

    filtered = [
        r
        for r in filtered
        if r["support"] >= min_support * 0.5 and r["confidence"] >= min_confidence * 0.5
    ][:50]

    metrics = bundle.get("metrics", {})
    conf = min(0.9, 0.4 + 0.02 * len(filtered))
    return tool_envelope(
        "basket_mining",
        status="ok" if filtered else "degraded",
        confidence=conf if filtered else 0.25,
        data=filtered,
        data_slice=(
            f"FP-Growth rules; sku={sku} category={category}; "
            f"showing {len(filtered)} of {metrics.get('n_rules', 0)} stored"
        ),
        error_reason=None if filtered else "no rules matched filters",
    )
