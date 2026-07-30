"""
Churn scoring — binary classifier on engagement features.

Churn proxy label (locked): no order activity in the last 14 days after prior activity.
Why 14 days: balances early warning vs noise for retail cadence; stated in EVAL_RESULTS.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.envelope import events_to_df, fetch_events, tool_envelope

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CHURN_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
CHURN_INACTIVE_DAYS = 14
FEATURE_COLS = [
    "recency_days",
    "frequency_30d",
    "monetary_30d",
    "view_count_30d",
    "cart_count_30d",
    "order_count_30d",
    "avg_order_value",
    "days_since_first",
]


def _build_account_features(df: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["account_id"] + FEATURE_COLS + ["churned"])

    as_of = as_of or df["timestamp"].max()
    cutoff_30 = as_of - pd.Timedelta(days=30)
    cutoff_churn = as_of - pd.Timedelta(days=CHURN_INACTIVE_DAYS)

    rows = []
    for account_id, g in df.groupby("account_id"):
        g30 = g[g["timestamp"] >= cutoff_30]
        orders = g[g["event_type"] == "order"]
        last_order = orders["timestamp"].max() if len(orders) else g["timestamp"].max()
        first_ts = g["timestamp"].min()

        monetary = 0.0
        if len(orders):
            monetary = float((orders["price"].fillna(0) * orders["quantity"].fillna(1)).sum())

        orders_30 = g30[g30["event_type"] == "order"]
        aov = float(
            (orders_30["price"].fillna(0) * orders_30["quantity"].fillna(1)).mean()
            if len(orders_30)
            else 0.0
        )

        churned = int(pd.isna(last_order) or last_order < cutoff_churn)
        # Only label accounts that had some history before the churn window
        history = g[g["timestamp"] < cutoff_churn]
        if history.empty:
            continue

        rows.append(
            {
                "account_id": account_id,
                "recency_days": float((as_of - last_order).total_seconds() / 86400.0),
                "frequency_30d": int(len(g30)),
                "monetary_30d": float(
                    (g30["price"].fillna(0) * g30["quantity"].fillna(1)).sum()
                ),
                "view_count_30d": int((g30["event_type"] == "view").sum()),
                "cart_count_30d": int((g30["event_type"] == "cart_add").sum()),
                "order_count_30d": int((g30["event_type"] == "order").sum()),
                "avg_order_value": aov if not np.isnan(aov) else 0.0,
                "days_since_first": float((as_of - first_ts).total_seconds() / 86400.0),
                "churned": churned,
                "region": g["region"].iloc[0] if "region" in g.columns else None,
            }
        )
    return pd.DataFrame(rows)


def train_churn(df: pd.DataFrame) -> dict:
    feats = _build_account_features(df)
    if len(feats) < 50:
        raise ValueError("Need >=50 labeled accounts for churn training")

    X = feats[FEATURE_COLS].fillna(0)
    y = feats["churned"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    # Prefer LightGBM when available; HistGradientBoosting is the Windows-safe fallback
    model = None
    backend = "sklearn_hgbt"
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        backend = "lightgbm"
    except Exception:
        model = HistGradientBoostingClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else 0.0,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "backend": backend,
        "inactive_days": CHURN_INACTIVE_DAYS,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_COLS, "metrics": metrics}, CHURN_MODEL_PATH)
    return metrics


def load_churn_bundle():
    if not CHURN_MODEL_PATH.exists():
        return None
    return joblib.load(CHURN_MODEL_PATH)


def _top_features(model, x_row: pd.DataFrame) -> list[dict]:
    names = FEATURE_COLS
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        order = np.argsort(-imp)[:3]
        return [{"feature": names[i], "importance": float(imp[i])} for i in order]
    # fallback: largest absolute feature values
    vals = x_row.iloc[0]
    order = np.argsort(-np.abs(vals.values))[:3]
    return [{"feature": names[i], "value": float(vals.values[i])} for i in order]


def run_churn_scoring(
    account_ids: list[str] | None = None,
    region: str | None = None,
) -> dict:
    bundle = load_churn_bundle()
    if bundle is None:
        return tool_envelope(
            "churn_scoring",
            status="error",
            confidence=0.0,
            data=[],
            data_slice="no model",
            error_reason="churn model not trained",
        )

    if account_ids:
        events = []
        for aid in account_ids[:50]:
            events.extend(fetch_events(account_id=aid, limit=2000))
    else:
        events = fetch_events(region=region, limit=20000)

    df = events_to_df(events)
    feats = _build_account_features(df)
    if feats.empty:
        return tool_envelope(
            "churn_scoring",
            status="degraded",
            confidence=0.2,
            data=[],
            data_slice=f"region={region} accounts={account_ids}",
            error_reason="no scorable accounts",
        )

    if account_ids:
        feats = feats[feats["account_id"].isin(account_ids)]

    model = bundle["model"]
    X = feats[FEATURE_COLS].fillna(0)
    proba = model.predict_proba(X)[:, 1]

    results = []
    for i, (_, row) in enumerate(feats.iterrows()):
        results.append(
            {
                "account_id": row["account_id"],
                "churn_probability": round(float(proba[i]), 4),
                "top_features": _top_features(model, X.iloc[[i]]),
                "region": row.get("region"),
            }
        )
    results.sort(key=lambda r: -r["churn_probability"])

    metrics = bundle.get("metrics", {})
    conf = float(metrics.get("precision", 0.6))
    return tool_envelope(
        "churn_scoring",
        status="ok",
        confidence=conf,
        data=results,
        data_slice=(
            f"14-day inactivity proxy; region={region}; "
            f"n={len(results)}; backend={metrics.get('backend')}"
        ),
    )
