"""
Per-SKU demand forecasting.

Why LightGBM/HistGradientBoosting on lag features: works without long history
required by SARIMA, trains offline, serves fast for demo.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

from app.envelope import events_to_df, fetch_events, tool_envelope

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FORECAST_MODEL_PATH = MODELS_DIR / "forecasting_model.joblib"


def _daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    orders = df[df["event_type"].isin(["order", "cart_add"])].copy()
    if orders.empty:
        return pd.DataFrame(columns=["sku", "date", "qty", "category"])
    orders["qty"] = orders["quantity"].fillna(1).astype(float)
    orders["date"] = orders["timestamp"].dt.floor("D")
    daily = (
        orders.groupby(["sku", "date"])
        .agg(qty=("qty", "sum"), category=("category", "first"))
        .reset_index()
    )
    return daily


def _make_supervised(daily: pd.DataFrame, lags: tuple[int, ...] = (1, 7, 14)) -> pd.DataFrame:
    frames = []
    for sku, g in daily.groupby("sku"):
        g = g.sort_values("date").set_index("date")
        # fill missing days with 0
        full = g.asfreq("D", fill_value=0)
        full["sku"] = sku
        full["category"] = g["category"].iloc[0] if "category" in g.columns else None
        for lag in lags:
            full[f"lag_{lag}"] = full["qty"].shift(lag)
        full["roll7"] = full["qty"].rolling(7, min_periods=1).mean()
        full["dow"] = full.index.dayofweek
        frames.append(full.reset_index())
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.dropna()


def train_forecasting(df: pd.DataFrame) -> dict:
    daily = _daily_demand(df)
    if daily.empty or daily["sku"].nunique() < 3:
        raise ValueError("Need order/cart events across multiple SKUs for forecasting")

    # Cap SKUs for training speed
    top_skus = daily.groupby("sku")["qty"].sum().nlargest(200).index
    daily = daily[daily["sku"].isin(top_skus)]
    supervised = _make_supervised(daily)
    if len(supervised) < 100:
        raise ValueError("Insufficient supervised rows for forecasting")

    feature_cols = [c for c in supervised.columns if c.startswith("lag_") or c in ("roll7", "dow")]
    X = supervised[feature_cols]
    y = supervised["qty"]

    split = int(len(supervised) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    backend = "sklearn_hgbt"
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(n_estimators=80, max_depth=5, learning_rate=0.05, verbose=-1)
        model.fit(X_train, y_train)
        backend = "lightgbm"
    except Exception:
        model = HistGradientBoostingRegressor(max_depth=5, random_state=42)
        model.fit(X_train, y_train)

    preds = np.clip(model.predict(X_test), 0, None)
    # avoid MAPE blow-up on zeros
    mask = y_test.values > 0
    mape = (
        float(mean_absolute_percentage_error(y_test.values[mask], preds[mask]))
        if mask.any()
        else 1.0
    )

    metrics = {
        "mape": mape,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_skus": int(daily["sku"].nunique()),
        "backend": backend,
        "feature_cols": feature_cols,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Store recent daily history for recursive forecast at serve time
    history = daily.copy()
    joblib.dump(
        {"model": model, "metrics": metrics, "history": history, "feature_cols": feature_cols},
        FORECAST_MODEL_PATH,
    )
    return metrics


def load_forecast_bundle():
    if not FORECAST_MODEL_PATH.exists():
        return None
    return joblib.load(FORECAST_MODEL_PATH)


def run_forecasting(
    sku: str | None = None,
    category: str | None = None,
    horizon_days: int = 14,
) -> dict:
    bundle = load_forecast_bundle()
    if bundle is None:
        return tool_envelope(
            "forecasting",
            status="error",
            confidence=0.0,
            data={},
            data_slice="no model",
            error_reason="forecasting model not trained",
        )

    events = fetch_events(sku=sku, limit=20000) if sku else fetch_events(limit=20000)
    df = events_to_df(events)
    if category and not df.empty:
        df = df[df["category"] == category]

    live_daily = _daily_demand(df)
    history: pd.DataFrame = bundle["history"]
    if not live_daily.empty:
        # Prefer live event-store slice when available
        history = pd.concat([history, live_daily], ignore_index=True)
        history = history.drop_duplicates(subset=["sku", "date"], keep="last")

    if sku:
        candidates = [sku]
    elif category:
        candidates = (
            history[history["category"] == category]
            .groupby("sku")["qty"]
            .sum()
            .nlargest(10)
            .index.tolist()
        )
    else:
        candidates = history.groupby("sku")["qty"].sum().nlargest(5).index.tolist()

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    series_out = []

    for target_sku in candidates:
        g = history[history["sku"] == target_sku].sort_values("date")
        if g.empty:
            continue
        qty_series = g.set_index("date")["qty"].asfreq("D", fill_value=0)
        forecasts = []
        working = qty_series.copy()
        last_date = working.index.max()
        for step in range(1, horizon_days + 1):
            next_date = last_date + pd.Timedelta(days=step)
            row = {
                "lag_1": float(working.iloc[-1]) if len(working) else 0.0,
                "lag_7": float(working.iloc[-7]) if len(working) >= 7 else float(working.mean()),
                "lag_14": float(working.iloc[-14]) if len(working) >= 14 else float(working.mean()),
                "roll7": float(working.iloc[-7:].mean()) if len(working) else 0.0,
                "dow": int(next_date.dayofweek),
            }
            # Only use columns the model knows
            x = pd.DataFrame([{c: row.get(c, 0.0) for c in feature_cols}])
            pred = float(max(0.0, model.predict(x)[0]))
            forecasts.append({"date": next_date.strftime("%Y-%m-%d"), "qty": round(pred, 2)})
            working.loc[next_date] = pred

        series_out.append({"sku": target_sku, "horizon_days": horizon_days, "forecast": forecasts})

    mape = float(bundle["metrics"].get("mape", 0.5))
    # MAPE can exceed 1.0 on sparse retail series — use capped inverse for confidence
    conf = float(max(0.35, min(0.9, 1.0 / (1.0 + mape))))
    status = "ok" if series_out else "degraded"
    return tool_envelope(
        "forecasting",
        status=status,
        confidence=conf if series_out else 0.2,
        data={"series": series_out, "mape": round(mape, 4)},
        data_slice=f"sku={sku} category={category} horizon={horizon_days} n={len(series_out)}",
        error_reason=None if series_out else "no matching SKUs",
    )
