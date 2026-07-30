"""
RFM segmentation via KMeans.

Why KMeans on RFM: interpretable B2B segments without labeled segment data.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.envelope import compute_rfm, events_to_df, fetch_events, tool_envelope

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SEGMENT_MODEL_PATH = MODELS_DIR / "segmentation_kmeans.joblib"
SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal",
    2: "At Risk",
    3: "Hibernating",
}


def load_segmentation_bundle():
    if not SEGMENT_MODEL_PATH.exists():
        return None
    return joblib.load(SEGMENT_MODEL_PATH)


def train_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    rfm = compute_rfm(df)
    if len(rfm) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} accounts for segmentation")

    X = rfm[["recency", "frequency", "monetary"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(Xs)

    centers = scaler.inverse_transform(km.cluster_centers_)
    order = np.argsort(-centers[:, 2])  # monetary desc
    label_map = {int(old): SEGMENT_LABELS[i] for i, old in enumerate(order)}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"scaler": scaler, "kmeans": km, "label_map": label_map, "n_clusters": n_clusters}
    joblib.dump(bundle, SEGMENT_MODEL_PATH)
    return {"n_accounts": len(rfm), "path": str(SEGMENT_MODEL_PATH), "label_map": label_map}


def run_segmentation(region: str | None = None, category: str | None = None) -> dict:
    bundle = load_segmentation_bundle()
    if bundle is None:
        return tool_envelope(
            "segmentation",
            status="error",
            confidence=0.0,
            data=[],
            data_slice="no model",
            error_reason="segmentation model not trained — run train_analytics_models.py",
        )

    events = fetch_events(region=region, limit=20000)
    df = events_to_df(events)
    if category and not df.empty and "category" in df.columns:
        df = df[df["category"] == category]

    rfm = compute_rfm(df)
    if rfm.empty:
        return tool_envelope(
            "segmentation",
            status="degraded",
            confidence=0.2,
            data=[],
            data_slice=f"region={region} category={category}",
            error_reason="no events found for filters",
        )

    X = rfm[["recency", "frequency", "monetary"]].values
    Xs = bundle["scaler"].transform(X)
    raw = bundle["kmeans"].predict(Xs)
    label_map = bundle["label_map"]

    results = []
    for idx, (_, row) in enumerate(rfm.iterrows()):
        cluster = int(raw[idx])
        results.append(
            {
                "account_id": row["account_id"],
                "segment": label_map.get(cluster, f"cluster_{cluster}"),
                "cluster_id": cluster,
                "rfm": {
                    "recency": round(float(row["recency"]), 2),
                    "frequency": int(row["frequency"]),
                    "monetary": round(float(row["monetary"]), 2),
                },
                "region": row.get("region"),
            }
        )

    conf = min(0.95, 0.5 + 0.01 * min(len(results), 45))
    return tool_envelope(
        "segmentation",
        status="ok",
        confidence=conf,
        data=results,
        data_slice=f"RFM from event-store region={region} category={category} n={len(results)}",
    )
