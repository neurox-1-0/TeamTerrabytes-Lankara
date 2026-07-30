"""
Hybrid recommender: implicit ALS collaborative filtering + content-based cosine,
blended by configurable weight (plan Day 3).

ALS uses Hu et al. confidence-weighted alternating least squares on CPU
(pure NumPy/SciPy). Falls back to item-item cosine if factorization fails.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "hybrid_recommender.joblib"


def _interaction_matrix(df: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    orders = df[df["event_type"].isin(["order", "cart_add", "view"])].dropna(subset=["sku", "account_id"])
    accounts = sorted(orders["account_id"].astype(str).unique().tolist())
    skus = sorted(orders["sku"].astype(str).unique().tolist())
    a_idx = {a: i for i, a in enumerate(accounts)}
    s_idx = {s: i for i, s in enumerate(skus)}

    weight = {"view": 1.0, "cart_add": 3.0, "order": 5.0}
    rows, cols, data = [], [], []
    for _, r in orders.iterrows():
        rows.append(a_idx[str(r["account_id"])])
        cols.append(s_idx[str(r["sku"])])
        data.append(weight.get(str(r["event_type"]), 1.0))
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(accounts), len(skus)))
    return mat, accounts, skus


def _content_sim(df: pd.DataFrame, skus: list[str]) -> np.ndarray:
    meta = (
        df.dropna(subset=["sku"])
        .groupby("sku")
        .agg(category=("category", "first"))
        .reindex(skus)
    )
    cats = meta["category"].fillna("unknown").astype(str)
    cat_codes = pd.Categorical(cats).codes
    n = len(skus)
    n_cats = int(cat_codes.max()) + 1 if n else 1
    rows = np.arange(n)
    X = sparse.csr_matrix((np.ones(n), (rows, cat_codes)), shape=(n, max(n_cats, 1)))
    X = normalize(X)
    return cosine_similarity(X)


def _item_item_collab(mat: sparse.csr_matrix) -> np.ndarray:
    item_mat = normalize(mat.T)
    return cosine_similarity(item_mat)


def _implicit_als(
    R: sparse.csr_matrix,
    n_factors: int = 32,
    n_iters: int = 12,
    reg: float = 0.1,
    alpha: float = 40.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Confidence-weighted ALS for implicit feedback (Hu, Koren, Volinsky 2008).

    Prefers the `implicit` library when installed; otherwise pure NumPy/SciPy.
    Returns (user_factors, item_factors, method_label).
    """
    n_users, n_items = R.shape
    # Try official implicit package first
    try:
        from implicit.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        model = AlternatingLeastSquares(
            factors=n_factors,
            regularization=reg,
            iterations=n_iters,
            random_state=seed,
            use_gpu=False,
        )
        # implicit expects item-user CSR (items × users)
        with threadpool_limits(limits=1, user_api="blas"):
            model.fit(R.T.tocsr())
        uf = np.asarray(model.user_factors, dtype=np.float64)
        itf = np.asarray(model.item_factors, dtype=np.float64)
        # Guard against API orientation surprises across implicit versions
        if uf.shape[0] == n_items and itf.shape[0] == n_users:
            uf, itf = itf, uf
        if uf.shape[0] != n_users or itf.shape[0] != n_items:
            raise ValueError(
                f"ALS factor shapes mismatch: users={uf.shape}, items={itf.shape}, "
                f"expected ({n_users},*), ({n_items},*)"
            )
        return uf, itf, "implicit_als"
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, size=(n_users, n_factors))
    Y = rng.normal(0, 0.01, size=(n_items, n_factors))

    # Preference P = 1 if interacted else 0; confidence C = 1 + alpha * r
    R = R.tocsr().astype(np.float64)
    Cu = []
    for u in range(n_users):
        start, end = R.indptr[u], R.indptr[u + 1]
        cols = R.indices[start:end]
        vals = R.data[start:end]
        Cu.append((cols, 1.0 + alpha * vals))

    Rt = R.T.tocsr()
    Ci = []
    for i in range(n_items):
        start, end = Rt.indptr[i], Rt.indptr[i + 1]
        rows = Rt.indices[start:end]
        vals = Rt.data[start:end]
        Ci.append((rows, 1.0 + alpha * vals))

    eye = np.eye(n_factors)
    for _ in range(n_iters):
        YtY = Y.T @ Y
        for u in range(n_users):
            cols, conf = Cu[u]
            if len(cols) == 0:
                X[u] = 0.0
                continue
            Y_u = Y[cols]
            C_u = np.asarray(conf)
            # A = YtY + Y_u.T (C-I) Y_u + reg I ; b = Y_u.T C p
            A = YtY + (Y_u.T * (C_u - 1.0)) @ Y_u + reg * eye
            b = (Y_u.T * C_u) @ np.ones(len(cols))
            X[u] = np.linalg.solve(A, b)

        XtX = X.T @ X
        for i in range(n_items):
            rows, conf = Ci[i]
            if len(rows) == 0:
                Y[i] = 0.0
                continue
            X_i = X[rows]
            C_i = np.asarray(conf)
            A = XtX + (X_i.T * (C_i - 1.0)) @ X_i + reg * eye
            b = (X_i.T * C_i) @ np.ones(len(rows))
            Y[i] = np.linalg.solve(A, b)

    return X, Y, "numpy_als"


def train_hybrid(
    df: pd.DataFrame,
    collab_weight: float = 0.6,
    n_factors: int = 32,
    n_iters: int = 12,
) -> dict:
    mat, accounts, skus = _interaction_matrix(df)
    if len(skus) < 5 or len(accounts) < 5:
        raise ValueError("Need more accounts/SKUs to train recommender")

    content_sim = _content_sim(df, skus)
    user_factors, item_factors, als_method = _implicit_als(
        mat, n_factors=n_factors, n_iters=n_iters
    )
    # Also keep item-item as similarity fallback / sku-seed path
    collab_sim = item_factors @ item_factors.T
    # Normalize for stable blend with content
    norms = np.linalg.norm(collab_sim, axis=1, keepdims=True) + 1e-9
    collab_sim = collab_sim / norms

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "accounts": accounts,
        "skus": skus,
        "user_item": mat,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "collab_sim": collab_sim,
        "content_sim": content_sim,
        "collab_weight": collab_weight,
        "account_index": {a: i for i, a in enumerate(accounts)},
        "sku_index": {s: i for i, s in enumerate(skus)},
        "method": als_method,
    }
    joblib.dump(bundle, MODEL_PATH)
    return {
        "n_accounts": len(accounts),
        "n_skus": len(skus),
        "collab_weight": collab_weight,
        "method": als_method,
        "n_factors": n_factors,
        "path": str(MODEL_PATH),
    }


def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def recommend(
    *,
    account_id: str | None = None,
    sku: str | None = None,
    k: int = 10,
    collab_weight: float | None = None,
) -> dict:
    """Return ranked SKUs with collab/content/blended scores + tool envelope fields."""
    bundle = load_bundle()
    if bundle is None:
        return {
            "tool": "recommender",
            "status": "error",
            "confidence": 0.0,
            "data": [],
            "data_slice": "no model",
            "error_reason": "hybrid recommender not trained",
        }

    content = bundle["content_sim"]
    skus = bundle["skus"]
    w = float(collab_weight if collab_weight is not None else bundle["collab_weight"])
    method = bundle.get("method", "item_item")
    user_factors = bundle.get("user_factors")
    item_factors = bundle.get("item_factors")
    collab = bundle["collab_sim"]

    if account_id and account_id in bundle["account_index"]:
        u = bundle["account_index"][account_id]
        hist = bundle["user_item"].getrow(u).toarray().ravel()
        if user_factors is not None and item_factors is not None:
            # Ensure orientation: user_factors[n_accounts], item_factors[n_skus]
            uf, itf = user_factors, item_factors
            if uf.shape[0] == len(skus) and itf.shape[0] == len(bundle["accounts"]):
                uf, itf = itf, uf
            scores_c = uf[u] @ itf.T
            method_tag = f"account_{method}"
        else:
            scores_c = hist @ collab
            method_tag = "account_item_item"
        scores_t = hist @ content
        if scores_c.shape != scores_t.shape:
            # Last-resort: fall back to item-item collab if ALS shapes disagree
            scores_c = hist @ collab
            method_tag = "account_item_item_fallback"
        # Normalize collaborative scores for blending
        if scores_c.max() > scores_c.min():
            scores_c_n = (scores_c - scores_c.min()) / (scores_c.max() - scores_c.min() + 1e-9)
        else:
            scores_c_n = scores_c
        if scores_t.max() > scores_t.min():
            scores_t_n = (scores_t - scores_t.min()) / (scores_t.max() - scores_t.min() + 1e-9)
        else:
            scores_t_n = scores_t
        blended = w * scores_c_n + (1 - w) * scores_t_n
        blended[hist > 0] = -1.0
        top = np.argsort(-blended)[:k]
        results = []
        for i in top:
            if blended[i] < 0:
                continue
            results.append(
                {
                    "sku": skus[int(i)],
                    "collab_score": round(float(scores_c[i]), 4),
                    "content_score": round(float(scores_t[i]), 4),
                    "blended_score": round(float(blended[i]), 4),
                    "method": method_tag,
                }
            )
        conf = 0.8 if results else 0.2
        return {
            "tool": "recommender",
            "status": "ok" if results else "degraded",
            "confidence": conf,
            "data": results,
            "data_slice": (
                f"account_id={account_id} k={k} w_collab={w} method={method_tag}"
            ),
            "error_reason": None if results else "no recommendations produced",
        }

    if sku and sku in bundle["sku_index"]:
        i0 = bundle["sku_index"][sku]
        if item_factors is not None:
            itf = item_factors
            if itf.shape[0] != len(skus) and user_factors is not None and user_factors.shape[0] == len(skus):
                itf = user_factors
            scores_c = itf[i0] @ itf.T
            method_tag = f"sku_{method}"
        else:
            scores_c = collab[i0]
            method_tag = "sku_item_item"
        scores_t = content[i0]
        if np.asarray(scores_c).shape != np.asarray(scores_t).shape:
            scores_c = collab[i0]
            method_tag = "sku_item_item_fallback"
        if scores_c.max() > scores_c.min():
            scores_c_n = (scores_c - scores_c.min()) / (scores_c.max() - scores_c.min() + 1e-9)
        else:
            scores_c_n = scores_c
        if scores_t.max() > scores_t.min():
            scores_t_n = (scores_t - scores_t.min()) / (scores_t.max() - scores_t.min() + 1e-9)
        else:
            scores_t_n = scores_t
        blended = w * scores_c_n + (1 - w) * scores_t_n
        blended[i0] = -1.0
        top = np.argsort(-blended)[:k]
        results = []
        for i in top:
            if blended[i] < 0:
                continue
            results.append(
                {
                    "sku": skus[int(i)],
                    "collab_score": round(float(scores_c[i]), 4),
                    "content_score": round(float(scores_t[i]), 4),
                    "blended_score": round(float(blended[i]), 4),
                    "method": method_tag,
                }
            )
        conf = 0.8 if results else 0.2
        return {
            "tool": "recommender",
            "status": "ok" if results else "degraded",
            "confidence": conf,
            "data": results,
            "data_slice": f"sku={sku} k={k} w_collab={w} method={method_tag}",
            "error_reason": None if results else "no recommendations produced",
        }

    pop = np.asarray(bundle["user_item"].sum(axis=0)).ravel()
    top = np.argsort(-pop)[:k]
    results = [
        {
            "sku": skus[int(i)],
            "collab_score": 0.0,
            "content_score": 0.0,
            "blended_score": round(float(pop[i]), 4),
            "method": "popularity_fallback",
        }
        for i in top
    ]
    return {
        "tool": "recommender",
        "status": "ok" if results else "degraded",
        "confidence": 0.55 if results else 0.2,
        "data": results,
        "data_slice": f"account_id={account_id} sku={sku} k={k} method=popularity_fallback",
        "error_reason": None if results else "no recommendations produced",
    }
