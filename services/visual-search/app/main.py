"""
Visual / substitute search (Day 6) — CLIP embeddings + Chroma vector store.

Index product images (H&M subset when present, else deterministic catalog
images generated from SKU attributes). Query by sku, text, or image_base64.
Falls back to TF-IDF text similarity if CLIP/Chroma unavailable.
"""
from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Lankara Visual Search", version="1.0.0")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
CATALOG_PATH = MODELS_DIR / "product_catalog.parquet"
IMAGE_DIR = MODELS_DIR / "catalog_images"
CHROMA_DIR = MODELS_DIR / "chroma_visual"
COLLECTION_NAME = "lankara_visual"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_events() -> Path | None:
    candidates = [
        _repo_root() / "data" / "processed" / "events_demo.parquet",
        Path("/app/data/processed/events_demo.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _hm_image_dirs() -> list[Path]:
    roots = [
        _repo_root() / "data" / "raw" / "hm" / "images",
        Path("/app/data/raw/hm/images"),
    ]
    return [p for p in roots if p.exists()]


def _build_catalog() -> pd.DataFrame:
    events_path = _repo_events()
    if events_path is None:
        rows = [
            {"sku": f"SKU-POP-{i:03d}", "category": "Fashion", "title": f"Fashion item {i}"}
            for i in range(25)
        ] + [
            {
                "sku": f"SKU-CAT-{i:04d}",
                "category": "Electronics",
                "title": f"Electronics item {i}",
            }
            for i in range(40)
        ]
        return pd.DataFrame(rows)

    df = pd.read_parquet(events_path)
    g = (
        df.dropna(subset=["sku"])
        .groupby("sku")
        .agg(category=("category", "first"), n=("event_id", "count"))
        .reset_index()
    )
    g["title"] = g.apply(
        lambda r: f"{r['category'] or 'Item'} product {r['sku']}", axis=1
    )
    return g[["sku", "category", "title"]].head(500)


def _color_for_sku(sku: str) -> tuple[int, int, int]:
    h = hashlib.md5(sku.encode()).hexdigest()
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _ensure_catalog_image(sku: str, category: str, title: str) -> Path:
    """Create a deterministic product image if H&M file missing (demo-capable)."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = IMAGE_DIR / f"{sku.replace('/', '_')}.jpg"
    if out.exists():
        return out

    # Prefer real H&M image if article id matches
    for root in _hm_image_dirs():
        # H&M layout: images/0xx/0xxxxxxxx.jpg
        candidates = list(root.rglob(f"*{sku.replace('HM-', '')}*.jpg"))[:1]
        if candidates:
            return candidates[0]

    from PIL import Image, ImageDraw, ImageFont

    r, g, b = _color_for_sku(sku)
    img = Image.new("RGB", (224, 224), (r, g, b))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 204, 204], outline=(255, 255, 255), width=3)
    label = f"{(category or 'Item')[:18]}\n{sku[:22]}"
    draw.multiline_text((28, 90), label, fill=(255, 255, 255), spacing=4)
    img.save(out, quality=85)
    return out


class _Index:
    def __init__(self) -> None:
        self.catalog = _build_catalog()
        self.skus = self.catalog["sku"].astype(str).tolist()
        texts = (
            self.catalog["title"].fillna("")
            + " "
            + self.catalog["category"].fillna("")
            + " "
            + self.catalog["sku"].astype(str)
        ).tolist()
        self.backend = "tfidf"
        self.matrix = None
        self.model = None
        self.collection = None
        self.image_paths: dict[str, str] = {}
        self.vectorizer = None

        for _, row in self.catalog.iterrows():
            sku = str(row["sku"])
            path = _ensure_catalog_image(sku, str(row.get("category") or ""), str(row.get("title") or ""))
            self.image_paths[sku] = str(path)

        use_clip = os.getenv("VISUAL_USE_CLIP", "1").lower() not in ("0", "false", "no")
        if use_clip:
            self._try_clip_chroma(texts)

        if self.backend == "tfidf":
            self.vectorizer = TfidfVectorizer(max_features=2048)
            self.matrix = self.vectorizer.fit_transform(texts)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.catalog.to_parquet(CATALOG_PATH, index=False)

    def _try_clip_chroma(self, texts: list[str]) -> None:
        try:
            from PIL import Image
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("clip-ViT-B-32")
            images = []
            for sku in self.skus:
                images.append(Image.open(self.image_paths[sku]).convert("RGB"))
            # Encode images with CLIP
            embeddings = np.asarray(
                self.model.encode(images, batch_size=16, normalize_embeddings=True)
            )

            try:
                import chromadb
                from chromadb.config import Settings

                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
                try:
                    client.delete_collection(COLLECTION_NAME)
                except Exception:
                    pass
                self.collection = client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                self.collection.add(
                    ids=self.skus,
                    embeddings=embeddings.tolist(),
                    documents=texts,
                    metadatas=[
                        {
                            "sku": s,
                            "category": str(self.catalog.iloc[i].get("category") or ""),
                            "title": str(self.catalog.iloc[i].get("title") or ""),
                            "image_path": self.image_paths[s],
                        }
                        for i, s in enumerate(self.skus)
                    ],
                )
                self.backend = "clip_chroma"
                self.matrix = embeddings
            except Exception:
                self.matrix = embeddings
                self.backend = "clip"
        except Exception:
            self.backend = "tfidf"
            self.model = None
            self.collection = None

    def encode_text(self, text: str) -> np.ndarray:
        if self.backend in ("clip", "clip_chroma") and self.model is not None:
            return np.asarray(self.model.encode([text], normalize_embeddings=True))
        return self.vectorizer.transform([text])

    def encode_image(self, img) -> np.ndarray:
        if self.backend not in ("clip", "clip_chroma") or self.model is None:
            raise RuntimeError("CLIP not loaded")
        return np.asarray(self.model.encode([img], normalize_embeddings=True))

    def similar_vec(self, query_vec: np.ndarray, k: int = 8, exclude: str | None = None) -> list[dict]:
        q = np.asarray(query_vec)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if self.collection is not None:
            res = self.collection.query(query_embeddings=q.tolist(), n_results=min(k + 2, len(self.skus)))
            out = []
            for i, sku in enumerate(res["ids"][0]):
                if exclude and sku == exclude:
                    continue
                meta = (res["metadatas"][0][i] or {}) if res.get("metadatas") else {}
                dist = res["distances"][0][i] if res.get("distances") else 0.0
                # cosine distance → similarity
                sim = 1.0 - float(dist) if dist is not None else 0.0
                out.append(
                    {
                        "sku": sku,
                        "category": meta.get("category"),
                        "title": meta.get("title"),
                        "similarity": round(sim, 4),
                        "image_path": meta.get("image_path"),
                    }
                )
                if len(out) >= k:
                    break
            return out

        sims = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(-sims)
        out = []
        for idx in order:
            sku = self.skus[int(idx)]
            if exclude and sku == exclude:
                continue
            row = self.catalog.iloc[int(idx)]
            out.append(
                {
                    "sku": sku,
                    "category": row.get("category"),
                    "title": row.get("title"),
                    "similarity": round(float(sims[idx]), 4),
                    "image_path": self.image_paths.get(sku),
                }
            )
            if len(out) >= k:
                break
        return out


@lru_cache(maxsize=1)
def get_index() -> _Index:
    return _Index()


class VisualSearchRequest(BaseModel):
    sku: str | None = None
    text_query: str | None = None
    image_base64: str | None = None
    k: int = Field(default=8, ge=1, le=30)


def _envelope(status: str, confidence: float, data, data_slice: str, error_reason=None):
    return {
        "tool": "visual_search",
        "status": status,
        "confidence": confidence,
        "data": data,
        "data_slice": data_slice,
        "error_reason": error_reason,
    }


@app.get("/health")
def health():
    try:
        idx = get_index()
        return {
            "status": "ok",
            "service": "visual-search",
            "ready": True,
            "backend": idx.backend,
            "n_skus": len(idx.skus),
            "chroma": idx.collection is not None,
        }
    except Exception as exc:
        return {"status": "ok", "service": "visual-search", "ready": False, "error": str(exc)}


@app.post("/visual-search")
def visual_search(body: VisualSearchRequest):
    idx = get_index()
    exclude = body.sku

    if body.image_base64:
        if idx.backend not in ("clip", "clip_chroma"):
            return _envelope(
                "degraded",
                0.25,
                [],
                "image query without CLIP",
                "CLIP not loaded. Set VISUAL_USE_CLIP=1 and install sentence-transformers.",
            )
        try:
            from PIL import Image

            raw = body.image_base64.split(",")[-1]
            img = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
            q = idx.encode_image(img)
            results = idx.similar_vec(q, k=body.k, exclude=exclude)
            return _envelope(
                "ok",
                0.85,
                results,
                f"CLIP image search backend={idx.backend} n={len(results)}",
            )
        except Exception as exc:
            return _envelope("degraded", 0.3, [], "clip image failed", str(exc))

    query_text = body.text_query
    if body.sku:
        row = idx.catalog[idx.catalog["sku"].astype(str) == str(body.sku)]
        if not row.empty:
            # Prefer image embedding of the seed SKU when CLIP is available
            if idx.backend in ("clip", "clip_chroma") and body.sku in idx.image_paths:
                try:
                    from PIL import Image

                    img = Image.open(idx.image_paths[body.sku]).convert("RGB")
                    q = idx.encode_image(img)
                    results = idx.similar_vec(q, k=body.k, exclude=exclude)
                    return _envelope(
                        "ok",
                        0.82,
                        results,
                        f"CLIP sku-image search sku={body.sku} backend={idx.backend}",
                    )
                except Exception:
                    pass
            query_text = f"{row.iloc[0]['title']} {row.iloc[0]['category']} {body.sku}"
        else:
            query_text = query_text or str(body.sku)

    if not query_text:
        return _envelope(
            "error",
            0.0,
            [],
            "missing query",
            "Provide sku, text_query, or image_base64",
        )

    q = idx.encode_text(query_text)
    results = idx.similar_vec(q, k=body.k, exclude=exclude)
    conf = 0.8 if idx.backend.startswith("clip") else 0.55
    return _envelope(
        "ok",
        conf,
        results,
        f"substitute search query={query_text[:80]!r} backend={idx.backend}",
    )
