# Evaluation Results

Offline metrics from training / eval scripts (demo source unless noted).

| Model | Metric | Value | Notes |
|-------|--------|-------|-------|
| Segmentation | accounts clustered | 800 | KMeans k=4 on RFM |
| Churn | Precision / Recall / AUC | ~1.0 / 1.0 / 1.0 | Demo labels separable; LightGBM; **14-day inactivity proxy** |
| Churn | Churn rate | 0.30 | Demo train set |
| Forecasting | MAPE | ~1.3 | Sparse daily series; confidence = `1/(1+MAPE)` |
| Basket mining | Top lift | ~8.5 | FP-Growth; 200 rules stored |
| Recommender | Method | **implicit ALS** + content | `implicit` library; NumPy ALS fallback; w_collab=0.6 |
| Recommender | Recall@10 | **0.239** (191/800) | Hold-out last SKU; `eval_recommender_recall.py` |
| Visual search | backend | CLIP + Chroma | `clip-ViT-B-32` embeddings in persistent Chroma; catalog images under `services/visual-search/models/catalog_images` |
| STT | engines | Whisper API → faster-whisper | Browser Web Speech still available on `/` |
| Translate | engines | NLLB → OPUS-MT → LLM → heuristic | `facebook/nllb-200-distilled-600M` |

## Assumptions (state to judges)

- **Churn proxy:** no `order` in last **14 days**, after prior activity.
- **Regions:** Colombo / Kandy / Galle / Jaffna synthesized in ETL.
- **Instacart / H&M:** ETL hooks ready; Kaggle terms may block raw download — demo parquet covers live demos.
- **Catalog images:** H&M images used when present; otherwise deterministic SKU-colored placeholders for CLIP indexing (documented, not hidden).
- **Approve:** simulated execution only.
- **PLAN:** native `bind_tools` when the LLM supports it; JSON-schema fallback for Gemini.

## Retrain / eval

```bash
python scripts/bootstrap.py
# or:
python data/scripts/train_analytics_models.py --source demo
python data/scripts/train_recommender.py
python data/scripts/generate_catalog_images.py
python data/scripts/eval_recommender_recall.py
```
