# Lankara — NeuroX 1.0 Phase 2 Implementation Plan
**Team Terrabytes · Solo build · 7 days · Goal: working, defensible, judge-proof autonomous agent**

This is the master plan. Feed sections to Cursor Pro one folder at a time (see "How to use this with Cursor" below). Do not paste the whole file into one Cursor session — Cursor works best with scoped context per service.

---

## 0. Non-negotiable scoring guardrails (from Phase 2 rubric)

Keep this list pinned. Every day's work gets checked against it before moving on.

| Judging criterion | Weight | What kills it |
|---|---|---|
| Autonomous Reasoning | 25% | Fixed pipeline, hardcoded tool order, no branching |
| B2B Impact & Viability | 25% | Vague or consumer-facing framing |
| Technical Architecture | 20% | Monolith, single API call disguised as "multi-tool" |
| Human-in-the-Loop | 15% | Cosmetic "confirm" button, no edit/reject path |
| Live Demo | 15% | Hardcoded/pre-fetched data presented as live |

**Five explicit failure modes to avoid** (from "What Will NOT Score Well"):
1. Chatbot only — the agent must *act* (produce a proposal), not just answer questions.
2. Fixed pipeline — tool order/selection must be decided at runtime by the LLM, not `if/else` on keyword.
3. Single API call — you need ≥3 genuinely independent tools, agent picks which to use.
4. Unexplained AI code — you must be able to explain every architectural decision, line-level, on demand. Cursor will generate a lot of code; **you review and understand it before moving to the next task**, not after.
5. Fake demo — the live run must hit the real seeded database, not a cached JSON blob.

---

## 1. Locked scope (per your answers)

- **Timeline:** 7 days, solo.
- **Tools integrated:** all 5 from the Phase 1 report — Analytics Pipelines (4 sub-models), Hybrid Recommender, Event Store, Visual Search (CLIP), Voice STT+Translation (Whisper/NLLB). Voice + visual search are Day 6 stretch — see the cut-scope trigger in Day 6.
- **Orchestrator:** LangGraph (Python), calling every tool over HTTP as an independent microservice — matches "Analytics pipelines API," "Hybrid recommender service (FastAPI)" etc. in your own report, so the Phase 1 story and Phase 2 build stay consistent for judges.
- **LLM provider:** Anthropic Claude via API (tool-use/function-calling) as primary reasoning engine — allowed per the "Stack & Technical Rules" slide. Keep an OpenAI fallback key wired in case of rate limits during the live demo (belt-and-braces, not double work — same call signature, swap via env var).

---

## 2. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (Next.js)                                          │
│  Goal input → live trace stream → Approval Queue → history   │
└───────────────────────────┬───────────────────────────────────┘
                             │ REST + WebSocket
┌───────────────────────────▼───────────────────────────────────┐
│  AGENT-CORE (FastAPI + LangGraph)                             │
│  PERCEIVE → PLAN → REASON → PROPOSE                            │
│  - ambiguity handler (ask vs scope-and-state)                  │
│  - retry → fallback → degrade → block ladder per tool call     │
│  - decision_trail persisted every step                         │
│  - Approval Queue endpoints (draft/review/execute/feedback)    │
└──┬────────────┬────────────┬────────────┬────────────┬────────┘
   │ HTTP        │ HTTP       │ HTTP       │ HTTP       │ HTTP
┌──▼───┐   ┌─────▼────┐  ┌────▼─────┐ ┌────▼──────┐ ┌───▼────────┐
│Event │   │Analytics │  │Recommender│ │  Visual   │ │STT+Transl. │
│Store │   │Pipelines │  │  (hybrid) │ │  Search   │ │(stretch)   │
│(PG)  │   │(4 models)│  │           │ │  (CLIP)   │ │            │
└──────┘   └──────────┘  └───────────┘ └───────────┘ └────────────┘
```

Every box below "AGENT-CORE" is a standalone FastAPI service in its own Docker container. The orchestrator never imports their code directly — it only calls their HTTP contract. This is what makes "Multi-Tool Integration" and "Technical Architecture" scoring genuine rather than cosmetic, and it's why a failure in one tool can't take the loop down.

---

## 3. Repo structure

```
lankara/
├── docker-compose.yml
├── .env.example
├── README.md                      ← quickstart for judges
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPLAIN.md                 ← answers to "The Core Test" questions
│   ├── EVAL_RESULTS.md
│   └── demo-script.md
├── data/
│   ├── raw/                       ← downloaded datasets (gitignored)
│   ├── notebooks/                 ← offline training notebooks
│   │   ├── 01_churn_model.ipynb
│   │   ├── 02_forecasting_model.ipynb
│   │   ├── 03_basket_mining.ipynb
│   │   └── 04_recommender_training.ipynb
│   └── scripts/
│       ├── download_datasets.py
│       └── etl_to_event_schema.py
├── services/
│   ├── event-store/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── models.py          ← SQLAlchemy models
│   │   │   ├── schemas.py         ← Pydantic
│   │   │   └── crud.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── analytics-pipeline/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── segmentation.py
│   │   │   ├── churn.py
│   │   │   ├── forecasting.py
│   │   │   └── basket_mining.py
│   │   ├── models/                ← saved .pkl / .joblib artifacts
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── recommender/
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   └── hybrid.py
│   │   ├── models/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── visual-search/             ← STRETCH
│   │   ├── app/main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── stt-translation/           ← STRETCH
│       ├── app/main.py
│       ├── Dockerfile
│       └── requirements.txt
├── agent-core/
│   ├── app/
│   │   ├── main.py                ← FastAPI + WebSocket endpoints
│   │   ├── graph.py                ← LangGraph state graph definition
│   │   ├── nodes/
│   │   │   ├── perceive.py
│   │   │   ├── plan.py
│   │   │   ├── reason.py
│   │   │   └── propose.py
│   │   ├── tools/
│   │   │   ├── base.py            ← retry/fallback/degrade/block wrapper
│   │   │   ├── analytics_tool.py
│   │   │   ├── recommender_tool.py
│   │   │   ├── visual_search_tool.py
│   │   │   └── stt_tool.py
│   │   ├── approval_queue.py      ← draft/review/execute/feedback
│   │   ├── trail.py                ← decision_trail persistence
│   │   └── db.py
│   ├── Dockerfile
│   └── requirements.txt
└── frontend/
    ├── app/
    │   ├── page.tsx                ← goal input + live trace
    │   ├── queue/page.tsx          ← Approval Queue
    │   └── history/[id]/page.tsx   ← decision trail explorer
    ├── components/
    ├── package.json
    └── Dockerfile
```

---

## 4. Shared contracts (define these FIRST, before any service code — this is what lets you build services in isolated Cursor sessions without them drifting apart)

### 4.1 Unified event schema (`event-store`)

```json
{
  "event_id": "uuid",
  "event_type": "view | cart_add | cart_remove | order | return | support_ticket | voice_search | visual_search",
  "account_id": "string",
  "sku": "string | null",
  "category": "string | null",
  "quantity": "int | null",
  "price": "float | null",
  "discount_applied": "float | null",
  "channel": "string | null",
  "reason_code": "string | null",
  "sentiment": "string | null",
  "session_id": "string",
  "device": "string | null",
  "region": "string | null",
  "timestamp": "iso8601"
}
```
Every dataset (RetailRocket, Online Retail II, Instacart, H&M) gets ETL'd into this one schema in `data/scripts/etl_to_event_schema.py`. This is the single most important file for consistency — build it Day 1 and don't touch the schema again.

### 4.2 Tool response envelope (every microservice returns this shape, always)

```json
{
  "tool": "churn_scoring",
  "status": "ok | degraded | error",
  "confidence": 0.0,
  "data": { "...tool-specific payload..." },
  "data_slice": "human-readable description of exact input used",
  "error_reason": "string | null"
}
```
This envelope is what makes the REASON stage and the evidence pack possible without special-casing every tool. Non-negotiable — every service must return it.

### 4.3 Proposal object (what PROPOSE emits into the Approval Queue)

```json
{
  "proposal_id": "uuid",
  "goal": "string",
  "action_type": "price_change | reorder | retention_campaign | report",
  "target": { "segment | sku_list | account_ids": "..." },
  "payload": { "...the actual drafted action..." },
  "evidence": [ /* array of tool response envelopes used */ ],
  "assumptions": ["string"],
  "confidence": 0.0,
  "status": "pending | approved | edited | rejected | executed",
  "created_at": "iso8601"
}
```

### 4.4 Decision trail entry (what gets streamed to frontend + persisted)

```json
{
  "run_id": "uuid",
  "step": "PERCEIVE | PLAN | REASON | PROPOSE",
  "content": "string (what happened, in plain language)",
  "tool_calls": [ {"tool": "...", "input": {}, "output": {}} ],
  "timestamp": "iso8601"
}
```

---

## 5. Data acquisition (Day 1 morning)

| Dataset | Source | Drives |
|---|---|---|
| RetailRocket | Kaggle: `retailrocket/ecommerce-dataset` | Clickstream events, churn labels |
| Online Retail II | UCI ML Repository | Segmentation, forecasting |
| Instacart Market Basket | Kaggle: `instacart-market-basket-analysis` | Basket mining |
| H&M Personalized Fashion | Kaggle: `h-and-m-personalized-fashion-recommendations` | Recommender + product images for visual search |

Use `kaggle` CLI (`pip install kaggle`, needs API token in `~/.kaggle/kaggle.json`). Download only the subset you need — for H&M, cap at ~5,000 products / ~2,000 images to keep visual search buildable in one day. Write `data/scripts/download_datasets.py` to automate this so a judge (or you, if your machine dies) can reproduce it.

---

## 6. Day-by-day plan with Cursor prompts

> **How to use this with Cursor:** open one service folder as its own workspace/context in Cursor. Paste the "Cursor prompt" for that day, then paste the relevant contract from Section 4. Review every file Cursor generates before running it — you need to be able to explain it Day 7.

### Day 1 — Foundation + Event Store
**Goal:** `docker-compose up` brings up Postgres with real seeded behavioral data, queryable via API.

Tasks:
1. Scaffold repo structure above. Write `docker-compose.yml` with services: `postgres`, `event-store` (build later services incrementally, comment out until ready).
2. Write `data/scripts/download_datasets.py`.
3. Write `data/scripts/etl_to_event_schema.py` — transforms all 4 raw datasets into the unified event schema (Section 4.1) and bulk-inserts into Postgres.
4. Build `services/event-store`: SQLAlchemy model matching 4.1, FastAPI with:
   - `POST /events` (single/bulk insert)
   - `GET /events?account_id=&region=&event_type=&since=` (filtered query)
   - `GET /accounts/{id}/events` (full account history)

**Cursor prompt (event-store):**
> Build a FastAPI microservice called `event-store` using SQLAlchemy + Postgres. Table `events` with this schema: [paste 4.1]. Endpoints: POST /events (accepts single object or array, bulk upsert), GET /events with query params account_id, region, event_type, sku, since (ISO date), limit — returns list of matching events, GET /accounts/{account_id}/events — full event history for one account. Include a Dockerfile and requirements.txt. Include Alembic-free auto-create-tables-on-startup for simplicity.

**Deliverable check:** `curl localhost:8001/events?region=Kandy` returns real rows from a real dataset, not mock data.

---

### Day 2 — Analytics Pipeline service (this is the highest-weight tool — do it early, do it well)
**Goal:** 4 working endpoints, each backed by an offline-trained model.

Tasks (train offline in notebooks, serve via FastAPI):
1. **Segmentation** — RFM features (Recency/Frequency/Monetary) computed from event-store data + KMeans clustering. Endpoint: `POST /segmentation {region?, category?}` → list of accounts with segment label + RFM scores.
2. **Churn scoring** — LightGBM/XGBoost classifier trained on RetailRocket engagement decay as churn proxy label. Endpoint: `POST /churn-scoring {account_ids[] | region}` → risk score 0-1 per account + top contributing features.
3. **Forecasting** — LightGBM or statsmodels SARIMA per-SKU demand forecast trained on Online Retail II. Endpoint: `POST /forecasting {sku | category, horizon_days}` → forecast series + MAPE-based confidence.
4. **Basket mining** — FP-Growth (mlxtend) on Instacart baskets. Endpoint: `POST /basket-mining {sku | category, min_support, min_confidence}` → association rules with lift.

Every endpoint returns the Section 4.2 envelope.

**Cursor prompt (per sub-model, run 4 times in the same session):**
> In `services/analytics-pipeline`, add a churn scoring module. Train a LightGBM binary classifier offline using data in `data/raw/retailrocket` — label churn as [define: e.g. no purchase event in the last N days after being active]. Save the trained model to `models/churn_model.pkl`. Add a FastAPI endpoint `POST /churn-scoring` that loads the model at startup, accepts {account_ids: string[]} or {region: string}, pulls recent features from the event-store service via HTTP, and returns the standard tool envelope: [paste 4.2] with data = [{account_id, churn_probability, top_features}].

**Deliverable check:** all 4 endpoints return real numbers computed from real trained models — not `random.random()`. Screenshot each response for `docs/EVAL_RESULTS.md`.

---

### Day 3 — Recommender + Agent Core skeleton
**Goal:** hybrid recommender working standalone; LangGraph agent can call at least one tool end-to-end.

Tasks:
1. `services/recommender`: collaborative filtering (implicit ALS on H&M interactions) + content-based fallback (cosine similarity on product category/attributes) blended with a configurable weight. Endpoint: `POST /recommend {account_id | sku, k}` → ranked SKU list with method breakdown (collab_score, content_score, blended_score).
2. `agent-core`: set up LangGraph `StateGraph` with 4 nodes (perceive, plan, reason, propose) per Section 3.1 of the report. Wire Claude tool-use so the **plan** node genuinely decides which tools to call — do not hardcode "always call segmentation then churn."
3. Build `tools/base.py`: a wrapper function `call_tool(name, url, payload)` implementing the retry → fallback → degrade → block ladder (Section 3.3 of the Phase 1 report) around every HTTP call to a microservice.

**Cursor prompt (agent-core skeleton):**
> Build a LangGraph StateGraph in Python with 4 nodes: perceive, plan, reason, propose. State object holds: goal (str), perceived_events (list), plan (list of tool calls), tool_results (list), proposal (dict), decision_trail (list). The `plan` node calls Claude (Anthropic API, tool-use/function-calling) giving it the goal plus a summary of perceived_events, and a manifest of available tools [analytics: segmentation/churn/forecasting/basket_mining, recommender], and asks it to decide which tools to call and in what order — return this as structured tool-call intents, not free text. The `reason` node combines tool_results, resolves conflicts, computes an overall confidence score. Every node appends a decision_trail entry using this schema: [paste 4.4]. Write `tools/base.py` with a call_tool function that: 1) tries the HTTP call once, 2) retries once on failure, 3) on second failure tries a fallback (coarser query / cached slice), 4) if still failing, marks the tool result status="degraded" and lowers overall confidence, 5) if the tool is essential and totally unavailable, raise a BlockedProposal exception with a human-readable reason instead of guessing.

**Deliverable check:** run `agent-core` with a hardcoded goal like `"Which SKUs need reordering in Kandy?"` — confirm in logs that Claude actually chose which tools to call (vary the goal, confirm the tool selection changes).

---

### Day 4 — Complete agentic loop + evidence pack + persistence
**Goal:** goal in → full trace → Proposal out, for 3+ distinct goal types, including one ambiguous input and one simulated tool failure.

Tasks:
1. Implement ambiguity handling in `plan.py`: before planning tools, Claude first classifies whether the goal is materially ambiguous (region/category/time window would change the answer). If yes → emit a clarifying-question decision_trail step and pause the run (frontend will surface this later). If no → agent scopes itself and states the assumption in the trail (this must show up later in the evidence pack, unmodified).
2. Implement `propose.py`: assembles the Proposal object (Section 4.3) from tool_results + reasoning, writes it to Postgres (`proposals` table) with status `pending`.
3. Implement `trail.py`: persist every decision_trail entry to a `decision_trail` table keyed by `run_id`, so any run is replayable later for judges.
4. Write 3 test goals and run them end-to-end via a CLI script `agent-core/run_goal.py "..."`:
   - Clear goal (e.g., "Draft reorder proposals for slow-moving SKUs in the Colombo region")
   - Ambiguous goal (e.g., "Sales look weak") — confirm it asks or scopes-and-states
   - Forced tool failure (temporarily stop the recommender container) — confirm retry → fallback → degrade shows up correctly in the trail, and the proposal ships with a lower confidence rather than failing silently

**Deliverable check:** for each of the 3 test goals, you can pull up the full decision_trail from Postgres and narrate it out loud without looking at code — this is your Day 7 "explain your agent" rehearsal starting early.

---

### Day 5 — Approval Queue + Frontend
**Goal:** full click-through demo works in a browser: type a goal, watch it reason, approve/edit/reject the result.

Tasks:
1. `approval_queue.py` in agent-core:
   - `GET /proposals?status=pending` 
   - `POST /proposals/{id}/approve` → status=executed, writes to `actions_executed` table (this is a **simulated** execution — a log row, not a real price change/email; be explicit about this to judges, it's expected)
   - `POST /proposals/{id}/edit` → accepts a modified payload, status=edited, re-runs confidence recompute if relevant
   - `POST /proposals/{id}/reject` → status=rejected, reason captured
   - Every edit/reject writes a row to `reviewer_feedback` (segment, action_type, original_payload, human_payload, reason) — this is what the `plan`/`propose` nodes read back as few-shot context next time a similar goal/segment comes up (implement as: prepend last 3 relevant feedback rows into the Claude prompt for propose). This operationalizes "compounding accuracy" from the report.
2. Frontend (Next.js + Tailwind, use `frontend-design` skill for styling choices):
   - `/` — goal input box, submit triggers agent-core run via WebSocket, trace streams in live as PERCEIVE/PLAN/REASON/PROPOSE cards appear one at a time (this *is* your live demo — it must look like reasoning happening, not a spinner).
   - `/queue` — Approval Queue: cards showing action_type, target, payload preview, confidence badge, an expandable evidence panel (chart via `recharts`, exact data slice as a table), Approve/Edit/Reject buttons.
   - `/history/[run_id]` — full replay of any past run's decision trail, for judges to dig into architecture after the demo.

**Cursor prompt (frontend):**
> Build a Next.js 14 app router frontend with Tailwind. Page 1 (`/`): a textarea for a goal, a submit button that opens a WebSocket to `ws://localhost:8000/ws/run` sending {goal}, and streams back decision_trail entries [paste 4.4] rendered as a vertical timeline of cards, one per step, appearing progressively (not all at once). Page 2 (`/queue`): fetches GET /proposals?status=pending, renders each Proposal [paste 4.3] as a card with an expandable evidence section (render evidence[].data as a small recharts bar/line chart where numeric, else a table), Approve/Edit/Reject buttons calling the respective POST endpoints. Page 3 (`/history/[run_id]`): fetches the full trail for a run_id and renders it the same way as page 1 but static/replay.

**Deliverable check:** full loop works with zero manual DB edits — goal typed in browser → visible reasoning → proposal in queue → approve button changes status.

---

### Day 6 — Stretch tools (visual search + voice/translation) OR hardening
**Cut-scope trigger:** if by end of Day 5 the core loop (Days 1–5) isn't rock solid — any of the 3 test goals from Day 4 doesn't run cleanly, or the frontend demo has visible bugs — **skip stretch tools entirely and spend Day 6 hardening the core loop instead.** A polished 3-tool agent beats a buggy 5-tool agent on every rubric line. Decide this at 9am Day 6, not at 9pm.

If proceeding with stretch:
1. `services/visual-search`: CLIP embeddings (use `sentence-transformers` `clip-ViT-B-32`, avoid raw `openai/clip` install friction) over the H&M image subset, stored in Chroma (simplest local vector DB, no extra infra). Endpoint: `POST /visual-search {sku | image_base64}` → visually similar/substitute SKUs. Wire as a genuinely optional tool the `plan` node can choose when a proposal needs a substitute SKU (e.g., reorder proposal where the exact SKU is out of stock upstream).
2. `services/stt-translation`: use the Anthropic/OpenAI-hosted Whisper API for STT (skip local model weights — saves a full day of GPU/dependency pain) and a lightweight translation call (either NLLB-200-distilled-600M locally if time allows, or a translation API) for Sinhala/Tamil/English. Endpoint: `POST /transcribe` (audio) → text + detected language; `POST /translate` → target text. Wire a mic button into the frontend goal input.

**Deliverable check:** both stretch tools are called by the agent *sometimes, not always* — prove this by running two goals side by side where only one triggers a visual-search call, showing genuine runtime tool selection rather than a forced pipeline.

---

### Day 7 — Evaluation, docs, and demo rehearsal
**Goal:** you can defend every claim in the Phase 1 report with a working artifact, and you've rehearsed the live run at least 3 times.

Tasks:
1. **Offline evaluation** — compute and record in `docs/EVAL_RESULTS.md`:
   - Recommender: Recall@K on held-out H&M interactions
   - Forecasting: MAPE per SKU category
   - Churn: Precision (and recall, for your own sanity) on held-out RetailRocket labels
   - Basket mining: support/confidence/lift for top rules
2. **`docs/EXPLAIN.md`** — write direct answers to the exact 4 judge questions from "The Core Test" slide, each with a pointer to the specific file/function that proves it:
   - *What makes it autonomous?* → point to `plan.py`'s Claude tool-use call, not hardcoded branches.
   - *How does the agent loop work?* → walk PERCEIVE→PLAN→REASON→PROPOSE with a real run_id from your Postgres trail.
   - *How does it choose tools?* → show two different goals producing two different tool-call sequences.
   - *How does it recover from failures?* → replay the Day 4 forced-failure test, show the degraded confidence in the trail.
3. **`docs/ARCHITECTURE.md`** — the diagram from Section 2 plus a one-paragraph rationale per service (why FastAPI, why LangGraph, why Postgres over a vector-only store, etc.) — this is your defense against "unexplained AI code."
4. **`docs/demo-script.md`** — scripted live-demo goals (2–3 unscripted-feeling but pre-tested prompts you know exercise different tool combinations), plus a fallback: a screen-recorded backup run in case of live network/API issues on demo day.
5. Full clean-clone test: `git clone` into a fresh directory, `docker-compose up`, confirm everything comes up green with zero manual steps beyond `.env` population.
6. Buffer time for whatever broke.

---

## 7. `.env.example`

```
ANTHROPIC_API_KEY=
OPENAI_API_KEY=              # fallback LLM + Whisper API if not running local STT
DATABASE_URL=postgresql://lankara:lankara@postgres:5432/lankara
EVENT_STORE_URL=http://event-store:8001
ANALYTICS_URL=http://analytics-pipeline:8002
RECOMMENDER_URL=http://recommender:8003
VISUAL_SEARCH_URL=http://visual-search:8004
STT_TRANSLATION_URL=http://stt-translation:8005
```

---

## 8. Explaining the code (mandatory prep, not optional polish)

Judges are explicitly told to check for "unexplained AI code." Do this as you go, not on Day 7:

- After each Cursor-generated file, add a 2–3 line docstring at the top explaining *why* this approach (not what the code does — Cursor's code is usually readable; the *why* is what you'll be asked).
- Keep a running note in `docs/EXPLAIN.md` of any place you accepted a Cursor suggestion you didn't fully understand — go back and understand it before Day 6. If you can't explain it by Day 6, simplify it rather than keep it as a black box, even if it "works."

---

## 9. Open items you'll need to decide as you go (not blocking, but flag them)

- Exact churn-label definition on RetailRocket (no explicit churn flag in the raw data — you'll need to define "no activity in N days" as a proxy and state this assumption explicitly in `EVAL_RESULTS.md`).
- Whether "region" (e.g. "Kandy") is synthesized — none of the 4 public datasets have Sri Lankan regional data natively, so you'll likely need to inject synthetic region tags onto accounts during ETL to match the report's worked example. Do this deliberately in `etl_to_event_schema.py` and document it as a demo-data decision, not hide it.
- GPU access for CLIP/Whisper if going local — if you don't have a GPU, prefer the API-based fallbacks called out in Day 6 to avoid losing a day to inference speed.

---

**Bottom line sequencing:** Days 1–5 build the thing that actually gets scored on 85% of the rubric (Autonomous Reasoning + B2B Impact + Architecture + HITL). Days 6–7 are stretch polish and defense prep. If anything slips, protect Days 1–5 and cut from Day 6 first.
