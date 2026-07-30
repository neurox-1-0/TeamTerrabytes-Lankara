# Architecture

## System diagram

```
┌─────────────────────────────────────────────────────────────┐
│  FRONTEND (Next.js)                                          │
│  Goal + mic → live trail (WS) → Approval Queue → history     │
└───────────────────────────┬───────────────────────────────────┘
                             │ REST + WebSocket
┌───────────────────────────▼───────────────────────────────────┐
│  AGENT-CORE (FastAPI + LangGraph)                             │
│  PERCEIVE → PLAN → REASON → PROPOSE                           │
│  retry/fallback/degrade/block · decision_trail · HITL queue   │
└──┬────────────┬────────────┬────────────┬────────────┬────────┘
   │ HTTP        │ HTTP       │ HTTP       │ HTTP       │ HTTP
┌──▼───┐   ┌─────▼────┐  ┌────▼─────┐ ┌────▼──────┐ ┌───▼────────┐
│Event │   │Analytics │  │Recommender│ │  Visual   │ │STT+Transl. │
│Store │   │(4 models)│  │  hybrid   │ │  Search   │ │            │
└──────┘   └──────────┘  └───────────┘ └───────────┘ └────────────┘
```

## Why these choices

| Choice | Rationale |
|--------|-----------|
| **LangGraph** | Explicit PERCEIVE→PLAN→REASON→PROPOSE stages judges can walk; tool choice lives in PLAN, not hidden if/else. |
| **FastAPI microservices** | Each tool is an independent HTTP service — matches Phase 1 “pipelines API / recommender service” story; one failure cannot import-crash the orchestrator. |
| **Postgres (+ SQLite fallback)** | Event store, proposals, and decision trails are queryable for live demos and judge replay. SQLite keeps Day 4–5 demos alive when Docker is down. |
| **LLM via AgentRouter + Gemini fallback** | AgentRouter is OpenAI-compatible (`https://agentrouter.org/v1`) for Claude/GPT; Vertex Gemini SA covers live demos if the gateway 401s. |
| **Tool envelope 4.2** | Uniform `{tool,status,confidence,data,data_slice,error_reason}` so REASON/PROPOSE never special-case each service. |
| **HITL Approval Queue** | Approve = *simulated* execution row (explicit to judges). Edit/reject write `reviewer_feedback` for compounding accuracy. |
| **Visual search (Day 6)** | **CLIP (`clip-ViT-B-32`) + Chroma** over catalog images (H&M when present, else generated placeholders). TF-IDF fallback if CLIP unavailable. |
| **STT (Day 6)** | OpenAI Whisper API → local **faster-whisper**; browser Web Speech on `/` as UX fallback. |
| **Translate (Day 6)** | **NLLB-200-distilled** → Helsinki OPUS-MT → LLM gateway → heuristic. |
| **Recommender** | **Implicit ALS** (`implicit` library / NumPy ALS) + content blend. |

## Ports

| Service | Port |
|---------|------|
| agent-core | 8000 (local conflict → 8010) |
| event-store | 8001 |
| analytics-pipeline | 8002 |
| recommender | 8003 |
| visual-search | 8004 |
| stt-translation | 8005 |
| frontend | 3000 (local conflict → 3005) |

## Key files

- Contracts: `shared/contracts.py`
- Graph: `agent-core/app/graph.py`
- Tools ladder: `agent-core/app/tools/base.py`
- HITL: `agent-core/app/approval_queue.py`
