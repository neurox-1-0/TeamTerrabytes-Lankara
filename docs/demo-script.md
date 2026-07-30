# Live Demo Script

Rehearse at least 3 times. Have this doc + Approval Queue open.

## Pre-flight (2 min)

```bash
# Prefer docker compose up — or local ports:
# analytics :8002, recommender :8003, visual :8004, stt :8005
# agent-core :8010, frontend :3005
curl http://127.0.0.1:8010/health
open http://127.0.0.1:3005
```

Confirm health shows `ready: true` (Gemini fallback is OK if AgentRouter 401s).

## Scripted goals (feel unscripted)

### 1) Reorder — exercises forecasting
**Say:** “Let’s see what the agent does for inventory in Kandy.”  
**Type:** `Draft reorder proposals for slow-moving SKUs in the Kandy region`  
**Expect:** PLAN → `forecasting` (maybe basket/recommend). Trail streams. Proposal `reorder` lands in queue.  
**Narrate:** “Tool order wasn’t hardcoded — PLAN asked the LLM.”

### 2) Churn — different tool sequence
**Type:** `Which accounts in Colombo are at highest churn risk?`  
**Expect:** `churn_scoring` (not the same as #1). Open evidence chart.  
**Approve** one proposal — explain execution is **simulated** (log row), as expected by rubric.

### 3) Ambiguity
**Type:** `Sales look weak`  
**Expect:** clarifying question, no silent guess.  
**Say:** “That’s ask vs scope-and-state from the Phase 1 design.”

### 4) Optional stretch (if 8004 up)
**Type:** `Find a substitute SKU similar to SKU-POP-000 for reorder`  
**Expect:** sometimes `visual_search` / `recommend` — prove optional tool use.

## Failure talking point (if asked)

Run beforehand: `python -m app.run_goal --suite --simulate-failure`  
Show degraded confidence in trail — recovery ladder in `tools/base.py`.

## Fallback if live LLM/network dies

1. Screen-recorded backup of a clean run (record during rehearsal).  
2. Offline: `python -m app.run_goal --suite` still hits local analytics models + SQLite trails.  
3. Queue still shows prior proposals from DB.

## Do not

- Pretend cached JSON is a live DB hit.  
- Hide that regions are ETL-synthesized.  
- Claim Approve sends a real email/price change.
