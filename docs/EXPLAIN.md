# EXPLAIN — Judge Q&A (The Core Test)

Direct answers with file pointers. Rehearse aloud using a real `run_id` from the Approval Queue “view trail” link.

---

## 1. What makes it autonomous?

The agent **decides which tools to call at runtime** via an LLM in the PLAN node — there is no fixed `if "churn" in goal: call_churn()` pipeline.

- **Proof:** `agent-core/app/nodes/plan.py` — prefers LangChain **`bind_tools` / native tool-calling** (`emit_plan`) when the provider supports it (Anthropic / OpenAI / AgentRouter); otherwise JSON-schema prompting. If the model returns empty tools while claiming non-ambiguous, it is **re-prompted once**; still empty → clarifying question (never a keyword-built tool list).
- **Provider:** `agent-core/app/llm.py` — AgentRouter (Claude/GPT) with automatic Gemini Vertex fallback.
- **Guardrail (not a pipeline):** if a Sri Lankan region is already named, we refuse to stall on clarifying questions; **tool choice remains LLM-driven**.

---

## 2. How does the agent loop work?

Explicit LangGraph stages:

`PERCEIVE → PLAN → REASON → PROPOSE`

| Step | File | Role |
|------|------|------|
| PERCEIVE | `agent-core/app/nodes/perceive.py` | Fetch event-store slice |
| PLAN | `agent-core/app/nodes/plan.py` | Ambiguity + tool selection |
| REASON | `agent-core/app/nodes/reason.py` | Execute tools, confidence |
| PROPOSE | `agent-core/app/nodes/propose.py` | Draft Proposal 4.3 |

Graph wiring: `agent-core/app/graph.py` → `run_agent()`.

**Replay:** every step is persisted (`app/trail.py`) and streamed on `ws://…/ws/run`. Open `/history/{run_id}` or `GET /runs/{run_id}/trail`.

---

## 3. How does it choose tools?

Show two goals side by side (Day 4 suite / live UI):

| Goal | Typical tools |
|------|----------------|
| “Draft reorder… in Colombo” | `forecasting` (+ optional basket/recommend) |
| “Highest churn risk in Kandy” | `churn_scoring` |
| “Find substitute SKU similar to …” | `visual_search` and/or `recommend` |
| “Sales look weak” | clarifying question (no tools yet) |

Manifest: `agent-core/app/tools/registry.py`.

---

## 4. How does it recover from failures?

Every HTTP tool call goes through `agent-core/app/tools/base.py` → `call_tool()`:

1. try  
2. retry  
3. coarser fallback payload  
4. `status=degraded` + lower confidence  
5. `BlockedProposal` only if marked essential and still dead  

**Demo:** `python -m app.run_goal --suite --simulate-failure`  
Trail shows `forecasting=degraded` / `churn_scoring=degraded` and a **lower-confidence proposal still ships** — no silent failure.

---

## Human-in-the-loop

Not a cosmetic confirm: `/queue` supports **Approve** (simulated execution log), **Edit** (payload rewrite + `reviewer_feedback`), **Reject** (reason stored). Feedback is prepended into future `propose()` prompts (`recent_feedback()`).

---

## Data honesty

- Regions Colombo/Kandy/Galle/Jaffna are **synthesized in ETL**.
- Churn label = **14-day inactivity** proxy.
- Demo parquet / models are trained artifacts — live inference hits event-store or offline parquet, not a canned JSON answer blob.
