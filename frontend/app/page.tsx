"use client";

import { useEffect, useRef, useState } from "react";
import { TrailTimeline } from "@/components/TrailTimeline";
import { AGENT_HTTP, agentWsUrl, type Proposal, type TrailEntry } from "@/lib/api";

const EXAMPLES = [
  "Draft reorder proposals for slow-moving SKUs in the Kandy region",
  "Which accounts in Colombo are at highest churn risk?",
  "Sales look weak",
  "Find a substitute SKU similar to SKU-POP-000 for reorder",
];

export default function HomePage() {
  const [goal, setGoal] = useState(EXAMPLES[0]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [trail, setTrail] = useState<TrailEntry[]>([]);
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [clarify, setClarify] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<string>("checking…");
  const wsRef = useRef<WebSocket | null>(null);

  const [listening, setListening] = useState(false);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    fetch(`${AGENT_HTTP}/health`)
      .then((r) => r.json())
      .then((d) =>
        setHealth(
          d.ready
            ? `agent-core ready (${d.llm_active || d.llm_provider || "llm"})`
            : `agent-core not ready: ${d.llm_error || "unknown"}`
        )
      )
      .catch(() => setHealth("agent-core unreachable — start it on :8010"));
  }, []);

  function toggleMic() {
    const SR =
      typeof window !== "undefined" &&
      ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition);
    if (!SR) {
      setError("Web Speech API not supported in this browser. Type the goal instead.");
      return;
    }
    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }
    const rec = new SR();
    recognitionRef.current = rec;
    rec.lang = "en-US";
    rec.interimResults = true;
    rec.onresult = (ev: any) => {
      let text = "";
      for (let i = 0; i < ev.results.length; i++) {
        text += ev.results[i][0].transcript;
      }
      setGoal(text.trim());
    };
    rec.onerror = () => setListening(false);
    rec.onend = () => setListening(false);
    rec.start();
    setListening(true);
  }

  function runAgent() {
    if (!goal.trim() || running) return;
    setRunning(true);
    setError(null);
    setTrail([]);
    setProposal(null);
    setClarify(null);
    setRunId(null);
    setStatus("Connecting…");

    const ws = new WebSocket(agentWsUrl("/ws/run"));
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("Running agent…");
      ws.send(JSON.stringify({ goal }));
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "status") setStatus(msg.message);
      if (msg.type === "trail") {
        setTrail((prev) => [...prev, msg.entry]);
      }
      if (msg.type === "done") {
        setRunId(msg.run_id || null);
        setClarify(msg.clarifying_question || null);
        setProposal(msg.proposal || null);
        setStatus("Complete");
        setRunning(false);
      }
      if (msg.type === "error") {
        setError(msg.message);
        setRunning(false);
        setStatus("Error");
      }
    };
    ws.onerror = () => {
      setError("WebSocket error — is agent-core running? (try :8010)");
      setRunning(false);
    };
    ws.onclose = () => {
      setRunning(false);
    };
  }

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <p className="text-xs uppercase tracking-[0.2em] text-emerald-500/80">Lankara · NeuroX</p>
        <h1 className="text-3xl font-semibold tracking-tight">Run the retail agent</h1>
        <p className="max-w-2xl text-sm text-slate-400">
          Type a B2B goal. Watch PERCEIVE → PLAN → REASON → PROPOSE stream in. Approvals happen in the queue.
        </p>
        <p className="text-xs text-slate-500">{health}</p>
      </section>

      <section className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <label className="text-sm text-slate-300">Goal</label>
        <textarea
          className="w-full rounded-xl border border-slate-700 bg-slate-950 p-4 text-sm outline-none ring-emerald-500/40 focus:ring-2"
          rows={4}
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          placeholder='e.g. "Draft reorder proposals for slow-moving SKUs in Kandy"'
        />
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              onClick={() => setGoal(ex)}
              className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400 hover:border-emerald-600 hover:text-emerald-300"
            >
              {ex.slice(0, 42)}…
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={runAgent}
            disabled={running || !goal.trim()}
            className="rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-40"
          >
            {running ? "Running…" : "Run Agent"}
          </button>
          <button
            type="button"
            onClick={toggleMic}
            className={`rounded-xl border px-4 py-2.5 text-sm ${
              listening
                ? "border-rose-500 bg-rose-500/20 text-rose-200"
                : "border-slate-600 text-slate-300"
            }`}
          >
            {listening ? "Stop mic" : "Mic (Web Speech)"}
          </button>
          <label className="cursor-pointer rounded-xl border border-slate-600 px-4 py-2.5 text-sm text-slate-300 hover:border-emerald-600">
            Upload audio → Whisper
            <input
              type="file"
              accept="audio/*,.webm,.wav,.mp3,.m4a"
              className="hidden"
              onChange={async (e) => {
                const file = e.target.files?.[0];
                if (!file) return;
                setStatus("Transcribing with Whisper…");
                setError(null);
                try {
                  const fd = new FormData();
                  fd.append("audio", file);
                  const sttBase =
                    process.env.NEXT_PUBLIC_STT_URL || "http://127.0.0.1:8005";
                  const r = await fetch(`${sttBase}/transcribe`, {
                    method: "POST",
                    body: fd,
                  });
                  const j = await r.json();
                  const text = j?.data?.text || "";
                  if (text) setGoal(text);
                  else setError(j?.error_reason || "Whisper returned empty text");
                  setStatus(text ? `STT ok (${j?.data?.engine || "whisper"})` : "STT failed");
                } catch (err: any) {
                  setError(String(err?.message || err));
                  setStatus("STT error");
                }
              }}
            />
          </label>
        </div>
        {status ? <p className="text-xs text-slate-400">{status}</p> : null}
        {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-medium">Decision trail</h2>
        <TrailTimeline entries={trail} />
      </section>

      {clarify ? (
        <section className="rounded-xl border border-amber-600/40 bg-amber-500/10 p-4 text-sm text-amber-100">
          <p className="font-medium">Clarifying question</p>
          <p className="mt-1 text-amber-50/90">{clarify}</p>
        </section>
      ) : null}

      {proposal ? (
        <section className="rounded-xl border border-emerald-700/40 bg-emerald-500/10 p-4 text-sm">
          <p className="font-medium text-emerald-200">Proposal drafted</p>
          <p className="mt-1 text-slate-300">
            {proposal.action_type} · conf {(proposal.confidence * 100).toFixed(0)}% ·{" "}
            <a className="text-emerald-400 underline" href="/queue">
              open approval queue
            </a>
            {runId ? (
              <>
                {" · "}
                <a className="text-emerald-400 underline" href={`/history/${runId}`}>
                  replay trail
                </a>
              </>
            ) : null}
          </p>
        </section>
      ) : null}
    </div>
  );
}
