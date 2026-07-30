"use client";

import type { TrailEntry } from "@/lib/api";

const STEP_COLOR: Record<string, string> = {
  PERCEIVE: "border-sky-500/60 bg-sky-500/10",
  PLAN: "border-amber-500/60 bg-amber-500/10",
  REASON: "border-violet-500/40 bg-violet-500/10",
  PROPOSE: "border-emerald-500/60 bg-emerald-500/10",
};

export function TrailTimeline({ entries }: { entries: TrailEntry[] }) {
  if (!entries.length) {
    return <p className="text-sm text-slate-500">No trail steps yet.</p>;
  }
  return (
    <ol className="space-y-3">
      {entries.map((e, i) => (
        <li
          key={`${e.step}-${i}-${e.timestamp || i}`}
          className={`rounded-xl border px-4 py-3 transition ${STEP_COLOR[e.step] || "border-slate-700 bg-slate-900"}`}
        >
          <div className="mb-1 flex items-center justify-between gap-3">
            <span className="text-xs font-semibold tracking-wider text-slate-200">
              {e.step}
            </span>
            {e.timestamp ? (
              <span className="text-[11px] text-slate-500">
                {new Date(e.timestamp).toLocaleTimeString()}
              </span>
            ) : null}
          </div>
          <p className="text-sm leading-relaxed text-slate-300">{e.content}</p>
          {e.tool_calls && e.tool_calls.length > 0 ? (
            <p className="mt-2 text-xs text-slate-500">
              tools: {e.tool_calls.map((t) => t.tool).filter(Boolean).join(", ")}
            </p>
          ) : null}
        </li>
      ))}
    </ol>
  );
}
