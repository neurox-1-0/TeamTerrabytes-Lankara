"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Proposal } from "@/lib/api";
import { approveProposal, editProposal, rejectProposal } from "@/lib/api";

function confidenceBadge(c: number) {
  const pct = Math.round(c * 100);
  const color =
    pct >= 70 ? "bg-emerald-500/20 text-emerald-300" : pct >= 40 ? "bg-amber-500/20 text-amber-300" : "bg-rose-500/20 text-rose-300";
  return (
    <span className={`rounded-md px-2 py-0.5 text-xs font-medium ${color}`}>
      {pct}% conf
    </span>
  );
}

function EvidencePanel({ evidence }: { evidence?: unknown[] }) {
  const chartData = useMemo(() => {
    if (!Array.isArray(evidence) || !evidence.length) return [];
    const first = evidence[0] as { data?: unknown; tool?: string };
    const data = first?.data;
    if (Array.isArray(data)) {
      return data.slice(0, 8).map((row: any, i: number) => ({
        name: String(row.account_id || row.sku || row.segment || `#${i + 1}`).slice(0, 12),
        value: Number(
          row.churn_probability ?? row.blended_score ?? row.lift ?? row.rfm?.monetary ?? 0
        ),
      })).filter((d) => !Number.isNaN(d.value) && d.value !== 0);
    }
    if (data && typeof data === "object" && Array.isArray((data as any).series)) {
      const series = (data as any).series[0]?.forecast || [];
      return series.slice(0, 10).map((p: any) => ({
        name: String(p.date || "").slice(5),
        value: Number(p.qty || 0),
      }));
    }
    return [];
  }, [evidence]);

  if (!evidence?.length) return <p className="text-xs text-slate-500">No evidence attached.</p>;

  return (
    <div className="space-y-3">
      {chartData.length > 0 ? (
        <div className="h-40 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} />
              <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} width={32} />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
              />
              <Bar dataKey="value" fill="#34d399" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : null}
      <pre className="max-h-40 overflow-auto rounded-lg bg-slate-950/80 p-3 text-[11px] text-slate-400">
        {JSON.stringify(evidence, null, 2).slice(0, 2500)}
      </pre>
    </div>
  );
}

export function ProposalCard({
  proposal,
  onChanged,
}: {
  proposal: Proposal;
  onChanged?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [editText, setEditText] = useState(JSON.stringify(proposal.payload || {}, null, 2));
  const [error, setError] = useState<string | null>(null);

  async function act(kind: "approve" | "reject" | "edit") {
    setBusy(true);
    setError(null);
    try {
      if (kind === "approve") await approveProposal(proposal.proposal_id);
      if (kind === "reject") {
        const reason = window.prompt("Rejection reason?", "Not actionable") || "rejected";
        await rejectProposal(proposal.proposal_id, reason);
      }
      if (kind === "edit") {
        const payload = JSON.parse(editText);
        await editProposal(proposal.proposal_id, payload, "edited in UI");
      }
      onChanged?.();
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-500">{proposal.action_type}</p>
          <h3 className="mt-1 text-base font-semibold text-slate-100">{proposal.goal}</h3>
          <p className="mt-1 text-xs text-slate-500">
            {proposal.proposal_id.slice(0, 8)}… · {proposal.status}
            {proposal.run_id ? (
              <>
                {" · "}
                <a className="text-emerald-400 hover:underline" href={`/history/${proposal.run_id}`}>
                  view trail
                </a>
              </>
            ) : null}
          </p>
        </div>
        {confidenceBadge(proposal.confidence || 0)}
      </div>

      <div className="mt-4 grid gap-2 text-sm text-slate-300">
        <p>
          <span className="text-slate-500">Target: </span>
          {JSON.stringify(proposal.target)}
        </p>
        <p className="line-clamp-3">
          <span className="text-slate-500">Payload: </span>
          {JSON.stringify(proposal.payload)}
        </p>
      </div>

      <button
        className="mt-3 text-xs text-emerald-400 hover:underline"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide evidence" : "Show evidence"}
      </button>
      {open ? (
        <div className="mt-3 space-y-3 border-t border-slate-800 pt-3">
          <EvidencePanel evidence={proposal.evidence} />
          <label className="block text-xs text-slate-500">Edit payload (JSON)</label>
          <textarea
            className="h-28 w-full rounded-lg border border-slate-700 bg-slate-950 p-2 font-mono text-xs"
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
          />
        </div>
      ) : null}

      {error ? <p className="mt-2 text-xs text-rose-400">{error}</p> : null}

      <div className="mt-4 flex flex-wrap gap-2">
        <button
          disabled={busy || proposal.status === "executed"}
          onClick={() => act("approve")}
          className="rounded-lg bg-emerald-600 px-3 py-1.5 text-sm font-medium disabled:opacity-40"
        >
          Approve
        </button>
        <button
          disabled={busy}
          onClick={() => act("edit")}
          className="rounded-lg border border-slate-600 px-3 py-1.5 text-sm disabled:opacity-40"
        >
          Save edit
        </button>
        <button
          disabled={busy || proposal.status === "rejected"}
          onClick={() => act("reject")}
          className="rounded-lg border border-rose-700/60 px-3 py-1.5 text-sm text-rose-300 disabled:opacity-40"
        >
          Reject
        </button>
      </div>
    </article>
  );
}
