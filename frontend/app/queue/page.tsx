"use client";

import { useCallback, useEffect, useState } from "react";
import { ProposalCard } from "@/components/ProposalCard";
import { fetchProposals, type Proposal } from "@/lib/api";

export default function QueuePage() {
  const [status, setStatus] = useState("pending");
  const [items, setItems] = useState<Proposal[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setItems(await fetchProposals(status));
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [status]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Approval Queue</h1>
          <p className="mt-1 text-sm text-slate-400">
            Human-in-the-loop review. Approve is a simulated execution log.
          </p>
        </div>
        <div className="flex gap-2">
          {["pending", "edited", "executed", "rejected", "all"].map((s) => (
            <button
              key={s}
              onClick={() => setStatus(s)}
              className={`rounded-lg px-3 py-1.5 text-xs ${
                status === s ? "bg-emerald-600 text-white" : "border border-slate-700 text-slate-400"
              }`}
            >
              {s}
            </button>
          ))}
          <button
            onClick={load}
            className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300"
          >
            Refresh
          </button>
        </div>
      </div>

      {loading ? <p className="text-sm text-slate-500">Loading…</p> : null}
      {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      {!loading && !items.length ? (
        <p className="text-sm text-slate-500">
          No proposals. Run a goal on the home page first.
        </p>
      ) : null}

      <div className="space-y-4">
        {items.map((p) => (
          <ProposalCard key={p.proposal_id} proposal={p} onChanged={load} />
        ))}
      </div>
    </div>
  );
}
