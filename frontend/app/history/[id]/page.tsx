"use client";

import { useEffect, useState } from "react";
import { TrailTimeline } from "@/components/TrailTimeline";
import { fetchTrail, type TrailEntry } from "@/lib/api";

export default function HistoryPage({ params }: { params: { id: string } }) {
  const [trail, setTrail] = useState<TrailEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTrail(params.id)
      .then(setTrail)
      .catch((e) => setError(e?.message || String(e)));
  }, [params.id]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Run history</h1>
        <p className="mt-1 text-sm text-slate-400">Decision trail replay for run {params.id}</p>
      </div>
      {error ? <p className="text-sm text-rose-400">{error}</p> : null}
      <TrailTimeline entries={trail} />
    </div>
  );
}
