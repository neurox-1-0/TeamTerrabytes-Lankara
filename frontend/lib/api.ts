export const AGENT_HTTP =
  process.env.NEXT_PUBLIC_AGENT_URL || "http://127.0.0.1:8000";

export function agentWsUrl(path: string = "/ws/run") {
  const http = AGENT_HTTP.replace(/\/$/, "");
  if (http.startsWith("https://")) return http.replace("https://", "wss://") + path;
  return http.replace("http://", "ws://") + path;
}

export type TrailEntry = {
  run_id: string;
  step: string;
  content: string;
  tool_calls?: Array<{ tool?: string; input?: unknown; output?: unknown }>;
  timestamp?: string;
};

export type Proposal = {
  proposal_id: string;
  run_id?: string;
  goal: string;
  action_type: string;
  target: Record<string, unknown>;
  payload: Record<string, unknown>;
  evidence?: unknown[];
  assumptions?: string[];
  confidence: number;
  status: string;
  created_at?: string;
};

export async function fetchProposals(status = "pending"): Promise<Proposal[]> {
  const r = await fetch(`${AGENT_HTTP}/proposals?status=${encodeURIComponent(status)}`, {
    cache: "no-store",
  });
  if (!r.ok) throw new Error(`Failed to load proposals (${r.status})`);
  const data = await r.json();
  return data.proposals || [];
}

export async function approveProposal(id: string) {
  const r = await fetch(`${AGENT_HTTP}/proposals/${id}/approve`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function rejectProposal(id: string, reason: string) {
  const r = await fetch(`${AGENT_HTTP}/proposals/${id}/reject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function editProposal(
  id: string,
  payload: Record<string, unknown>,
  reason?: string
) {
  const r = await fetch(`${AGENT_HTTP}/proposals/${id}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ payload, reason }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function fetchTrail(runId: string): Promise<TrailEntry[]> {
  const r = await fetch(`${AGENT_HTTP}/runs/${runId}/trail`, { cache: "no-store" });
  if (!r.ok) throw new Error(`Trail not found (${r.status})`);
  const data = await r.json();
  return data.trail || [];
}
