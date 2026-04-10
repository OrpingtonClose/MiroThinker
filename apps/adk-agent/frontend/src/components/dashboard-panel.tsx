"use client";

import { useState, useEffect } from "react";
import {
  useDashboardSSE,
  type ConnectionStatus,
} from "@/hooks/useDashboardSSE";
import type {
  DashboardSnapshot,
  Phase,
  RawEvent,
  LLMCall,
  ToolCall,
  CorpusUpdate,
} from "@/types/dashboard";

// ── Helpers ──────────────────────────────────────────────────────
function fmt(n: number | undefined, decimals = 1): string {
  if (n === undefined || n === null) return "0";
  return n.toFixed(decimals);
}

function kFmt(n: number | undefined): string {
  if (!n) return "0";
  if (n >= 1000) return (n / 1000).toFixed(1) + "K";
  return String(n);
}

function elapsed(secs: number | undefined): string {
  if (!secs) return "0s";
  if (secs < 60) return secs.toFixed(1) + "s";
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function eventColor(type: string): string {
  if (type.includes("tool")) return "text-amber-400 bg-amber-500/10 border-amber-500/30";
  if (type.includes("llm") || type.includes("model")) return "text-blue-400 bg-blue-500/10 border-blue-500/30";
  if (type.includes("phase")) return "text-purple-400 bg-purple-500/10 border-purple-500/30";
  if (type.includes("corpus")) return "text-green-400 bg-green-500/10 border-green-500/30";
  if (type.includes("error") || type.includes("stall")) return "text-red-400 bg-red-500/10 border-red-500/30";
  return "text-slate-400 bg-slate-500/10 border-slate-500/30";
}

// ── Live Timer ───────────────────────────────────────────────────
function LiveTimer({ startedAt }: { startedAt?: number }) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 100);
    return () => clearInterval(id);
  }, []);
  if (!startedAt) return null;
  const secs = Math.max(0, now - startedAt);
  return <span className="tabular-nums font-mono">{elapsed(secs)}</span>;
}

// ── Connection Badge ─────────────────────────────────────────────
function ConnectionBadge({ status }: { status: ConnectionStatus }) {
  const styles: Record<ConnectionStatus, string> = {
    connected: "bg-green-500",
    connecting: "bg-yellow-500 animate-pulse",
    disconnected: "bg-red-500",
  };
  return (
    <div className="flex items-center gap-1.5 text-xs text-slate-400">
      <span className={`h-2 w-2 rounded-full ${styles[status]}`} />
      {status === "connected" ? "Live" : status}
    </div>
  );
}

// ── KPI Card (expandable) ────────────────────────────────────────
function KPICard({
  label,
  value,
  detail,
  color = "text-blue-400",
}: {
  label: string;
  value: string | number;
  detail?: React.ReactNode;
  color?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <button
      onClick={() => detail && setOpen(!open)}
      className={`text-left bg-slate-800/60 border border-slate-700/50 rounded-lg p-3 transition-all
        ${detail ? "cursor-pointer hover:border-slate-500/50 hover:bg-slate-800/80" : "cursor-default"}
        ${open ? "ring-1 ring-blue-500/30" : ""}`}
    >
      <div className="text-xs text-slate-500 mb-0.5">{label}</div>
      <div className={`text-xl font-bold tabular-nums ${color}`}>{value}</div>
      {open && detail && (
        <div className="mt-2 pt-2 border-t border-slate-700/50 text-xs text-slate-400">
          {detail}
        </div>
      )}
    </button>
  );
}

// ── Phase Row (expandable) ───────────────────────────────────────
function PhaseRow({ phase, startedAt }: { phase: Phase; startedAt?: number }) {
  const [open, setOpen] = useState(false);
  const statusColor =
    phase.outcome === "ok" || phase.outcome === "completed"
      ? "bg-green-500"
      : phase.outcome === "error"
      ? "bg-red-500"
      : "bg-blue-500 animate-pulse";

  return (
    <div
      onClick={() => setOpen(!open)}
      className="cursor-pointer bg-slate-800/40 border border-slate-700/40 rounded-lg p-2.5 hover:border-slate-600/50 transition-all"
    >
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full flex-shrink-0 ${statusColor}`} />
        <span className="text-sm font-medium text-slate-200 flex-1">
          {phase.agent}
        </span>
        <span className="text-xs text-slate-500">{phase.phase}</span>
        <span className="text-xs font-mono text-slate-400">
          {fmt(phase.elapsed)}s
        </span>
      </div>
      {open && (
        <div className="mt-2 pt-2 border-t border-slate-700/40 text-xs text-slate-500 space-y-1">
          <div>Phase: <span className="text-slate-300">{phase.phase}</span></div>
          <div>Agent: <span className="text-slate-300">{phase.agent}</span></div>
          <div>Duration: <span className="text-slate-300">{fmt(phase.elapsed, 2)}s</span></div>
          <div>Outcome: <span className={phase.outcome === "error" ? "text-red-400" : "text-green-400"}>{phase.outcome || "running"}</span></div>
        </div>
      )}
    </div>
  );
}

// ── Event Row (expandable) ───────────────────────────────────────
function EventRow({ event, startedAt }: { event: RawEvent; startedAt?: number }) {
  const [open, setOpen] = useState(false);
  const ts = startedAt ? event.timestamp - startedAt : event.timestamp;

  return (
    <div
      onClick={() => setOpen(!open)}
      className="cursor-pointer hover:bg-slate-800/40 rounded px-2 py-1 transition-colors"
    >
      <div className="flex items-center gap-2 text-xs">
        <span className="text-slate-600 font-mono w-14 text-right flex-shrink-0">
          +{fmt(ts)}s
        </span>
        <span className={`px-1.5 py-0.5 rounded border text-xs font-medium ${eventColor(event.event_type)}`}>
          {event.event_type}
        </span>
        <span className="text-slate-500 truncate flex-1">{event.agent}</span>
        <span className="text-slate-600 truncate max-w-32">{event.phase}</span>
      </div>
      {open && (
        <div className="ml-16 mt-1 mb-1 bg-slate-800/60 rounded p-2 text-xs">
          <pre className="text-slate-400 whitespace-pre-wrap break-all max-h-40 overflow-auto">
            {JSON.stringify(event.data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ── Tool Call Row (expandable) ───────────────────────────────────
function ToolRow({ tool }: { tool: ToolCall }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      onClick={() => setOpen(!open)}
      className="cursor-pointer hover:bg-slate-800/40 rounded px-2 py-1.5 transition-colors"
    >
      <div className="flex items-center gap-2 text-xs">
        <span className="text-amber-400 font-medium">{tool.tool_name}</span>
        <span className="text-slate-500">{tool.agent}</span>
        <span className="ml-auto text-slate-400 font-mono">{fmt(tool.duration_secs)}s</span>
        {tool.error && <span className="text-red-400">err</span>}
      </div>
      {open && (
        <div className="mt-1 bg-slate-800/60 rounded p-2 text-xs space-y-1">
          <div className="text-slate-500">Result: <span className="text-slate-300">{kFmt(tool.result_chars)} chars</span></div>
          {tool.was_compressed && <div className="text-yellow-500">Compressed</div>}
          {tool.error && <div className="text-red-400">Error: {tool.error}</div>}
        </div>
      )}
    </div>
  );
}

// ── LLM Call Row (expandable) ────────────────────────────────────
function LLMRow({ call }: { call: LLMCall }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      onClick={() => setOpen(!open)}
      className="cursor-pointer hover:bg-slate-800/40 rounded px-2 py-1.5 transition-colors"
    >
      <div className="flex items-center gap-2 text-xs">
        <span className="text-blue-400 font-medium">{call.agent}</span>
        <span className="ml-auto text-slate-400 font-mono">{fmt(call.duration_secs)}s</span>
        <span className="text-slate-500">{kFmt(call.completion_tokens_est)} tok</span>
      </div>
      {open && (
        <div className="mt-1 bg-slate-800/60 rounded p-2 text-xs space-y-1">
          <div className="text-slate-500">Duration: <span className="text-slate-300">{fmt(call.duration_secs, 2)}s</span></div>
          <div className="text-slate-500">Completion tokens: <span className="text-slate-300">{call.completion_tokens_est}</span></div>
        </div>
      )}
    </div>
  );
}

// ── Corpus Chart (expandable iterations) ─────────────────────────
function CorpusTimeline({ updates }: { updates: CorpusUpdate[] }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  if (!updates?.length) return null;

  const maxTotal = Math.max(...updates.map((u) => u.total), 1);

  return (
    <div className="space-y-1.5">
      {updates.map((u, i) => (
        <div key={i}>
          <div
            onClick={() => setOpenIdx(openIdx === i ? null : i)}
            className="cursor-pointer hover:bg-slate-800/40 rounded px-2 py-1 transition-colors flex items-center gap-2"
          >
            <span className="text-xs text-slate-500 w-10">Iter {u.iteration}</span>
            <div className="flex-1 bg-slate-800 rounded-full h-3 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-green-600 to-green-400 transition-all"
                style={{ width: `${(u.admitted / maxTotal) * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono text-green-400 w-12 text-right">{u.admitted}</span>
            <span className="text-xs text-slate-600">/</span>
            <span className="text-xs font-mono text-slate-400 w-12">{u.total}</span>
          </div>
          {openIdx === i && (
            <div className="ml-12 bg-slate-800/60 rounded p-2 text-xs space-y-1 mb-1">
              <div className="text-slate-500">Iteration: <span className="text-slate-300">{u.iteration}</span></div>
              <div className="text-slate-500">Admitted atoms: <span className="text-green-400">{u.admitted}</span></div>
              <div className="text-slate-500">Total candidates: <span className="text-slate-300">{u.total}</span></div>
              <div className="text-slate-500">Admission rate: <span className="text-blue-400">{((u.admitted / Math.max(u.total, 1)) * 100).toFixed(1)}%</span></div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Pending Operations ───────────────────────────────────────────
function PendingOps({ tools, llm }: { tools?: string[]; llm?: string[] }) {
  if (!tools?.length && !llm?.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {llm?.map((id, i) => (
        <span key={`l-${i}`} className="inline-flex items-center gap-1 text-xs bg-blue-500/10 border border-blue-500/30 text-blue-400 px-2 py-0.5 rounded-full">
          <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
          LLM {id.slice(0, 8)}
        </span>
      ))}
      {tools?.map((name, i) => (
        <span key={`t-${i}`} className="inline-flex items-center gap-1 text-xs bg-amber-500/10 border border-amber-500/30 text-amber-400 px-2 py-0.5 rounded-full">
          <span className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse" />
          {name}
        </span>
      ))}
    </div>
  );
}

// ── Collapsible Section ──────────────────────────────────────────
function Section({
  title,
  count,
  defaultOpen = true,
  children,
}: {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-slate-900/60 border border-slate-800/60 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-slate-800/30 transition-colors"
      >
        <span className="text-sm font-semibold text-slate-300">{title}</span>
        <div className="flex items-center gap-2">
          {count !== undefined && (
            <span className="text-xs bg-slate-700/60 text-slate-400 px-1.5 py-0.5 rounded">
              {count}
            </span>
          )}
          <span className="text-slate-500 text-xs">{open ? "collapse" : "expand"}</span>
        </div>
      </button>
      {open && <div className="px-4 pb-3">{children}</div>}
    </div>
  );
}

// ── Main Dashboard Panel ─────────────────────────────────────────
export function DashboardPanel() {
  const { snapshot: snap, status, reconnect } = useDashboardSSE();

  const isIdle = !snap || snap.status === "idle";
  const isRunning = snap && !isIdle && !snap.finalized;

  return (
    <div className="h-full overflow-y-auto space-y-4 p-4 bg-slate-950">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-slate-200">Pipeline Dashboard</h2>
        <div className="flex items-center gap-3">
          <ConnectionBadge status={status} />
          <button
            onClick={reconnect}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            reconnect
          </button>
        </div>
      </div>

      {/* Idle state */}
      {isIdle && (
        <div className="flex flex-col items-center justify-center py-16 text-slate-600">
          <div className="text-4xl mb-3">brain</div>
          <div className="text-sm font-medium">No active pipeline run</div>
          <div className="text-xs mt-1">
            Send a research question via the chat to start
          </div>
        </div>
      )}

      {/* Active run */}
      {snap && !isIdle && (
        <>
          {/* Query banner */}
          <div className="bg-slate-800/50 rounded-xl p-3 border-l-4 border-indigo-500">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="text-xs text-slate-500 mb-1">
                  Session {snap.session_id?.slice(0, 8)}
                  {" — "}
                  {isRunning ? (
                    <span className="text-green-400 font-medium">Running</span>
                  ) : snap.finalized ? (
                    <span className="text-blue-400 font-medium">Completed</span>
                  ) : (
                    "Unknown"
                  )}
                </div>
                <div className="text-sm text-slate-300 italic truncate">
                  {snap.query}
                </div>
              </div>
              <div className="text-right flex-shrink-0">
                <div className="text-2xl font-bold text-indigo-400">
                  <LiveTimer startedAt={snap.started_at} />
                </div>
                <div className="text-xs text-slate-500">elapsed</div>
              </div>
            </div>
          </div>

          {/* Rate metrics bar */}
          {(snap.llm_per_min || snap.tools_per_min || snap.secs_since_activity !== undefined) ? (
            <div className="flex gap-3 text-xs">
              {snap.llm_per_min !== undefined && (
                <div className="bg-blue-500/10 border border-blue-500/20 text-blue-400 px-2.5 py-1 rounded-lg">
                  {fmt(snap.llm_per_min)} LLM/min
                </div>
              )}
              {snap.tools_per_min !== undefined && (
                <div className="bg-amber-500/10 border border-amber-500/20 text-amber-400 px-2.5 py-1 rounded-lg">
                  {fmt(snap.tools_per_min)} tools/min
                </div>
              )}
              {snap.secs_since_activity !== undefined && (
                <div className="bg-slate-700/50 border border-slate-600/30 text-slate-400 px-2.5 py-1 rounded-lg">
                  {fmt(snap.secs_since_activity)}s since activity
                </div>
              )}
            </div>
          ) : null}

          {/* Alerts */}
          {snap.stalled && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-sm px-3 py-2 rounded-lg">
              Pipeline stalled — no events received recently
            </div>
          )}
          {snap.thinker_escalated && (
            <div className="bg-green-500/10 border border-green-500/30 text-green-400 text-sm px-3 py-2 rounded-lg">
              Thinker signalled EVIDENCE_SUFFICIENT — loop will exit
            </div>
          )}

          {/* Pending operations */}
          <PendingOps tools={snap.pending_tools} llm={snap.pending_llm} />

          {/* KPI Grid */}
          <div className="grid grid-cols-2 gap-2">
            <KPICard
              label="Tool Calls"
              value={snap.kpi?.tool_calls || 0}
              color="text-amber-400"
              detail={
                snap.kpi?.tool_errors ? (
                  <span className="text-red-400">{snap.kpi.tool_errors} errors</span>
                ) : undefined
              }
            />
            <KPICard
              label="LLM Calls"
              value={snap.kpi?.llm_calls || 0}
              color="text-blue-400"
              detail={
                <div>
                  <div>Prompt: {kFmt(snap.kpi?.prompt_tokens_est)} tok</div>
                  <div>Completion: {kFmt(snap.kpi?.completion_tokens_est)} tok</div>
                </div>
              }
            />
            <KPICard
              label="Corpus Atoms"
              value={snap.kpi?.corpus_atoms || 0}
              color="text-green-400"
              detail={snap.kpi?.dedup_blocks ? <span>{snap.kpi.dedup_blocks} dedup blocks</span> : undefined}
            />
            <KPICard
              label="Output"
              value={kFmt(snap.kpi?.text_chars)}
              color="text-purple-400"
              detail={
                snap.kpi?.reasoning_chars ? (
                  <span>Reasoning: {kFmt(snap.kpi.reasoning_chars)} chars</span>
                ) : undefined
              }
            />
          </div>

          {/* Phases */}
          {snap.phases && snap.phases.length > 0 && (
            <Section title="Phases" count={snap.phases.length}>
              <div className="space-y-1.5">
                {snap.phases.map((p, i) => (
                  <PhaseRow key={i} phase={p} startedAt={snap.started_at} />
                ))}
              </div>
            </Section>
          )}

          {/* Corpus Growth */}
          {snap.corpus_history && snap.corpus_history.length > 0 && (
            <Section title="Corpus Growth" count={snap.corpus_history.length}>
              <CorpusTimeline updates={snap.corpus_history} />
            </Section>
          )}

          {/* Recent Tool Calls */}
          {snap.recent_tools && snap.recent_tools.length > 0 && (
            <Section title="Recent Tools" count={snap.recent_tools.length}>
              <div className="space-y-0.5">
                {snap.recent_tools.map((t, i) => (
                  <ToolRow key={i} tool={t} />
                ))}
              </div>
            </Section>
          )}

          {/* Recent LLM Calls */}
          {snap.recent_llm && snap.recent_llm.length > 0 && (
            <Section title="Recent LLM Calls" count={snap.recent_llm.length}>
              <div className="space-y-0.5">
                {snap.recent_llm.map((c, i) => (
                  <LLMRow key={i} call={c} />
                ))}
              </div>
            </Section>
          )}

          {/* Activity Feed */}
          {snap.recent_events && snap.recent_events.length > 0 && (
            <Section title="Activity Feed" count={snap.recent_events.length} defaultOpen={false}>
              <div className="space-y-0.5 max-h-80 overflow-y-auto">
                {[...snap.recent_events].reverse().map((e, i) => (
                  <EventRow key={i} event={e} startedAt={snap.started_at} />
                ))}
              </div>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
