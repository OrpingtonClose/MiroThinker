import { useEffect, useState, useRef } from "react";
import {
  Activity,
  Cpu,
  Database,
  Clock,
  Wrench,
  Brain,
  AlertTriangle,
  CheckCircle,
  Wifi,
  WifiOff,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Zap,
  FileText,
  Search,
  History,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts";
import "./App.css";
import { PipelineGraph } from "./components/PipelineGraph";
import { useSSE, fetchRuns, fetchRunDetail } from "./hooks/useSSE";
import type {
  ConnectionStatus,
} from "./hooks/useSSE";
import type {
  RunListItem,
  RunDetail,
  ToolCall,
  Phase,
  RawEvent,
} from "./types";

// ── Color palette ────────────────────────────────────────────────
const COLORS = {
  accent: "#58a6ff",
  success: "#3fb950",
  warning: "#d29922",
  danger: "#f85149",
  info: "#79c0ff",
  purple: "#a371f7",
  muted: "#8b949e",
};

const PIE_COLORS = [
  "#58a6ff",
  "#3fb950",
  "#d29922",
  "#f85149",
  "#a371f7",
  "#79c0ff",
  "#f0883e",
  "#db61a2",
  "#7ee787",
  "#56d4dd",
];

// ── Status badge ─────────────────────────────────────────────────
function StatusBadge({ status }: { status: ConnectionStatus }) {
  const config = {
    connected: {
      icon: <Wifi size={14} />,
      text: "Live",
      cls: "bg-green-500/20 text-green-400 border-green-500/30",
    },
    connecting: {
      icon: <RefreshCw size={14} className="animate-spin" />,
      text: "Connecting",
      cls: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
    },
    disconnected: {
      icon: <WifiOff size={14} />,
      text: "Disconnected",
      cls: "bg-red-500/20 text-red-400 border-red-500/30",
    },
  };
  const c = config[status];
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${c.cls}`}
    >
      {c.icon} {c.text}
    </span>
  );
}

// ── KPI Card ─────────────────────────────────────────────────────
function KPICard({
  label,
  value,
  icon,
  color = COLORS.accent,
  subtitle,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
  subtitle?: string;
}) {
  return (
    <div className="bg-gray-800/50 border border-gray-700/50 rounded-xl p-4 hover:border-gray-600/50 transition-colors">
      <div className="flex items-center gap-2 mb-2">
        <span style={{ color }}>{icon}</span>
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
          {label}
        </span>
      </div>
      <div className="text-2xl font-bold" style={{ color }}>
        {value}
      </div>
      {subtitle && (
        <div className="text-xs text-gray-500 mt-1">{subtitle}</div>
      )}
    </div>
  );
}

// ── Phase timeline ───────────────────────────────────────────────
function PhaseTimeline({ phases }: { phases: Phase[] }) {
  if (!phases.length) return null;
  const total = phases.reduce((s, p) => s + p.elapsed, 0) || 1;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <Clock size={16} className="text-blue-400" /> Pipeline Phases
      </h3>
      <div className="flex rounded-lg overflow-hidden h-8">
        {phases.map((p, i) => {
          const pct = Math.max((p.elapsed / total) * 100, 2);
          const bg =
            p.outcome === "ok"
              ? "bg-green-500/60"
              : p.outcome === "error"
              ? "bg-red-500/60"
              : p.outcome === "disconnected"
              ? "bg-yellow-500/60"
              : "bg-blue-500/40";
          return (
            <div
              key={i}
              className={`${bg} flex items-center justify-center text-xs font-medium text-white/90 border-r border-gray-900/30 transition-all hover:brightness-125 cursor-default`}
              style={{ width: `${pct}%` }}
              title={`${p.phase} (${p.agent}) — ${p.elapsed.toFixed(1)}s — ${p.outcome || "running"}`}
            >
              {pct > 8 && (
                <span className="truncate px-1">
                  {p.phase.replace("ag_ui_request", "request")}
                </span>
              )}
            </div>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-3 text-xs text-gray-400">
        {phases.map((p, i) => (
          <span key={i} className="flex items-center gap-1">
            <span
              className={`w-2 h-2 rounded-full ${
                p.outcome === "ok"
                  ? "bg-green-500"
                  : p.outcome === "error"
                  ? "bg-red-500"
                  : "bg-blue-400"
              }`}
            />
            {p.phase.replace("ag_ui_request", "request")} ({p.elapsed.toFixed(1)}s)
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Tool calls table ─────────────────────────────────────────────
function ToolCallsTable({ tools }: { tools: ToolCall[] }) {
  const [expanded, setExpanded] = useState(false);
  if (!tools.length) return null;
  const display = expanded ? tools : tools.slice(0, 20);

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <Wrench size={16} className="text-yellow-400" /> Tool Calls ({tools.length})
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-500 uppercase tracking-wider border-b border-gray-700/50">
              <th className="py-2 px-3 text-left">#</th>
              <th className="py-2 px-3 text-left">Tool</th>
              <th className="py-2 px-3 text-left">Agent</th>
              <th className="py-2 px-3 text-right">Duration</th>
              <th className="py-2 px-3 text-right">Result</th>
              <th className="py-2 px-3 text-left">Args</th>
              <th className="py-2 px-3 text-left">Flags</th>
            </tr>
          </thead>
          <tbody>
            {display.map((t, i) => (
              <tr
                key={i}
                className={`border-b border-gray-800/50 hover:bg-gray-800/30 ${
                  t.error ? "bg-red-900/10" : ""
                }`}
              >
                <td className="py-1.5 px-3 text-gray-500">{i + 1}</td>
                <td className="py-1.5 px-3">
                  <code className="bg-gray-800 px-1.5 py-0.5 rounded text-blue-300">
                    {t.tool_name}
                  </code>
                </td>
                <td className="py-1.5 px-3 text-gray-400">{t.agent}</td>
                <td className="py-1.5 px-3 text-right font-mono text-gray-300">
                  {t.duration_secs.toFixed(2)}s
                </td>
                <td className="py-1.5 px-3 text-right text-gray-400">
                  {(t.result_chars || 0).toLocaleString()}
                </td>
                <td className="py-1.5 px-3 text-gray-500 max-w-xs truncate">
                  {t.args_summary}
                </td>
                <td className="py-1.5 px-3 flex gap-1 flex-wrap">
                  {t.was_dedup_blocked && (
                    <span className="bg-yellow-500/20 text-yellow-400 px-1.5 py-0.5 rounded text-xs">
                      DEDUP
                    </span>
                  )}
                  {t.arg_fix_applied && (
                    <span className="bg-blue-500/20 text-blue-400 px-1.5 py-0.5 rounded text-xs">
                      ARG-FIX
                    </span>
                  )}
                  {t.was_compressed && (
                    <span className="bg-purple-500/20 text-purple-400 px-1.5 py-0.5 rounded text-xs">
                      COMPRESSED
                    </span>
                  )}
                  {t.error && (
                    <span className="bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded text-xs">
                      {t.error.slice(0, 40)}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {tools.length > 20 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1"
        >
          {expanded ? (
            <>
              <ChevronDown size={14} /> Show less
            </>
          ) : (
            <>
              <ChevronRight size={14} /> Show all {tools.length} tool calls
            </>
          )}
        </button>
      )}
    </div>
  );
}

// ── Tool breakdown pie chart ─────────────────────────────────────
function ToolBreakdownChart({
  toolSummary,
}: {
  toolSummary: Record<string, { count: number; total_duration: number; errors: number }>;
}) {
  const entries = Object.entries(toolSummary)
    .map(([name, s]) => ({ name: name.replace(/^(brave_|firecrawl_|web_search_|kagi_)/, ""), fullName: name, ...s }))
    .sort((a, b) => b.count - a.count);

  if (!entries.length) return null;

  return (
    <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/30">
      <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <Search size={16} className="text-yellow-400" /> Tool Breakdown
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={entries}
            dataKey="count"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
            innerRadius={40}
            paddingAngle={2}
            label={({ name, count }) => `${name} (${count})`}
            labelLine={false}
          >
            {entries.map((_, i) => (
              <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: "#161b22",
              border: "1px solid #30363d",
              borderRadius: 8,
              color: "#e6edf3",
            }}
            formatter={(value: number, name: string) => [
              `${value} calls`,
              name,
            ]}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Tool duration bar chart ──────────────────────────────────────
function ToolDurationChart({
  toolSummary,
}: {
  toolSummary: Record<string, { count: number; total_duration: number; errors: number }>;
}) {
  const data = Object.entries(toolSummary)
    .map(([name, s]) => ({
      name: name.replace(/^(brave_|firecrawl_|web_search_|kagi_)/, ""),
      avgDuration: s.count > 0 ? s.total_duration / s.count : 0,
      totalDuration: s.total_duration,
      count: s.count,
    }))
    .sort((a, b) => b.totalDuration - a.totalDuration)
    .slice(0, 10);

  if (!data.length) return null;

  return (
    <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/30">
      <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <Clock size={16} className="text-blue-400" /> Tool Duration (avg seconds)
      </h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
          <XAxis type="number" tick={{ fill: "#8b949e", fontSize: 11 }} />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: "#8b949e", fontSize: 11 }}
            width={80}
          />
          <Tooltip
            contentStyle={{
              background: "#161b22",
              border: "1px solid #30363d",
              borderRadius: 8,
              color: "#e6edf3",
            }}
            formatter={(value: number) => [`${value.toFixed(2)}s`, "Avg"]}
          />
          <Bar dataKey="avgDuration" fill={COLORS.accent} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Corpus growth chart ──────────────────────────────────────────
function CorpusChart({
  updates,
}: {
  updates: { admitted: number; total: number; iteration: number }[];
}) {
  if (!updates.length) return null;
  const data = updates.map((u) => ({
    name: `Iter ${u.iteration}`,
    total: u.total,
    admitted: u.admitted,
  }));

  return (
    <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/30">
      <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <Database size={16} className="text-purple-400" /> Corpus Growth
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
          <XAxis dataKey="name" tick={{ fill: "#8b949e", fontSize: 11 }} />
          <YAxis tick={{ fill: "#8b949e", fontSize: 11 }} />
          <Tooltip
            contentStyle={{
              background: "#161b22",
              border: "1px solid #30363d",
              borderRadius: 8,
              color: "#e6edf3",
            }}
          />
          <Area
            type="monotone"
            dataKey="total"
            stroke={COLORS.purple}
            fill={COLORS.purple}
            fillOpacity={0.15}
            name="Total atoms"
          />
          <Area
            type="monotone"
            dataKey="admitted"
            stroke={COLORS.success}
            fill={COLORS.success}
            fillOpacity={0.15}
            name="Admitted"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Event stream ─────────────────────────────────────────────────
function EventStream({ events, startedAt }: { events: RawEvent[]; startedAt: number }) {
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const display = expanded ? events.slice(-500) : events.slice(-50);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [display.length]);

  const badgeColor = (type: string) => {
    if (type.includes("tool")) return "bg-yellow-500/20 text-yellow-400";
    if (type.includes("llm")) return "bg-blue-500/20 text-blue-400";
    if (type.includes("phase")) return "bg-green-500/20 text-green-400";
    if (type.includes("corpus")) return "bg-purple-500/20 text-purple-400";
    if (type.includes("stall") || type.includes("force")) return "bg-red-500/20 text-red-400";
    return "bg-gray-500/20 text-gray-400";
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Activity size={16} className="text-green-400" /> Event Stream (
          {events.length})
        </h3>
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-blue-400 hover:text-blue-300"
        >
          {expanded ? "Show last 50" : `Show last 500`}
        </button>
      </div>
      <div
        ref={scrollRef}
        className="max-h-96 overflow-y-auto rounded-lg bg-gray-900/50 border border-gray-700/30"
      >
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-gray-900">
            <tr className="text-gray-500 uppercase tracking-wider border-b border-gray-700/50">
              <th className="py-1.5 px-2 text-left w-20">Time</th>
              <th className="py-1.5 px-2 text-left w-32">Event</th>
              <th className="py-1.5 px-2 text-left w-24">Agent</th>
              <th className="py-1.5 px-2 text-left w-24">Phase</th>
              <th className="py-1.5 px-2 text-left">Data</th>
            </tr>
          </thead>
          <tbody>
            {display.map((e, i) => {
              const ts = startedAt ? e.timestamp - startedAt : e.timestamp;
              return (
                <tr
                  key={i}
                  className="border-b border-gray-800/30 hover:bg-gray-800/20"
                >
                  <td className="py-1 px-2 font-mono text-gray-500">
                    +{ts.toFixed(1)}s
                  </td>
                  <td className="py-1 px-2">
                    <span
                      className={`px-1.5 py-0.5 rounded text-xs font-medium ${badgeColor(e.event_type)}`}
                    >
                      {e.event_type}
                    </span>
                  </td>
                  <td className="py-1 px-2 text-gray-400">{e.agent}</td>
                  <td className="py-1 px-2 text-gray-500">{e.phase}</td>
                  <td className="py-1 px-2 text-gray-500 max-w-md truncate font-mono">
                    {JSON.stringify(e.data).slice(0, 120)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Run history sidebar item ─────────────────────────────────────
function RunItem({
  run,
  active,
  onClick,
}: {
  run: RunListItem;
  active: boolean;
  onClick: () => void;
}) {
  const date = run.started_at
    ? new Date(run.started_at * 1000).toLocaleString()
    : "";
  return (
    <button
      onClick={onClick}
      className={`w-full text-left p-3 rounded-lg border transition-colors ${
        active
          ? "bg-blue-500/10 border-blue-500/30"
          : "bg-gray-800/30 border-gray-700/30 hover:border-gray-600/30"
      }`}
    >
      <div className="text-xs text-gray-400 mb-1">{date}</div>
      <div className="text-sm text-gray-200 line-clamp-2">
        {run.query || "(no query)"}
      </div>
      <div className="flex gap-3 mt-1.5 text-xs text-gray-500">
        <span>{run.elapsed_secs?.toFixed(0)}s</span>
        <span>{run.kpi?.tool_calls || 0} tools</span>
        <span>{run.kpi?.llm_calls || 0} LLM</span>
      </div>
    </button>
  );
}

// ── Alerts / banners ─────────────────────────────────────────────
function AlertBanner({
  type,
  children,
}: {
  type: "danger" | "success" | "warning";
  children: React.ReactNode;
}) {
  const styles = {
    danger: "bg-red-500/10 border-red-500/30 text-red-400",
    success: "bg-green-500/10 border-green-500/30 text-green-400",
    warning: "bg-yellow-500/10 border-yellow-500/30 text-yellow-400",
  };
  const icons = {
    danger: <AlertTriangle size={16} />,
    success: <CheckCircle size={16} />,
    warning: <AlertTriangle size={16} />,
  };
  return (
    <div
      className={`flex items-center gap-2 px-4 py-3 rounded-lg border text-sm font-medium ${styles[type]}`}
    >
      {icons[type]} {children}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────
function App() {
  const { snapshot, status, reconnect } = useSSE();
  const [view, setView] = useState<"live" | "history">("live");
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<RunDetail | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);

  // Load run history
  useEffect(() => {
    if (view === "history") {
      fetchRuns()
        .then(setRuns)
        .catch(() => setRuns([]));
    }
  }, [view]);

  const loadRun = async (sessionId: string) => {
    setLoadingRun(true);
    try {
      const data = await fetchRunDetail(sessionId);
      setSelectedRun(data);
    } catch {
      setSelectedRun(null);
    }
    setLoadingRun(false);
  };

  // Determine what data to show
  const isLive = view === "live";
  const snap = isLive ? snapshot : null;
  const detail = isLive ? null : selectedRun;
  const isIdle = snap?.status === "idle";
  const isRunning = snap && !isIdle && !snap.finalized;
  const isFinalized = snap?.finalized;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-200">
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Brain size={24} className="text-blue-400" />
            <h1 className="text-lg font-bold text-gray-100">
              MiroThinker Dashboard
            </h1>
          </div>
          <div className="flex items-center gap-4">
            {/* View toggle */}
            <div className="flex bg-gray-800 rounded-lg p-0.5">
              <button
                onClick={() => setView("live")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                  isLive
                    ? "bg-blue-500/20 text-blue-400"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                <Zap size={13} /> Live
              </button>
              <button
                onClick={() => setView("history")}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors flex items-center gap-1.5 ${
                  !isLive
                    ? "bg-blue-500/20 text-blue-400"
                    : "text-gray-400 hover:text-gray-200"
                }`}
              >
                <History size={13} /> History
              </button>
            </div>
            <StatusBadge status={status} />
            <button
              onClick={reconnect}
              className="text-gray-500 hover:text-gray-300 transition-colors"
              title="Reconnect"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto px-6 py-6 space-y-6">
        {/* ── Live View ───────────────────────────────────────── */}
        {isLive && (
          <>
            {/* Query banner */}
            {snap && !isIdle && (
              <div className="bg-gray-800/40 rounded-xl p-4 border-l-4 border-blue-500">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-xs text-gray-500 mb-1">
                      Session {snap.session_id?.slice(0, 8)} —{" "}
                      {isRunning ? (
                        <span className="text-green-400 font-medium">
                          Running
                        </span>
                      ) : isFinalized ? (
                        <span className="text-blue-400 font-medium">
                          Completed
                        </span>
                      ) : (
                        "Unknown"
                      )}
                    </div>
                    <div className="text-sm text-gray-300 italic">
                      {snap.query}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-blue-400">
                      {snap.elapsed_secs?.toFixed(1)}s
                    </div>
                    <div className="text-xs text-gray-500">elapsed</div>
                  </div>
                </div>
              </div>
            )}

            {/* Idle state */}
            {isIdle && (
              <div className="flex flex-col items-center justify-center py-24 text-gray-500">
                <Brain size={48} className="mb-4 text-gray-700" />
                <div className="text-lg font-medium">
                  No active pipeline run
                </div>
                <div className="text-sm mt-1">
                  Send a query to the AG-UI endpoint to see live data here
                </div>
              </div>
            )}

            {/* Alerts */}
            {snap?.stalled && (
              <AlertBanner type="danger">
                Pipeline stalled — no events received recently
              </AlertBanner>
            )}
            {snap?.thinker_escalated && (
              <AlertBanner type="success">
                Thinker signalled EVIDENCE_SUFFICIENT — loop will exit
              </AlertBanner>
            )}

            {/* Pipeline Execution Graph */}
            {snap && !isIdle && (
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
                  <Activity size={16} className="text-blue-400" /> Pipeline Execution
                </h3>
                <PipelineGraph
                  phases={snap.phases || []}
                  currentPhase={snap.current_phase || ""}
                  finalized={!!snap.finalized}
                  isIdle={false}
                />
              </div>
            )}

            {/* KPI Grid */}
            {snap && !isIdle && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                  <KPICard
                    label="Tool Calls"
                    value={snap.kpi?.tool_calls || 0}
                    icon={<Wrench size={18} />}
                    color={COLORS.warning}
                  />
                  <KPICard
                    label="LLM Calls"
                    value={snap.kpi?.llm_calls || 0}
                    icon={<Cpu size={18} />}
                    color={COLORS.info}
                  />
                  <KPICard
                    label="ADK Events"
                    value={snap.kpi?.adk_events || 0}
                    icon={<Activity size={18} />}
                    color={COLORS.accent}
                  />
                  <KPICard
                    label="Corpus Atoms"
                    value={snap.kpi?.corpus_atoms || 0}
                    icon={<Database size={18} />}
                    color={COLORS.purple}
                  />
                  <KPICard
                    label="Output"
                    value={`${((snap.kpi?.text_chars || 0) / 1000).toFixed(1)}K`}
                    icon={<FileText size={18} />}
                    color={COLORS.success}
                    subtitle="chars generated"
                  />
                  <KPICard
                    label="Dedup Blocks"
                    value={snap.kpi?.dedup_blocks || 0}
                    icon={<AlertTriangle size={18} />}
                    color={
                      (snap.kpi?.dedup_blocks || 0) > 0
                        ? COLORS.warning
                        : COLORS.muted
                    }
                  />
                </div>

                {/* Phase timeline */}
                <PhaseTimeline phases={snap.phases || []} />
              </>
            )}
          </>
        )}

        {/* ── History View ────────────────────────────────────── */}
        {!isLive && (
          <div className="grid grid-cols-12 gap-6">
            {/* Sidebar: run list */}
            <div className="col-span-3 space-y-2">
              <h3 className="text-sm font-semibold text-gray-400 flex items-center gap-2">
                <History size={16} /> Past Runs
              </h3>
              {runs.length === 0 && (
                <div className="text-xs text-gray-600 py-4 text-center">
                  No saved runs found
                </div>
              )}
              {runs.map((r) => (
                <RunItem
                  key={r.session_id}
                  run={r}
                  active={selectedRun?.session_id === r.session_id}
                  onClick={() => loadRun(r.session_id.slice(0, 8))}
                />
              ))}
            </div>

            {/* Main: run detail */}
            <div className="col-span-9 space-y-6">
              {loadingRun && (
                <div className="text-center py-12 text-gray-500">
                  <RefreshCw size={24} className="animate-spin mx-auto mb-2" />
                  Loading run...
                </div>
              )}
              {!loadingRun && !detail && (
                <div className="text-center py-12 text-gray-600">
                  Select a run from the sidebar
                </div>
              )}
              {detail && <RunDetailView detail={detail} />}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

// ── Run detail view (for history) ────────────────────────────────
function RunDetailView({ detail }: { detail: RunDetail }) {
  const kpi = detail.kpi;
  const started = detail.started_at
    ? new Date(detail.started_at * 1000).toLocaleString()
    : "";

  return (
    <div className="space-y-6">
      {/* Query */}
      <div className="bg-gray-800/40 rounded-xl p-4 border-l-4 border-blue-500">
        <div className="text-xs text-gray-500 mb-1">
          {started} — Session {detail.session_id?.slice(0, 8)}
        </div>
        <div className="text-sm text-gray-300 italic">{detail.query}</div>
        <div className="text-xs text-gray-500 mt-2">
          Elapsed: {detail.elapsed_secs?.toFixed(1)}s — Output:{" "}
          {(detail.result_length || 0).toLocaleString()} chars
        </div>
      </div>

      {/* Alerts */}
      {detail.stall_events?.length > 0 && (
        <AlertBanner type="danger">
          Pipeline stalled at {detail.stall_events[0].agent}
        </AlertBanner>
      )}
      {detail.thinker_escalated && (
        <AlertBanner type="success">
          Thinker signalled EVIDENCE_SUFFICIENT
        </AlertBanner>
      )}

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
        <KPICard
          label="Tool Calls"
          value={kpi?.tool_calls || 0}
          icon={<Wrench size={16} />}
          color={COLORS.warning}
        />
        <KPICard
          label="LLM Calls"
          value={kpi?.llm_calls || 0}
          icon={<Cpu size={16} />}
          color={COLORS.info}
        />
        <KPICard
          label="Events"
          value={kpi?.adk_events || 0}
          icon={<Activity size={16} />}
          color={COLORS.accent}
        />
        <KPICard
          label="Tool Errors"
          value={kpi?.tool_errors || 0}
          icon={<AlertTriangle size={16} />}
          color={(kpi?.tool_errors || 0) > 0 ? COLORS.danger : COLORS.muted}
        />
        <KPICard
          label="Prompt Tokens"
          value={`${((kpi?.prompt_tokens_est || 0) / 1000).toFixed(0)}K`}
          icon={<FileText size={16} />}
          color={COLORS.accent}
        />
        <KPICard
          label="Completion"
          value={`${((kpi?.completion_tokens_est || 0) / 1000).toFixed(0)}K`}
          icon={<FileText size={16} />}
          color={COLORS.success}
        />
        <KPICard
          label="Reasoning"
          value={`${((kpi?.reasoning_chars || 0) / 1000).toFixed(0)}K`}
          icon={<Brain size={16} />}
          color={COLORS.purple}
          subtitle="chars"
        />
        <KPICard
          label="Elapsed"
          value={`${(kpi?.elapsed_secs || 0).toFixed(0)}s`}
          icon={<Clock size={16} />}
          color={COLORS.accent}
        />
      </div>

      {/* Pipeline Execution Graph */}
      {detail.phases?.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
            <Activity size={16} className="text-blue-400" /> Pipeline Execution
          </h3>
          <PipelineGraph
            phases={detail.phases.map((p) => ({
              phase: p.phase,
              agent: p.agent,
              elapsed: (p.end_time || detail.finalized_at) - p.start_time,
              outcome: p.outcome,
            }))}
            currentPhase=""
            finalized={true}
            isIdle={false}
          />
        </div>
      )}

      {/* Phase timeline */}
      {detail.phases?.length > 0 && (
        <PhaseTimeline
          phases={detail.phases.map((p) => ({
            phase: p.phase,
            agent: p.agent,
            elapsed: (p.end_time || detail.finalized_at) - p.start_time,
            outcome: p.outcome,
          }))}
        />
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {detail.tool_summary && (
          <ToolBreakdownChart toolSummary={detail.tool_summary} />
        )}
        {detail.tool_summary && (
          <ToolDurationChart toolSummary={detail.tool_summary} />
        )}
      </div>

      {/* Corpus growth */}
      {detail.corpus_updates?.length > 0 && (
        <CorpusChart updates={detail.corpus_updates} />
      )}

      {/* Tool calls table */}
      {detail.tool_calls?.length > 0 && (
        <ToolCallsTable tools={detail.tool_calls} />
      )}

      {/* Event stream */}
      {detail.events?.length > 0 && (
        <EventStream events={detail.events} startedAt={detail.started_at} />
      )}
    </div>
  );
}

export default App;
