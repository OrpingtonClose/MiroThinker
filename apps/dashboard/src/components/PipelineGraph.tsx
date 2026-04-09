import { useMemo, useCallback } from "react";
import {
  ReactFlow,
  Background,
  type Node,
  type Edge,
  Position,
  Handle,
  type NodeProps,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Brain,
  Search,
  FlaskConical,
  FileText,
  RotateCw,
  Sparkles,
} from "lucide-react";
import type { Phase } from "../types";

// ── Node status derivation from phases ──────────────────────────
type NodeStatus = "idle" | "active" | "completed" | "error";

interface PipelineNodeData {
  label: string;
  description: string;
  icon: React.ReactNode;
  status: NodeStatus;
  elapsed?: number;
  iteration?: number;
  [key: string]: unknown;
}

// ── Custom pipeline node ────────────────────────────────────────
function PipelineNode({ data }: NodeProps<Node<PipelineNodeData>>) {
  const statusStyles: Record<NodeStatus, string> = {
    idle: "border-gray-600/50 bg-gray-800/60 text-gray-400",
    active:
      "border-blue-400 bg-blue-500/15 text-blue-300 shadow-lg shadow-blue-500/20 ring-2 ring-blue-400/30",
    completed: "border-green-500/60 bg-green-500/10 text-green-300",
    error: "border-red-500/60 bg-red-500/10 text-red-300",
  };

  const iconStyles: Record<NodeStatus, string> = {
    idle: "text-gray-500",
    active: "text-blue-400 animate-pulse",
    completed: "text-green-400",
    error: "text-red-400",
  };

  const dotStyles: Record<NodeStatus, string> = {
    idle: "bg-gray-600",
    active: "bg-blue-400 animate-ping",
    completed: "bg-green-500",
    error: "bg-red-500",
  };

  return (
    <div
      className={`relative rounded-xl border-2 px-4 py-3 min-w-40 transition-all duration-300 ${statusStyles[data.status]}`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-gray-600 !border-gray-500 !w-2 !h-2"
      />

      {/* Status dot */}
      <div className="absolute -top-1 -right-1">
        <span className="relative flex h-3 w-3">
          {data.status === "active" && (
            <span
              className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${dotStyles[data.status]}`}
            />
          )}
          <span
            className={`relative inline-flex rounded-full h-3 w-3 ${dotStyles[data.status]}`}
          />
        </span>
      </div>

      <div className="flex items-center gap-2 mb-1">
        <span className={iconStyles[data.status]}>{data.icon}</span>
        <span className="font-semibold text-sm">{data.label}</span>
      </div>
      <div className="text-xs opacity-60">{data.description}</div>

      {data.elapsed !== undefined && data.elapsed > 0 && (
        <div className="text-xs font-mono mt-1 opacity-70">
          {data.elapsed.toFixed(1)}s
        </div>
      )}

      {data.iteration !== undefined && (
        <div className="absolute -bottom-1 -left-1 bg-purple-500/20 text-purple-300 text-xs px-1.5 py-0.5 rounded-full border border-purple-500/30 font-medium">
          iter {data.iteration}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-gray-600 !border-gray-500 !w-2 !h-2"
      />
    </div>
  );
}

// ── Loop container node ─────────────────────────────────────────
function LoopContainerNode({ data }: NodeProps<Node<PipelineNodeData>>) {
  return (
    <div className="rounded-2xl border-2 border-dashed border-purple-500/30 bg-purple-500/5 px-6 py-4 min-w-96">
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-purple-500 !border-purple-400 !w-2 !h-2"
      />
      <div className="flex items-center gap-2 mb-1 text-purple-300">
        <RotateCw size={14} />
        <span className="text-xs font-semibold uppercase tracking-wider">
          {data.label}
        </span>
        {data.iteration !== undefined && (
          <span className="text-xs bg-purple-500/20 px-1.5 py-0.5 rounded-full border border-purple-500/30">
            Iteration {data.iteration}/3
          </span>
        )}
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-purple-500 !border-purple-400 !w-2 !h-2"
      />
    </div>
  );
}

const nodeTypes = {
  pipeline: PipelineNode,
  loopContainer: LoopContainerNode,
};

// ── Derive node statuses from SSE phases ────────────────────────
function deriveStatuses(
  phases: Phase[],
  currentPhase: string,
  finalized: boolean
): Record<string, { status: NodeStatus; elapsed: number; iteration: number }> {
  const result: Record<
    string,
    { status: NodeStatus; elapsed: number; iteration: number }
  > = {
    thinker: { status: "idle", elapsed: 0, iteration: 0 },
    researcher: { status: "idle", elapsed: 0, iteration: 0 },
    loop_synthesiser: { status: "idle", elapsed: 0, iteration: 0 },
    swarm: { status: "idle", elapsed: 0, iteration: 0 },
    synthesiser: { status: "idle", elapsed: 0, iteration: 0 },
  };

  // Track the current iteration for each agent by counting completed phases
  let thinkerIter = 0;
  let researcherIter = 0;
  let loopSynthIter = 0;

  for (const p of phases) {
    const agent = p.agent?.toLowerCase() || "";
    const phase = p.phase?.toLowerCase() || "";

    // Map phase/agent names to our node keys
    let key = "";
    if (agent.includes("thinker") && !agent.includes("synthesiser")) {
      key = "thinker";
      thinkerIter++;
      result[key].iteration = thinkerIter;
    } else if (agent.includes("researcher") || agent.includes("executor")) {
      key = "researcher";
      researcherIter++;
      result[key].iteration = researcherIter;
    } else if (
      agent.includes("loop_synthesiser") ||
      (agent.includes("synthesiser") && phase.includes("loop"))
    ) {
      key = "loop_synthesiser";
      loopSynthIter++;
      result[key].iteration = loopSynthIter;
    } else if (
      agent.includes("swarm") ||
      phase.includes("swarm")
    ) {
      key = "swarm";
    } else if (agent.includes("synthesiser")) {
      key = "synthesiser";
    }

    if (!key) continue;

    result[key].elapsed += p.elapsed || 0;

    if (p.outcome === "ok" || p.outcome === "completed") {
      result[key].status = "completed";
    } else if (p.outcome === "error") {
      result[key].status = "error";
    } else {
      // Still running
      result[key].status = "active";
    }
  }

  // If currentPhase includes an agent name, mark it as active
  if (currentPhase && !finalized) {
    const cp = currentPhase.toLowerCase();
    if (cp.includes("thinker") && !cp.includes("synth")) {
      result.thinker.status = "active";
    } else if (cp.includes("researcher") || cp.includes("executor")) {
      result.researcher.status = "active";
    } else if (cp.includes("loop_synth")) {
      result.loop_synthesiser.status = "active";
    } else if (cp.includes("swarm")) {
      result.swarm.status = "active";
    } else if (cp.includes("synth")) {
      result.synthesiser.status = "active";
    }
  }

  // If finalized, mark everything as completed
  if (finalized) {
    for (const k of Object.keys(result)) {
      if (result[k].status === "active") {
        result[k].status = "completed";
      }
    }
  }

  return result;
}

// ── Build the graph ─────────────────────────────────────────────
interface PipelineGraphProps {
  phases: Phase[];
  currentPhase: string;
  finalized: boolean;
  isIdle: boolean;
}

export function PipelineGraph({
  phases,
  currentPhase,
  finalized,
  isIdle,
}: PipelineGraphProps) {
  const statuses = useMemo(
    () => deriveStatuses(phases, currentPhase, finalized),
    [phases, currentPhase, finalized]
  );

  // Current loop iteration (max of all agent iterations)
  const currentIteration = Math.max(
    statuses.thinker.iteration,
    statuses.researcher.iteration,
    statuses.loop_synthesiser.iteration,
    1
  );

  const nodes: Node<PipelineNodeData>[] = useMemo(
    () => [
      // Loop container (background group)
      {
        id: "loop-container",
        type: "loopContainer",
        position: { x: 60, y: 60 },
        data: {
          label: "Research Loop",
          description: "max 3 iterations",
          icon: <RotateCw size={16} />,
          status: isIdle ? "idle" : finalized ? "completed" : "active",
          iteration: currentIteration,
        },
        style: { width: 380, height: 340 },
        selectable: false,
        draggable: false,
      },
      // Thinker
      {
        id: "thinker",
        type: "pipeline",
        position: { x: 170, y: 100 },
        data: {
          label: "Thinker",
          description: "Pure reasoning, strategy",
          icon: <Brain size={16} />,
          status: isIdle ? "idle" : statuses.thinker.status,
          elapsed: statuses.thinker.elapsed,
          iteration: statuses.thinker.iteration || undefined,
        },
      },
      // Researcher
      {
        id: "researcher",
        type: "pipeline",
        position: { x: 170, y: 210 },
        data: {
          label: "Researcher",
          description: "Tool execution, expansion",
          icon: <Search size={16} />,
          status: isIdle ? "idle" : statuses.researcher.status,
          elapsed: statuses.researcher.elapsed,
          iteration: statuses.researcher.iteration || undefined,
        },
      },
      // Loop synthesiser
      {
        id: "loop-synth",
        type: "pipeline",
        position: { x: 170, y: 320 },
        data: {
          label: "Loop Synthesiser",
          description: "Fermentation, re-ingestion",
          icon: <FlaskConical size={16} />,
          status: isIdle ? "idle" : statuses.loop_synthesiser.status,
          elapsed: statuses.loop_synthesiser.elapsed,
          iteration: statuses.loop_synthesiser.iteration || undefined,
        },
      },
      // Swarm synthesis
      {
        id: "swarm",
        type: "pipeline",
        position: { x: 170, y: 450 },
        data: {
          label: "Flock Swarm",
          description: "Gossip protocol synthesis",
          icon: <Sparkles size={16} />,
          status: isIdle ? "idle" : statuses.swarm.status,
          elapsed: statuses.swarm.elapsed,
        },
      },
      // Final synthesiser
      {
        id: "synthesiser",
        type: "pipeline",
        position: { x: 170, y: 550 },
        data: {
          label: "Final Synthesiser",
          description: "Polished report output",
          icon: <FileText size={16} />,
          status: isIdle ? "idle" : statuses.synthesiser.status,
          elapsed: statuses.synthesiser.elapsed,
        },
      },
    ],
    [statuses, isIdle, finalized, currentIteration]
  );

  const edges: Edge[] = useMemo(
    () => [
      {
        id: "e-thinker-researcher",
        source: "thinker",
        target: "researcher",
        animated: statuses.researcher.status === "active",
        style: {
          stroke:
            statuses.researcher.status === "active"
              ? "#58a6ff"
              : statuses.researcher.status === "completed"
              ? "#3fb950"
              : "#30363d",
          strokeWidth: 2,
        },
        label: "strategy",
        labelStyle: { fill: "#8b949e", fontSize: 10 },
        labelBgStyle: { fill: "#0d1117", fillOpacity: 0.8 },
      },
      {
        id: "e-researcher-loopsynth",
        source: "researcher",
        target: "loop-synth",
        animated: statuses.loop_synthesiser.status === "active",
        style: {
          stroke:
            statuses.loop_synthesiser.status === "active"
              ? "#58a6ff"
              : statuses.loop_synthesiser.status === "completed"
              ? "#3fb950"
              : "#30363d",
          strokeWidth: 2,
        },
        label: "findings",
        labelStyle: { fill: "#8b949e", fontSize: 10 },
        labelBgStyle: { fill: "#0d1117", fillOpacity: 0.8 },
      },
      // Loop-back edge: loop_synthesiser → thinker
      {
        id: "e-loopsynth-thinker",
        source: "loop-synth",
        target: "thinker",
        type: "smoothstep",
        animated:
          statuses.thinker.status === "active" &&
          statuses.thinker.iteration > 1,
        style: {
          stroke:
            statuses.thinker.status === "active" &&
            statuses.thinker.iteration > 1
              ? "#a371f7"
              : "#30363d",
          strokeWidth: 2,
          strokeDasharray: "5 5",
        },
        label: "re-ingested corpus",
        labelStyle: { fill: "#a371f7", fontSize: 9 },
        labelBgStyle: { fill: "#0d1117", fillOpacity: 0.8 },
        sourcePosition: Position.Left,
        targetPosition: Position.Left,
      },
      // Loop → swarm
      {
        id: "e-loop-swarm",
        source: "loop-container",
        target: "swarm",
        animated: statuses.swarm.status === "active",
        style: {
          stroke:
            statuses.swarm.status === "active"
              ? "#58a6ff"
              : statuses.swarm.status === "completed"
              ? "#3fb950"
              : "#30363d",
          strokeWidth: 2,
        },
        label: "corpus",
        labelStyle: { fill: "#8b949e", fontSize: 10 },
        labelBgStyle: { fill: "#0d1117", fillOpacity: 0.8 },
      },
      // Swarm → final synthesiser
      {
        id: "e-swarm-synth",
        source: "swarm",
        target: "synthesiser",
        animated: statuses.synthesiser.status === "active",
        style: {
          stroke:
            statuses.synthesiser.status === "active"
              ? "#58a6ff"
              : statuses.synthesiser.status === "completed"
              ? "#3fb950"
              : "#30363d",
          strokeWidth: 2,
        },
        label: "swarm report",
        labelStyle: { fill: "#8b949e", fontSize: 10 },
        labelBgStyle: { fill: "#0d1117", fillOpacity: 0.8 },
      },
    ],
    [statuses]
  );

  const onInit = useCallback(
    (instance: { fitView: () => void }) => instance.fitView(),
    []
  );

  return (
    <div className="w-full h-96 rounded-xl border border-gray-700/30 bg-gray-900/50 overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={onInit}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        zoomOnScroll={true}
        minZoom={0.5}
        maxZoom={1.5}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#21262d"
        />
      </ReactFlow>
    </div>
  );
}
