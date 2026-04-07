"use client";

import { ResearchFindings } from "@/components/research-findings";
import { AgentState } from "@/lib/types";
import {
  useCoAgent,
  useFrontendTool,
  useRenderToolCall,
} from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotSidebar } from "@copilotkit/react-ui";
import { useState } from "react";

export default function MiroThinkerPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  // Frontend Action: let the agent change the UI theme color
  useFrontendTool({
    name: "setThemeColor",
    parameters: [
      {
        name: "themeColor",
        description: "The theme color to set. Pick professional dark-theme-friendly colors.",
        required: true,
      },
    ],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main
      style={
        { "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties
      }
    >
      <CopilotSidebar
        disableSystemMessage={true}
        clickOutsideToClose={false}
        defaultOpen={true}
        labels={{
          title: "MiroThinker",
          initial:
            "Deep research agent powered by AG-UI. Ask any research question.",
        }}
        suggestions={[
          {
            title: "Deep Research",
            message:
              "Research the latest advances in quantum computing error correction.",
          },
          {
            title: "Market Analysis",
            message:
              "Find the most promising AI startups in Europe founded in the last 2 years.",
          },
          {
            title: "Technical Review",
            message:
              "Compare WebAssembly vs native for edge computing workloads.",
          },
        ]}
      >
        <MainContent themeColor={themeColor} />
      </CopilotSidebar>
    </main>
  );
}

function MainContent({ themeColor }: { themeColor: string }) {
  // Shared State: bidirectional state sync between agent and frontend
  const { state, setState } = useCoAgent<AgentState>({
    name: "research_agent",
    initialState: {
      findings: [],
      execution_mode: "report",
      sources: [],
    },
  });

  // Generative UI: render tool calls as rich UI components
  useRenderToolCall(
    {
      name: "brave_web_search",
      description: "Search the web using Brave Search.",
      parameters: [
        { name: "query", type: "string", required: true },
      ],
      render: ({ args, result }) => {
        return (
          <ToolCallCard
            title="Web Search"
            icon="magnifying-glass"
            query={args.query}
            result={result}
            themeColor={themeColor}
          />
        );
      },
    },
    [themeColor],
  );

  useRenderToolCall(
    {
      name: "firecrawl_scrape",
      description: "Scrape a web page using Firecrawl.",
      parameters: [
        { name: "url", type: "string", required: true },
      ],
      render: ({ args, result }) => {
        return (
          <ToolCallCard
            title="Page Scrape"
            icon="document"
            query={args.url}
            result={result}
            themeColor={themeColor}
          />
        );
      },
    },
    [themeColor],
  );

  useRenderToolCall(
    {
      name: "web_search_exa",
      description: "Search with Exa neural search.",
      parameters: [
        { name: "query", type: "string", required: true },
      ],
      render: ({ args, result }) => {
        return (
          <ToolCallCard
            title="Exa Search"
            icon="sparkle"
            query={args.query}
            result={result}
            themeColor={themeColor}
          />
        );
      },
    },
    [themeColor],
  );

  useRenderToolCall(
    {
      name: "kagi_search",
      description: "Search with Kagi.",
      parameters: [
        { name: "query", type: "string", required: true },
      ],
      render: ({ args, result }) => {
        return (
          <ToolCallCard
            title="Kagi Search"
            icon="globe"
            query={args.query}
            result={result}
            themeColor={themeColor}
          />
        );
      },
    },
    [themeColor],
  );

  return (
    <div className="min-h-screen flex justify-center items-start pt-12 px-4 bg-slate-900">
      <ResearchFindings state={state} setState={setState} />
    </div>
  );
}

// Generative UI component for rendering tool calls as cards
function ToolCallCard({
  title,
  icon,
  query,
  result,
  themeColor,
}: {
  title: string;
  icon: string;
  query?: string;
  result?: string;
  themeColor: string;
}) {
  const iconMap: Record<string, string> = {
    "magnifying-glass": "\uD83D\uDD0D",
    document: "\uD83D\uDCC4",
    sparkle: "\u2728",
    globe: "\uD83C\uDF10",
  };

  return (
    <div
      style={{ borderColor: themeColor }}
      className="border-l-4 bg-slate-800/50 rounded-lg p-3 my-2"
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-lg">{iconMap[icon] || "\uD83D\uDD27"}</span>
        <span className="text-sm font-semibold text-slate-300">{title}</span>
      </div>
      {query && (
        <p className="text-xs text-slate-400 mb-1 truncate">
          {query}
        </p>
      )}
      {result && (
        <p className="text-xs text-slate-500 line-clamp-3">
          {typeof result === "string" ? result : JSON.stringify(result)}
        </p>
      )}
      {!result && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <span className="animate-pulse">Running...</span>
        </div>
      )}
    </div>
  );
}
