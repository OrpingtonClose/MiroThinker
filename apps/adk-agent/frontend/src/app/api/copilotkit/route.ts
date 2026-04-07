import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest } from "next/server";

// Use the empty adapter since we only have the ADK agent.
const serviceAdapter = new ExperimentalEmptyAdapter();

// AG-UI client connects to the MiroThinker FastAPI AG-UI endpoint.
const agentUrl = process.env.AGENT_URL || "http://localhost:8000/";

const runtime = new CopilotRuntime({
  agents: {
    research_agent: new HttpAgent({ url: agentUrl }),
  },
});

// Next.js API route that proxies CopilotKit requests to the ADK agent.
export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
