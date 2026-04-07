import type { Metadata } from "next";

import { CopilotKit } from "@copilotkit/react-core";
import "./globals.css";
import "@copilotkit/react-ui/styles.css";

export const metadata: Metadata = {
  title: "MiroThinker - Deep Research Agent",
  description:
    "AG-UI powered deep research agent with streaming, tool calls, and generative UI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <CopilotKit runtimeUrl="/api/copilotkit" agent="research_agent">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
