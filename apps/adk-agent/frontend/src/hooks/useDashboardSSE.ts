"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { DashboardSnapshot } from "@/types/dashboard";

export type ConnectionStatus = "connected" | "connecting" | "disconnected";

// SSE URL resolution:
// - On localhost, connect directly to :8000 (avoids Next.js proxy SSE buffering)
// - On tunnel/deployed URLs, use the /agui/ proxy path (only port 3000 is exposed)
function getSSEUrl(): string {
  if (typeof window === "undefined") return "http://localhost:8000/dashboard/stream";
  const host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") {
    return `http://${host}:8000/dashboard/stream`;
  }
  // Tunnel or deployed: use Next.js proxy path (same origin)
  return `${window.location.origin}/agui/dashboard/stream`;
}

const SSE_URL = getSSEUrl();

export function useDashboardSSE() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const esRef = useRef<EventSource | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (retryRef.current) {
      clearTimeout(retryRef.current);
      retryRef.current = null;
    }
    if (esRef.current) {
      esRef.current.close();
    }
    setStatus("connecting");

    const es = new EventSource(SSE_URL);
    esRef.current = es;

    es.onopen = () => setStatus("connected");

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as DashboardSnapshot;
        setSnapshot(data);
        setStatus("connected");
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      es.close();
      setStatus("disconnected");
      // Auto-retry in 2s
      retryRef.current = setTimeout(connect, 2000);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      esRef.current?.close();
      if (retryRef.current) clearTimeout(retryRef.current);
    };
  }, [connect]);

  return { snapshot, status, reconnect: connect };
}
