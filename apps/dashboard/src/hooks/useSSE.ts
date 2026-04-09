import { useEffect, useRef, useState, useCallback } from "react";
import type { DashboardSnapshot } from "../types";

const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

export function useSSE() {
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

    const es = new EventSource(`${API}/dashboard/stream`);
    esRef.current = es;

    es.onopen = () => {
      setStatus("connected");
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as DashboardSnapshot;
        setSnapshot(data);
        setStatus("connected");
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      setStatus("disconnected");
      es.close();
      esRef.current = null;
      // Retry after 3s
      retryRef.current = setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (esRef.current) esRef.current.close();
      if (retryRef.current) clearTimeout(retryRef.current);
    };
  }, [connect]);

  return { snapshot, status, reconnect: connect };
}

/** Fetch a specific run detail */
export async function fetchRunDetail(sessionId: string) {
  const res = await fetch(`${API}/dashboard/run/${sessionId}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** Fetch all runs list */
export async function fetchRuns() {
  const res = await fetch(`${API}/dashboard/runs`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.runs || [];
}

/** Fetch latest run */
export async function fetchLatest() {
  const res = await fetch(`${API}/dashboard/latest`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
