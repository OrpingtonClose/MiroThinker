"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { DashboardSnapshot } from "@/types/dashboard";

export type ConnectionStatus = "connected" | "connecting" | "disconnected";

// On localhost we can hit :8000 directly for real-time SSE.
// On tunnel/deployed URLs only port 3000 is exposed, so we poll
// the /agui/dashboard/latest JSON endpoint every second instead.
function isLocal(): boolean {
  if (typeof window === "undefined") return true;
  const h = window.location.hostname;
  return h === "localhost" || h === "127.0.0.1";
}

function sseUrl(): string {
  if (typeof window === "undefined") return "http://localhost:8000/dashboard/stream";
  return `http://${window.location.hostname}:8000/dashboard/stream`;
}

function pollUrl(): string {
  if (typeof window === "undefined") return "http://localhost:8000/dashboard/latest";
  if (isLocal()) return `http://${window.location.hostname}:8000/dashboard/latest`;
  return `${window.location.origin}/agui/dashboard/latest`;
}

export function useDashboardSSE() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const esRef = useRef<EventSource | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const cleanup = useCallback(() => {
    if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null; }
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (esRef.current) { esRef.current.close(); esRef.current = null; }
  }, []);

  // ── Polling mode (tunnel / deployed) ──────────────────────────
  const startPolling = useCallback(() => {
    cleanup();
    setStatus("connecting");
    const url = pollUrl();
    const poll = async () => {
      try {
        const r = await fetch(url);
        if (!r.ok) { setStatus("disconnected"); return; }
        const data = (await r.json()) as DashboardSnapshot;
        setSnapshot(data);
        setStatus("connected");
      } catch {
        setStatus("disconnected");
      }
    };
    poll();
    pollRef.current = setInterval(poll, 1000);
  }, [cleanup]);

  // ── SSE mode (localhost) ──────────────────────────────────────
  const startSSE = useCallback(() => {
    cleanup();
    setStatus("connecting");
    const es = new EventSource(sseUrl());
    esRef.current = es;
    es.onopen = () => setStatus("connected");
    es.onmessage = (ev) => {
      try {
        setSnapshot(JSON.parse(ev.data) as DashboardSnapshot);
        setStatus("connected");
      } catch { /* ignore */ }
    };
    es.onerror = () => {
      es.close();
      esRef.current = null;
      setStatus("disconnected");
      retryRef.current = setTimeout(startSSE, 2000);
    };
  }, [cleanup]);

  // ── Boot ──────────────────────────────────────────────────────
  useEffect(() => {
    if (isLocal()) startSSE(); else startPolling();
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reconnect = useCallback(() => {
    if (isLocal()) startSSE(); else startPolling();
  }, [startSSE, startPolling]);

  return { snapshot, status, reconnect };
}
