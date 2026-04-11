"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import type { DashboardSnapshot } from "@/types/dashboard";

export type ConnectionStatus = "connected" | "connecting" | "disconnected";

/**
 * Maximum consecutive SSE failures before switching to polling.
 * SSE often fails over public internet (proxy buffering, tunnel timeouts)
 * so we fall back to polling `/dashboard/latest` every second.
 */
const SSE_MAX_FAILURES = 3;
const POLL_INTERVAL_MS = 1000;
const SSE_RETRY_MS = 3000;

/** Derive the API base URL from env or the current page origin. */
function apiBase(): string {
  if (typeof window === "undefined") return "http://localhost:8000";
  const env = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (env) return env.replace(/\/+$/, "");
  return window.location.origin;
}

function sseUrl(): string {
  return `${apiBase()}/dashboard/stream`;
}

function pollUrl(): string {
  return `${apiBase()}/dashboard/latest`;
}

export function useDashboardSSE() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const esRef = useRef<EventSource | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sseFailures = useRef(0);

  const cleanup = useCallback(() => {
    if (retryRef.current) { clearTimeout(retryRef.current); retryRef.current = null; }
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (esRef.current) { esRef.current.close(); esRef.current = null; }
  }, []);

  // ── Polling fallback (public internet / after SSE failures) ───
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
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS);
  }, [cleanup]);

  // ── SSE mode (primary) ────────────────────────────────────────
  const startSSE = useCallback(() => {
    cleanup();
    setStatus("connecting");

    const es = new EventSource(sseUrl());
    esRef.current = es;

    es.onopen = () => {
      setStatus("connected");
      sseFailures.current = 0;
    };

    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as DashboardSnapshot;
        setSnapshot(data);
        setStatus("connected");
        sseFailures.current = 0;
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      setStatus("disconnected");
      es.close();
      esRef.current = null;

      sseFailures.current += 1;

      if (sseFailures.current >= SSE_MAX_FAILURES) {
        // SSE keeps failing — switch to polling (common over public internet)
        console.warn(
          `[Dashboard] SSE failed ${sseFailures.current} times, switching to polling fallback`
        );
        startPolling();
      } else {
        retryRef.current = setTimeout(startSSE, SSE_RETRY_MS);
      }
    };
  }, [cleanup, startPolling]);

  // ── Boot ──────────────────────────────────────────────────────
  useEffect(() => {
    startSSE(); // always try SSE first, fall back to polling on failure
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reconnect = useCallback(() => {
    sseFailures.current = 0;
    startSSE(); // always try SSE first on manual reconnect
  }, [startSSE]);

  return { snapshot, status, reconnect };
}
