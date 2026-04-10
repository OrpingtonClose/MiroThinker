// Shared agent state synced via CopilotKit useCoAgent
export interface AgentState {
  findings: string[];
  execution_mode: string;
  sources: string[];
}

// Dashboard SSE snapshot from /dashboard/stream
export interface DashboardSnapshot {
  status: string;
  session_id?: string;
  query?: string;
  started_at?: number;
  elapsed_secs?: number;
  finalized?: boolean;
  stalled?: boolean;
  thinker_escalated?: boolean;
  current_phase?: string;
  message?: string;

  kpi?: {
    tool_calls: number;
    llm_calls: number;
    adk_events: number;
    corpus_atoms: number;
    text_chars: number;
    dedup_blocks: number;
    tool_errors: number;
    prompt_tokens_est: number;
    completion_tokens_est: number;
    reasoning_chars: number;
    elapsed_secs: number;
  };

  phases?: Phase[];
  recent_events?: RawEvent[];
  recent_llm?: LLMCall[];
  recent_tools?: ToolCall[];
  pending_tools?: string[];
  pending_llm?: string[];
  corpus_history?: CorpusUpdate[];
  llm_per_min?: number;
  tools_per_min?: number;
  secs_since_activity?: number;
}

export interface Phase {
  phase: string;
  agent: string;
  elapsed: number;
  outcome?: string;
}

export interface RawEvent {
  event_type: string;
  timestamp: number;
  agent: string;
  phase: string;
  data: Record<string, unknown>;
}

export interface LLMCall {
  agent: string;
  duration_secs: number;
  completion_tokens_est: number;
  end_time: number;
}

export interface ToolCall {
  tool_name: string;
  agent: string;
  duration_secs: number;
  result_chars: number;
  error: string;
  was_compressed: boolean;
}

export interface CorpusUpdate {
  iteration: number;
  admitted: number;
  total: number;
}
