/** SSE snapshot shape from /dashboard/stream */
export interface DashboardSnapshot {
  session_id: string;
  query: string;
  elapsed_secs: number;
  current_phase: string;
  phases: Phase[];
  kpi: KPI;
  thinker_escalated: boolean;
  stalled: boolean;
  finalized: boolean;
  event_count: number;
  status?: "idle";
  message?: string;
}

export interface Phase {
  phase: string;
  agent: string;
  elapsed: number;
  outcome: string;
}

export interface KPI {
  adk_events: number;
  tool_calls: number;
  llm_calls: number;
  text_chars: number;
  reasoning_chars: number;
  dedup_blocks: number;
  arg_fixes: number;
  bad_results: number;
  compressions: number;
  keep_k_trims: number;
  corpus_atoms: number;
}

/** Full run detail from /dashboard/latest or /dashboard/run/:id */
export interface RunDetail {
  session_id: string;
  query: string;
  started_at: number;
  finalized_at: number;
  elapsed_secs: number;
  result_length: number;
  phases: PhaseDetail[];
  kpi: RunKPI;
  algorithms: Algorithms;
  tool_calls: ToolCall[];
  tool_summary: Record<string, ToolSummaryEntry>;
  llm_calls: LLMCall[];
  corpus_updates: CorpusUpdate[];
  stall_events: StallEvent[];
  thinker_escalated: boolean;
  thinker_escalate_time: number;
  events: RawEvent[];
}

export interface PhaseDetail {
  phase: string;
  agent: string;
  start_time: number;
  end_time: number;
  outcome: string;
}

export interface RunKPI extends KPI {
  tool_errors: number;
  llm_errors: number;
  prompt_tokens_est: number;
  completion_tokens_est: number;
  elapsed_secs: number;
}

export interface Algorithms {
  dedup_blocks: DedupBlock[];
  arg_fixes: ArgFix[];
  bad_results: BadResult[];
  compressions: Compression[];
  keep_k_trims: KeepKTrim[];
  force_ends: ForceEnd[];
}

export interface DedupBlock {
  tool_name: string;
  query_key: string;
  consecutive: number;
  timestamp: number;
}

export interface ArgFix {
  tool_name: string;
  fixes: string[];
  timestamp: number;
}

export interface BadResult {
  tool_name: string;
  error: string;
  timestamp: number;
}

export interface Compression {
  tool_name: string;
  original_chars: number;
  compressed_chars: number;
  ratio: number;
  timestamp: number;
}

export interface KeepKTrim {
  kept: number;
  omitted: number;
  utilisation: number;
  timestamp: number;
}

export interface ForceEnd {
  estimated_tokens: number;
  timestamp: number;
}

export interface ToolCall {
  tool_name: string;
  agent: string;
  args_summary: string;
  start_time: number;
  end_time: number;
  duration_secs: number;
  result_chars: number;
  error: string;
  was_dedup_blocked: boolean;
  arg_fix_applied: boolean;
  was_compressed: boolean;
  original_chars: number;
}

export interface ToolSummaryEntry {
  count: number;
  total_duration: number;
  errors: number;
  total_result_chars: number;
}

export interface LLMCall {
  agent: string;
  start_time: number;
  end_time: number;
  duration_secs: number;
  prompt_tokens_est: number;
  completion_tokens_est: number;
}

export interface CorpusUpdate {
  admitted: number;
  total: number;
  iteration: number;
  timestamp: number;
}

export interface StallEvent {
  agent: string;
  event_count: number;
  timeout: number;
  timestamp: number;
}

export interface RawEvent {
  event_type: string;
  timestamp: number;
  agent: string;
  phase: string;
  data: Record<string, unknown>;
}

export interface RunListItem {
  file: string;
  session_id: string;
  query: string;
  elapsed_secs: number;
  started_at: number;
  kpi: Partial<KPI>;
}
