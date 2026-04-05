/**
 * MiroThinker ADK Dashboard — Main Application Logic.
 *
 * Connects to the SSE event stream or loads a static report,
 * routes events to the correct panel modules, and manages
 * sidebar navigation.
 */

const App = {
  // State
  sessionId: null,
  mode: 'live',  // 'live' or 'static'
  sseClient: null,
  startTime: null,
  elapsedTimer: null,

  kpi: {
    turn: 0, toolCalls: 0, toolErrors: 0, llmCalls: 0,
    promptTokens: 0, completionTokens: 0, retryAttempts: 0,
    intermediateAnswers: 0, elapsed: 0, dedupBlocks: 0,
  },

  // Accumulate all received events for client-side post-hoc stats
  _events: [],

  // ------------------------------------------------------------------
  // Initialisation
  // ------------------------------------------------------------------

  init() {
    // Parse URL params
    const params = new URLSearchParams(window.location.search);
    this.sessionId = params.get('session');
    const reportFile = params.get('report');

    // Init panels
    KPIPanel.init(document.getElementById('kpi-container'));
    TimelinePanel.init(document.getElementById('timeline-container'));
    AgentGraphPanel.init(document.getElementById('agent-graph-container'));
    ToolCallsPanel.init(document.getElementById('tool-calls-container'));
    ContextPanel.init(document.getElementById('context-container'));
    AlgorithmsPanel.init(document.getElementById('algorithms-container'));
    AnswersPanel.init(document.getElementById('answers-container'));

    // Sidebar navigation
    this._initSidebar();

    // Start elapsed timer
    this.startTime = Date.now();
    this.elapsedTimer = setInterval(() => this._updateElapsed(), 1000);

    if (reportFile) {
      this._loadReport(reportFile);
    } else if (this.sessionId) {
      this._connectSSE(this.sessionId);
    } else {
      this._loadSessionList();
    }
  },

  // ------------------------------------------------------------------
  // Sidebar
  // ------------------------------------------------------------------

  _initSidebar() {
    const items = document.querySelectorAll('.sidebar-item');
    items.forEach(item => {
      item.addEventListener('click', () => {
        // Deactivate all
        items.forEach(i => i.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

        // Activate clicked
        item.classList.add('active');
        const target = item.dataset.panel;
        const panel = document.getElementById(`panel-${target}`);
        if (panel) panel.classList.add('active');
      });
    });

    // Default: show timeline
    const defaultItem = document.querySelector('.sidebar-item[data-panel="timeline"]');
    if (defaultItem) defaultItem.click();
  },

  // ------------------------------------------------------------------
  // SSE Connection (live mode)
  // ------------------------------------------------------------------

  _connectSSE(sessionId) {
    this.mode = 'live';
    this._updateStatus('live', `Connected to ${sessionId}`);

    this.sseClient = new SSEClient(
      `/api/events/${sessionId}`,
      (event) => this._handleEvent(event),
      () => this._onStreamEnd(),
      (err) => this._updateStatus('error', 'Connection lost, reconnecting...')
    );
    this.sseClient.connect();
  },

  _onStreamEnd() {
    this.mode = 'static';
    this._updateStatus('ended', 'Session complete');
    clearInterval(this.elapsedTimer);
    // Freeze elapsed at final value
    this.kpi.elapsed = (Date.now() - this.startTime) / 1000;
    KPIPanel.update(this.kpi);

    // Try server-side metrics first, fall back to client-side synthesis
    this._loadFinalMetrics(this.sessionId);
  },

  // ------------------------------------------------------------------
  // Event Routing
  // ------------------------------------------------------------------

  _handleEvent(event) {
    // Accumulate for client-side post-hoc stats
    this._events.push(event);

    // Always add to timeline
    TimelinePanel.addEntry(event);

    const d = event.data || {};

    switch (event.type) {
      case 'turn_start':
        this.kpi.turn = event.turn || d.turn || this.kpi.turn + 1;
        break;

      case 'tool_call_start':
        this.kpi.toolCalls++;
        ToolCallsPanel.addCall({
          call_id: d.call_id || '',
          tool_name: d.tool_name || '',
          agent_name: event.agent || '',
          turn: event.turn || 0,
          arguments_summary: d.arguments_summary || '',
          arg_fix_applied: d.arg_fix_applied || false,
        });
        break;

      case 'tool_call_end':
        ToolCallsPanel.updateCall(d.call_id || '', {
          duration_secs: d.duration_secs,
          result_size_chars: d.result_size_chars || 0,
          error: d.error || '',
          was_bad_result: d.was_bad_result || false,
        });
        if (d.error) this.kpi.toolErrors++;
        break;

      case 'llm_call_start':
        this.kpi.llmCalls++;
        this.kpi.promptTokens += d.estimated_prompt_tokens || 0;
        break;

      case 'llm_call_end':
        this.kpi.completionTokens += d.completion_tokens_est || 0;
        break;

      case 'dedup_blocked':
        this.kpi.dedupBlocks++;
        AlgorithmsPanel.addEvent(event);
        break;

      case 'dedup_allowed':
        AlgorithmsPanel.addEvent(event);
        break;

      case 'arg_fix_applied':
        AlgorithmsPanel.addEvent(event);
        break;

      case 'bad_result':
        AlgorithmsPanel.addEvent(event);
        break;

      case 'context_trimmed':
        ContextPanel.addTrim(d);
        AlgorithmsPanel.addEvent(event);
        break;

      case 'force_end':
        ContextPanel.updateTokenEstimate(d.estimated_tokens || 0, d.max_tokens || 0);
        AlgorithmsPanel.addEvent(event);
        break;

      case 'boxed_extracted':
        this.kpi.intermediateAnswers++;
        AnswersPanel.addAnswer(event);
        AlgorithmsPanel.addEvent(event);
        break;

      case 'retry_attempt':
        this.kpi.retryAttempts = d.attempt_number || this.kpi.retryAttempts + 1;
        AlgorithmsPanel.addEvent(event);
        break;

      case 'failure_summary':
        AlgorithmsPanel.addEvent(event);
        break;

      case 'agent_start':
        AgentGraphPanel.agentStarted(event.agent || d.agent_name || '');
        break;

      case 'agent_end':
        AgentGraphPanel.agentEnded(event.agent || d.agent_name || '');
        break;
    }

    // Update KPI cards
    KPIPanel.update(this.kpi);
  },

  // ------------------------------------------------------------------
  // Post-hoc mode
  // ------------------------------------------------------------------

  async _loadFinalMetrics(sessionId) {
    // Try server-side metrics first
    try {
      const resp = await fetch(`/api/metrics/${sessionId}`);
      if (resp.ok) {
        const metrics = await resp.json();
        this._populateFromMetrics(metrics);
        return;
      }
    } catch (e) {
      console.warn('Live metrics unavailable:', e.message);
    }
    // Try saved reports
    try {
      const reportResp = await fetch('/api/reports');
      if (reportResp.ok) {
        const reports = await reportResp.json();
        const match = reports.find(r => r.filename.includes(sessionId));
        if (match) {
          const dataResp = await fetch(`/api/reports/${match.filename}`);
          if (dataResp.ok) {
            const metrics = await dataResp.json();
            this._populateFromMetrics(metrics);
            return;
          }
        }
      }
    } catch (e) {
      console.warn('Saved reports unavailable:', e.message);
    }
    // Final fallback: build post-hoc stats from client-side accumulated events
    console.info('Building post-hoc stats from client-side event data');
    this._buildPostHocFromClientState();
  },

  _buildPostHocFromClientState() {
    // Count algorithm events from accumulated SSE events
    let dedupBlocks = 0, contextTrims = 0, argFixes = 0;
    let badResults = 0, retryAttempts = 0, intermediateAnswers = 0, forceEnds = 0;
    const toolUsage = {};

    for (const evt of this._events) {
      const d = evt.data || {};
      switch (evt.type) {
        case 'dedup_blocked': dedupBlocks++; break;
        case 'context_trimmed': contextTrims++; break;
        case 'arg_fix_applied': argFixes++; break;
        case 'bad_result': badResults++; break;
        case 'retry_attempt': retryAttempts = Math.max(retryAttempts, d.attempt_number || 0); break;
        case 'boxed_extracted': intermediateAnswers++; break;
        case 'force_end': forceEnds++; break;
        case 'tool_call_end': {
          const name = d.tool_name || 'unknown';
          if (!toolUsage[name]) toolUsage[name] = { count: 0, total_duration: 0, errors: 0 };
          toolUsage[name].count++;
          toolUsage[name].total_duration += d.duration_secs || 0;
          if (d.error) toolUsage[name].errors++;
          break;
        }
      }
    }

    // Count tool_call_start for tools that never got a tool_call_end
    const startCounts = {};
    for (const evt of this._events) {
      if (evt.type === 'tool_call_start') {
        const name = (evt.data || {}).tool_name || 'unknown';
        startCounts[name] = (startCounts[name] || 0) + 1;
      }
    }
    // Use start counts if no end events were captured for a tool
    for (const [name, startCount] of Object.entries(startCounts)) {
      if (!toolUsage[name]) {
        toolUsage[name] = { count: startCount, total_duration: 0, errors: 0 };
      } else if (toolUsage[name].count < startCount) {
        toolUsage[name].count = startCount;
      }
    }

    const metrics = {
      algorithm_stats: {
        dedup_blocks_saved: dedupBlocks,
        context_trims_performed: contextTrims,
        arg_fixes_applied: argFixes,
        bad_results_caught: badResults,
        retry_attempts_used: retryAttempts,
        intermediate_answers_extracted: intermediateAnswers,
        force_end_triggered: forceEnds,
      },
      llm_summary: {
        total_calls: this.kpi.llmCalls,
        total_prompt_tokens_est: this.kpi.promptTokens,
        total_completion_tokens_est: this.kpi.completionTokens,
        avg_duration_secs: 0,
      },
      tool_summary: toolUsage,
    };

    // Calculate avg LLM duration from events
    let llmDurations = [];
    let llmStart = null;
    for (const evt of this._events) {
      if (evt.type === 'llm_call_start') llmStart = evt.timestamp;
      if (evt.type === 'llm_call_end' && llmStart) {
        llmDurations.push(evt.timestamp - llmStart);
        llmStart = null;
      }
    }
    if (llmDurations.length > 0) {
      metrics.llm_summary.avg_duration_secs =
        llmDurations.reduce((a, b) => a + b, 0) / llmDurations.length;
    }

    this._renderPostHocStats(metrics);
  },

  async _loadReport(filename) {
    this.mode = 'static';
    this._updateStatus('ended', `Report: ${filename}`);
    clearInterval(this.elapsedTimer);

    try {
      const resp = await fetch(`/api/reports/${filename}`);
      if (!resp.ok) throw new Error('Report not found');
      const metrics = await resp.json();
      this._populateFromMetrics(metrics);
    } catch (e) {
      console.error('Failed to load report:', e);
    }
  },

  _populateFromMetrics(metrics) {
    // KPI
    const kpi = metrics.kpi || {};
    this.kpi = {
      turn: kpi.turns || 0,
      toolCalls: kpi.tool_calls || 0,
      toolErrors: kpi.tool_errors || 0,
      llmCalls: kpi.llm_calls || 0,
      promptTokens: kpi.prompt_tokens_est || 0,
      completionTokens: kpi.completion_tokens_est || 0,
      retryAttempts: kpi.retry_attempts || 0,
      intermediateAnswers: kpi.intermediate_answers || 0,
      elapsed: kpi.elapsed_secs || 0,
      dedupBlocks: (metrics.algorithm_stats || {}).dedup_blocks_saved || 0,
    };
    KPIPanel.update(this.kpi);

    // Algorithm events
    AlgorithmsPanel.loadFromMetrics(metrics);

    // Answers
    AnswersPanel.loadFromMetrics(metrics);

    // Tool calls
    (metrics.tool_calls || []).forEach(tc => {
      ToolCallsPanel.addCall(tc);
    });

    // Context trims
    const trims = (metrics.algorithms || {}).context_trims || [];
    trims.forEach(t => ContextPanel.addTrim(t));

    // Post-hoc stats card
    this._renderPostHocStats(metrics);
  },

  _renderPostHocStats(metrics) {
    const container = document.getElementById('posthoc-container');
    if (!container) return;

    const stats = metrics.algorithm_stats || {};
    const summary = metrics.llm_summary || {};

    let html = '<div class="card"><div class="card-title">Algorithm Effectiveness (Post-Hoc)</div>';

    const rows = [
      ['Dedup guard saved', `${stats.dedup_blocks_saved || 0} redundant tool calls`],
      ['Context trimming omitted', `results across ${stats.context_trims_performed || 0} trim events`],
      ['Arg fix corrected', `${stats.arg_fixes_applied || 0} parameter mistakes`],
      ['Bad result detection prevented', `${stats.bad_results_caught || 0} error propagations`],
      ['Context compression used', `${stats.retry_attempts_used || 0} retry attempts`],
      ['Intermediate answers extracted', `${stats.intermediate_answers_extracted || 0}`],
      ['Force-end triggered', `${stats.force_end_triggered || 0} times`],
    ];

    rows.forEach(([label, value]) => {
      html += `<div class="stat-row"><span class="stat-label">${label}</span><span class="stat-value">${value}</span></div>`;
    });

    html += '</div>';

    // LLM summary
    html += '<div class="card"><div class="card-title">LLM Summary</div>';
    html += `<div class="stat-row"><span class="stat-label">Total LLM calls</span><span class="stat-value">${summary.total_calls || 0}</span></div>`;
    html += `<div class="stat-row"><span class="stat-label">Prompt tokens (est.)</span><span class="stat-value">${summary.total_prompt_tokens_est || 0}</span></div>`;
    html += `<div class="stat-row"><span class="stat-label">Completion tokens (est.)</span><span class="stat-value">${summary.total_completion_tokens_est || 0}</span></div>`;
    html += `<div class="stat-row"><span class="stat-label">Avg LLM duration</span><span class="stat-value">${(summary.avg_duration_secs || 0).toFixed(2)}s</span></div>`;
    html += '</div>';

    // Tool usage distribution
    const toolSummary = metrics.tool_summary || {};
    if (Object.keys(toolSummary).length > 0) {
      html += '<div class="card"><div class="card-title">Tool Usage Distribution</div>';
      const maxCount = Math.max(...Object.values(toolSummary).map(s => s.count || 0), 1);
      Object.entries(toolSummary)
        .sort((a, b) => (b[1].count || 0) - (a[1].count || 0))
        .forEach(([name, s]) => {
          const pct = ((s.count || 0) / maxCount) * 100;
          const avgDur = s.count ? ((s.total_duration || 0) / s.count).toFixed(2) : '0';
          html += `<div style="margin-bottom:0.5rem;">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;">
              <span>${name}</span>
              <span style="color:var(--text-muted)">${s.count}x &middot; avg ${avgDur}s ${s.errors ? '&middot; <span style="color:var(--red)">' + s.errors + ' errors</span>' : ''}</span>
            </div>
            <div class="bar-container"><div class="bar bar-blue" style="width:${pct}%">${s.count}</div></div>
          </div>`;
        });
      html += '</div>';
    }

    container.innerHTML = html;
  },

  // ------------------------------------------------------------------
  // Session list
  // ------------------------------------------------------------------

  async _loadSessionList() {
    try {
      const resp = await fetch('/api/sessions');
      const sessions = await resp.json();
      if (sessions.length > 0) {
        // Auto-connect to the first session
        this.sessionId = sessions[0].session_id;
        window.history.replaceState(null, '', `?session=${this.sessionId}`);
        this._connectSSE(this.sessionId);
      } else {
        // Check for saved reports
        const reportResp = await fetch('/api/reports');
        const reports = await reportResp.json();
        if (reports.length > 0) {
          this._loadReport(reports[reports.length - 1].filename);
        } else {
          this._updateStatus('idle', 'No sessions active. Start the agent to see events.');
        }
      }
    } catch (e) {
      this._updateStatus('error', 'Cannot connect to dashboard server');
    }
  },

  // ------------------------------------------------------------------
  // UI Helpers
  // ------------------------------------------------------------------

  _updateStatus(status, text) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-text');
    if (dot) {
      dot.className = `status-dot ${status}`;
    }
    if (label) {
      label.textContent = text;
    }
  },

  _updateElapsed() {
    if (this.mode !== 'live') return;
    this.kpi.elapsed = (Date.now() - this.startTime) / 1000;
    KPIPanel.update(this.kpi);
  },
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
