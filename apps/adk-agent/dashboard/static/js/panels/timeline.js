/**
 * Turn Timeline Panel — chronological view of turns with algorithm events.
 */

const TimelinePanel = {
  _el: null,
  _entries: [],
  _startTime: null,

  init(container) {
    this._el = container;
    this._entries = [];
    this._startTime = Date.now() / 1000;
    this._el.innerHTML = '<div class="no-data">Waiting for events...</div>';
  },

  addEntry(event) {
    if (!this._startTime) this._startTime = event.timestamp;

    const entry = {
      type: event.type,
      timestamp: event.timestamp,
      turn: event.turn,
      agent: event.agent || '',
      data: event.data || {},
      relTime: (event.timestamp - this._startTime).toFixed(2),
    };
    this._entries.push(entry);
    this._renderEntry(entry);
  },

  _renderEntry(entry) {
    // Remove no-data placeholder
    const noData = this._el.querySelector('.no-data');
    if (noData) noData.remove();

    const div = document.createElement('div');
    const cssClass = this._getEntryClass(entry.type);
    div.className = `timeline-entry ${cssClass}`;
    div.onclick = () => div.classList.toggle('expanded');

    const { icon, label, detail } = this._formatEntry(entry);

    div.innerHTML = `
      <div class="timeline-time">+${entry.relTime}s &middot; Turn ${entry.turn} ${entry.agent ? '&middot; ' + entry.agent : ''}</div>
      <div class="timeline-content">${icon} ${label}</div>
      <div class="timeline-detail">${detail}</div>
    `;
    this._el.appendChild(div);
    // Auto-scroll to bottom
    this._el.scrollTop = this._el.scrollHeight;
  },

  _getEntryClass(type) {
    const map = {
      tool_call_start: 'tool', tool_call_end: 'tool',
      dedup_blocked: 'error', dedup_allowed: 'algorithm',
      arg_fix_applied: 'algorithm', bad_result: 'error',
      context_trimmed: 'algorithm', force_end: 'error',
      boxed_extracted: 'answer', retry_attempt: 'algorithm',
      failure_summary: 'algorithm',
      llm_call_start: '', llm_call_end: '',
      turn_start: '', turn_end: '',
      agent_start: '', agent_end: '',
    };
    return map[type] || '';
  },

  _formatEntry(entry) {
    const d = entry.data;
    switch (entry.type) {
      case 'turn_start':
        return { icon: '', label: `<strong>Turn ${entry.turn}</strong> started`, detail: '' };
      case 'tool_call_start':
        return {
          icon: '', label: `Tool: <strong>${d.tool_name || '?'}</strong>`,
          detail: d.arguments_summary ? `<pre>${this._esc(d.arguments_summary)}</pre>` : ''
        };
      case 'tool_call_end':
        return {
          icon: '', label: `Tool done: <strong>${d.tool_name || '?'}</strong> (${d.duration_secs || '?'}s)`,
          detail: d.error ? `<span style="color:var(--red)">Error: ${this._esc(d.error)}</span>` :
                  `Result: ${d.result_size_chars || 0} chars`
        };
      case 'dedup_blocked':
        return {
          icon: '', label: `<span class="badge badge-red">DEDUP</span> Blocked: ${this._esc(d.tool_name || '')}`,
          detail: `Query key: ${this._esc(d.query_key || '')} (seen ${d.previous_count || 0}x)`
        };
      case 'dedup_allowed':
        return {
          icon: '', label: `<span class="badge badge-orange">DEDUP ESC</span> Allowed duplicate: ${this._esc(d.tool_name || '')}`,
          detail: `Escape hatch after ${d.consecutive_errors || 0} consecutive errors`
        };
      case 'arg_fix_applied':
        return {
          icon: '', label: `<span class="badge badge-yellow">ARG FIX</span> ${this._esc(d.tool_name || '')}`,
          detail: `${this._esc(d.fix_description || '')}`
        };
      case 'bad_result':
        return {
          icon: '', label: `<span class="badge badge-orange">BAD RESULT</span> ${this._esc(d.tool_name || '')}`,
          detail: `Reason: ${this._esc(d.reason || d.error_message || '')}`
        };
      case 'context_trimmed':
        return {
          icon: '', label: `<span class="badge badge-blue">TRIM</span> Kept ${d.kept_results || 0}/${d.total_tool_results || 0} tool results`,
          detail: `Omitted ${d.omitted_count || 0} older results`
        };
      case 'force_end':
        return {
          icon: '', label: `<span class="badge badge-red">FORCE END</span> Context overflow`,
          detail: `Est. tokens: ${d.estimated_tokens || '?'} / ${d.max_tokens || '?'}`
        };
      case 'boxed_extracted':
        return {
          icon: '', label: `<span class="badge badge-green">BOXED</span> Answer found`,
          detail: `<code>${this._esc(d.content || '')}</code>`
        };
      case 'retry_attempt':
        return {
          icon: '', label: `<span class="badge badge-purple">RETRY</span> Attempt ${d.attempt_number || '?'}/${d.max_attempts || '?'}`,
          detail: d.failure_summary_preview ? `Summary: ${this._esc(d.failure_summary_preview)}` : ''
        };
      case 'failure_summary':
        return {
          icon: '', label: `<span class="badge badge-purple">FAILURE</span> Summary generated`,
          detail: `<pre>${this._esc((d.summary || '').substring(0, 500))}</pre>`
        };
      case 'llm_call_start':
        return { icon: '', label: `LLM call started`, detail: '' };
      case 'llm_call_end':
        return {
          icon: '', label: `LLM call done (${d.duration_secs || '?'}s)`,
          detail: `Tokens: ~${d.prompt_tokens_est || 0} in / ~${d.completion_tokens_est || 0} out`
        };
      case 'agent_start':
        return { icon: '', label: `Agent <strong>${entry.agent}</strong> started`, detail: '' };
      case 'agent_end':
        return { icon: '', label: `Agent <strong>${entry.agent}</strong> ended`, detail: '' };
      default:
        return { icon: '', label: entry.type, detail: JSON.stringify(d) };
    }
  },

  _esc(s) {
    const el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
  },

  clear() {
    this._entries = [];
    this._startTime = null;
    this._el.innerHTML = '<div class="no-data">Waiting for events...</div>';
  }
};

window.TimelinePanel = TimelinePanel;
