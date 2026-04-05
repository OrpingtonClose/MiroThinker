/**
 * Algorithm Activity Panel — chronological log of all algorithm events.
 */

const AlgorithmsPanel = {
  _el: null,
  _events: [],

  init(container) {
    this._el = container;
    this._events = [];
    this._el.innerHTML = '<div class="no-data">No algorithm events yet</div>';
  },

  addEvent(event) {
    const entry = this._formatEvent(event);
    if (!entry) return;
    this._events.push(entry);
    this._renderLatest(entry);
  },

  _formatEvent(event) {
    const d = event.data || {};
    switch (event.type) {
      case 'dedup_blocked':
        return {
          badge: 'badge-red', badgeText: 'DEDUP',
          algorithm: 'Algorithm 2 — Dedup Guard',
          detail: `Blocked duplicate query: ${d.tool_name || '?'} "${(d.query_key || '').substring(0, 80)}"`,
          meta: `Turn ${event.turn} &middot; Seen ${d.previous_count || 0}x before`,
          timestamp: event.timestamp,
        };
      case 'dedup_allowed':
        return {
          badge: 'badge-orange', badgeText: 'ESCAPE',
          algorithm: 'Algorithm 2 — Dedup Guard',
          detail: `Allowed duplicate through (escape hatch): ${d.tool_name || '?'}`,
          meta: `Turn ${event.turn} &middot; ${d.consecutive_errors || 0} consecutive errors`,
          timestamp: event.timestamp,
        };
      case 'bad_result':
        return {
          badge: 'badge-orange', badgeText: 'BAD RESULT',
          algorithm: 'Algorithm 4 — Bad Result Detection',
          detail: `${d.tool_name || '?'}: ${d.reason || d.error_message || 'bad result detected'}`,
          meta: `Turn ${event.turn}`,
          timestamp: event.timestamp,
        };
      case 'context_trimmed':
        return {
          badge: 'badge-blue', badgeText: 'TRIM',
          algorithm: 'Algorithm 5 — Keep-K-Recent',
          detail: `Kept ${d.kept_results || 0}/${d.total_tool_results || 0} tool results, omitted ${d.omitted_count || 0}`,
          meta: `Turn ${event.turn}`,
          timestamp: event.timestamp,
        };
      case 'force_end':
        return {
          badge: 'badge-red', badgeText: 'FORCE END',
          algorithm: 'Algorithm 5 — Context Overflow',
          detail: `Context ~${d.estimated_tokens || '?'} tokens exceeds ${d.max_tokens || '?'} — forcing final answer`,
          meta: `Turn ${event.turn}`,
          timestamp: event.timestamp,
        };
      case 'retry_attempt':
        return {
          badge: 'badge-purple', badgeText: 'RETRY',
          algorithm: 'Algorithm 6 — Context Compression',
          detail: `Attempt ${d.attempt_number || '?'}/${d.max_attempts || '?'}`,
          meta: d.failure_summary_preview || '',
          timestamp: event.timestamp,
        };
      case 'failure_summary':
        return {
          badge: 'badge-purple', badgeText: 'FAILURE',
          algorithm: 'Algorithm 6 — Failure Summary',
          detail: `Summary: ${(d.summary || '').substring(0, 150)}${(d.summary || '').length > 150 ? '...' : ''}`,
          meta: `Turn ${event.turn}`,
          timestamp: event.timestamp,
        };
      case 'boxed_extracted':
        return {
          badge: 'badge-green', badgeText: 'BOXED',
          algorithm: 'Algorithm 7 — Boxed Extraction',
          detail: `Intermediate answer: "${(d.content || '').substring(0, 100)}"`,
          meta: `Turn ${event.turn}`,
          timestamp: event.timestamp,
        };
      case 'arg_fix_applied':
        return {
          badge: 'badge-yellow', badgeText: 'ARG FIX',
          algorithm: 'Algorithm 8 — Arg Fix',
          detail: `Fixed ${d.tool_name || '?'}: ${d.fix_description || ''}`,
          meta: `Turn ${event.turn}`,
          timestamp: event.timestamp,
        };
      default:
        return null;
    }
  },

  _renderLatest(entry) {
    // Remove no-data placeholder
    const noData = this._el.querySelector('.no-data');
    if (noData) noData.remove();

    const div = document.createElement('div');
    div.className = 'algo-event';
    div.innerHTML = `
      <span class="badge ${entry.badge}">${entry.badgeText}</span>
      <div>
        <div class="algo-event-detail">${entry.detail}</div>
        <div class="algo-event-meta">${entry.algorithm} &middot; ${entry.meta}</div>
      </div>
    `;
    this._el.appendChild(div);
    this._el.scrollTop = this._el.scrollHeight;
  },

  /** Load events from a full metrics dict (post-hoc mode). */
  loadFromMetrics(metrics) {
    this._events = [];
    this._el.innerHTML = '';

    const algos = metrics.algorithms || {};
    const allEvents = [];

    // Collect all algorithm events with timestamps
    (algos.dedup_blocks || []).forEach(e =>
      allEvents.push({ type: 'dedup_blocked', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.dedup_escapes || []).forEach(e =>
      allEvents.push({ type: 'dedup_allowed', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.arg_fixes || []).forEach(e =>
      allEvents.push({ type: 'arg_fix_applied', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.bad_results || []).forEach(e =>
      allEvents.push({ type: 'bad_result', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.context_trims || []).forEach(e =>
      allEvents.push({ type: 'context_trimmed', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.force_ends || []).forEach(e =>
      allEvents.push({ type: 'force_end', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.boxed_answers || []).forEach(e =>
      allEvents.push({ type: 'boxed_extracted', data: e, timestamp: e.timestamp, turn: e.turn }));
    (algos.retry_attempts || []).forEach(e =>
      allEvents.push({ type: 'retry_attempt', data: e, timestamp: e.timestamp, turn: 0 }));

    // Sort by timestamp
    allEvents.sort((a, b) => (a.timestamp || 0) - (b.timestamp || 0));

    if (allEvents.length === 0) {
      this._el.innerHTML = '<div class="no-data">No algorithm events recorded</div>';
      return;
    }

    allEvents.forEach(e => this.addEvent(e));
  }
};

window.AlgorithmsPanel = AlgorithmsPanel;
