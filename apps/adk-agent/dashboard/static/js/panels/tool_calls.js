/**
 * Tool Calls Panel — table of all tool calls with timing and status.
 */

const ToolCallsPanel = {
  _el: null,
  _calls: [],

  init(container) {
    this._el = container;
    this._calls = [];
    this._render();
  },

  addCall(data) {
    this._calls.push(data);
    this._render();
  },

  updateCall(callId, updates) {
    const call = this._calls.find(c => c.call_id === callId);
    if (call) {
      Object.assign(call, updates);
      this._render();
    }
  },

  _render() {
    if (this._calls.length === 0) {
      this._el.innerHTML = '<div class="no-data">No tool calls yet</div>';
      return;
    }

    let html = `<table>
      <thead><tr>
        <th>#</th><th>Tool</th><th>Agent</th><th>Turn</th>
        <th>Duration</th><th>Result</th><th>Status</th>
      </tr></thead><tbody>`;

    this._calls.forEach((c, i) => {
      const status = c.error
        ? '<span class="badge badge-red">Error</span>'
        : c.was_dedup_blocked
          ? '<span class="badge badge-orange">Dedup</span>'
          : c.was_bad_result
            ? '<span class="badge badge-orange">Bad</span>'
            : c.duration_secs !== undefined
              ? '<span class="badge badge-green">OK</span>'
              : '<span class="badge badge-muted">Running</span>';

      const flags = [];
      if (c.arg_fix_applied) flags.push('<span class="badge badge-yellow">ArgFix</span>');
      if (c.was_dedup_blocked) flags.push('<span class="badge badge-red">Dedup</span>');

      html += `<tr>
        <td>${i + 1}</td>
        <td><strong>${this._esc(c.tool_name || '?')}</strong> ${flags.join(' ')}</td>
        <td>${this._esc(c.agent_name || '')}</td>
        <td>${c.turn || ''}</td>
        <td>${c.duration_secs !== undefined ? c.duration_secs.toFixed(2) + 's' : '...'}</td>
        <td>${c.result_size_chars ? c.result_size_chars + ' chars' : c.error ? this._esc(c.error).substring(0, 50) : '—'}</td>
        <td>${status}</td>
      </tr>`;
    });

    html += '</tbody></table>';

    // Summary
    const total = this._calls.length;
    const errors = this._calls.filter(c => c.error || c.was_bad_result).length;
    const dedup = this._calls.filter(c => c.was_dedup_blocked).length;
    const avgDuration = this._calls.filter(c => c.duration_secs).reduce((s, c) => s + c.duration_secs, 0) / (total || 1);

    html += `<div style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">
      Total: ${total} &middot; Errors: ${errors} &middot; Dedup blocked: ${dedup} &middot; Avg duration: ${avgDuration.toFixed(2)}s
    </div>`;

    this._el.innerHTML = html;
  },

  _esc(s) {
    const el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
  }
};

window.ToolCallsPanel = ToolCallsPanel;
