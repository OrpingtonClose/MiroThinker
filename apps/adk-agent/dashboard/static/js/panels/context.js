/**
 * Context Window Panel — visualization of message history and trimming.
 */

const ContextPanel = {
  _el: null,
  _trims: [],
  _tokenEstimate: 0,
  _maxTokens: 128000,

  init(container) {
    this._el = container;
    this._trims = [];
    this._render();
  },

  addTrim(data) {
    this._trims.push(data);
    this._render();
  },

  updateTokenEstimate(estimated, max) {
    this._tokenEstimate = estimated;
    this._maxTokens = max || this._maxTokens;
    this._render();
  },

  _render() {
    let html = '';

    // Token usage bar
    const pct = this._maxTokens > 0
      ? Math.min(100, (this._tokenEstimate / this._maxTokens) * 100)
      : 0;
    const barColor = pct > 90 ? 'bar-red' : pct > 70 ? 'bar-orange' : 'bar-blue';

    html += '<div class="card">';
    html += '<div class="card-title">Context Window Usage</div>';
    html += `<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:0.25rem;">`;
    html += `<span>~${this._formatTokens(this._tokenEstimate)} tokens</span>`;
    html += `<span style="color:var(--text-muted)">${this._formatTokens(this._maxTokens)} max</span>`;
    html += `</div>`;
    html += `<div class="bar-container"><div class="bar ${barColor}" style="width:${pct}%">${pct.toFixed(0)}%</div></div>`;
    html += '</div>';

    // Context composition visualization
    if (this._trims.length > 0) {
      const latest = this._trims[this._trims.length - 1];
      const kept = latest.kept_results || 0;
      const omitted = latest.omitted_count || 0;
      const total = kept + omitted;

      html += '<div class="card">';
      html += '<div class="card-title">Message Composition</div>';
      html += '<div class="context-bar">';

      // First message (user task) — always kept
      html += '<div class="context-block user" style="height:100%;flex:2;" title="User task (always kept)"></div>';

      // Tool results
      if (total > 0) {
        for (let i = 0; i < total; i++) {
          const isKept = i >= omitted;
          const cls = isKept ? 'tool-kept' : 'tool-omitted';
          const h = isKept ? 80 + Math.random() * 20 : 20;
          const title = isKept ? `Tool result ${i + 1} (kept)` : `Tool result ${i + 1} (omitted)`;
          html += `<div class="context-block ${cls}" style="height:${h}%;" title="${title}"></div>`;
        }
      }

      html += '</div>';

      // Legend
      html += '<div class="context-legend">';
      html += '<span><span class="context-legend-dot" style="background:var(--green)"></span>User task</span>';
      html += '<span><span class="context-legend-dot" style="background:var(--cyan)"></span>Kept results (' + kept + ')</span>';
      html += '<span><span class="context-legend-dot" style="background:var(--surface2)"></span>Omitted (' + omitted + ')</span>';
      html += '</div>';
      html += '</div>';
    }

    // Trim history table
    if (this._trims.length > 0) {
      html += '<div class="card">';
      html += '<div class="card-title">Trim History</div>';
      html += '<table><thead><tr><th>Turn</th><th>Total Results</th><th>Kept</th><th>Omitted</th></tr></thead><tbody>';
      this._trims.forEach(t => {
        html += `<tr>
          <td>${t.turn || '?'}</td>
          <td>${t.total_tool_results || '?'}</td>
          <td>${t.kept_results || '?'}</td>
          <td>${t.omitted_count || '?'}</td>
        </tr>`;
      });
      html += '</tbody></table></div>';
    }

    if (!html) {
      html = '<div class="no-data">No context trimming events yet</div>';
    }

    this._el.innerHTML = html;
  },

  _formatTokens(n) {
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return n.toString();
  }
};

window.ContextPanel = ContextPanel;
