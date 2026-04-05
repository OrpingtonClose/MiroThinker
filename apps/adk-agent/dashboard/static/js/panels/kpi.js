/**
 * KPI Cards Panel — top-row stat cards.
 */

const KPIPanel = {
  _el: null,

  init(container) {
    this._el = container;
    this._render({
      turn: 0, toolCalls: 0, toolErrors: 0, llmCalls: 0,
      promptTokens: 0, completionTokens: 0, retryAttempts: 0,
      intermediateAnswers: 0, elapsed: 0, dedupBlocks: 0,
    });
  },

  update(kpi) {
    this._render(kpi);
  },

  _render(k) {
    const cards = [
      { value: k.turn || 0, label: 'Turn', cls: '' },
      { value: k.toolCalls || 0, label: 'Tool Calls', cls: '' },
      { value: k.toolErrors || 0, label: 'Tool Errors', cls: k.toolErrors > 0 ? 'red' : '' },
      { value: k.llmCalls || 0, label: 'LLM Calls', cls: '' },
      { value: this._formatTokens(k.promptTokens || 0), label: 'Prompt Tokens', cls: '' },
      { value: this._formatTokens(k.completionTokens || 0), label: 'Completion Tokens', cls: '' },
      { value: k.dedupBlocks || 0, label: 'Dedup Blocks', cls: k.dedupBlocks > 0 ? 'orange' : '' },
      { value: `${k.retryAttempts || 0}`, label: 'Retry Attempts', cls: k.retryAttempts > 1 ? 'yellow' : '' },
      { value: k.intermediateAnswers || 0, label: 'Boxed Answers', cls: k.intermediateAnswers > 0 ? 'green' : '' },
      { value: this._formatDuration(k.elapsed || 0), label: 'Elapsed', cls: '' },
    ];

    this._el.innerHTML = cards.map(c => `
      <div class="kpi-card">
        <div class="kpi-value ${c.cls}">${c.value}</div>
        <div class="kpi-label">${c.label}</div>
      </div>
    `).join('');
  },

  _formatTokens(n) {
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return n;
  },

  _formatDuration(secs) {
    if (secs < 60) return secs.toFixed(1) + 's';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}m ${s}s`;
  }
};

window.KPIPanel = KPIPanel;
