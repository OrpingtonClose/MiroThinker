/**
 * Intermediate Answers Panel — list of \\boxed{} answers extracted during execution.
 */

const AnswersPanel = {
  _el: null,
  _answers: [],

  init(container) {
    this._el = container;
    this._answers = [];
    this._render();
  },

  addAnswer(data) {
    this._answers.push({
      content: data.content || data.data?.content || '?',
      turn: data.turn || data.data?.turn || 0,
      timestamp: data.timestamp || 0,
      isIntermediate: true,
    });
    this._render();
  },

  setFinalAnswer(answer) {
    this._answers.push({
      content: answer,
      turn: 'final',
      timestamp: Date.now() / 1000,
      isIntermediate: false,
    });
    this._render();
  },

  _render() {
    if (this._answers.length === 0) {
      this._el.innerHTML = '<div class="no-data">No \\boxed{} answers extracted yet</div>';
      return;
    }

    let html = '';

    // Show all answers, highlight the latest as current fallback
    this._answers.forEach((a, i) => {
      const isLatest = i === this._answers.length - 1;
      const cls = isLatest ? 'answer-item latest' : 'answer-item';
      const label = a.isIntermediate
        ? `<span class="badge badge-blue">Turn ${a.turn}</span>`
        : `<span class="badge badge-green">FINAL</span>`;

      html += `
        <div class="${cls}">
          <div>
            ${label}
            <span class="answer-content">${this._esc(a.content)}</span>
          </div>
          <div class="answer-meta">
            ${isLatest && a.isIntermediate ? '<span class="badge badge-green" style="font-size:0.6rem;">Current fallback</span>' : ''}
          </div>
        </div>
      `;
    });

    // Summary
    html += `<div style="margin-top:0.75rem;font-size:0.75rem;color:var(--text-muted);">
      ${this._answers.length} answer(s) extracted during execution
    </div>`;

    this._el.innerHTML = html;
  },

  /** Load answers from a metrics dict (post-hoc mode). */
  loadFromMetrics(metrics) {
    this._answers = [];
    const algos = metrics.algorithms || {};
    (algos.boxed_answers || []).forEach(a => {
      this._answers.push({
        content: a.content || '?',
        turn: a.turn || 0,
        timestamp: a.timestamp || 0,
        isIntermediate: true,
      });
    });
    this._render();
  },

  _esc(s) {
    const el = document.createElement('span');
    el.textContent = s;
    return el.innerHTML;
  }
};

window.AnswersPanel = AnswersPanel;
