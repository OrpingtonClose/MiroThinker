/**
 * Agent Hierarchy Panel — live agent tree with active/completed highlighting.
 */

const AgentGraphPanel = {
  _el: null,
  _agents: {},  // name -> { status: 'idle'|'active'|'completed', children: [] }

  init(container) {
    this._el = container;
    this._agents = {
      'research_agent': { status: 'idle', children: ['browsing_agent', 'summary_agent'] },
      'browsing_agent': { status: 'idle', children: [] },
      'summary_agent': { status: 'idle', children: [] },
    };
    this._render();
  },

  agentStarted(name) {
    if (this._agents[name]) {
      this._agents[name].status = 'active';
    } else {
      this._agents[name] = { status: 'active', children: [] };
    }
    this._render();
  },

  agentEnded(name) {
    if (this._agents[name]) {
      this._agents[name].status = 'completed';
    }
    this._render();
  },

  reset() {
    Object.keys(this._agents).forEach(k => this._agents[k].status = 'idle');
    this._render();
  },

  _render() {
    let html = '<div class="card"><div class="card-title">Agent Hierarchy</div>';
    html += '<div style="text-align:center;padding:1rem 0;">';

    // Main agent
    html += this._renderNode('research_agent');
    html += '<div style="margin:0.5rem 0;color:var(--text-muted);font-size:1.2rem;">';
    html += '&#8595; delegates to &#8595;';
    html += '</div>';
    html += '<div style="display:flex;justify-content:center;gap:2rem;">';
    html += this._renderNode('browsing_agent');
    html += this._renderNode('summary_agent');
    html += '</div>';

    html += '</div></div>';

    // Active agents list
    const active = Object.entries(this._agents)
      .filter(([, v]) => v.status === 'active')
      .map(([k]) => k);
    if (active.length > 0) {
      html += `<div style="font-size:0.8rem;color:var(--text-muted);margin-top:0.5rem;">`;
      html += `Active: ${active.map(a => `<span class="badge badge-green">${a}</span>`).join(' ')}`;
      html += '</div>';
    }

    this._el.innerHTML = html;
  },

  _renderNode(name) {
    const agent = this._agents[name];
    const status = agent ? agent.status : 'idle';
    const label = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `<span class="agent-node ${status}">${label}</span>`;
  }
};

window.AgentGraphPanel = AgentGraphPanel;
