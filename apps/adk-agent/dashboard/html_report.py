# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""Generate a self-contained HTML dashboard from finalized pipeline data.

The output is a single ``.html`` file with embedded CSS and JS (Chart.js
loaded from CDN).  It can be opened directly in any browser — no build
step, no server, works offline (except for the CDN chart library).

Sections:
  - Header: query, timing, phase outcomes
  - KPI cards: events, tool calls, LLM calls, corpus atoms, elapsed time
  - Timeline: Gantt-style chart of phases over wall-clock time
  - Tool breakdown: table of each tool call with timing + args
  - Algorithm activity: dedup blocks, arg fixes, bad results, compressions
  - Corpus growth: chart of atoms admitted per iteration
  - Event stream: scrollable log of all raw events
"""

from __future__ import annotations

import html
import json
import time
from typing import Any


def generate_dashboard_html(data: dict[str, Any]) -> str:
    """Return a self-contained HTML string for the pipeline dashboard."""
    query = html.escape(data.get("query", "")[:200])
    session_id = html.escape(data.get("session_id", "")[:16])
    elapsed = data.get("elapsed_secs", 0)
    kpi = data.get("kpi", {})
    phases = data.get("phases", [])
    tool_calls = data.get("tool_calls", [])
    tool_summary = data.get("tool_summary", {})
    algorithms = data.get("algorithms", {})
    corpus_updates = data.get("corpus_updates", [])
    llm_calls = data.get("llm_calls", [])
    events = data.get("events", [])
    stall_events = data.get("stall_events", [])
    thinker_escalated = data.get("thinker_escalated", False)
    started_at = data.get("started_at", 0)

    # Format started_at as human-readable
    started_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(started_at))

    # Build phase rows
    phase_rows = ""
    for p in phases:
        dur = p.get("end_time", 0) - p.get("start_time", 0)
        if dur < 0:
            dur = 0
        outcome_class = "success" if p.get("outcome") == "ok" else "warning"
        phase_rows += f"""
        <tr>
          <td><strong>{html.escape(p.get('phase', ''))}</strong></td>
          <td>{html.escape(p.get('agent', ''))}</td>
          <td>{dur:.1f}s</td>
          <td><span class="badge {outcome_class}">{html.escape(p.get('outcome', 'running'))}</span></td>
        </tr>"""

    # Build tool call rows
    tool_rows = ""
    for i, t in enumerate(tool_calls):
        error_class = ' class="error-row"' if t.get("error") else ""
        badges = ""
        if t.get("was_dedup_blocked"):
            badges += '<span class="badge warning">DEDUP</span> '
        if t.get("arg_fix_applied"):
            badges += '<span class="badge info">ARG-FIX</span> '
        if t.get("was_compressed"):
            badges += f'<span class="badge info">COMPRESSED {t.get("original_chars", 0)}→{t.get("result_chars", 0)}</span> '
        if t.get("error"):
            badges += f'<span class="badge danger">{html.escape(t["error"][:80])}</span>'

        tool_rows += f"""
        <tr{error_class}>
          <td>{i + 1}</td>
          <td><code>{html.escape(t.get('tool_name', ''))}</code></td>
          <td>{html.escape(t.get('agent', ''))}</td>
          <td class="mono">{t.get('duration_secs', 0):.2f}s</td>
          <td>{t.get('result_chars', 0):,}</td>
          <td class="args-cell">{html.escape(t.get('args_summary', '')[:120])}</td>
          <td>{badges}</td>
        </tr>"""

    # Build tool summary rows
    tool_summary_rows = ""
    for name, s in sorted(tool_summary.items(), key=lambda x: -x[1].get("count", 0)):
        avg = s["total_duration"] / max(s["count"], 1)
        tool_summary_rows += f"""
        <tr>
          <td><code>{html.escape(name)}</code></td>
          <td>{s['count']}</td>
          <td>{s['total_duration']:.2f}s</td>
          <td>{avg:.2f}s</td>
          <td>{s.get('total_result_chars', 0):,}</td>
          <td>{s.get('errors', 0)}</td>
        </tr>"""

    # Build LLM call rows
    llm_rows = ""
    for i, r in enumerate(llm_calls):
        llm_rows += f"""
        <tr>
          <td>{i + 1}</td>
          <td>{html.escape(r.get('agent', ''))}</td>
          <td class="mono">{r.get('duration_secs', 0):.2f}s</td>
          <td>{r.get('prompt_tokens_est', 0):,}</td>
          <td>{r.get('completion_tokens_est', 0):,}</td>
        </tr>"""

    # Build algorithm activity
    algo_sections = ""

    dedup_blocks = algorithms.get("dedup_blocks", [])
    if dedup_blocks:
        rows = ""
        for d in dedup_blocks:
            rows += f"""<tr>
              <td><code>{html.escape(d.get('tool_name', ''))}</code></td>
              <td>{html.escape(d.get('query_key', '')[:80])}</td>
              <td>{d.get('consecutive', 0)}</td>
            </tr>"""
        algo_sections += f"""
        <h3>Dedup Blocks ({len(dedup_blocks)})</h3>
        <table><thead><tr><th>Tool</th><th>Query Key</th><th>Consecutive</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    arg_fixes = algorithms.get("arg_fixes", [])
    if arg_fixes:
        rows = ""
        for a in arg_fixes:
            rows += f"""<tr>
              <td><code>{html.escape(a.get('tool_name', ''))}</code></td>
              <td>{html.escape(', '.join(a.get('fixes', [])))}</td>
            </tr>"""
        algo_sections += f"""
        <h3>Arg Fixes ({len(arg_fixes)})</h3>
        <table><thead><tr><th>Tool</th><th>Fixes Applied</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    compressions = algorithms.get("compressions", [])
    if compressions:
        rows = ""
        for c in compressions:
            rows += f"""<tr>
              <td><code>{html.escape(c.get('tool_name', ''))}</code></td>
              <td>{c.get('original_chars', 0):,}</td>
              <td>{c.get('compressed_chars', 0):,}</td>
              <td>{c.get('ratio', 0):.1%}</td>
            </tr>"""
        algo_sections += f"""
        <h3>Compressions ({len(compressions)})</h3>
        <table><thead><tr><th>Tool</th><th>Original</th><th>Compressed</th><th>Ratio</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    keep_k_trims = algorithms.get("keep_k_trims", [])
    if keep_k_trims:
        rows = ""
        for k in keep_k_trims:
            rows += f"""<tr>
              <td>{k.get('kept', 0)}</td>
              <td>{k.get('omitted', 0)}</td>
              <td>{k.get('utilisation', 0):.1%}</td>
            </tr>"""
        algo_sections += f"""
        <h3>Keep-K Trims ({len(keep_k_trims)})</h3>
        <table><thead><tr><th>Kept</th><th>Omitted</th><th>Utilisation</th></tr></thead>
        <tbody>{rows}</tbody></table>"""

    if not algo_sections:
        algo_sections = "<p class='muted'>No algorithm interventions during this run.</p>"

    # Corpus growth data for chart
    corpus_labels = json.dumps([f"iter {c.get('iteration', i)}" for i, c in enumerate(corpus_updates)])
    corpus_data = json.dumps([c.get("total", 0) for c in corpus_updates])
    corpus_admitted = json.dumps([c.get("admitted", 0) for c in corpus_updates])

    # Timeline data for chart
    timeline_labels = json.dumps([p.get("phase", "") for p in phases])
    timeline_starts = json.dumps([
        round(p.get("start_time", started_at) - started_at, 1) for p in phases
    ])
    timeline_durations = json.dumps([
        round((p.get("end_time", 0) or time.time()) - p.get("start_time", started_at), 1)
        for p in phases
    ])

    # Event stream (last 500)
    event_rows = ""
    display_events = events[-500:] if len(events) > 500 else events
    for e in display_events:
        ts = e.get("timestamp", 0) - started_at
        event_rows += f"""<tr>
          <td class="mono">{ts:+.2f}s</td>
          <td><span class="badge {_event_badge_class(e.get('event_type', ''))}">{html.escape(e.get('event_type', ''))}</span></td>
          <td>{html.escape(e.get('agent', ''))}</td>
          <td>{html.escape(e.get('phase', ''))}</td>
          <td class="args-cell">{html.escape(json.dumps(e.get('data', {}), default=str)[:200])}</td>
        </tr>"""

    # Stall warning banner
    stall_banner = ""
    if stall_events:
        s = stall_events[0]
        stall_banner = f"""
        <div class="banner danger">
          Pipeline stalled at {s.get('agent', 'unknown')} after {s.get('event_count', 0)} events
          (timeout: {s.get('timeout', 0):.0f}s)
        </div>"""

    escalation_banner = ""
    if thinker_escalated:
        esc_time = data.get("thinker_escalate_time", 0) - started_at
        escalation_banner = f"""
        <div class="banner success">
          Thinker signalled EVIDENCE_SUFFICIENT at +{esc_time:.1f}s
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MiroThinker Pipeline Dashboard — {session_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --success: #3fb950; --warning: #d29922; --danger: #f85149;
    --info: #79c0ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: 20px; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.6em; margin-bottom: 4px; }}
  h2 {{ font-size: 1.3em; margin: 24px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ font-size: 1.1em; margin: 16px 0 8px; color: var(--accent); }}
  .header {{ margin-bottom: 20px; }}
  .header .meta {{ color: var(--muted); font-size: 0.9em; }}
  .query {{ background: var(--surface); padding: 12px; border-radius: 6px;
            border-left: 3px solid var(--accent); margin: 8px 0; font-style: italic; }}
  .kpi-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin: 16px 0;
  }}
  .kpi-card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; text-align: center;
  }}
  .kpi-card .value {{ font-size: 2em; font-weight: 700; color: var(--accent); }}
  .kpi-card .label {{ font-size: 0.8em; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 0.85em;
  }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--surface); color: var(--muted); font-weight: 600; text-transform: uppercase;
       font-size: 0.75em; letter-spacing: 0.5px; }}
  tr:hover {{ background: rgba(88, 166, 255, 0.05); }}
  .error-row {{ background: rgba(248, 81, 73, 0.1); }}
  .mono {{ font-family: 'SF Mono', 'Fira Code', monospace; }}
  code {{ background: var(--surface); padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;
  }}
  .badge.success {{ background: rgba(63, 185, 80, 0.2); color: var(--success); }}
  .badge.warning {{ background: rgba(210, 153, 34, 0.2); color: var(--warning); }}
  .badge.danger {{ background: rgba(248, 81, 73, 0.2); color: var(--danger); }}
  .badge.info {{ background: rgba(121, 192, 255, 0.2); color: var(--info); }}
  .badge.phase_start {{ background: rgba(88, 166, 255, 0.2); color: var(--accent); }}
  .badge.phase_end {{ background: rgba(63, 185, 80, 0.2); color: var(--success); }}
  .badge.adk_event {{ background: rgba(139, 148, 158, 0.15); color: var(--muted); }}
  .badge.tool_call_start {{ background: rgba(210, 153, 34, 0.2); color: var(--warning); }}
  .badge.tool_call_end {{ background: rgba(210, 153, 34, 0.2); color: var(--warning); }}
  .badge.llm_call_start {{ background: rgba(121, 192, 255, 0.2); color: var(--info); }}
  .badge.llm_call_end {{ background: rgba(121, 192, 255, 0.2); color: var(--info); }}
  .badge.thinker_escalate {{ background: rgba(63, 185, 80, 0.2); color: var(--success); }}
  .badge.stall_detected {{ background: rgba(248, 81, 73, 0.2); color: var(--danger); }}
  .badge.corpus_update {{ background: rgba(163, 113, 247, 0.2); color: #a371f7; }}
  .banner {{
    padding: 12px 16px; border-radius: 6px; margin: 12px 0; font-weight: 600;
  }}
  .banner.danger {{ background: rgba(248, 81, 73, 0.15); border: 1px solid var(--danger); color: var(--danger); }}
  .banner.success {{ background: rgba(63, 185, 80, 0.15); border: 1px solid var(--success); color: var(--success); }}
  .args-cell {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .chart-container {{ background: var(--surface); border-radius: 8px; padding: 16px; margin: 12px 0; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 800px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .event-scroll {{ max-height: 500px; overflow-y: auto; }}
  .muted {{ color: var(--muted); }}
  .section {{ margin-bottom: 24px; }}
  canvas {{ max-height: 300px; }}
</style>
</head>
<body>

<div class="header">
  <h1>MiroThinker Pipeline Dashboard</h1>
  <div class="meta">Session: {session_id} | Started: {started_str} | Elapsed: {elapsed:.1f}s</div>
  <div class="query">{query}</div>
</div>

{stall_banner}
{escalation_banner}

<div class="kpi-grid">
  <div class="kpi-card"><div class="value">{kpi.get('adk_events', 0)}</div><div class="label">ADK Events</div></div>
  <div class="kpi-card"><div class="value">{kpi.get('tool_calls', 0)}</div><div class="label">Tool Calls</div></div>
  <div class="kpi-card"><div class="value">{kpi.get('llm_calls', 0)}</div><div class="label">LLM Calls</div></div>
  <div class="kpi-card"><div class="value">{kpi.get('prompt_tokens_est', 0):,}</div><div class="label">Prompt Tokens</div></div>
  <div class="kpi-card"><div class="value">{kpi.get('completion_tokens_est', 0):,}</div><div class="label">Completion Tokens</div></div>
  <div class="kpi-card"><div class="value">{kpi.get('text_chars', 0):,}</div><div class="label">Text Chars</div></div>
  <div class="kpi-card"><div class="value">{kpi.get('reasoning_chars', 0):,}</div><div class="label">Reasoning Chars</div></div>
  <div class="kpi-card"><div class="value">{elapsed:.1f}s</div><div class="label">Wall Clock</div></div>
</div>

<div class="section">
  <h2>Phases</h2>
  <table>
    <thead><tr><th>Phase</th><th>Agent</th><th>Duration</th><th>Outcome</th></tr></thead>
    <tbody>{phase_rows if phase_rows else '<tr><td colspan="4" class="muted">No phases recorded</td></tr>'}</tbody>
  </table>
</div>

<div class="section">
  <h2>Charts</h2>
  <div class="chart-row">
    <div class="chart-container">
      <h3>Phase Timeline</h3>
      <canvas id="timelineChart"></canvas>
    </div>
    <div class="chart-container">
      <h3>Corpus Growth</h3>
      <canvas id="corpusChart"></canvas>
    </div>
  </div>
</div>

<div class="section">
  <h2>Tool Summary</h2>
  <table>
    <thead><tr><th>Tool</th><th>Calls</th><th>Total Time</th><th>Avg Time</th><th>Result Chars</th><th>Errors</th></tr></thead>
    <tbody>{tool_summary_rows if tool_summary_rows else '<tr><td colspan="6" class="muted">No tool calls</td></tr>'}</tbody>
  </table>
</div>

<div class="section">
  <h2>Tool Calls ({len(tool_calls)})</h2>
  <table>
    <thead><tr><th>#</th><th>Tool</th><th>Agent</th><th>Duration</th><th>Result</th><th>Args</th><th>Status</th></tr></thead>
    <tbody>{tool_rows if tool_rows else '<tr><td colspan="7" class="muted">No tool calls</td></tr>'}</tbody>
  </table>
</div>

<div class="section">
  <h2>LLM Calls ({len(llm_calls)})</h2>
  <table>
    <thead><tr><th>#</th><th>Agent</th><th>Duration</th><th>Prompt Tokens</th><th>Completion Tokens</th></tr></thead>
    <tbody>{llm_rows if llm_rows else '<tr><td colspan="5" class="muted">No LLM calls</td></tr>'}</tbody>
  </table>
</div>

<div class="section">
  <h2>Algorithm Activity</h2>
  {algo_sections}
</div>

<div class="section">
  <h2>Event Stream ({len(events)} events)</h2>
  <div class="event-scroll">
    <table>
      <thead><tr><th>Time</th><th>Type</th><th>Agent</th><th>Phase</th><th>Data</th></tr></thead>
      <tbody>{event_rows if event_rows else '<tr><td colspan="5" class="muted">No events</td></tr>'}</tbody>
    </table>
  </div>
</div>

<script>
// Phase timeline chart
const timelineCtx = document.getElementById('timelineChart');
if (timelineCtx) {{
  const labels = {timeline_labels};
  const starts = {timeline_starts};
  const durations = {timeline_durations};
  new Chart(timelineCtx, {{
    type: 'bar',
    data: {{
      labels: labels,
      datasets: [
        {{
          label: 'Start offset (s)',
          data: starts,
          backgroundColor: 'transparent',
          borderWidth: 0,
        }},
        {{
          label: 'Duration (s)',
          data: durations,
          backgroundColor: ['rgba(88, 166, 255, 0.6)', 'rgba(63, 185, 80, 0.6)',
                           'rgba(210, 153, 34, 0.6)', 'rgba(163, 113, 247, 0.6)'],
          borderRadius: 4,
        }}
      ]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{
          stacked: true,
          title: {{ display: true, text: 'Seconds', color: '#8b949e' }},
          grid: {{ color: '#30363d' }},
          ticks: {{ color: '#8b949e' }},
        }},
        y: {{
          stacked: true,
          grid: {{ display: false }},
          ticks: {{ color: '#e6edf3' }},
        }}
      }}
    }}
  }});
}}

// Corpus growth chart
const corpusCtx = document.getElementById('corpusChart');
if (corpusCtx) {{
  const corpusLabels = {corpus_labels};
  const corpusTotal = {corpus_data};
  const corpusAdmitted = {corpus_admitted};
  if (corpusLabels.length > 0) {{
    new Chart(corpusCtx, {{
      type: 'line',
      data: {{
        labels: corpusLabels,
        datasets: [
          {{
            label: 'Total Atoms',
            data: corpusTotal,
            borderColor: '#a371f7',
            backgroundColor: 'rgba(163, 113, 247, 0.1)',
            fill: true,
            tension: 0.3,
          }},
          {{
            label: 'Admitted This Iter',
            data: corpusAdmitted,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88, 166, 255, 0.1)',
            fill: true,
            tension: 0.3,
          }}
        ]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
        scales: {{
          x: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }} }},
          y: {{ grid: {{ color: '#30363d' }}, ticks: {{ color: '#8b949e' }},
               title: {{ display: true, text: 'Atoms', color: '#8b949e' }} }}
        }}
      }}
    }});
  }} else {{
    corpusCtx.parentElement.innerHTML += '<p class="muted" style="text-align:center">No corpus data</p>';
  }}
}}
</script>

</body>
</html>"""


def _event_badge_class(event_type: str) -> str:
    """Return CSS class for event type badge."""
    return event_type if event_type in (
        "phase_start", "phase_end", "adk_event",
        "tool_call_start", "tool_call_end",
        "llm_call_start", "llm_call_end",
        "thinker_escalate", "stall_detected", "corpus_update",
    ) else "info"
