"""Self-contained HTML report for an investigation session.

Page structure (rendered when the agent calls ``write_report``):

* ``<header>``: the hero stat for the best primary metric against baseline, with
  a tally panel on the right (attempts, branches, leaves, session id)
* narrative section: the markdown the agent wrote
* exploration map: a sparkline of the primary metric over attempts and a
  colour-coded SVG tree of the parent/child structure with a legend
* attempt details: collapsible cards with rationale, parameter changes,
  metrics, and any plots referenced by ``record_attempt``
* footer: a link to the full session transcript, which ``render_report``
  writes alongside the report so the link always resolves
"""

from __future__ import annotations

import base64
import contextlib
import html
import json
import mimetypes
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from jutul_agent.trace import Event, schema
from jutul_agent.transcript.attempts import Attempt, build_attempt_tree
from jutul_agent.transcript.markdown_html import render_markdown_html

# The report is opened from disk in the user's browser and embeds agent-authored
# content (narrative, raw ``html`` blocks, ``custom_css``) that may be steered by
# untrusted input the agent read. This Content-Security-Policy makes the page
# inert and offline: no scripts (the report has none) and no network. Images and
# fonts are inlined as ``data:`` URIs, and styles are inline only. So a custom
# block can restyle and lay out, but cannot run JS, phone home, or load a pixel.
_CSP = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
    "font-src data:; base-uri 'none'; form-action 'none'"
)

_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --bg-soft: #f8fafc;
  --bg-card: #ffffff;
  --bd: #e2e8f0;
  --bd-soft: #edf2f7;
  --mu: #64748b;
  --fg: #0f172a;
  --ac: #2563eb;
  --ac-soft: rgba(37, 99, 235, 0.08);
  --tone-baseline: #c7d2fe;
  --tone-improved: #a7f3d0;
  --tone-regressed: #fecaca;
  --tone-neutral: #e2e8f0;
  --pos: #059669;
  --neg: #dc2626;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1419;
    --bg-soft: #1a2332;
    --bg-card: #131a25;
    --bd: #334155;
    --bd-soft: #1e293b;
    --mu: #94a3b8;
    --fg: #e2e8f0;
    --ac-soft: rgba(37, 99, 235, 0.18);
    --tone-baseline: #4338ca;
    --tone-improved: #047857;
    --tone-regressed: #b91c1c;
    --tone-neutral: #334155;
  }
}
* { box-sizing: border-box; }
body {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  max-width: 960px;
  margin: 2.5rem auto;
  padding: 0 1.25rem 3rem;
  line-height: 1.55;
  color: var(--fg);
  background: var(--bg);
}
header {
  display: flex;
  flex-wrap: wrap;
  gap: 1.5rem;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--bd);
}
header h1 { margin: 0 0 0.4rem; font-size: 1.55rem; letter-spacing: -0.01em; }
.hero { display: flex; flex-direction: column; min-width: 0; flex: 1 1 18rem; }
.hero .stat-label {
  font-size: 0.72rem;
  color: var(--mu);
  font-weight: 600;
}
.hero .stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--ac);
  line-height: 1.1;
  margin: 0.1rem 0;
}
.hero .stat-baseline { font-size: 0.85rem; color: var(--mu); }
.hero .stat-delta { font-size: 0.9rem; font-weight: 600; margin-top: 0.1rem; }
.hero .stat-delta.pos { color: var(--pos); }
.hero .stat-delta.neg { color: var(--neg); }
.summary-side {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  align-items: flex-end;
  text-align: right;
  font-size: 0.85rem;
  color: var(--mu);
}
.summary-side .sim { color: var(--fg); font-weight: 600; }
.tally {
  display: inline-flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.35rem;
}
.tally span {
  background: var(--bg-soft);
  border: 1px solid var(--bd);
  border-radius: 999px;
  padding: 0.15rem 0.7rem;
  font-size: 0.78rem;
  color: var(--fg);
}
.summary-side .sid code {
  background: var(--bg-soft);
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
  font-size: 0.78rem;
}
.results .hero {
  background: var(--ac-soft);
  border: 1px solid var(--bd);
  border-radius: 8px;
  padding: 0.9rem 1.1rem;
  flex: 0 0 auto;
}
.results .hero .stat-value { font-size: 1.6rem; }
section { margin: 2rem 0; }
section > h2 {
  font-size: 1.05rem;
  margin: 0 0 0.7rem;
  letter-spacing: -0.005em;
  color: var(--fg);
}
section > h2.section-sub { font-size: 0.95rem; margin-top: 1.5rem; color: var(--mu); }
.narrative {
  background: var(--bg-card);
  border: 1px solid var(--bd);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
}
.narrative > :first-child { margin-top: 0; }
.narrative > :last-child { margin-bottom: 0; }
.narrative img { max-width: 100%; height: auto; border-radius: 6px; margin: 0.6rem 0; }
.figure-block, .metrics-block, .table-block { margin: 2rem 0; }
.figure-block figure { margin: 0; }
.figure-block img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--bd);
  border-radius: 8px;
}
.figure-block figcaption { color: var(--mu); font-size: 0.85rem; margin-top: 0.4rem; }
.kv-grid { display: flex; flex-wrap: wrap; gap: 0.75rem; }
.kv {
  display: flex;
  flex-direction: column;
  background: var(--bg-card);
  border: 1px solid var(--bd);
  border-radius: 8px;
  padding: 0.6rem 0.9rem;
  min-width: 8rem;
}
.kv-label { font-size: 0.72rem; color: var(--mu); font-weight: 600; }
.kv-value { font-size: 1.4rem; font-weight: 700; color: var(--ac); }
.table-block table { width: 100%; border-collapse: collapse; }
.table-block th, .table-block td {
  border: 1px solid var(--bd);
  padding: 0.4rem 0.6rem;
  text-align: left;
}
.table-block th { background: var(--bg-soft); }
.narrative h1, .narrative h2, .narrative h3 { margin-top: 1.2rem; }
.narrative h1 { font-size: 1.3rem; }
.narrative h2 { font-size: 1.1rem; }
.narrative h3 { font-size: 1rem; }
.narrative p { margin: 0.55rem 0; }
.narrative pre {
  background: var(--bg-soft);
  border: 1px solid var(--bd-soft);
  border-radius: 6px;
  padding: 0.7rem 0.9rem;
  overflow-x: auto;
}
.metric-chart { margin: 0 0 1rem; }
.metric-chart h3 {
  margin: 0 0 0.35rem;
  font-size: 0.82rem;
  color: var(--mu);
  font-weight: 600;
}
.spark {
  display: block;
  width: 100%;
  max-width: 720px;
  height: 96px;
  border: 1px solid var(--bd-soft);
  border-radius: 8px;
  background: var(--bg-soft);
}
.spark .area { fill: var(--ac-soft); }
.spark .line { stroke: var(--ac); stroke-width: 1.6; fill: none; }
.spark .dot { fill: var(--ac); }
.spark .dot.best { fill: var(--ac); stroke: var(--bg-card); stroke-width: 2.2; }
.spark text { font-size: 10px; fill: var(--mu); }
.spark .baseline-line {
  stroke: var(--mu); stroke-width: 1; stroke-dasharray: 4 4; opacity: 0.55;
}
.tree-wrap {
  border: 1px solid var(--bd-soft);
  border-radius: 8px;
  background: var(--bg-soft);
  margin: 0 0 1rem;
  overflow: hidden;
}
.tree {
  display: block;
  width: 100%;
  height: auto;
  padding: 0.5rem 0.25rem 0.25rem;
}
.tree .edge { fill: none; stroke: var(--mu); stroke-width: 1.2; opacity: 0.5; }
.tree .box { stroke: var(--bd); stroke-width: 1.2; }
.tree .box.baseline { fill: var(--tone-baseline); }
.tree .box.improved { fill: var(--tone-improved); }
.tree .box.regressed { fill: var(--tone-regressed); }
.tree .box.neutral { fill: var(--tone-neutral); }
.tree .box.best { stroke: var(--ac); stroke-width: 2.2; }
.tree a:hover .box { stroke: var(--ac); }
.tree text { font-size: 11px; fill: #0f172a; }
.tree text.title { font-weight: 700; }
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.85rem;
  padding: 0.6rem 0.9rem;
  font-size: 0.78rem;
  color: var(--mu);
  border-top: 1px solid var(--bd-soft);
  background: var(--bg);
}
.legend .swatch { display: inline-flex; align-items: center; gap: 0.4rem; }
.legend .swatch::before {
  content: "";
  width: 0.85rem;
  height: 0.85rem;
  border-radius: 999px;
  background: var(--swatch, var(--ac));
  border: 1px solid rgba(15, 23, 42, 0.15);
}
.legend .swatch.baseline { --swatch: var(--tone-baseline); }
.legend .swatch.improved { --swatch: var(--tone-improved); }
.legend .swatch.regressed { --swatch: var(--tone-regressed); }
.legend .swatch.neutral { --swatch: var(--tone-neutral); }
.legend .swatch.best { --swatch: var(--ac); }
details {
  border: 1px solid var(--bd);
  border-left: 3px solid var(--bd);
  border-radius: 8px;
  padding: 0.6rem 0.85rem;
  margin: 0.55rem 0;
  background: var(--bg-card);
  transition: background 0.15s;
}
details.best {
  border-left-color: var(--ac);
  background: var(--ac-soft);
}
details[open] { background: var(--bg-soft); }
summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--fg);
  list-style: none;
}
summary::-webkit-details-marker { display: none; }
summary .idx { display: inline-block; font-weight: 700; color: var(--ac); margin-right: 0.4rem; }
summary .parent { color: var(--mu); font-weight: 500; margin-right: 0.4rem; }
summary .metrics { color: var(--mu); font-weight: 400; margin-left: 0.5rem; font-size: 0.92em; }
summary .badge {
  display: inline-block;
  margin-left: 0.5rem;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 0.1rem 0.45rem;
  border-radius: 999px;
  background: var(--ac);
  color: #fff;
  vertical-align: 0.05em;
}
details .id-chip {
  display: inline-block;
  background: var(--ac-soft);
  color: var(--ac);
  padding: 0.1rem 0.5rem;
  border-radius: 999px;
  font-family: ui-monospace, monospace;
  font-size: 0.75em;
  font-weight: 600;
  margin-right: 0.4rem;
}
details h4 {
  margin: 0.9rem 0 0.3rem;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--mu);
}
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.25rem 0; }
th, td {
  border-bottom: 1px solid var(--bd-soft);
  padding: 0.35rem 0.55rem;
  text-align: left;
  vertical-align: top;
}
th {
  color: var(--mu);
  font-weight: 600;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
img {
  max-width: 100%;
  border: 1px solid var(--bd-soft);
  border-radius: 6px;
  margin-top: 0.5rem;
  background: var(--bg);
}
code {
  font-family: ui-monospace, monospace;
  font-size: 0.85em;
  background: var(--bg-soft);
  padding: 0.05rem 0.3rem;
  border-radius: 4px;
}
footer {
  color: var(--mu);
  font-size: 0.85rem;
  margin-top: 2.5rem;
  padding-top: 0.85rem;
  border-top: 1px solid var(--bd-soft);
}
footer a { color: var(--ac); text-decoration: none; }
footer a:hover { text-decoration: underline; }
"""


def render_report(
    events: Iterable[Event],
    out_path: Path,
    *,
    narrative_markdown: str = "",
    title: str | None = None,
    session_id: str | None = None,
    simulator: str | None = None,
    artifact_dirs: Sequence[Path] | None = None,
    blocks: Sequence[dict[str, Any]] | None = None,
    custom_css: str = "",
) -> str:
    """Write the report HTML and return the document string.

    ``blocks`` lets the agent compose the report from typed building blocks
    (``prose``, ``figure``, ``metrics``, ``table``, ``exploration``) in any
    order, with its own titles, and nothing is forced. When ``blocks`` is omitted,
    the report falls back to the calibration template: the narrative followed by
    the metric/exploration/attempt blocks built from any logged attempts (each
    hides itself when there is nothing to show).
    """

    event_list = list(events)
    flat = _flatten(build_attempt_tree(event_list))
    sid = session_id or _event_field(event_list, schema.SESSION_START, "session_id")
    sim = simulator or _event_field(
        event_list, schema.SESSION_START, "simulator", default="unknown"
    )
    heading = title or f"{sim} investigation report"
    dirs = artifact_dirs or ()

    if blocks is not None:
        body = [_render_block(block, flat=flat, artifact_dirs=dirs) for block in blocks]
    else:
        body = [
            _render_narrative(narrative_markdown, dirs),
            _render_results(flat),
            _render_exploration(flat),
            _render_attempt_details(flat, dirs),
        ]

    sections = [_render_header(heading, sim, sid, flat), *body, _render_footer(sid)]

    doc = (
        '<!doctype html>\n<html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(heading)}</title>"
        f"<style>{_STYLES}{custom_css}</style>"
        "</head><body>" + "".join(part for part in sections if part) + "</body></html>"
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")

    # Write the transcript the footer links to right next to the report, so the
    # link resolves immediately. write_report runs mid-turn, before the model's
    # closing message, so this is a snapshot; the session refreshes it once the
    # turn settles (see Session.refresh_report_transcripts). A transcript hiccup
    # must never fail the report write.
    with contextlib.suppress(Exception):
        write_sidecar_transcript(out.parent, event_list, artifact_dirs=dirs)

    return doc


# ---------------------------------------------------------------------------
# Tree / metric helpers


def _flatten(roots: list[Attempt]) -> list[Attempt]:
    flat: list[Attempt] = []

    def walk(node: Attempt) -> None:
        flat.append(node)
        for child in node.children:
            walk(child)

    for root in roots:
        walk(root)
    return flat


def _event_field(events: list[Event], kind: str, key: str, *, default: str = "") -> str:
    for event in events:
        if event.kind == kind:
            return str(event.payload.get(key) or default)
    return default


def _primary_key(metrics: dict) -> str:
    for key in metrics:
        lowered = key.lower()
        if "rmse" in lowered or "mae" in lowered:
            return key
    return next(iter(metrics))


def _best_attempt(flat: list[Attempt]) -> Attempt | None:
    key: str | None = None
    best: tuple[Attempt, float] | None = None
    for node in flat:
        if not node.metrics:
            continue
        if key is None:
            key = _primary_key(node.metrics)
        if key not in node.metrics:
            continue
        val = float(node.metrics[key])
        if best is None or val < best[1]:
            best = (node, val)
    return best[0] if best else None


def _baseline_metric(flat: list[Attempt], key: str) -> float | None:
    for node in flat:
        if key in node.metrics:
            try:
                return float(node.metrics[key])
            except (TypeError, ValueError):
                return None
    return None


def _branch_leaf_counts(flat: list[Attempt]) -> tuple[int, int]:
    branches = sum(1 for node in flat if node.children)
    leaves = sum(1 for node in flat if not node.children)
    return branches, leaves


def _fmt(value: float) -> str:
    return f"{value:.4g}"


# ---------------------------------------------------------------------------
# Header (hero stat + tally panel)


def _render_header(heading: str, simulator: str, session_id: str, flat: list[Attempt]) -> str:
    side = _render_summary_side(simulator, session_id, flat)
    return f'<header><div class="hero-wrap"><h1>{html.escape(heading)}</h1></div>{side}</header>'


def _render_results(flat: list[Attempt]) -> str:
    """A results summary card placed near the attempt list, not at the top.

    Shows the best primary metric and its delta vs baseline. Hidden when
    there is no useful comparison (single attempt, or only one attempt has
    the metric).
    """

    best = _best_attempt(flat)
    if best is None or not best.metrics:
        return ""
    key = _primary_key(best.metrics)
    value = float(best.metrics[key])
    baseline = _baseline_metric(flat, key)
    attempts_with_key = sum(1 for node in flat if key in node.metrics)

    # No point in a "results" card if we have nothing to compare against.
    if attempts_with_key < 2 or baseline is None or baseline == value:
        return ""

    parts = [
        f'<span class="stat-label">{html.escape(key)} (best)</span>',
        f'<span class="stat-value">{html.escape(_fmt(value))}</span>',
        f'<span class="stat-baseline">baseline {html.escape(_fmt(baseline))}</span>',
    ]
    if baseline != 0:
        pct = (value - baseline) / abs(baseline) * 100
        cls = "neg" if value > baseline else "pos"
        sign = "+" if pct > 0 else ""
        parts.append(f'<span class="stat-delta {cls}">{sign}{pct:.1f}% vs baseline</span>')
    return (
        '<section class="results">'
        "<h2>Results</h2>"
        f'<div class="hero">{"".join(parts)}</div>'
        "</section>"
    )


def _render_summary_side(simulator: str, session_id: str, flat: list[Attempt]) -> str:
    pills: list[str] = []
    if flat:
        n = len(flat)
        pills.append(f"<span>{n} attempt{'s' if n != 1 else ''}</span>")
        branches, leaves = _branch_leaf_counts(flat)
        if branches:
            pills.append(f"<span>{branches} branch{'es' if branches != 1 else ''}</span>")
        # "1 leaf" is implied by "1 attempt"; only show when it adds info.
        if leaves and n > 1:
            leaf_label = "leaf" if leaves == 1 else "leaves"
            pills.append(f"<span>{leaves} {leaf_label}</span>")
    parts: list[str] = [f'<div class="sim">{html.escape(simulator)}</div>']
    if pills:
        parts.append(f'<div class="tally">{"".join(pills)}</div>')
    if session_id:
        parts.append(f'<div class="sid">Session <code>{html.escape(session_id)}</code></div>')
    return f'<div class="summary-side">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Narrative


def _render_narrative(markdown: str, artifact_dirs: Sequence[Path] = ()) -> str:
    if not markdown.strip():
        return ""
    body = _embed_narrative_images(render_markdown_html(markdown), artifact_dirs)
    return f'<section class="narrative">{body}</section>'


def _embed_narrative_images(html_body: str, artifact_dirs: Sequence[Path]) -> str:
    """Inline images the narrative references so the report stays one file.

    The agent can drop ``![caption](artifacts/plot.png)`` into its write-up to
    put a figure at the top; the path resolves against the run's artifact dirs
    and the bytes embed as a data URI, the same way the attempt plots do. A
    reference that doesn't resolve is left as-is rather than dropped.
    """

    def _inline(match: re.Match[str]) -> str:
        src = match.group(1)
        if src.startswith("data:"):
            return match.group(0)
        resolved = _resolve_plot(src, artifact_dirs)
        if resolved is None:
            return match.group(0)
        mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        data = base64.standard_b64encode(resolved.read_bytes()).decode("ascii")
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', _inline, html_body)


# ---------------------------------------------------------------------------
# Composable building blocks (the agent picks and orders these)


def _render_block(
    block: dict[str, Any], *, flat: list[Attempt], artifact_dirs: Sequence[Path]
) -> str:
    """Render one agent-composed report block; unknown types are skipped."""
    if not isinstance(block, dict):
        return ""
    btype = str(block.get("type") or "")
    if btype == "prose":
        return _render_narrative(str(block.get("markdown") or block.get("md") or ""), artifact_dirs)
    if btype == "figure":
        return _render_figure_block(block, artifact_dirs)
    if btype == "metrics":
        return _render_metrics_block(block)
    if btype == "table":
        return _render_table_block(block)
    if btype == "exploration":
        return (
            _render_results(flat)
            + _render_exploration(flat)
            + _render_attempt_details(flat, artifact_dirs)
        )
    if btype == "html":
        # Escape hatch: the agent supplies raw HTML for a fully custom block,
        # paired with `custom_css` on the report when it needs its own styling.
        return f'<section class="html-block">{block.get("html") or ""}</section>'
    return ""


def _render_figure_block(block: dict[str, Any], artifact_dirs: Sequence[Path]) -> str:
    path = str(block.get("path") or "")
    if not path:
        return ""
    title = str(block.get("title") or "")
    caption = str(block.get("caption") or "")
    head = f"<h2>{html.escape(title)}</h2>" if title else ""
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return (
        f'<section class="figure-block">{head}'
        f"<figure>{_embed_plot(path, artifact_dirs)}{cap}</figure></section>"
    )


def _render_metrics_block(block: dict[str, Any]) -> str:
    """A titled set of label/value pairs, the flexible-name generalisation of the
    calibration hero stat."""
    rows = block.get("rows")
    if not isinstance(rows, dict) or not rows:
        return ""
    title = str(block.get("title") or "")
    head = f"<h2>{html.escape(title)}</h2>" if title else ""
    cells = "".join(
        f'<div class="kv"><span class="kv-label">{html.escape(str(key))}</span>'
        f'<span class="kv-value">{html.escape(_fmt_cell(value))}</span></div>'
        for key, value in rows.items()
    )
    return f'<section class="metrics-block">{head}<div class="kv-grid">{cells}</div></section>'


def _render_table_block(block: dict[str, Any]) -> str:
    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        return ""
    title = str(block.get("title") or "")
    headers = block.get("headers") if isinstance(block.get("headers"), list) else []
    head = f"<h2>{html.escape(title)}</h2>" if title else ""
    thead = (
        "<thead><tr>"
        + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        + "</tr></thead>"
        if headers
        else ""
    )
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt_cell(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
        if isinstance(row, list)
    )
    return (
        f'<section class="table-block">{head}'
        f"<table>{thead}<tbody>{body_rows}</tbody></table></section>"
    )


def _fmt_cell(value: Any) -> str:
    return _fmt(value) if isinstance(value, float) else str(value)


# ---------------------------------------------------------------------------
# Exploration map (metric chart + colour-coded tree + legend)


def _render_exploration(flat: list[Attempt]) -> str:
    if not flat or len(flat) <= 1:
        return ""
    best = _best_attempt(flat)
    best_id = best.id if best else None
    index = {node.id: i + 1 for i, node in enumerate(flat)}

    chart = _render_metric_chart(flat, index, best_id)
    tree, legend = _render_tree(flat, index, best_id)
    if not (chart or tree):
        return ""

    return (
        '<section class="exploration"><h2>Exploration map</h2>'
        f"{chart}"
        f'<div class="tree-wrap">{tree}{legend}</div>'
        "</section>"
    )


def _render_metric_chart(
    flat: list[Attempt],
    index: dict[str, int],
    best_id: str | None,
) -> str:
    key: str | None = None
    pts: list[tuple[int, float, str]] = []
    for node in flat:
        if not node.metrics:
            continue
        if key is None:
            key = _primary_key(node.metrics)
        if key in node.metrics:
            pts.append((index[node.id], float(node.metrics[key]), node.id))
    if len(pts) < 2:
        return ""

    width, height = 720, 96
    pad_l, pad_r, pad_t, pad_b = 40, 30, 12, 22
    ys = [p[1] for p in pts]
    vmin, vmax = min(ys), max(ys)
    span = max(vmax - vmin, 1e-12)
    n = len(pts)

    def x_of(i: int) -> float:
        return pad_l + (i / max(n - 1, 1)) * (width - pad_l - pad_r)

    def y_of(v: float) -> float:
        return pad_t + (1 - (v - vmin) / span) * (height - pad_t - pad_b)

    coords = [(x_of(i), y_of(v)) for i, (_, v, _) in enumerate(pts)]
    line_path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area_path = (
        line_path
        + f" L{coords[-1][0]:.1f},{height - pad_b:.1f}"
        + f" L{coords[0][0]:.1f},{height - pad_b:.1f} Z"
    )

    baseline = pts[0][1]
    baseline_y = y_of(baseline)
    baseline_svg = (
        f'<line class="baseline-line" x1="{pad_l:.1f}" x2="{width - pad_r:.1f}" '
        f'y1="{baseline_y:.1f}" y2="{baseline_y:.1f}"/>'
        f'<text x="{width - pad_r - 4:.1f}" y="{baseline_y - 4:.1f}" '
        f'text-anchor="end">baseline</text>'
    )

    dots: list[str] = []
    x_labels: list[str] = []
    for i, (idx, v, aid) in enumerate(pts):
        cx, cy = coords[i]
        is_best = aid == best_id
        dots.append(
            f'<circle class="dot {"best" if is_best else ""}" '
            f'cx="{cx:.1f}" cy="{cy:.1f}" r="{3.4 if is_best else 2.6}">'
            f"<title>#{idx} · {html.escape(_fmt(v))}</title></circle>"
        )
        x_labels.append(
            f'<text x="{cx:.1f}" y="{height - 6:.1f}" text-anchor="middle">#{idx}</text>'
        )

    y_max_label = (
        f'<text x="{pad_l - 4:.1f}" y="{y_of(vmax) + 3:.1f}" text-anchor="end">'
        f"{html.escape(_fmt(vmax))}</text>"
    )
    y_min_label = (
        f'<text x="{pad_l - 4:.1f}" y="{y_of(vmin) + 3:.1f}" text-anchor="end">'
        f"{html.escape(_fmt(vmin))}</text>"
    )

    return (
        f'<div class="metric-chart">'
        f"<h3>{html.escape(key or '')} by attempt</h3>"
        f'<svg class="spark" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="{html.escape(key or "metric")} per attempt">'
        f'<path class="area" d="{area_path}"/>'
        f'<path class="line" d="{line_path}"/>'
        f"{baseline_svg}"
        f"{''.join(dots)}{y_max_label}{y_min_label}{''.join(x_labels)}"
        "</svg></div>"
    )


def _tone_for(node: Attempt, parent: Attempt | None, key: str | None) -> str:
    if parent is None:
        return "baseline"
    if key is None or key not in node.metrics or key not in parent.metrics:
        return "neutral"
    a = float(node.metrics[key])
    b = float(parent.metrics[key])
    if a < b:
        return "improved"
    if a > b:
        return "regressed"
    return "neutral"


def _render_tree(
    flat: list[Attempt],
    index: dict[str, int],
    best_id: str | None,
) -> tuple[str, str]:
    """Return (tree-svg, legend-html). Both empty when there are no attempts."""

    if not flat:
        return "", ""

    by_id = {node.id: node for node in flat}
    depth: dict[str, int] = {}
    for node in flat:
        if node.parent_id and node.parent_id in by_id:
            depth[node.id] = depth[node.parent_id] + 1
        else:
            depth[node.id] = 0
    rows: dict[int, list[Attempt]] = {}
    for node in flat:
        rows.setdefault(depth[node.id], []).append(node)
    max_depth = max(rows)

    # Pick the chart's primary key so the tree's tone reflects the same metric.
    chart_key: str | None = None
    for node in flat:
        if node.metrics:
            chart_key = _primary_key(node.metrics)
            break

    box_w, box_h = 168, 56
    row_h = 96
    margin = 20
    widest = max(len(nodes) for nodes in rows.values())
    width = max(620, widest * (box_w + 36))
    height = (max_depth + 1) * row_h + 2 * margin - (row_h - box_h)

    pos: dict[str, tuple[float, float]] = {}
    for d, nodes in rows.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = ((i + 1) / (n + 1)) * width
            y = margin + d * row_h
            pos[node.id] = (x, y)

    edges: list[str] = []
    for node in flat:
        if node.parent_id and node.parent_id in pos:
            px, py = pos[node.parent_id]
            cx, cy = pos[node.id]
            py_bottom = py + box_h
            mid = (py_bottom + cy) / 2
            edges.append(
                f'<path class="edge" d="M{px:.1f},{py_bottom:.1f} '
                f'C{px:.1f},{mid:.1f} {cx:.1f},{mid:.1f} {cx:.1f},{cy:.1f}"/>'
            )

    boxes: list[str] = []
    for node in flat:
        x, y = pos[node.id]
        parent = by_id.get(node.parent_id) if node.parent_id else None
        tone = _tone_for(node, parent, chart_key)
        is_best = node.id == best_id
        cls = f"box {tone}" + (" best" if is_best else "")
        title = _truncate((node.notes or node.rationale or "Attempt").strip(), 28)
        metric = _primary_metric_text(node)
        boxes.append(
            f'<a href="#attempt-{index[node.id]}">'
            f"<title>{html.escape(title)}</title>"
            f'<rect class="{cls}" x="{x - box_w / 2:.1f}" y="{y:.1f}" '
            f'width="{box_w}" height="{box_h}" rx="8" ry="8"/>'
            f'<text class="title" x="{x:.1f}" y="{y + 21:.0f}" text-anchor="middle">'
            f"#{index[node.id]} {html.escape(title)}</text>"
            f'<text x="{x:.1f}" y="{y + 39:.0f}" text-anchor="middle">'
            f"{html.escape(metric)}</text>"
            "</a>"
        )

    svg = (
        f'<svg class="tree" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="Attempt tree">'
        f"{''.join(edges)}{''.join(boxes)}"
        "</svg>"
    )
    legend = (
        '<div class="legend">'
        '<span class="swatch baseline">Baseline</span>'
        '<span class="swatch improved">Improved vs parent</span>'
        '<span class="swatch regressed">Regressed vs parent</span>'
        '<span class="swatch neutral">No change</span>'
        '<span class="swatch best">Best</span>'
        "</div>"
    )
    return svg, legend


def _primary_metric_text(node: Attempt) -> str:
    if not node.metrics:
        return ""
    key = _primary_key(node.metrics)
    return f"{key}={_fmt(float(node.metrics[key]))}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Attempt details


def _render_attempt_details(flat: list[Attempt], artifact_dirs: Sequence[Path]) -> str:
    if not flat:
        return ""
    index = {node.id: i + 1 for i, node in enumerate(flat)}
    best = _best_attempt(flat)
    best_id = best.id if best else None
    items = "".join(
        _render_attempt(node, index, artifact_dirs, is_best=node.id == best_id) for node in flat
    )
    return f'<section class="attempt-details"><h2>Attempt details</h2>{items}</section>'


def _render_attempt(
    node: Attempt,
    index: dict[str, int],
    artifact_dirs: Sequence[Path],
    *,
    is_best: bool = False,
) -> str:
    title = (node.notes or node.rationale or "Attempt").strip().splitlines()[0]
    parent_html = (
        f'<span class="parent">from #{index[node.parent_id]}</span>'
        if node.parent_id and node.parent_id in index
        else ""
    )
    metric_pairs = sorted(node.metrics.items())
    if metric_pairs:
        primary_key = _primary_key(node.metrics)
        primary_val = node.metrics.get(primary_key)
        if primary_val is not None:
            metric_text = html.escape(f"{primary_key}={_fmt(float(primary_val))}")
            metrics_inline = f'<span class="metrics">{metric_text}</span>'
        else:
            metrics_inline = ""
    else:
        metrics_inline = ""
    badge = '<span class="badge">best</span>' if is_best else ""
    summary = (
        f'<span class="idx">#{index[node.id]}</span>'
        f"{parent_html}"
        f"{html.escape(title)}"
        f"{metrics_inline}"
        f"{badge}"
    )

    body: list[str] = []
    id_chip = f'<span class="id-chip">{html.escape(node.id[:8])}</span>'
    body.append(f"<p>{id_chip}<strong>Attempt id:</strong> <code>{html.escape(node.id)}</code></p>")
    if node.rationale and node.rationale.strip() != title:
        body.append(f"<p>{html.escape(node.rationale)}</p>")
    if node.parameters_changed:
        # Drop rows where old == new so the table only shows real edits.
        changed = [
            (k, v) for k, v in sorted(node.parameters_changed.items()) if not _is_param_noop(v)
        ]
        if changed:
            rows = "".join(
                f"<tr><td><code>{html.escape(k)}</code></td>"
                f"<td>{html.escape(_fmt_param(v))}</td></tr>"
                for k, v in changed
            )
            body.append("<h4>Parameter changes</h4>")
            body.append(f"<table>{rows}</table>")
    if metric_pairs:
        m_rows = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(_fmt(v))}</td></tr>"
            for k, v in metric_pairs
        )
        body.append("<h4>Metrics</h4>")
        body.append(f"<table>{m_rows}</table>")
    if node.plot_artifact_path:
        body.append(_embed_plot(node.plot_artifact_path, artifact_dirs))

    anchor = f"attempt-{index[node.id]}"
    cls = "best" if is_best else ""
    return (
        f'<details id="{anchor}" class="{cls}"><summary>{summary}</summary>'
        f"{''.join(body)}</details>"
    )


def _fmt_param(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]} → {value[1]}"
    if isinstance(value, dict) and {"old", "new"} <= value.keys():
        return f"{value['old']} → {value['new']}"
    return json.dumps(value, default=str)


def _is_param_noop(value) -> bool:
    """True when an ``old → new`` change has identical sides (just noise)."""

    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0] == value[1]
    if isinstance(value, dict) and {"old", "new"} <= value.keys():
        return value["old"] == value["new"]
    return False


# ---------------------------------------------------------------------------
# Plot embedding + footer


def _embed_plot(path: str, artifact_dirs: Sequence[Path]) -> str:
    resolved = _resolve_plot(path, artifact_dirs)
    if resolved is None:
        return f"<p><em>Plot not found: <code>{html.escape(path)}</code></em></p>"
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    data = base64.standard_b64encode(resolved.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{data}" alt="Attempt plot">'


def _resolve_plot(path: str, artifact_dirs: Sequence[Path]) -> Path | None:
    raw = Path(path)
    if raw.is_absolute() and raw.is_file():
        return raw
    for root in artifact_dirs:
        candidate = root / path
        if candidate.is_file():
            return candidate
        if path.startswith("artifacts/"):
            tail = root / raw.name
            if tail.is_file():
                return tail
    return None


_TRANSCRIPT_FILENAME = "transcript.html"


def write_sidecar_transcript(
    report_dir: Path, events: Sequence[Event], artifact_dirs: Sequence[Path] = ()
) -> None:
    """Write the transcript a report links to, beside the report (same folder).

    The sidecar sits inside the artifacts folder, so its recorded ``artifacts/…``
    image paths don't resolve relative to it; inlining the images (resolved via
    the report dir and any extra ``artifact_dirs``) makes the transcript show its
    plots wherever it is opened.
    """
    from jutul_agent.transcript.html import render_html

    dirs = [report_dir, *artifact_dirs]
    (report_dir / _TRANSCRIPT_FILENAME).write_text(
        render_html(list(events), artifact_dirs=dirs), encoding="utf-8"
    )


def _render_footer(session_id: str) -> str:
    # ``render_report`` writes the transcript next to the report, so this link
    # resolves on its own from the same directory, with no manual `/transcript` needed.
    return (
        f'<footer>Also see the <a href="{_TRANSCRIPT_FILENAME}">'
        "full session transcript</a>.</footer>"
    )
