# ╔══════════════════════════════════════╗
# ║  Ryan Wetzstein                      ║
# ║  Nexis ML                            ║
# ║  2026                                ║
# ╚══════════════════════════════════════╝

"""Self-contained HTML report for a finished run.

Reads only the on-disk run store (metrics.jsonl + summary/config + the
confusion-matrix / image-grid artifacts) and emits a single standalone
.html file — inline SVG charts, a metrics summary, the confusion matrix,
the latest sample-prediction grid (base64-embedded), generated-text
samples, and the config. No torch, no network, no external assets, so it
opens anywhere and is easy to share or archive.
"""

from __future__ import annotations

import base64
import html
import json
import os
from typing import Any

from . import __version__
from .run_store import read_json  # shared file reader (was a local copy)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _decimate(
    steps: list[float], values: list[float], buckets: int = 300
) -> list[tuple[float, float]]:
    """Min/max bin a series to at most ~2*buckets points so a long run
    still renders as a small inline SVG without losing spikes."""
    n = len(steps)
    if n <= buckets * 2:
        return list(zip(steps, values))
    per = n / buckets
    out: list[tuple[float, float]] = []
    for b in range(buckets):
        start = int(b * per)
        end = min(int((b + 1) * per), n)
        if start >= end:
            continue
        lo = hi = start
        for j in range(start + 1, end):
            if values[j] < values[lo]:
                lo = j
            if values[j] > values[hi]:
                hi = j
        a, c = min(lo, hi), max(lo, hi)
        out.append((steps[a], values[a]))
        if c != a:
            out.append((steps[c], values[c]))
    return out


def _svg_chart(name: str, steps: list[float], values: list[float]) -> str:
    w, h, pad = 560, 140, 24
    pts = _decimate(steps, values)
    if not pts:
        return ""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def px(x: float) -> float:
        return pad + (x - min_x) / span_x * (w - pad * 2)

    def py(v: float) -> float:
        return pad + (1 - (v - min_y) / span_y) * (h - pad * 2)

    poly = " ".join(f"{px(x):.1f},{py(v):.1f}" for x, v in pts)
    return (
        f'<div class="chart"><div class="chart-head"><span>{html.escape(name)}</span>'
        f"<span class=mono>{values[-1]:.4g}</span></div>"
        f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<line x1="{pad}" y1="{h / 2:.0f}" x2="{w - pad}" y2="{h / 2:.0f}" class="grid"/>'
        f'<polyline points="{poly}" class="line"/></svg>'
        f'<div class="chart-foot mono"><span>{min_y:.4g}</span>'
        f"<span>{max_y:.4g}</span></div></div>"
    )


def _confusion_table(cm: dict[str, Any]) -> str:
    # Color math kept in sync with the live panel's cellColor
    # (Nexis src/modules/ml/MlPanel.tsx) so the report and panel agree.
    labels = [str(x) for x in cm.get("labels", [])]
    matrix = cm.get("matrix", [])
    if not labels or not matrix:
        return ""
    mx = max((max(r) for r in matrix if r), default=0)
    head = "".join(f"<th>{html.escape(c)}</th>" for c in labels)
    rows = ""
    for i, row in enumerate(matrix):
        cells = ""
        for j, v in enumerate(row):
            alpha = 0 if not mx or v == 0 else 0.12 + (v / mx) * 0.73
            rgb = "16,185,129" if i == j else "244,63,94"
            bg = f"background:rgba({rgb},{alpha:.3f})" if v else ""
            cells += f'<td style="{bg}">{v}</td>'
        rows += f"<tr><th>{html.escape(labels[i])}</th>{cells}</tr>"
    return (
        '<table class="cm"><tr><th></th>'
        + head
        + "</tr>"
        + rows
        + "</table><p class=hint>Rows = actual, columns = predicted; "
        "green diagonal is correct.</p>"
    )


def _data_uri_png(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return None


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;
margin:2rem auto;padding:0 1rem;color:#1c1c1e;background:#fafaf8;line-height:1.5}
h1{font-size:20px;font-weight:600;margin:0 0 .25rem}h2{font-size:14px;font-weight:600;
margin:1.75rem 0 .5rem;text-transform:uppercase;letter-spacing:.04em;color:#666}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0 0}
.chip{background:#eceae3;border-radius:6px;padding:2px 8px;font-size:12px;color:#444}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
table{border-collapse:collapse;width:100%;font-size:13px}
.summary td,.summary th{border-bottom:1px solid #e6e4dc;padding:4px 8px;text-align:left}
.summary th{color:#666;font-weight:500}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}
.chart-head{display:flex;justify-content:space-between;font-size:12px;color:#555;margin-bottom:2px}
.chart svg{width:100%;height:120px;background:#fff;border:1px solid #ececE4;border-radius:6px}
.chart-foot{display:flex;justify-content:space-between;font-size:10px;color:#999}
.line{fill:none;stroke:#6366f1;stroke-width:1.5}.grid{stroke:#eee}
.cm td,.cm th{border:1px solid #eee;padding:4px 8px;text-align:center;font-size:12px}
.cm th{color:#666;background:#f4f2ec}
.hint{font-size:11px;color:#999;margin:.25rem 0 0}
pre{background:#f4f2ec;border-radius:8px;padding:.75rem;overflow-x:auto;font-size:12px}
.sample{background:#fff;border:1px solid #ececE4;border-radius:8px;padding:.5rem .75rem;
margin:.5rem 0;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12px}
img{max-width:100%;border-radius:8px;border:1px solid #ececE4}
footer{margin:2rem 0 1rem;font-size:11px;color:#aaa}
"""


def build_html_report(run_dir: str) -> str:
    run_id = os.path.basename(os.path.normpath(run_dir))
    events = _read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    summary = read_json(os.path.join(run_dir, "summary.json")) or {}
    config = read_json(os.path.join(run_dir, "config.json")) or {}

    series: dict[str, tuple[list[float], list[float]]] = {}
    samples: list[dict[str, Any]] = []
    cm_path: str | None = None
    grid_path: str | None = None
    for ev in events:
        kind = ev.get("ev")
        if kind == "metric":
            name = ev.get("name")
            value = ev.get("value")
            step = ev.get("step", 0)
            if isinstance(name, str) and isinstance(value, (int, float)):
                s = series.setdefault(name, ([], []))
                s[0].append(step)
                s[1].append(value)
        elif kind == "sample":
            samples.append(ev)
        elif kind == "artifact":
            if ev.get("kind") == "confusion-matrix":
                cm_path = ev.get("path")
            elif ev.get("kind") == "image-grid":
                grid_path = ev.get("path")

    esc = html.escape
    parts: list[str] = []
    parts.append(f"<h1>{esc(summary.get('name') or run_id)}</h1>")
    parts.append(
        f'<div class="mono" style="color:#888;font-size:12px">{esc(run_id)}</div>'
    )

    chips = []
    status = summary.get("status")
    if status:
        chips.append(f"status: {esc(str(status))}")
    if summary.get("device"):
        chips.append(f"device: {esc(str(summary['device']))}")
    if summary.get("lastEpoch"):
        chips.append(f"{summary['lastEpoch']} passes")
    if summary.get("finishedAt"):
        chips.append(esc(str(summary["finishedAt"])[:19].replace("T", " ")))
    if chips:
        parts.append(
            '<div class="chips">'
            + "".join(f"<span class=chip>{c}</span>" for c in chips)
            + "</div>"
        )

    metrics = summary.get("metrics") or {}
    if isinstance(metrics, dict) and metrics:
        rows = "".join(
            f"<tr><th>{esc(name)}</th><td class=mono>{st.get('last', ''):.4g}</td>"
            f"<td class=mono>{st.get('min', ''):.4g}</td>"
            f"<td class=mono>{st.get('max', ''):.4g}</td></tr>"
            for name, st in sorted(metrics.items())
            if isinstance(st, dict) and isinstance(st.get("last"), (int, float))
        )
        parts.append("<h2>Summary</h2>")
        parts.append(
            '<table class="summary"><tr><th>metric</th><th>last</th><th>min</th>'
            "<th>max</th></tr>" + rows + "</table>"
        )

    if series:
        parts.append("<h2>Curves</h2><div class=charts>")
        for name in sorted(series):
            steps, values = series[name]
            parts.append(_svg_chart(name, steps, values))
        parts.append("</div>")

    if cm_path:
        cm = read_json(cm_path)
        if isinstance(cm, dict):
            table = _confusion_table(cm)
            if table:
                parts.append("<h2>Confusion matrix</h2>" + table)

    if grid_path:
        uri = _data_uri_png(grid_path)
        if uri:
            parts.append(f'<h2>Sample predictions</h2><img src="{uri}" alt="samples"/>')

    if samples:
        parts.append("<h2>Generated samples</h2>")
        for ev in samples[-5:]:
            text = str(ev.get("output", ""))
            parts.append(f'<div class="sample">{esc(text)}</div>')

    if config:
        parts.append(
            "<h2>Config</h2><pre>" + esc(json.dumps(config, indent=2)) + "</pre>"
        )

    parts.append(f"<footer>Generated by nexis-ml {esc(__version__)}</footer>")

    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        f"<title>{esc(run_id)} — nexis-ml report</title><style>{_CSS}</style></head>"
        "<body>" + "".join(parts) + "</body></html>"
    )


def write_html_report(run_dir: str, out: str | None = None) -> str:
    """Build the report and write it; returns the output path."""
    target = out or os.path.join(run_dir, "report.html")
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(build_html_report(run_dir))
    os.replace(tmp, target)
    return target
