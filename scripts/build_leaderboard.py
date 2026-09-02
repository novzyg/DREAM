"""Build the DREAM leaderboard static page from the survey LaTeX source.

Parses the three benchmark result tables (all-level / ATC-3 / ATC-4) in
main_v5.tex and renders docs/index.html. Best / second / third results per
column are highlighted following the \\best / \\second / \\third macros in the
paper.

Usage:
    python scripts/build_leaderboard.py [--tex main_v5.tex] [--out docs/index.html]
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import re
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

METHODS = ["LR", "ECC", "RETAIN", "LEAP", "GAMENet", "G-BERT", "CompNet",
           "MICRON", "SafeDrug", "COGNet", "PREMIER", "4SDrug", "DrugRec",
           "REFINE", "MedRec", "MoleRec", "Carmen", "OntoPath", "VITA",
           "DRecHGR", "RAREMed", "MR-DTR", "TEMPT", "ARMR", "SSPNet",
           "LAMO", "FLAME"]

GRANULARITIES = ["all", "atc3", "atc4"]
GRAN_LABELS = {"all": "All-level", "atc3": "ATC-3", "atc4": "ATC-4"}
DATASETS = ["mimic-iii", "mimic-iv", "eicu"]
DATASET_LABELS = {"mimic-iii": "MIMIC-III", "mimic-iv": "MIMIC-IV", "eicu": "eICU"}
METRICS = ["jaccard", "prauc", "f1", "ddi"]
METRIC_LABELS = {
    "jaccard": "Jaccard (%) &uarr;",
    "prauc": "PRAUC (%) &uarr;",
    "f1": "F1 (%) &uarr;",
    "ddi": "DDI (%) &darr;",
}
RANK_STYLES = {"best": "best", "second": "second", "third": "third"}


def parse_latex_tables(tex_path: str) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Parse the three result tables.

    Returns {granularity: {dataset: {method: {metric: {mean, std, rank}}}}}.
    """
    with open(tex_path, encoding="utf-8") as handle:
        text = handle.read()

    blocks = re.findall(
        r"\\begin\{tabular\}\{lcccccccccccc\}(.*?)\\end\{tabular\}",
        text, re.S,
    )
    if len(blocks) != 3:
        raise ValueError(f"expected 3 result tables, found {len(blocks)}")

    tables: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for gran, block in zip(GRANULARITIES, blocks):
        rows: Dict[str, List[Optional[dict]]] = {}
        for chunk in re.split(r"\\\\", block):
            cell = chunk.replace("\n", " ")
            cell = re.sub(r"\\rowcolor\{[^}]*\}", " ", cell)
            cell = re.sub(r"\\(midrule|toprule|bottomrule)", " ", cell).strip()
            if not cell or "\\multicolumn" in cell or "\\multirow" in cell:
                continue
            parts = [p.strip() for p in cell.split("&")]
            name = re.sub(r"\\cite\{[^}]*\}", "", parts[0]).strip()
            if name not in METHODS:
                continue
            values: List[Optional[dict]] = []
            ok = True
            for raw in parts[1:]:
                if "--" in raw:
                    values.append(None)
                    continue
                match = re.search(
                    r"(\d+(?:\.\d+)?)\\pm(\d+(?:\.\d+)?)", raw)
                if not match:
                    ok = False
                    break
                rank = None
                for macro, style in RANK_STYLES.items():
                    if f"\\{macro}" in raw:
                        rank = style
                        break
                values.append({
                    "mean": float(match.group(1)),
                    "std": float(match.group(2)),
                    "rank": rank,
                })
            if ok and len(values) == 12:
                rows[name] = values
        if len(rows) != len(METHODS):
            missing = [m for m in METHODS if m not in rows]
            raise ValueError(f"{gran}: missing methods {missing}")

        gran_data: Dict[str, Dict[str, dict]] = {}
        for di, dataset in enumerate(DATASETS):
            ds_rows = {}
            for method in METHODS:
                cells = rows[method][di * 4:(di + 1) * 4]
                ds_rows[method] = {
                    metric: cells[mi] for mi, metric in enumerate(METRICS)
                }
            gran_data[dataset] = ds_rows
        tables[gran] = gran_data
    return tables


def page_json(tables) -> dict:
    """Flatten parsed tables into the JSON payload embedded in the page."""
    payload = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "granularities": [
            {"id": g, "label": GRAN_LABELS[g]} for g in GRANULARITIES
        ],
        "datasets": [
            {"id": d, "label": DATASET_LABELS[d]} for d in DATASETS
        ],
        "metrics": [
            {"id": m, "label": html.unescape(METRIC_LABELS[m])}
            for m in METRICS
        ],
        "methods": METHODS,
        "results": tables,
    }
    return payload


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DREAM Leaderboard &mdash; Drug Recommendation Evaluation Across Multiple Settings</title>
<style>
:root {
  --bg: #f6f8fa; --card: #ffffff; --ink: #1f2328; --muted: #656d76;
  --line: #d8dee4; --accent: #0969da; --accent-soft: #ddf4ff;
  --gold: #1a7f37; --silver: #0969da; --bronze: #9a6700;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.5;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 24px 20px 64px; }
header { text-align: center; margin-bottom: 20px; }
header h1 { font-size: 1.55rem; margin: 0 0 6px; letter-spacing: -0.01em; }
header p { color: var(--muted); margin: 0; }
header .badges { margin-top: 10px; }
header .badges a {
  display: inline-block; padding: 3px 10px; margin: 2px; border-radius: 12px;
  background: var(--accent-soft); color: var(--accent); text-decoration: none;
  font-size: 0.8rem; font-weight: 600;
}
.controls { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 18px 0 6px; }
.seg { display: flex; background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.seg button {
  border: 0; background: transparent; padding: 7px 16px; cursor: pointer;
  font: inherit; color: var(--muted); font-weight: 600;
}
.seg button + button { border-left: 1px solid var(--line); }
.seg button.on { background: var(--accent); color: #fff; }
.note { text-align: center; color: var(--muted); font-size: 0.82rem; margin: 10px 0 4px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; white-space: nowrap; }
th { position: sticky; top: 0; background: #f0f3f6; cursor: pointer; user-select: none; font-size: 0.86rem; }
th.method, td.method { text-align: left; position: sticky; left: 0; background: inherit; }
thead th.method { background: #f0f3f6; z-index: 3; }
tbody td.method { background: var(--card); font-weight: 600; z-index: 2; }
tbody tr:nth-child(even) td.method { background: #fafbfc; }
th.sorted::after { content: " \\2193"; color: var(--accent); }
th.sorted.rev::after { content: " \\2191"; color: var(--accent); }
td .val { font-weight: 600; }
td .std { color: var(--muted); font-size: 0.78rem; }
td.best .val { color: var(--gold); }
td.second .val { color: var(--silver); }
td.third .val { color: var(--bronze); }
td.best { background: rgba(26,127,55,0.07); }
td.second { background: rgba(9,105,218,0.06); }
td.third { background: rgba(154,103,0,0.06); }
td.na { color: var(--muted); }
footer { text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 26px; }
footer a { color: var(--accent); text-decoration: none; }
.legend { display: flex; gap: 14px; justify-content: center; font-size: 0.8rem; color: var(--muted); margin: 8px 0 2px; }
.legend span i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }
.legend .l-best i { background: rgba(26,127,55,0.45); }
.legend .l-second i { background: rgba(9,105,218,0.40); }
.legend .l-third i { background: rgba(154,103,0,0.40); }
@media (max-width: 720px) { .wrap { padding: 12px 8px 48px; } th, td { padding: 6px 8px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>DREAM Leaderboard</h1>
    <p>Drug Recommendation Evaluation Across Multiple settings &mdash; 27 methods &times; 3 datasets &times; 3 medication granularities</p>
    <div class="badges">
      <a href="https://github.com/novzyg/DREAM">GitHub</a>
      <a href="https://github.com/novzyg/DREAM#citation">Citation</a>
    </div>
  </header>

  <div class="controls">
    <div class="seg" id="gran-seg"></div>
    <div class="seg" id="ds-seg"></div>
  </div>
  <div class="legend">
    <span class="l-best"><i></i>best</span>
    <span class="l-second"><i></i>second</span>
    <span class="l-third"><i></i>third</span>
    <span>&nbsp;&mdash;&nbsp; mean &plusmn; std over 5 seeds</span>
  </div>
  <p class="note" id="note"></p>

  <div class="card"><table id="board"><thead></thead><tbody></tbody></table></div>

  <footer>
    Generated <span id="gen-time"></span> from the survey
    <em>&ldquo;EHR-Based Medication Recommendation: A Stage-Oriented Survey and Unified Benchmark&rdquo;</em>.
    &ldquo;--&rdquo; denotes results not reported due to incompatibility between fine-grained text-based outputs and aggregated ATC labels.
  </footer>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const state = { gran: DATA.granularities[0].id, ds: DATA.datasets[0].id,
                sortKey: "jaccard", sortDir: -1 };

function makeSeg(el, items, key) {
  el.innerHTML = "";
  items.forEach(it => {
    const b = document.createElement("button");
    b.textContent = it.label; b.dataset.id = it.id;
    b.onclick = () => { state[key] = it.id; render(); };
    el.appendChild(b);
  });
}

function render() {
  makeSeg(document.getElementById("gran-seg"), DATA.granularities, "gran");
  makeSeg(document.getElementById("ds-seg"), DATA.datasets, "ds");
  document.querySelectorAll("#gran-seg button").forEach(b => b.classList.toggle("on", b.dataset.id === state.gran));
  document.querySelectorAll("#ds-seg button").forEach(b => b.classList.toggle("on", b.dataset.id === state.ds));

  const note = document.getElementById("note");
  const granLabel = DATA.granularities.find(g => g.id === state.gran).label;
  const dsLabel = DATA.datasets.find(d => d.id === state.ds).label;
  note.textContent = granLabel + " medication space \\u00b7 " + dsLabel +
    " \\u00b7 click a column header to sort";

  const rows = DATA.results[state.gran][state.ds];
  const methods = DATA.methods.slice().sort((a, b) => {
    const ra = rows[a][state.sortKey], rb = rows[b][state.sortKey];
    const va = ra ? ra.mean : null, vb = rb ? rb.mean : null;
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    return (va - vb) * state.sortDir * (state.sortKey === "ddi" ? -1 : 1);
  });

  const thead = document.querySelector("#board thead");
  const tbody = document.querySelector("#board tbody");
  let h = "<tr><th class='method'>Method</th>";
  DATA.metrics.forEach(m => {
    const cls = state.sortKey === m.id ? "sorted" + (state.sortDir === 1 ? " rev" : "") : "";
    h += `<th class="${cls}" data-metric="${m.id}">${m.label}</th>`;
  });
  h += "</tr>";
  thead.innerHTML = h;
  thead.querySelectorAll("th[data-metric]").forEach(th => {
    th.onclick = () => {
      const m = th.dataset.metric;
      if (state.sortKey === m) { state.sortDir *= -1; }
      else { state.sortKey = m; state.sortDir = (m === "ddi") ? 1 : -1; }
      render();
    };
  });

  let b = "";
  methods.forEach(method => {
    b += `<tr><td class="method">${method}</td>`;
    DATA.metrics.forEach(m => {
      const cell = rows[method][m.id];
      if (!cell) { b += `<td class="na">--</td>`; return; }
      const cls = cell.rank ? ` class="${cell.rank}"` : "";
      b += `<td${cls}><span class="val">${cell.mean.toFixed(2)}</span> <span class="std">&plusmn;${cell.std.toFixed(2)}</span></td>`;
    });
    b += "</tr>";
  });
  tbody.innerHTML = b;

  document.getElementById("gen-time").textContent = DATA.generated;
}

render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default=os.path.join(PROJECT_ROOT, "main_v5.tex"),
                        help="Path to the survey LaTeX source")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "docs", "index.html"),
                        help="Output HTML path")
    args = parser.parse_args()

    tables = parse_latex_tables(args.tex)
    payload = page_json(tables)

    doc = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(doc)
    print(f"wrote {args.out}")
    total = sum(
        1 for g in tables.values() for d in g.values()
        for m in d.values() for c in m.values() if c
    )
    print(f"cells parsed: {total}")


if __name__ == "__main__":
    main()
