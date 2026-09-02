"""Build the DREAM project page (GitHub Pages) from the survey LaTeX source.

Parses main_v5.tex — title, authors, abstract, keywords, and the three
benchmark result tables — and renders docs/index.html, a single-file project
page with: hero, abstract, news, taxonomy figure, interactive leaderboard,
dataset statistics, benchmark protocol, and citation.

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
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

GITHUB_URL = "https://github.com/novzyg/DREAM"
PAPER_URL = ""  # placeholder; update when the paper is public

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
    "jaccard": "Jaccard (%) ↑",
    "prauc": "PRAUC (%) ↑",
    "f1": "F1 (%) ↑",
    "ddi": "DDI (%) ↓",
}
RANK_STYLES = {"best": "best", "second": "second", "third": "third"}

NEWS = [
    {"date": "2026-09", "text": "DREAM v1 released: 22 integrated models, "
     "3 datasets × 3 medication granularities, unified CLI, and multi-GPU "
     "batch scheduling."},
    {"date": "2026-09", "text": "Leaderboard published with benchmark "
     "results for 27 methods."},
]

DATASET_STATS = [
    {"dataset": "eICU", "level": "All-level", "patients": "10,568", "visits": "23,080",
     "diagnoses": "2,575", "medications": "155", "procedures": "2,054", "ddi": "0.0781"},
    {"dataset": "eICU", "level": "ATC-3", "patients": "10,526", "visits": "22,988",
     "diagnoses": "2,572", "medications": "75", "procedures": "2,052", "ddi": "0.5137"},
    {"dataset": "eICU", "level": "ATC-4", "patients": "10,526", "visits": "22,988",
     "diagnoses": "2,572", "medications": "107", "procedures": "2,052", "ddi": "0.3071"},
    {"dataset": "MIMIC-III", "level": "All-level", "patients": "6,360", "visits": "16,976",
     "diagnoses": "4,672", "medications": "718", "procedures": "1,420", "ddi": "0.0819"},
    {"dataset": "MIMIC-III", "level": "ATC-3", "patients": "6,359", "visits": "16,974",
     "diagnoses": "4,672", "medications": "148", "procedures": "1,420", "ddi": "0.5465"},
    {"dataset": "MIMIC-III", "level": "ATC-4", "patients": "6,359", "visits": "16,974",
     "diagnoses": "4,672", "medications": "317", "procedures": "1,420", "ddi": "0.3279"},
    {"dataset": "MIMIC-IV", "level": "All-level", "patients": "8,949", "visits": "24,106",
     "diagnoses": "11,030", "medications": "877", "procedures": "4,810", "ddi": "0.0589"},
    {"dataset": "MIMIC-IV", "level": "ATC-3", "patients": "8,946", "visits": "24,100",
     "diagnoses": "11,029", "medications": "158", "procedures": "4,810", "ddi": "0.4872"},
    {"dataset": "MIMIC-IV", "level": "ATC-4", "patients": "8,946", "visits": "24,100",
     "diagnoses": "11,029", "medications": "348", "procedures": "4,810", "ddi": "0.2836"},
]

PROTOCOL = [
    ("Data processing", "Raw drug names are mapped to DrugBank identifiers; "
     "compound medications are decomposed into single-ingredient drugs when "
     "possible. DrugBank provides ATC codes and molecular structures; the DDI "
     "adjacency matrix is built from DDInter."),
    ("Data splits & seeds", "Patient-level split 66% / 17% / 17% "
     "(train / validation / test), identical across models. All experiments "
     "are repeated with 5 random seeds; results are reported as mean ± std."),
    ("Training", "Adam optimizer, at most 50 epochs, single-patient-level "
     "batching. Hyperparameters selected by grid search on validation "
     "Jaccard. Early stopping after 10 epochs without validation-Jaccard "
     "improvement; the best-validation checkpoint is used for testing."),
    ("Metrics", "Accuracy: Jaccard, PRAUC, F1 (plus per-example "
     "precision/recall). Safety: DDI rate — the proportion of known "
     "drug–drug interaction pairs among all predicted drug pairs."),
]

BIBTEX = """@misc{zhen2026dream,
  title  = {EHR-Based Medication Recommendation: A Stage-Oriented Survey and Unified Benchmark},
  author = {Zhen, Yeguang and Wang, Shoujin and Li, Yishuo and Hu, Liang and Lian, Defu and Wang, Cong and Lu, Wenpeng},
  year   = {2026},
  url    = {https://github.com/novzyg/DREAM}
}"""


# ---------------------------------------------------------------------------
# LaTeX parsing
# ---------------------------------------------------------------------------

def parse_meta(tex_path: str) -> dict:
    with open(tex_path, encoding="utf-8") as handle:
        text = handle.read()

    meta: dict = {}

    title = re.search(r"\\title\{(.+?)\}", text, re.S)
    meta["title"] = re.sub(r"\s+", " ", title.group(1)).strip() if title else ""

    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    if abstract:
        body = abstract.group(1)
        body = "\n".join(
            line for line in body.splitlines() if not line.strip().startswith("%")
        )
        body = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", body)
        body = re.sub(r"\s+", " ", body).strip()
        meta["abstract"] = body
    else:
        meta["abstract"] = ""

    keywords = re.search(r"\\keywords\{(.+?)\}", text, re.S)
    meta["keywords"] = [
        k.strip() for k in keywords.group(1).split(",")
    ] if keywords else []

    authors: List[dict] = []
    start = text.find("\\title{")
    end = text.find("\\maketitle")
    block = text[start:end] if start != -1 and end != -1 else text

    names = re.findall(r"\\author\{(.+?)\}", block, re.S)
    emails = re.findall(r"\\email\{(.+?)\}", block, re.S)
    affiliations = re.findall(r"\\affiliation\{(.*?)\}", block, re.S)
    corresponding = "Corresponding Author" in block

    for index, name in enumerate(names):
        name = re.sub(r"\s+", " ", name).strip()
        email = emails[index].strip() if index < len(emails) else ""
        inst = ""
        country = ""
        if index < len(affiliations):
            affil = re.sub(r"%.*", "", affiliations[index])
            inst_match = re.search(r"\\institution\{(.*?)\}", affil, re.S)
            if inst_match:
                inst = re.sub(r"\s+", " ", inst_match.group(1)).strip()
            country_match = re.search(r"\\country\{(.+?)\}", affil)
            if country_match:
                country = country_match.group(1).strip()
        authors.append({
            "name": name,
            "email": email,
            "institution": inst,
            "country": country,
        })
    meta["authors"] = authors
    meta["corresponding"] = corresponding
    return meta


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
                match = re.search(r"(\d+(?:\.\d+)?)\\pm(\d+(?:\.\d+)?)", raw)
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


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def build_payload(meta: dict, tables) -> dict:
    return {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "title": meta["title"],
        "abstract": meta["abstract"],
        "keywords": meta["keywords"],
        "authors": meta["authors"],
        "githubUrl": GITHUB_URL,
        "paperUrl": PAPER_URL,
        "news": NEWS,
        "stats": DATASET_STATS,
        "protocol": [{"title": t, "text": x} for t, x in PROTOCOL],
        "bibtex": BIBTEX,
        "granularities": [
            {"id": g, "label": GRAN_LABELS[g]} for g in GRANULARITIES
        ],
        "datasets": [{"id": d, "label": DATASET_LABELS[d]} for d in DATASETS],
        "metrics": [{"id": m, "label": METRIC_LABELS[m]} for m in METRICS],
        "methods": METHODS,
        "results": tables,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>DREAM — Drug Recommendation Evaluation Across Multiple settings</title>
<style>
:root {
  --bg: #f6f8fa; --card: #ffffff; --ink: #1f2328; --muted: #656d76;
  --line: #d8dee4; --accent: #0969da; --accent-soft: #ddf4ff;
  --gold: #1a7f37; --silver: #0969da; --bronze: #9a6700;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: -apple-system, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
  font-size: 15px; line-height: 1.6;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 20px 64px; }

/* nav */
nav {
  position: sticky; top: 0; z-index: 50; background: rgba(255,255,255,0.92);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--line);
}
nav .inner { max-width: 1080px; margin: 0 auto; padding: 10px 20px; display: flex; align-items: center; gap: 18px; }
nav .brand { font-weight: 800; font-size: 1.05rem; color: var(--ink); text-decoration: none; letter-spacing: -0.02em; }
nav .brand span { color: var(--accent); }
nav a.navlink { color: var(--muted); text-decoration: none; font-weight: 600; font-size: 0.9rem; }
nav a.navlink:hover { color: var(--accent); }
nav .spacer { flex: 1; }

/* hero */
header.hero { text-align: center; padding: 52px 0 30px; }
header.hero h1 { font-size: 1.85rem; line-height: 1.3; margin: 0 0 14px; letter-spacing: -0.02em; max-width: 900px; margin-left: auto; margin-right: auto; }
header.hero .authors { color: var(--ink); margin: 14px 0 4px; font-size: 0.95rem; }
header.hero .authors sup { color: var(--muted); }
header.hero .insts { color: var(--muted); font-size: 0.8rem; margin: 4px 0 0; max-width: 760px; margin-left: auto; margin-right: auto; }
.btns { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 22px; }
.btn {
  display: inline-block; padding: 8px 18px; border-radius: 8px; text-decoration: none;
  font-weight: 700; font-size: 0.9rem; border: 1px solid var(--line); background: var(--card); color: var(--ink);
}
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
.btn:hover { opacity: 0.88; }
.kw { margin-top: 16px; color: var(--muted); font-size: 0.82rem; }
.kw b { color: var(--ink); font-weight: 600; }

/* sections */
section { margin-top: 46px; }
h2.sec {
  font-size: 1.3rem; letter-spacing: -0.01em; margin: 0 0 6px;
  display: flex; align-items: center; gap: 10px;
}
h2.sec::before { content: ""; width: 5px; height: 22px; background: var(--accent); border-radius: 3px; }
p.secdesc { color: var(--muted); margin: 0 0 18px; font-size: 0.92rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 22px 26px; }

/* abstract */
#abstract .card { font-size: 0.96rem; line-height: 1.75; }

/* news */
ul.news { list-style: none; padding: 0; margin: 0; }
ul.news li { display: flex; gap: 14px; padding: 9px 0; border-bottom: 1px dashed var(--line); }
ul.news li:last-child { border-bottom: 0; }
ul.news .date { color: var(--accent); font-weight: 700; white-space: nowrap; font-size: 0.85rem; padding-top: 2px; }

/* figures */
figure { margin: 0; }
figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 10px; background: #fff; }

/* leaderboard */
.controls { display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin: 16px 0 6px; }
.seg { display: flex; background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: transparent; padding: 7px 16px; cursor: pointer; font: inherit; color: var(--muted); font-weight: 600; }
.seg button + button { border-left: 1px solid var(--line); }
.seg button.on { background: var(--accent); color: #fff; }
.note { text-align: center; color: var(--muted); font-size: 0.82rem; margin: 10px 0 4px; }
.boardcard { background: var(--card); border: 1px solid var(--line); border-radius: 10px; overflow: auto; max-height: 78vh; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 12px; text-align: right; white-space: nowrap; }
th { position: sticky; top: 0; background: #f0f3f6; cursor: pointer; user-select: none; font-size: 0.86rem; }
th.method, td.method { text-align: left; position: sticky; left: 0; background: inherit; }
thead th.method { background: #f0f3f6; z-index: 3; }
tbody td.method { background: var(--card); font-weight: 600; z-index: 2; }
tbody tr:nth-child(even) td.method { background: #fafbfc; }
th.sorted::after { content: " \2193"; color: var(--accent); }
th.sorted.rev::after { content: " \2191"; color: var(--accent); }
td .val { font-weight: 600; }
td .std { color: var(--muted); font-size: 0.78rem; }
td.best .val { color: var(--gold); }
td.second .val { color: var(--silver); }
td.third .val { color: var(--bronze); }
td.best { background: rgba(26,127,55,0.07); }
td.second { background: rgba(9,105,218,0.06); }
td.third { background: rgba(154,103,0,0.06); }
td.na { color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; justify-content: center; font-size: 0.8rem; color: var(--muted); margin: 8px 0 2px; }
.legend span i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }
.legend .l-best i { background: rgba(26,127,55,0.45); }
.legend .l-second i { background: rgba(9,105,218,0.40); }
.legend .l-third i { background: rgba(154,103,0,0.40); }

/* stats table */
.statscard { overflow: auto; }
#stats table { font-size: 0.88rem; }
#stats th, #stats td { text-align: right; }
#stats th.method, #stats td.method { text-align: left; }

/* protocol */
.protocol { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }
.protocol .p-item { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.protocol .p-item h3 { margin: 0 0 8px; font-size: 0.98rem; }
.protocol .p-item p { margin: 0; color: var(--muted); font-size: 0.86rem; }

/* citation */
#cite pre {
  background: #f0f3f6; border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; font-size: 0.82rem; overflow: auto;
}
.copybtn {
  float: right; border: 1px solid var(--line); background: var(--card); border-radius: 6px;
  padding: 3px 12px; cursor: pointer; font: inherit; font-size: 0.78rem; font-weight: 700; color: var(--accent);
}
.copybtn:hover { background: var(--accent-soft); }

footer { text-align: center; color: var(--muted); font-size: 0.82rem; margin-top: 52px; }
footer a { color: var(--accent); text-decoration: none; }
@media (max-width: 720px) {
  .wrap { padding: 0 10px 48px; }
  header.hero h1 { font-size: 1.4rem; }
  th, td { padding: 6px 8px; }
  nav .inner { gap: 12px; }
}
</style>
</head>
<body>

<nav><div class="inner">
  <a class="brand" href="#top">DREAM<span>.</span></a>
  <a class="navlink" href="#abstract">Abstract</a>
  <a class="navlink" href="#taxonomy">Taxonomy</a>
  <a class="navlink" href="#leaderboard">Leaderboard</a>
  <a class="navlink" href="#stats">Datasets</a>
  <a class="navlink" href="#protocol">Protocol</a>
  <a class="navlink" href="#cite">Citation</a>
  <div class="spacer"></div>
  <a class="navlink" id="nav-paper" href="#">Paper</a>
  <a class="navlink" href="__GITHUB__">GitHub</a>
</div></nav>

<div class="wrap" id="top">

<header class="hero">
  <h1 id="hero-title"></h1>
  <div class="authors" id="hero-authors"></div>
  <div class="insts" id="hero-insts"></div>
  <div class="btns">
    <a class="btn primary" id="btn-paper" href="#">Paper (coming soon)</a>
    <a class="btn" href="__GITHUB__">GitHub Repository</a>
    <a class="btn" href="#leaderboard">Leaderboard</a>
    <a class="btn" href="#cite">BibTeX</a>
  </div>
  <div class="kw" id="hero-kw"></div>
</header>

<section id="abstract">
  <h2 class="sec">Abstract</h2>
  <p class="secdesc">Automatically parsed from the survey manuscript.</p>
  <div class="card" id="abstract-body"></div>
</section>

<section id="news">
  <h2 class="sec">News</h2>
  <div class="card"><ul class="news" id="news-list"></ul></div>
</section>

<section id="taxonomy">
  <h2 class="sec">Taxonomy</h2>
  <p class="secdesc">The survey organizes EHR-based medication recommendation into three functional stages — patient state modeling, medication knowledge modeling, and prescription decision modeling.</p>
  <div class="card" style="padding:14px;">
    <figure>
      <img src="fig/taxonomy.png" alt="Stage-oriented taxonomy of medication recommendation systems"/>
      <figcaption style="color:var(--muted);font-size:0.8rem;text-align:center;margin-top:8px;">
        Stage-oriented taxonomy of medication recommendation systems.
      </figcaption>
    </figure>
  </div>
</section>

<section id="leaderboard">
  <h2 class="sec">Leaderboard</h2>
  <p class="secdesc">Benchmark results: mean ± std over 5 seeds. Click a column header to sort. "Best / second / third" follow the rankings reported in the survey.</p>
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
  <div class="boardcard"><table id="board"><thead></thead><tbody></tbody></table></div>
</section>

<section id="stats">
  <h2 class="sec">Datasets</h2>
  <p class="secdesc">Basic statistics of the processed datasets under different medication granularity levels.</p>
  <div class="card statscard"><table id="stats-table">
    <thead><tr>
      <th class="method">Dataset</th><th>Level</th><th>Patients</th><th>Visits</th>
      <th>Diagnoses</th><th>Medications</th><th>Procedures</th><th>DDI Rate</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>
</section>

<section id="protocol">
  <h2 class="sec">Benchmark Protocol</h2>
  <p class="secdesc">All models are evaluated under an identical protocol to ensure fair comparison.</p>
  <div class="protocol" id="protocol-grid"></div>
</section>

<section id="cite">
  <h2 class="sec">Citation</h2>
  <p class="secdesc">If you use DREAM in your research, please cite our survey.</p>
  <pre id="bibtex"></pre>
</section>

<footer>
  Generated <span id="gen-time"></span> ·
  <a href="__GITHUB__">novzyg/DREAM</a> ·
  DREAM: Drug Recommendation Evaluation Across Multiple settings
</footer>

</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);

/* hero */
document.getElementById("hero-title").textContent = DATA.title;
const uniqueInsts = [];
DATA.authors.forEach(a => {
  const inst = a.institution.split(";")[0].trim();
  if (inst && !uniqueInsts.includes(inst)) uniqueInsts.push(inst);
});
document.getElementById("hero-authors").innerHTML = DATA.authors
  .map((a, i) => htmlEscape(a.name) + "<sup>" + (uniqueInsts.indexOf(a.institution.split(";")[0].trim()) + 1) + "</sup>")
  .join(", ");
document.getElementById("hero-insts").innerHTML = uniqueInsts
  .map((s, i) => "<sup>" + (i + 1) + "</sup>" + htmlEscape(s))
  .join(" &nbsp; ");
function htmlEscape(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* paper buttons */
if (DATA.paperUrl) {
  const pb = document.getElementById("btn-paper");
  pb.href = DATA.paperUrl; pb.textContent = "Paper";
  const np = document.getElementById("nav-paper");
  np.href = DATA.paperUrl;
} else {
  document.getElementById("nav-paper").style.display = "none";
}

/* keywords */
document.getElementById("hero-kw").innerHTML = DATA.keywords.length
  ? "<b>Keywords:</b> " + htmlEscape(DATA.keywords.join(" · "))
  : "";

/* abstract */
document.getElementById("abstract-body").textContent = DATA.abstract;

/* news */
document.getElementById("news-list").innerHTML = DATA.news
  .map(n => `<li><span class="date">${htmlEscape(n.date)}</span><span>${htmlEscape(n.text)}</span></li>`)
  .join("");

/* stats */
document.querySelector("#stats-table tbody").innerHTML = DATA.stats
  .map(s => `<tr><td class="method">${htmlEscape(s.dataset)}</td><td>${htmlEscape(s.level)}</td><td>${s.patients}</td><td>${s.visits}</td><td>${s.diagnoses}</td><td>${s.medications}</td><td>${s.procedures}</td><td>${s.ddi}</td></tr>`)
  .join("");

/* protocol */
document.getElementById("protocol-grid").innerHTML = DATA.protocol
  .map(p => `<div class="p-item"><h3>${htmlEscape(p.title)}</h3><p>${htmlEscape(p.text)}</p></div>`)
  .join("");

/* bibtex */
const bibEl = document.getElementById("bibtex");
bibEl.textContent = DATA.bibtex;
const btn = document.createElement("button");
btn.className = "copybtn"; btn.textContent = "Copy";
btn.onclick = () => {
  navigator.clipboard.writeText(DATA.bibtex).then(() => {
    btn.textContent = "Copied!"; setTimeout(() => btn.textContent = "Copy", 1500);
  });
};
bibEl.prepend(btn);

/* leaderboard */
const state = { gran: DATA.granularities[0].id, ds: DATA.datasets[0].id,
                sortKey: "jaccard", sortDir: -1 };

function makeSeg(el, items, key) {
  el.innerHTML = "";
  items.forEach(it => {
    const b = document.createElement("button");
    b.textContent = it.label; b.dataset.id = it.id;
    b.onclick = () => { state[key] = it.id; renderBoard(); };
    el.appendChild(b);
  });
}

function renderBoard() {
  makeSeg(document.getElementById("gran-seg"), DATA.granularities, "gran");
  makeSeg(document.getElementById("ds-seg"), DATA.datasets, "ds");
  document.querySelectorAll("#gran-seg button").forEach(b => b.classList.toggle("on", b.dataset.id === state.gran));
  document.querySelectorAll("#ds-seg button").forEach(b => b.classList.toggle("on", b.dataset.id === state.ds));

  const granLabel = DATA.granularities.find(g => g.id === state.gran).label;
  const dsLabel = DATA.datasets.find(d => d.id === state.ds).label;
  document.getElementById("note").textContent = granLabel + " medication space \u00b7 " + dsLabel +
    " \u00b7 click a column header to sort";

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
      renderBoard();
    };
  });

  let b = "";
  methods.forEach(method => {
    b += `<tr><td class="method">${htmlEscape(method)}</td>`;
    DATA.metrics.forEach(m => {
      const cell = rows[method][m.id];
      if (!cell) { b += `<td class="na">--</td>`; return; }
      const cls = cell.rank ? ` class="${cell.rank}"` : "";
      b += `<td${cls}><span class="val">${cell.mean.toFixed(2)}</span> <span class="std">&plusmn;${cell.std.toFixed(2)}</span></td>`;
    });
    b += "</tr>";
  });
  tbody.innerHTML = b;
}

renderBoard();
document.getElementById("gen-time").textContent = DATA.generated;
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

    meta = parse_meta(args.tex)
    tables = parse_latex_tables(args.tex)
    payload = build_payload(meta, tables)

    doc = (HTML_TEMPLATE
           .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
           .replace("__GITHUB__", GITHUB_URL))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(doc)

    print(f"wrote {args.out}")
    print(f"  title:    {meta['title'][:60]}...")
    print(f"  authors:  {len(meta['authors'])}")
    print(f"  abstract: {len(meta['abstract'])} chars")
    print(f"  keywords: {meta['keywords']}")
    cells = sum(1 for g in tables.values() for d in g.values()
                for m in d.values() for c in m.values() if c)
    print(f"  cells:    {cells}")


if __name__ == "__main__":
    main()
