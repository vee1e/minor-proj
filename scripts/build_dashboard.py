# ruff: noqa: E501, E741, B018, F841
"""Build a self-contained HTML dashboard from the evaluation artifacts.

Reads artifacts/datasets/dataset_summary.json, artifacts/reports/results.json,
artifacts/models/theta.json and portfolio.json, and the test set from
dataset.parquet; writes artifacts/reports/dashboard.html (single file, no
external dependencies).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
DATASETS = ROOT / "artifacts" / "datasets"
MODELS = ROOT / "artifacts" / "models"


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


def main() -> None:
    summary = json.loads((DATASETS / "dataset_summary.json").read_text())
    results = _load("results.json")
    theta = json.loads((MODELS / "theta.json").read_text())
    portfolio = json.loads((MODELS / "portfolio.json").read_text())

    test = pd.read_parquet(DATASETS / "dataset.parquet")
    test = test[test["split"] == "test"].copy()
    solvers = portfolio.get("solvers", [
        "minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
        "maplecm", "kissat404"])
    test = test.head(48)

    lane = []
    for fid, row in test.iterrows():
        cells = []
        for s in solvers:
            t = float(row[f"{s}_time_s"])
            cells.append({"solver": s, "time": t})
        lane.append({"formula": str(fid), "cells": cells,
                     "selector": "selector",
                     "vbs": row["best_solver"]})

    data = {
        "summary": summary,
        "results": results,
        "theta": theta,
        "portfolio": portfolio,
        "lane": lane,
        "solvers": solvers,
    }
    (REPORTS / "dashboard.html").write_text(_render(data))
    print(f"dashboard -> {REPORTS / 'dashboard.html'}")


def _render(data: dict) -> str:
    return _HTML.replace("__DATA_BLOB__", json.dumps(data, default=str))


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>satsel — solver selection dashboard</title>
<style>
  :root{
    --paper:#F3EFE6; --paper2:#ECE6D8; --ink:#1C2B33; --ink2:#4A5A63;
    --blue:#2E5E7A; --blue-soft:#DCE8EF; --verdict:#C2442E; --verdict-soft:#F3DCD6;
    --solved:#2F6B4F; --solved-soft:#DCE9E0; --rule:#D8D0BE; --white:#FDFCF8;
    --serif:Georgia,"Times New Roman",serif;
    --sans:-apple-system,BlinkMacSystemFont,"Helvetica Neue","Segoe UI",sans-serif;
    --mono:"SF Mono","JetBrains Mono","Cascadia Code",ui-monospace,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
       -webkit-font-smoothing:antialiased;line-height:1.55}
  a{color:var(--blue)}
  .wrap{max-width:1000px;margin:0 auto;padding:0 24px}

  /* hero */
  .hero{padding:64px 0 40px;border-bottom:1px solid var(--rule)}
  .eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;
           text-transform:uppercase;color:var(--blue);margin:0 0 18px}
  h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,5vw,46px);
     line-height:1.1;margin:0 0 10px;letter-spacing:-.01em}
  .sub{font-size:17px;color:var(--ink2);max-width:62ch;margin:0 0 26px}
  .verdict{display:inline-flex;align-items:baseline;gap:10px;flex-wrap:wrap;
           font-family:var(--mono);font-size:14px;color:var(--verdict);
           background:var(--verdict-soft);border:1px solid var(--verdict);
           border-radius:999px;padding:8px 16px}
  .verdict b{font-size:18px}

  /* truth lane */
  .lane-section{margin-top:8px}
  .lane{display:grid;gap:2px;margin:22px 0 8px;overflow-x:auto}
  .lane-row{display:grid;gap:2px;align-items:center}
  .lane-cell{height:18px;border-radius:2px;position:relative}
  .lane-cell::after{content:"";position:absolute;inset:1px;border-radius:2px}
  .lane-cell.sel::after{border:2px solid var(--verdict)}
  .lane-cell.vbs::after{border:2px solid var(--blue)}
  .lane-cell.both::after{border:2px solid var(--ink)}
  .lane-meta{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;color:var(--ink2)}
  .lane-meta i{width:10px;height:10px;border-radius:2px;display:inline-block;vertical-align:-1px;margin-right:5px}

  /* sections */
  .section{padding:44px 0 8px;border-bottom:1px solid var(--rule)}
  .sec-head{display:flex;align-items:baseline;gap:14px;margin:0 0 22px}
  .sec-no{font-family:var(--mono);font-size:13px;color:var(--blue)}
  h2{font-family:var(--serif);font-weight:600;font-size:26px;margin:0;letter-spacing:-.01em}
  .sec-note{color:var(--ink2);max-width:70ch;margin:6px 0 0 42px;font-size:14.5px}

  /* cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:0 0 18px}
  .card{background:var(--white);border:1px solid var(--rule);border-radius:8px;padding:16px 16px 14px}
  .card .k{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink2);margin:0 0 8px}
  .card .v{font-family:var(--mono);font-size:26px;font-weight:600;line-height:1;margin:0}
  .card .d{font-size:12.5px;color:var(--ink2);margin:8px 0 0}
  .card.lead .v{color:var(--blue)}
  .card.bad .v{color:var(--verdict)}
  .card.good .v{color:var(--solved)}

  /* bars */
  .bars{margin:6px 0 20px}
  .bar-row{display:grid;grid-template-columns:150px 1fr 90px;gap:12px;align-items:center;margin:9px 0}
  .bar-row .name{font-size:13.5px;color:var(--ink2);text-align:right;overflow:hidden;text-overflow:ellipsis}
  .bar-row .track{background:var(--paper2);border-radius:3px;height:20px;position:relative;overflow:hidden}
  .bar-row .fill{height:100%;border-radius:3px;background:var(--blue)}
  .bar-row .fill.vbs{background:var(--ink)}
  .bar-row .fill.bad{background:var(--verdict)}
  .bar-row .val{font-family:var(--mono);font-size:13px}
  .bar-row .val small{color:var(--ink2)}
  .legend{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:12px;color:var(--ink2);margin:0 0 8px 162px}
  .legend i{width:12px;height:12px;border-radius:2px;display:inline-block;vertical-align:-1px;margin-right:6px}

  /* tables */
  table{width:100%;border-collapse:collapse;font-size:14px;margin:8px 0 20px}
  th{font-family:var(--mono);font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);text-align:left;padding:8px 10px;border-bottom:2px solid var(--rule)}
  td{padding:9px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
  td.num,th.num{font-family:var(--mono);text-align:right}
  .ok{color:var(--solved);font-weight:600}
  .no{color:var(--verdict);font-weight:600}
  .dim{color:var(--ink2)}
  .pill{display:inline-block;font-family:var(--mono);font-size:11.5px;padding:2px 9px;border-radius:999px;border:1px solid var(--rule)}
  .pill.win{color:var(--solved);border-color:var(--solved);background:var(--solved-soft)}
  .pill.lose{color:var(--verdict);border-color:var(--verdict);background:var(--verdict-soft)}
  .pill.tie{color:var(--ink2);border-color:var(--rule);background:var(--white)}

  footer{padding:36px 0 60px;color:var(--ink2);font-size:13.5px}
  footer .mono{font-family:var(--mono);font-size:12px}

  @media (prefers-reduced-motion:no-preference){
    .reveal{opacity:0;transform:translateY(10px);animation:rise .5s ease forwards}
    @keyframes rise{to{opacity:1;transform:none}}
  }
  @media (max-width:640px){
    .bar-row{grid-template-columns:110px 1fr 80px}
    .legend{margin-left:0}
    .sec-note{margin-left:0}
  }
</style>
</head>
<body>
<div class="hero">
  <div class="wrap">
    <p class="eyebrow reveal">satsel &middot; ML-driven SAT solver selection</p>
    <h1 class="reveal">Which solver, per formula.</h1>
    <p class="sub reveal">Every formula is profiled against seven SAT solvers. A model learns to
      predict the fastest one, and a single call runs it. This dashboard shows the pre-registered
      evaluation on the profiled corpus.</p>
    <div class="verdict reveal" id="verdict"></div>
  </div>
</div>

<div class="wrap">
  <section class="lane-section">
    <div class="lane-meta reveal" style="margin-top:26px">
      <span><i style="background:var(--solved-soft)"></i>fast</span>
      <span><i style="background:var(--blue-soft)"></i>mid</span>
      <span><i style="background:var(--verdict-soft)"></i>slow / timeout</span>
      <span><i style="border:2px solid var(--verdict)"></i>selector pick</span>
      <span><i style="border:2px solid var(--blue)"></i>VBS pick</span>
    </div>
    <div class="lane reveal" id="lane" style="grid-template-columns:minmax(150px,1fr) repeat(7,minmax(26px,1fr))"></div>
    <p class="lane-meta" style="margin:4px 0 0">Each row is one test formula; each column is a solver. Darker means slower.
      The red ring is the model's pick, the blue ring is the best possible solver for that formula.</p>
  </section>

  <section class="section" id="s1">
    <div class="wrap">
      <div class="sec-head reveal"><span class="sec-no">01</span><h2>Corpus</h2></div>
      <div class="cards reveal" id="corpus-cards"></div>
      <table class="reveal" id="family-table"></table>
    </div>
  </section>

  <section class="section" id="s2">
    <div class="wrap">
      <div class="sec-head reveal"><span class="sec-no">02</span><h2>Systems on the test set</h2></div>
      <p class="sec-note reveal">PAR-2 is the mean solve time with a timeout counted as double the limit.
        Lower is better. The selector includes its own feature-extraction and inference overhead.</p>
      <div class="legend reveal"><span><i style="background:var(--blue)"></i>system</span>
        <span><i style="background:var(--ink)"></i>VBS ceiling</span>
        <span><i style="background:var(--verdict)"></i>worse than best single solver</span></div>
      <div class="bars reveal" id="systems-bars"></div>
    </div>
  </section>

  <section class="section" id="s3">
    <div class="wrap">
      <div class="sec-head reveal"><span class="sec-no">03</span><h2>Planned comparisons</h2></div>
      <p class="sec-note reveal">Paired bootstrap on per-instance PAR-2 differences, stratified by family,
        10,000 resamples, Holm-Bonferroni adjusted across the four planned comparisons. A win needs the
        confidence interval to exclude zero and the adjusted test to reject.</p>
      <table class="reveal" id="comparison-table"></table>
    </div>
  </section>

  <section class="section" id="s4">
    <div class="wrap">
      <div class="sec-head reveal"><span class="sec-no">04</span><h2>Fallback threshold &amp; portfolio</h2></div>
      <div class="cards reveal" id="theta-cards"></div>
      <div class="reveal" id="theta-curve"></div>
      <table class="reveal" id="portfolio-table"></table>
    </div>
  </section>

  <footer>
    <div class="wrap">
      <p class="mono">generated from artifacts/reports/results.json &middot;
        artifacts/models/theta.json &amp; portfolio.json &middot; artifacts/datasets/dataset.parquet</p>
      <p>Method: <a href="DEVELOPMENT_PLAN.md">DEVELOPMENT_PLAN.md</a> and
        <a href="EVAL_PROTOCOL.md">EVAL_PROTOCOL.md</a>. This dashboard is a rendering of the
        pre-registered report; the numbers here are not a substitute for reading the protocol.</p>
    </div>
  </footer>
</div>

<script type="application/json" id="DATA">__DATA_BLOB__</script>
<script>
const D = JSON.parse(document.getElementById("DATA").textContent);
const fmt = (x, d=3) => (x===null||x===undefined||isNaN(x)) ? "&ndash;" : Number(x).toFixed(d);
const cells = (k) => `<div class="card${k.cls?" "+k.cls:""}"><p class="k">${k.k}</p><p class="v">${k.v}</p>${k.d?`<p class="d">${k.d}</p>`:""}</div>`;

/* verdict */
(function(){
  const r = D.results;
  const sys = Object.fromEntries(r.systems.map(s=>[s.system,s]));
  const sel = sys["selector"], best = sys["best_average"];
  if (!sel || !best) { document.getElementById("verdict").textContent = "evaluation incomplete"; return; }
  const ratio = sel.par2 / best.par2;
  const el = document.getElementById("verdict");
  const w = ratio < 1;
  el.className = "verdict " + (w ? "" : "");
  el.innerHTML = w
    ? `Selector PAR-2 <b>${fmt(sel.par2)}</b> beats the best single solver <b>${fmt(best.par2)}</b> &mdash; ${fmt(ratio,2)}&times;`
    : `On this corpus the selector is <b>${fmt(ratio,2)}&times;</b> slower than the best single solver &mdash; an honest null result, not a win`;
})();

/* truth lane */
(function(){
  const lane = document.getElementById("lane");
  const rows = D.lane;
  const maxT = Math.max(...rows.flatMap(r=>r.cells.map(c=>c.time)));
  rows.forEach((r,i)=>{
    const row = document.createElement("div");
    row.className = "lane-row";
    row.style.gridColumn = "1 / -1";
    row.style.gridTemplateColumns = "minmax(150px,1fr) repeat(7,minmax(26px,1fr))";
    const label = document.createElement("div");
    label.style.fontFamily = "var(--mono)"; label.style.fontSize="10.5px";
    label.style.color="var(--ink2)"; label.style.overflow="hidden";
    label.style.textOverflow="ellipsis"; label.style.whiteSpace="nowrap";
    label.textContent = r.formula;
    row.appendChild(label);
    r.cells.forEach(c=>{
      const cell = document.createElement("div");
      cell.className = "lane-cell";
      const t = c.time;
      let color = "var(--solved-soft)";
      if (t > maxT*0.66) color = "var(--verdict-soft)";
      else if (t > maxT*0.33) color = "var(--blue-soft)";
      cell.style.background = color;
      if (r.selector === c.solver) cell.classList.add("sel");
      if (r.vbs === c.solver) cell.classList.add("vbs");
      cell.title = `${c.solver} ${fmt(t,4)}s`;
      row.appendChild(cell);
    });
    lane.appendChild(row);
  });
})();

/* corpus cards */
(function(){
  const s = D.summary;
  const cards = [
    {k:"formulas profiled", v:fmt(s.n_id,0), d:"ID + " + fmt(s.n_ood,0) + " held out as OOD"},
    {k:"train / val / test", v:`${fmt(s.n_train,0)} / ${fmt(s.n_val,0)} / ${fmt(s.n_test,0)}`, d:"cluster-safe 70 / 15 / 15"},
    {k:"families", v:fmt(Object.keys(s.families).length,0), d:"VBS solve rate " + fmt(s.ood_vbs_solve_rate) + " on OOD"},
    {k:"duplicates dropped", v:fmt(s.n_deduped,0), d:s.cluster_leak ? "leak detected" : "no cluster leak"},
  ];
  document.getElementById("corpus-cards").innerHTML = cards.map(cells).join("");
  const rows = Object.entries(s.families).map(([f,v])=>`
    <tr><td>${f}</td><td class="num">${fmt(v.n,0)}</td>
    <td class="num">${fmt(v.n_train,0)}</td><td class="num">${fmt(v.n_val,0)}</td>
    <td class="num">${fmt(v.n_test,0)}</td><td class="num">${fmt(v.vbs_solve_rate)}</td></tr>`).join("");
  document.getElementById("family-table").innerHTML =
    `<thead><tr><th>family</th><th class="num">n</th><th class="num">train</th><th class="num">val</th>
     <th class="num">test</th><th class="num">VBS solve rate</th></tr></thead><tbody>${rows}</tbody>`;
})();

/* systems bars */
(function(){
  const r = D.results;
  const order = ["vbs","best_average","fallback","majority","classical","selector","satzilla","random"];
  const sys = Object.fromEntries(r.systems.map(s=>[s.system,s]));
  const best = sys["best_average"] ? sys["best_average"].par2 : 1;
  const maxP = Math.max(...order.map(o=>sys[o]?sys[o].par2:0));
  const rows = order.filter(o=>sys[o]).map(o=>{
    const s = sys[o];
    const cls = o==="vbs"?"vbs":(o==="selector"||o==="classical")?(s.par2>best?"bad":""):(o==="satzilla"&&s.par2>best?"bad":"");
    return `<div class="bar-row"><div class="name">${o}</div>
      <div class="track"><div class="fill ${cls}" style="width:${Math.max(1,(s.par2/maxP)*100)}%"></div></div>
      <div class="val">${fmt(s.par2)} <small>&middot; ${fmt(s.solve_rate)}</small></div></div>`;
  }).join("");
  document.getElementById("systems-bars").innerHTML = rows;
})();

/* comparisons */
(function(){
  const r = D.results;
  const rows = (r.comparisons||[]).map(c=>{
    const win = c.ci_excludes_zero && c.mean_diff<0 && c.holm_reject;
    const lose = c.ci_excludes_zero && c.mean_diff>0 && c.holm_reject;
    const pill = win?`<span class="pill win">selector wins</span>`
              : lose?`<span class="pill lose">selector worse</span>`
              : `<span class="pill tie">no significant difference</span>`;
    return `<tr>
      <td>${c.comparison}<br><span class="dim" style="font-size:12px">${c.system_a} vs ${c.system_b}</span></td>
      <td class="num">${fmt(c.mean_diff)}</td>
      <td class="num">[${fmt(c.ci_low)}, ${fmt(c.ci_high)}]</td>
      <td class="num">${fmt(c.wilcoxon_p,4)}</td>
      <td class="num">${fmt(c.mcnemar_p,4)}</td>
      <td>${pill}</td></tr>`;
  }).join("");
  document.getElementById("comparison-table").innerHTML =
    `<thead><tr><th>comparison</th><th class="num">mean diff</th><th class="num">95% CI</th>
     <th class="num">Wilcoxon p</th><th class="num">McNemar p</th><th>verdict</th></tr></thead><tbody>${rows}</tbody>`;
})();

/* theta + portfolio */
(function(){
  const th = D.theta, p = D.portfolio;
  const bestAvg = p.best_average_solver;
  const cards = [
    {k:"theta (fallback cutoff)", v:fmt(th.theta,2), d:`fallback: ${th.fallback_solver||bestAvg}`},
    {k:"best average solver", v:bestAvg, d:"lowest mean time on train+val"},
    {k:"mean best-solver median", v:fmt(p.mean_best_solver_median,4)+"s", d:"feature budget anchor"},
    {k:"oracle gap", v:fmt(D.results.oracle_gap_selector,2), d:"selector / VBS, capped"},
  ];
  document.getElementById("theta-cards").innerHTML = cards.map(cells).join("");

  const curve = th.curve||[];
  const maxP = Math.max(...curve.map(c=>c.val_par2));
  const rows = curve.map(c=>`
    <div class="bar-row"><div class="name">theta ${fmt(c.theta,2)}</div>
    <div class="track"><div class="fill" style="width:${Math.max(1,(c.val_par2/maxP)*100)}%"></div></div>
    <div class="val">${fmt(c.val_par2)} <small>&middot; ${fmt(c.coverage)} cov</small></div></div>`).join("");
  document.getElementById("theta-curve").innerHTML = rows;

  const share = p.unique_share||{};
  const prows = Object.entries(share).map(([s,v])=>`
    <tr><td>${s}${s===bestAvg?' <span class="pill" style="color:var(--blue)">best avg</span>':''}</td>
    <td class="num">${fmt(v,4)}</td></tr>`).join("");
  document.getElementById("portfolio-table").innerHTML =
    `<thead><tr><th>solver</th><th class="num">strictly-best share</th></tr></thead><tbody>${prows}</tbody>`;
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
