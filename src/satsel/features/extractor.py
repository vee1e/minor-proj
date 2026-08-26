"""Feature extraction: turn a CNF into a fixed-size numeric vector.

Groups are extracted cheapest-first so a hard time budget can be respected:
syntactic (O(n+m)), then graph (near-linear), then a short in-process solver
probe stopped by the solver's own interrupt(). Any group that would exceed the
remaining budget is marked missing.
"""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from satsel.core.dimacs import Cnf

FEATURE_SCHEMA_VERSION = "1.1"
PROBE_BUDGET_S = 0.2
# Measured cost of the graph group per literal (pure Python, near-linear in
# total literals): ~1.4e-6 s/literal on this machine, kept at 3e-6 for
# headroom. Used only to decide whether the group fits in the budget.
GRAPH_S_PER_LITERAL = 3.0e-6
# interrupt() sets a flag the solver checks periodically, so it does not stop
# instantly. Measured overshoot is a near-constant ~15 ms across budgets from
# 20 ms to 200 ms (solver construction + interrupt latency + teardown), so the
# probe is given a margin plus only a fraction of the time remaining. The
# budget is therefore respected within roughly one interrupt interval, which is
# the strongest guarantee available without killing the process.
PROBE_STOP_MARGIN_S = 0.02
PROBE_BUDGET_FRACTION = 0.8
# MiniSat: one of the few pysat backends whose bindings implement interrupt(),
# which an in-process budgeted probe depends on. CaDiCaL and Lingeling raise
# NotImplementedError and would run unbounded; kissat404 segfaults in-process
# on some Windows builds. See core.solvers.INTERRUPTIBLE_SOLVERS.
PROBE_SOLVER = "minisat22"


@dataclass(frozen=True)
class FeatureVector:
    values: tuple[float, ...]
    names: tuple[str, ...]
    missing: tuple[bool, ...]
    cost_s: float


def _syntactic(cnf: Cnf, clens: list[int], var_occ: list[int]) -> dict[str, float]:
    nvars = cnf.nvars
    nclauses = cnf.nclauses
    nvars = max(nvars, 1)
    nclauses = max(nclauses, 1)
    f: dict[str, float] = {}

    def stat(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0,
                    "q25": 0.0, "q75": 0.0}
        s = sorted(vals)
        lo = s[0]
        hi = s[-1]
        mid = statistics.median(s)
        mean = sum(s) / len(s)
        var = sum((x - mean) ** 2 for x in s) / len(s)
        q25 = s[len(s) // 4]
        q75 = s[(3 * len(s)) // 4]
        return {"mean": mean, "median": mid, "max": hi, "min": lo,
                "std": math.sqrt(var), "q25": q25, "q75": q75}

    cs = stat([float(x) for x in clens])
    for k, v in cs.items():
        f[f"syn_clen_{k}"] = v
    os_ = stat([float(o) for o in var_occ])
    for k, v in os_.items():
        f[f"syn_varocc_{k}"] = v

    f["syn_nvars"] = float(nvars)
    f["syn_nclauses"] = float(nclauses)
    f["syn_ratio"] = nclauses / nvars

    buckets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for clen in clens:
        buckets[min(clen, 6) - 1 if clen <= 6 else 5] += 1.0
    for i, v in enumerate(buckets):
        f[f"syn_clen_share_{i+1 if i < 5 else 6}"] = v / nclauses

    ob = [0.0] * 6
    for c in var_occ:
        ob[min(c, 5)] += 1.0
    for i, v in enumerate(ob):
        f[f"syn_varocc_share_{i if i < 5 else 5}"] = v / nvars

    pos_lits = 0
    total_lits = 0
    units = 0
    binaries = 0
    horn1 = 0
    horn2 = 0
    for cl in cnf.clauses:
        clen = len(cl)
        total_lits += clen
        npos = sum(1 for lit in cl if lit > 0)
        pos_lits += npos
        if npos <= 1:
            horn1 += 1
        if npos >= 2:
            horn2 += 1
        if clen == 1:
            units += 1
        elif clen == 2:
            binaries += 1
    f["syn_pos_lit_share"] = pos_lits / max(total_lits, 1)
    f["syn_horn_share"] = horn1 / nclauses
    f["syn_nonhorn_share"] = horn2 / nclauses
    f["syn_unit_count"] = units / nclauses
    f["syn_binary_count"] = binaries / nclauses
    f["syn_avg_lits"] = total_lits / nclauses
    f["syn_unit_per_var"] = units / nvars
    f["syn_binary_per_var"] = binaries / nvars

    pos_only = 0
    neg_only = 0
    once = 0
    used = 0
    pos_occ = [0] * (cnf.nvars + 1)
    neg_occ = [0] * (cnf.nvars + 1)
    for cl in cnf.clauses:
        for lit in cl:
            if lit > 0:
                pos_occ[lit] += 1
            else:
                neg_occ[-lit] += 1
    for v in range(1, cnf.nvars + 1):
        p, ng = pos_occ[v], neg_occ[v]
        o = p + ng
        if o > 0:
            used += 1
        if o == 1:
            once += 1
        if p > 0 and ng == 0:
            pos_only += 1
        elif ng > 0 and p == 0:
            neg_only += 1
    f["syn_pos_only_share"] = pos_only / nvars
    f["syn_neg_only_share"] = neg_only / nvars
    f["syn_once_share"] = once / nvars
    f["syn_used_var_share"] = used / nvars
    return f


def _graph(cnf: Cnf, clens: list[int], var_occ: list[int]) -> dict[str, float]:
    nvars = max(cnf.nvars, 1)
    f: dict[str, float] = {}

    # variable-variable co-occurrence graph (edges from clauses with len <= 12)
    deg = [0] * (cnf.nvars + 1)
    pairs: Counter = Counter()
    for cl in cnf.clauses:
        clen = len(cl)
        if clen <= 1 or clen > 12:
            continue
        vs = sorted(set(abs(lit) for lit in cl))
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                u, w = vs[a], vs[b]
                pairs[(u, w)] += 1
                deg[u] += 1
                deg[w] += 1
    deg = deg[1:]
    edges = len(pairs)
    possible = max(nvars * (nvars - 1) / 2.0, 1.0)
    f["gr_density"] = edges / possible
    dstat = _stat([float(d) for d in deg])
    for k, v in dstat.items():
        f[f"gr_vv_deg_{k}"] = v
    dbuckets = [0.0] * 6
    for d in deg:
        dbuckets[min(d, 5)] += 1.0
    for i, v in enumerate(dbuckets):
        f[f"gr_vv_deg_share_{i if i < 5 else 5}"] = v / nvars
    nconn, largest = _components(cnf.nvars, pairs)
    f["gr_components_per_var"] = nconn / nvars
    f["gr_largest_comp"] = largest / nvars
    if deg:
        mean_deg = sum(deg) / len(deg)
        second = sum(d * d for d in deg) / len(deg)
        f["gr_second_moment"] = second
        f["gr_mean_neighbor_deg"] = second / max(mean_deg, 1e-9)
    else:
        f["gr_second_moment"] = 0.0
        f["gr_mean_neighbor_deg"] = 0.0

    # structural: mean clause length at incident clauses (var side)
    wsum = [0.0] * (cnf.nvars + 1)
    for cl in cnf.clauses:
        clen = len(cl)
        for lit in cl:
            wsum[abs(lit)] += clen
    ws = [wsum[v] / var_occ[v - 1] if var_occ[v - 1] else 0.0
          for v in range(1, cnf.nvars + 1)]
    f["gr_mean_incident_clen"] = sum(ws) / nvars

    in_long = 0
    for cl in cnf.clauses:
        if len(cl) >= 4:
            in_long += len(cl)
    f["gr_long_lit_share"] = in_long / max(sum(clens), 1)

    f["gr_complexity"] = math.log1p(cnf.nvars) * math.log1p(cnf.nclauses)
    return f


def _stat(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
    s = sorted(vals)
    mean = sum(s) / len(s)
    var = sum((x - mean) ** 2 for x in s) / len(s)
    return {"mean": mean, "median": statistics.median(s), "max": s[-1],
            "min": s[0], "std": math.sqrt(var)}


def _components(nvars: int, pairs: Counter) -> tuple[int, int]:
    parent = list(range(nvars + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (u, w) in pairs:
        union(u, w)
    sizes: Counter = Counter()
    for v in range(1, nvars + 1):
        sizes[find(v)] += 1
    return len(sizes), max(sizes.values()) if sizes else 0


PROBE_FEATURES = (
    "probe_runtime_s", "probe_solved", "probe_satisfiable",
    "probe_conflicts_log", "probe_decisions_log", "probe_propagations_log",
    "probe_restarts_log", "probe_props_per_dec", "probe_confl_per_dec",
    "probe_decisions_per_var",
)


def _probe(cnf: Cnf, budget_s: float, seed: int) -> dict[str, float]:
    """Short DPLL probe, run in this process under the solver's own interrupt.

    This used to write the formula to a temp file and spawn
    `python -m satsel.profiling.worker`, costing roughly 0.45 s of interpreter
    startup and re-parsing per instance. On an easy formula that is several
    hundred times the solve time it was measuring, and it broke the
    tau_feat <= 5% budget the evaluation protocol promises. Running in-process
    reduces it to the probe budget itself.

    Beyond timing, the solver's own counters give the search-space statistics
    the feature taxonomy asks for (Nudelman et al.: DPLL probe depth and
    unit-propagation behaviour). Counts are log-compressed because they span
    orders of magnitude across families, and the ratios are scale-free so they
    stay comparable between a 100-variable and a 100,000-variable formula.
    """
    from satsel.core.solvers import solve_probe

    out = solve_probe(cnf.clauses, cnf.nvars, PROBE_SOLVER, budget_s, seed=seed)
    if not out.ok:
        # A failed probe must look missing, not like a real run whose counters
        # happened to be zero; otherwise training learns a fake pattern.
        return {}

    f: dict[str, float] = {}
    f["probe_runtime_s"] = float(min(out.runtime_s, budget_s))
    if out.solved:
        f["probe_solved"] = 1.0
        f["probe_satisfiable"] = 1.0 if out.satisfiable else 0.0

    st = out.stats
    conflicts = float(st.get("conflicts", 0))
    decisions = float(st.get("decisions", 0))
    propagations = float(st.get("propagations", 0))
    restarts = float(st.get("restarts", 0))

    f["probe_conflicts_log"] = math.log1p(max(conflicts, 0.0))
    f["probe_decisions_log"] = math.log1p(max(decisions, 0.0))
    f["probe_propagations_log"] = math.log1p(max(propagations, 0.0))
    f["probe_restarts_log"] = math.log1p(max(restarts, 0.0))
    f["probe_props_per_dec"] = propagations / max(decisions, 1.0)
    f["probe_confl_per_dec"] = conflicts / max(decisions, 1.0)
    f["probe_decisions_per_var"] = decisions / max(float(cnf.nvars), 1.0)
    return f

def _group_names_for(cnf: Cnf) -> dict[str, list[str]]:
    clens = [len(cl) for cl in cnf.clauses]
    var_occ = [0] * (cnf.nvars + 1)
    for cl in cnf.clauses:
        for lit in cl:
            var_occ[abs(lit)] += 1
    var_occ = var_occ[1:]
    return {
        "syntactic": list(_syntactic(cnf, clens, var_occ)),
        "graph": list(_graph(cnf, clens, var_occ)),
        "probe": list(PROBE_FEATURES),
    }


_GROUP_NAMES: dict[str, list[str]] = _group_names_for(
    Cnf(nvars=3, nclauses=2, clauses=((1, 2, 3), (-1, 3)), family="", source=""))


def extract_features(cnf: Cnf, budget_s: float, seed: int = 0, *,
                     enable_probe: bool = True) -> FeatureVector:
    """Extract features within a hard budget (cheapest groups first).

    Every feature in the fixed schema always appears; a group skipped by the
    budget (or by `enable_probe=False`) contributes its features with value
    0.0 and missing=True.

    Timing uses perf_counter, not monotonic: monotonic has a ~15.6 ms
    resolution on Windows, which is coarse enough that a 200 ms budget was
    being computed from an elapsed time that frequently read as zero.
    """
    t0 = time.perf_counter()
    clens = [len(cl) for cl in cnf.clauses]
    var_occ = [0] * (cnf.nvars + 1)
    for cl in cnf.clauses:
        for lit in cl:
            var_occ[abs(lit)] += 1
    var_occ = var_occ[1:]

    ran_syntactic = True
    syn = _syntactic(cnf, clens, var_occ)

    # Groups are atomic: the budget gates whether a group is *entered*, because
    # a half-computed group yields no usable features. So entry needs a cost
    # estimate, not just "is there time left" -- the graph pass is near-linear
    # in total literals, and entering it with 5 ms left used to blow the budget
    # by 60 ms on a 250-variable formula. GRAPH_S_PER_LITERAL is measured; the
    # 2x headroom covers slower machines.
    n_literals = sum(clens)
    elapsed = time.perf_counter() - t0
    predicted_graph_s = GRAPH_S_PER_LITERAL * max(n_literals, 1) * 2.0
    ran_graph = elapsed + predicted_graph_s <= budget_s
    graph = _graph(cnf, clens, var_occ) if ran_graph else {}
    remaining = budget_s - (time.perf_counter() - t0)
    probe_budget = min(PROBE_BUDGET_S,
                       max(0.0, remaining * PROBE_BUDGET_FRACTION - PROBE_STOP_MARGIN_S))
    ran_probe = enable_probe and probe_budget >= 0.02
    probe = _probe(cnf, probe_budget, seed) if ran_probe else {}
    if ran_probe and not probe:
        # probe failed (non-interruptible solver or exception): mark missing
        ran_probe = False

    names: list[str] = []
    values: list[float] = []
    missing: list[bool] = []
    for gname, ran in (("syntactic", ran_syntactic), ("graph", ran_graph), ("probe", ran_probe)):
        feats = (syn if gname == "syntactic" else graph if gname == "graph" else probe)
        if ran:
            for name, val in feats.items():
                names.append(name)
                values.append(float(val))
                missing.append(False)
        else:
            for name in _GROUP_NAMES[gname]:
                names.append(name)
                values.append(0.0)
                missing.append(True)
    return FeatureVector(values=tuple(values), names=tuple(names),
                         missing=tuple(missing), cost_s=time.perf_counter() - t0)


FEATURE_NAMES: tuple[str, ...] = tuple(
    name for gname in ("syntactic", "graph", "probe") for name in _GROUP_NAMES[gname]
)
FEATURE_GROUPS: dict[str, list[str]] = {k: list(v) for k, v in _GROUP_NAMES.items()}


def extract_one(cnf: Cnf, budget_s: float, seed: int = 0, *,
                enable_probe: bool = True) -> dict:
    fv = extract_features(cnf, budget_s, seed, enable_probe=enable_probe)
    out = {"formula_id": cnf.name, "feat_cost_s": fv.cost_s,
           "missing_frac": sum(fv.missing) / max(len(fv.missing), 1)}
    for name, val, miss in zip(fv.names, fv.values, fv.missing, strict=True):
        out[f"f_{name}"] = 0.0 if miss else val
    return out


def _stable_seed(formula_id: str) -> int:
    """Stable per-formula seed. str.__hash__ is salted per process, so it
    would give each pool worker a different seed and differ across reruns."""
    return int(hashlib.md5(formula_id.encode()).hexdigest()[:8], 16)


def _extract_row_worker(args) -> dict:
    from satsel.core.dimacs import parse_cnf

    row, budget, enable_probe = args
    c = parse_cnf(Path(row["path"]).read_bytes())
    c.name = row["formula_id"]
    rec = extract_one(c, budget, seed=_stable_seed(row["formula_id"]),
                      enable_probe=enable_probe)
    rec["formula_id"] = row["formula_id"]
    return rec


def batch_extract(cfg: dict, manifest_path: str | Path, workers: int = 4,
                  enable_probe: bool = True) -> None:
    """Extract features for every formula in the manifest; write parquet +
    schema json into cfg['paths']['features']."""
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    manifest = Path(manifest_path)
    df = pd.read_parquet(manifest)
    out_dir = Path(cfg["paths"]["features"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same budget formula the oracle uses at inference time, so training and
    # serving features are extracted under identical conditions. The previous
    # `frac * 0.5` here disagreed with feat_budget_s whenever the portfolio's
    # mean best-solver median exceeded 4 s.
    from satsel.core.config import feat_budget_s

    portfolio_path = Path(cfg["paths"]["models"]) / "portfolio.json"
    mean_best = 0.5
    try:
        if portfolio_path.exists():
            mean_best = float(json.loads(portfolio_path.read_text())
                              .get("mean_best_solver_median", 0.5))
    except Exception:  # noqa: BLE001  # fall back to the default mean
        pass
    budget = feat_budget_s(cfg, mean_best)

    args = [(r, budget, enable_probe) for r in df.to_dict("records")]
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_extract_row_worker, a) for a in args]
        for fut in as_completed(futs):
            records.append(fut.result())
    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values("formula_id").reset_index(drop=True)
    out_df.to_parquet(out_dir / "features.parquet", index=False)

    schema = {
        "schema_version": str(cfg.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": {k: v for k, v in FEATURE_GROUPS.items()},
    }

    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))
