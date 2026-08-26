"""GraSS-style heterogeneous GNN solver selector.

Follows the tripartite formulation of Zhang et al., "GraSS: Combining Graph
Neural Networks with Expert Knowledge for SAT Solver Selection" (KDD 2024),
which the project synopsis names as the primary model architecture.

The formula becomes a heterogeneous graph over three node types:

    var     one node per variable
    lit     two nodes per variable (positive and negative)
    clause  one node per clause

with four relations:

    (lit,    'in',    clause)   literal occurs in clause
    (clause, 'has',   lit)      reverse, so clauses inform literals
    (var,    'owns',  lit)      variable to each of its two polarities
    (lit,    'of',    var)      reverse, so polarity evidence reaches variables

Why the literal layer matters: the earlier implementation here built a plain
variable-clause bipartite graph and indexed edges by ``abs(lit)``, which threw
away the sign of every occurrence. A model built that way cannot distinguish
(x OR y) from (NOT x OR NOT y). Polarity is the dominant structural signal in a
CNF, and giving each literal its own node is precisely how GraSS preserves it.

Three GraSS ingredients are implemented here:

 1. domain-specific node features per type (not a shared degree scalar),
 2. positional encodings on clause nodes, and
 3. a runtime-sensitive loss: the expected cost of the predicted distribution,
    which is PAR-2 proportional by construction.

Formulas above the configured node cap are routed to the classical path at
inference time rather than being padded or truncated.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from satsel.models.classical import SOLVERS

# Node types and relations, in a fixed order so saved state dicts stay loadable.
NODE_TYPES = ("var", "lit", "clause")
RELATIONS = (
    ("lit", "in", "clause"),
    ("clause", "has", "lit"),
    ("var", "owns", "lit"),
    ("lit", "of", "var"),
)

# Feature widths per node type. Keep in sync with `cnf_to_hetero`.
CLAUSE_PE_DIM = 8
IN_DIMS = {"var": 6, "lit": 5, "clause": 3 + CLAUSE_PE_DIM}


def _torch():
    """Import torch with thread limits set before first use."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    return torch


def resolve_device(cfg: dict | None = None):
    """Pick the compute device.

    Order of precedence: the SATSEL_DEVICE environment variable, then
    cfg['gnn_device'], then CUDA when available, then CPU. Set SATSEL_DEVICE=cpu
    to force CPU on a machine that has a GPU (useful when profiling runs are
    sharing the box and you want the GPU left alone).
    """
    torch = _torch()
    want = os.environ.get("SATSEL_DEVICE") or (cfg or {}).get("gnn_device") or "auto"
    want = str(want).lower()
    if want.startswith("cuda") and torch.cuda.is_available():
        return torch.device(want)
    if want == "cpu":
        return torch.device("cpu")
    if want == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device) -> str:
    torch = _torch()
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"{device} ({name}, {total:.0f} GB)"
    return f"{device} (CPU, {os.cpu_count()} cores visible)"


def _load_cnf(row) -> object:
    from satsel.core.dimacs import parse_cnf

    path = row.get("path") if hasattr(row, "get") else row.get("path", "")
    if not path:
        raise ValueError("dataset row has no 'path' column; rerun the build step")
    return parse_cnf(Path(path).read_bytes())


def _clause_positional_encoding(nclauses: int, dim: int = CLAUSE_PE_DIM) -> np.ndarray:
    """Sinusoidal encoding of each clause's index in the formula.

    GraSS adds positional information to clause nodes because clause order is
    not arbitrary in structured instances: encoders emit related constraints in
    contiguous blocks, so nearby indices tend to be semantically related. The
    encoding is normalised by formula length, so it describes relative position
    and stays comparable across formulas of very different sizes.
    """
    if nclauses <= 0:
        return np.zeros((0, dim), dtype=np.float32)
    pos = np.arange(nclauses, dtype=np.float32)[:, None] / max(nclauses - 1, 1)
    freqs = np.power(10000.0, -np.arange(0, dim, 2, dtype=np.float32) / dim)
    angles = pos * freqs[None, :] * (2.0 * math.pi)
    pe = np.zeros((nclauses, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)
    pe[:, 1::2] = np.cos(angles)
    return pe


def _keep_columns(x: np.ndarray, keep: list[int]) -> np.ndarray:
    """Zero every column except `keep`, preserving shape."""
    out = np.zeros_like(x)
    for c in keep:
        if c < x.shape[1]:
            out[:, c] = x[:, c]
    return out


def _lit_node(lit: int, nvars: int) -> int:
    """Map a DIMACS literal to a literal-node index.

    Positive literal of variable v -> v-1; negative literal -> nvars + v-1.
    """
    v = abs(lit) - 1
    return v if lit > 0 else nvars + v


def cnf_to_hetero(cnf, nvars: int, node_feats: str = "full"):
    """Build the tripartite HeteroData for one formula.

    `node_feats` selects how much expert knowledge the model is given, which is
    what the feature-free comparison in objective O6 turns on:

      "full"       domain-designed features per node type (the default)
      "structural" degree only; everything else zeroed
      "none"       constant 1.0 -- the model sees pure topology and must learn
                   everything from message passing alone

    Dimensionality is identical in every mode, so a state dict trained under
    one mode still loads under another and only the inputs differ.

    Returns a `torch_geometric.data.HeteroData` with `var`, `lit` and `clause`
    node stores and the four relations above.
    """
    torch = _torch()
    from torch_geometric.data import HeteroData

    clauses = cnf.clauses
    nclauses = max(cnf.nclauses, 0)
    nvars = max(nvars, 1)

    pos_occ = np.zeros(nvars, dtype=np.float32)
    neg_occ = np.zeros(nvars, dtype=np.float32)
    lit_src: list[int] = []
    cls_dst: list[int] = []
    clause_len = np.zeros(nclauses, dtype=np.float32)
    clause_pos_frac = np.zeros(nclauses, dtype=np.float32)

    for ci, cl in enumerate(clauses):
        n_pos = 0
        for lit in cl:
            v = abs(lit) - 1
            if v < 0 or v >= nvars:
                continue
            if lit > 0:
                pos_occ[v] += 1
                n_pos += 1
            else:
                neg_occ[v] += 1
            lit_src.append(_lit_node(lit, nvars))
            cls_dst.append(ci)
        clause_len[ci] = len(cl)
        clause_pos_frac[ci] = n_pos / max(len(cl), 1)

    deg = pos_occ + neg_occ
    deg_scale = float(deg.max()) or 1.0
    len_scale = float(clause_len.max()) or 1.0

    # var features: degree, polarity split, balance, size context, index.
    var_x = np.zeros((nvars, IN_DIMS["var"]), dtype=np.float32)
    var_x[:, 0] = deg / deg_scale
    var_x[:, 1] = pos_occ / deg_scale
    var_x[:, 2] = neg_occ / deg_scale
    var_x[:, 3] = (pos_occ - neg_occ) / np.maximum(deg, 1.0)   # signed balance
    var_x[:, 4] = np.minimum(deg, 1.0)                          # occurs at all
    var_x[:, 5] = np.arange(nvars, dtype=np.float32) / max(nvars - 1, 1)

    # lit features: own occurrences, polarity flag, share of its variable.
    lit_occ = np.concatenate([pos_occ, neg_occ])
    own_deg = np.concatenate([deg, deg])
    lit_x = np.zeros((2 * nvars, IN_DIMS["lit"]), dtype=np.float32)
    lit_x[:, 0] = lit_occ / deg_scale
    lit_x[:nvars, 1] = 1.0                                       # positive
    lit_x[nvars:, 2] = 1.0                                       # negative
    lit_x[:, 3] = lit_occ / np.maximum(own_deg, 1.0)             # share of var
    lit_x[:, 4] = np.minimum(lit_occ, 1.0)                       # pure-literal cue

    # clause features: length, polarity mix, unit/binary cue, position.
    clause_x = np.zeros((nclauses, IN_DIMS["clause"]), dtype=np.float32)
    if nclauses:
        clause_x[:, 0] = clause_len / len_scale
        clause_x[:, 1] = clause_pos_frac
        clause_x[:, 2] = (clause_len <= 2).astype(np.float32)
        clause_x[:, 3:] = _clause_positional_encoding(nclauses)

    if node_feats == "structural":
        # Keep the degree signal only; zero the designed features.
        var_x = _keep_columns(var_x, [0])
        lit_x = _keep_columns(lit_x, [0])
        clause_x = _keep_columns(clause_x, [0])
    elif node_feats == "none":
        var_x = np.ones_like(var_x)
        lit_x = np.ones_like(lit_x)
        clause_x = np.ones_like(clause_x)
    elif node_feats != "full":
        raise ValueError(f"unknown node_feats mode {node_feats!r}")

    data = HeteroData()
    data["var"].x = torch.from_numpy(var_x)
    data["lit"].x = torch.from_numpy(lit_x)
    data["clause"].x = torch.from_numpy(clause_x)

    lit_t = torch.tensor(lit_src, dtype=torch.long)
    cls_t = torch.tensor(cls_dst, dtype=torch.long)
    data["lit", "in", "clause"].edge_index = torch.stack([lit_t, cls_t])
    data["clause", "has", "lit"].edge_index = torch.stack([cls_t, lit_t])

    var_ids = torch.arange(nvars, dtype=torch.long).repeat(2)
    lit_ids = torch.arange(2 * nvars, dtype=torch.long)
    data["var", "owns", "lit"].edge_index = torch.stack([var_ids, lit_ids])
    data["lit", "of", "var"].edge_index = torch.stack([lit_ids, var_ids])
    return data


def cnf_to_graph(cnf, nvars: int, node_feats=None):
    """Backwards-compatible shim returning (x, edge_index, edge_attr).

    Retained so older callers and the layer-ablation notebooks keep working;
    new code should use `cnf_to_hetero`, which preserves literal polarity.
    """
    data = cnf_to_hetero(cnf, nvars)
    torch = _torch()
    x = torch.cat([data["var"].x[:, :1], data["lit"].x[:, :1],
                   data["clause"].x[:, :1]], dim=0)
    ei = data["lit", "in", "clause"].edge_index
    if node_feats == "no_feats":
        x = torch.ones((x.size(0), 1), dtype=torch.float)
    return x, ei, None


def _gnn_model(n_layers: int, n_out: int, hidden: int = 64):
    """Heterogeneous GraSS-style encoder with a solver-selection head."""
    torch = _torch()
    import torch.nn as nn
    from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

    class GraSS(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleDict(
                {t: nn.Linear(IN_DIMS[t], hidden) for t in NODE_TYPES})
            self.convs = nn.ModuleList()
            for _ in range(max(n_layers, 1)):
                self.convs.append(HeteroConv(
                    {rel: SAGEConv((hidden, hidden), hidden) for rel in RELATIONS},
                    aggr="sum"))
            self.norm = nn.ModuleDict(
                {t: nn.LayerNorm(hidden) for t in NODE_TYPES})
            self.head = nn.Sequential(
                nn.Linear(hidden * len(NODE_TYPES), hidden), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(hidden, n_out))

        def forward(self, x_dict, edge_index_dict, batch_dict):
            h = {t: torch.relu(self.proj[t](x)) for t, x in x_dict.items()}
            for conv in self.convs:
                out = conv(h, edge_index_dict)
                # HeteroConv omits node types with no incoming edge this layer;
                # carry those forward unchanged rather than dropping them.
                h = {t: torch.relu(self.norm[t](out.get(t, h[t]))) for t in h}
            pooled = [global_mean_pool(h[t], batch_dict[t]) for t in NODE_TYPES]
            return self.head(torch.cat(pooled, dim=1))

    return GraSS()


def _batch_from_rows(rows: list, device, node_feats: str = "full"):
    """Build one batched HeteroData plus the per-row cost matrix."""
    torch = _torch()
    from torch_geometric.data import Batch

    graphs = []
    costs = []
    for row in rows:
        cnf = _load_cnf(row)
        graphs.append(cnf_to_hetero(cnf, cnf.nvars, node_feats))
        costs.append([float(row[f"{s}_weight"]) for s in SOLVERS])
    batch = Batch.from_data_list(graphs).to(device)
    cost = torch.tensor(costs, dtype=torch.float, device=device)
    return batch, cost


def _training_rows(dataset, cap: int):
    """Train on informative rows, mirroring the classical selector.

    The previous filter here was `tie_flag == 0`. On an easy corpus almost
    every formula is a tie, so that filter emptied the training set and the
    model collapsed onto one solver. `informative` (a real runtime spread
    across the portfolio) is the filter the classical path settled on, and the
    two must agree or their results are not comparable.
    """
    train = dataset[(dataset["split"] == "train") & dataset["vbs_solved"]].copy()
    if "informative" in train.columns:
        usable = train[train["informative"].astype(bool)]
    else:
        usable = train[train["tie_flag"] == 0]
    note = ""
    if len(usable) < 30:
        note = (f"only {len(usable)} informative training rows; "
                f"falling back to all {len(train)} vbs_solved rows")
        usable = train
    usable = usable[usable["nvars"] + usable["nclauses"] <= cap]
    return usable, note


def _minibatches(rows: list, batch_size: int, rng):
    order = rng.permutation(len(rows))
    for i in range(0, len(order), batch_size):
        yield [rows[j] for j in order[i:i + batch_size]]


def train_gnn(dataset, cfg: dict, seed: int, epochs: int = 40) -> dict:
    torch = _torch()

    cap = int(cfg.get("gnn_node_cap", 500000))
    device = resolve_device(cfg)
    train_df, note = _training_rows(dataset, cap)
    if train_df.empty:
        raise ValueError("no training rows within the GNN node cap")

    val = dataset[(dataset["split"] == "val") & dataset["vbs_solved"]].copy()
    val = val[val["nvars"] + val["nclauses"] <= cap]

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        # Batches vary in size per formula, so let cuDNN re-tune rather than
        # locking in an algorithm chosen for the first batch shape.
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

    rows = train_df.to_dict("records")
    batch_size = int(cfg.get("gnn_batch_size", 16))
    lr = float(cfg.get("gnn_lr", 1e-3))
    warnings: list[str] = [note] if note else []

    n_layers_choices = [int(x) for x in cfg.get("gnn_layer_ablation", [1, 2, 4])]
    # Objective O6 asks how the GNN compares against a feature-free approach.
    # Running both here answers it directly and on identical splits: "full" is
    # the expert-knowledge model, "none" sees pure topology. The gap between
    # them is the value of the hand-designed node features, which is exactly
    # the feature-engineering-bottleneck question Zhang et al. (2026) raise.
    feats_choices = [str(x) for x in cfg.get("gnn_node_feats_ablation", ["full", "none"])]
    best_state = None
    best_par2 = float("inf")
    best_layers = n_layers_choices[0]
    best_feats = feats_choices[0]
    ablation: list[dict] = []

    for node_feats in feats_choices:
        for n_layers in n_layers_choices:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = _gnn_model(n_layers, len(SOLVERS)).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            last_loss = float("nan")
            for _epoch in range(epochs):
                model.train()
                for batch_rows in _minibatches(rows, batch_size, rng):
                    batch, cost = _batch_from_rows(batch_rows, device, node_feats)
                    opt.zero_grad()
                    logits = model(batch.x_dict, batch.edge_index_dict,
                                   {t: batch[t].batch for t in NODE_TYPES})
                    p = torch.softmax(logits, dim=1)
                    loss = (p * cost).sum(dim=1).mean()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.detach().cpu())
            vp, hist = _val_par2(model, val, cap, device, node_feats)
            ablation.append({"node_feats": node_feats, "n_layers": n_layers,
                             "val_par2": vp, "final_train_loss": last_loss,
                             "choice_hist": hist})
            if vp < best_par2:
                best_par2, best_layers, best_feats = vp, n_layers, node_feats
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise ValueError("GNN training produced no usable model")

    # Headline comparison: best configuration per node-feature mode.
    feature_free: dict[str, float] = {}
    for mode in feats_choices:
        scores = [e["val_par2"] for e in ablation
                  if e["node_feats"] == mode and np.isfinite(e["val_par2"])]
        if scores:
            feature_free[mode] = float(min(scores))
    if "full" in feature_free and "none" in feature_free:
        delta = feature_free["none"] - feature_free["full"]
        warnings.append(
            f"node features change validation PAR-2 by {delta:+.4f}s "
            f"(full={feature_free['full']:.4f}s, feature-free={feature_free['none']:.4f}s)")

    max_share = 0.0
    for entry in ablation:
        if (entry["n_layers"] == best_layers
                and entry.get("node_feats") == best_feats and entry["choice_hist"]):
            total = sum(entry["choice_hist"].values()) or 1
            max_share = max(entry["choice_hist"].values()) / total
    if max_share > 0.9:
        warnings.append(
            f"GNN predicts one solver on {max_share:.0%} of validation rows; "
            "treat this as a collapsed selector, not a result")

    out_dir = Path(cfg["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "architecture": "grass-hetero-tripartite",
        "n_layers": best_layers,
        "node_feats": best_feats,
        "hidden": 64,
        "solver_list": list(SOLVERS),
        "in_dims": dict(IN_DIMS),
        "node_cap": cap,
        "val_par2": float(best_par2),
        "seed": int(seed),
        "epochs": int(epochs),
        "device_trained_on": str(device),
        "n_train": int(len(rows)),
        "layer_ablation": ablation,
        "feature_free_comparison": feature_free,
        "warnings": warnings,
    }, out_dir / "gnn.pt")
    return {"val_par2": float(best_par2), "n_layers": best_layers,
            "node_feats": best_feats, "device": describe_device(device),
            "n_train": len(rows), "layer_ablation": ablation,
            "feature_free_comparison": feature_free, "warnings": warnings}


def _val_par2(model, val, cap: int, device,
              node_feats: str = "full") -> tuple[float, dict]:
    """Validation PAR-2 plus the histogram of solvers the model chose."""
    torch = _torch()
    if val.empty:
        return float("inf"), {}
    model.eval()
    total, count = 0.0, 0
    hist: dict[str, int] = {}
    with torch.no_grad():
        for _fid, row in val.iterrows():
            if row["nvars"] + row["nclauses"] > cap:
                continue
            cnf = _load_cnf(row)
            batch, _ = _batch_from_rows([row.to_dict()], device, node_feats)
            logits = model(batch.x_dict, batch.edge_index_dict,
                           {t: batch[t].batch for t in NODE_TYPES})
            pred = SOLVERS[int(torch.argmax(logits, dim=1)[0])]
            hist[pred] = hist.get(pred, 0) + 1
            t = float(row[f"{pred}_time_s"])
            total += t if bool(row[f"{pred}_solved"]) else 2.0 * t
            count += 1
            del cnf
    return total / max(count, 1), hist


def predict_gnn(model_state: dict, cnf, nvars: int, solver_list) -> tuple[str, float]:
    """Predict a solver for one formula. Signature is stable for callers."""
    torch = _torch()
    from torch_geometric.data import Batch

    solvers = list(solver_list) or list(SOLVERS)
    device = resolve_device(None)
    n_layers = int(model_state.get("n_layers", 2))
    model = _gnn_model(n_layers, len(solvers),
                       hidden=int(model_state.get("hidden", 64)))
    model.load_state_dict(model_state["model_state"])
    model.to(device)
    model.eval()

    node_feats = str(model_state.get("node_feats", "full"))
    batch = Batch.from_data_list([cnf_to_hetero(cnf, nvars, node_feats)]).to(device)
    with torch.no_grad():
        logits = model(batch.x_dict, batch.edge_index_dict,
                       {t: batch[t].batch for t in NODE_TYPES})
        p = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(p))
    return solvers[idx], float(p[idx])
