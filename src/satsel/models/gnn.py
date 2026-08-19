"""Graph neural network selector.

Reads each formula as a variable-clause bipartite graph and trains with the
expected-cost loss: L = mean over batch of sum_s p(s|f) * w(i,s). Formulas
larger than the node cap are routed to the classical path at inference time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from satsel.models.classical import SOLVERS


def _load_cnf(row) -> object:
    from satsel.core.dimacs import parse_cnf

    path = row.get("path") if hasattr(row, "get") else row.get("path", "")
    if not path:
        raise ValueError("dataset row has no 'path' column; rerun the build step")
    return parse_cnf(Path(path).read_bytes())


def cnf_to_graph(cnf, nvars: int, node_feats=None):
    """Return (x, edge_index, edge_attr) for the variable-clause bipartite
    graph. Variables are nodes 0..nvars-1; clauses are nodes
    nvars..nvars+nclauses-1."""
    import torch

    clauses = cnf.clauses
    nclauses = cnf.nclauses
    n = nvars + nclauses
    var_deg = [0] * nvars
    for cl in clauses:
        for lit in cl:
            var_deg[abs(lit) - 1] += 1

    src: list[int] = []
    dst: list[int] = []
    clause_idx = nvars
    for cl in clauses:
        for lit in cl:
            src.append(abs(lit) - 1)
            dst.append(clause_idx)
        clause_idx += 1

    cl_len = [len(cl) for cl in clauses]
    cl_sign = [sum(1 for lit in cl if lit > 0) / max(len(cl), 1) for cl in clauses]

    x = np.zeros((n, 4), dtype=np.float32)
    for v in range(nvars):
        x[v, 0] = var_deg[v]
        x[v, 3] = var_deg[v]
    for i in range(nclauses):
        x[nvars + i, 1] = cl_len[i]
        x[nvars + i, 2] = cl_sign[i]
        x[nvars + i, 3] = cl_len[i]

    if node_feats == "degree_only":
        keep = np.zeros((n, 1), dtype=np.float32)
        keep[:, 0] = x[:, 3]
        x = keep
    elif node_feats == "no_feats":
        x = np.ones((n, 1), dtype=np.float32)

    x_t = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x_t, edge_index, None


def _gnn_model(in_dim: int, n_layers: int, n_out: int, hidden: int = 64):
    import torch
    import torch.nn as nn
    import torch_geometric.nn as gnn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            self.convs.append(gnn.GCNConv(in_dim, hidden))
            for _ in range(n_layers - 1):
                self.convs.append(gnn.GCNConv(hidden, hidden))
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                      nn.Linear(64, n_out))

        def forward(self, x, edge_index, batch):
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            x = gnn.global_mean_pool(x, batch)
            return self.head(x)

    return Net()


def _single_data(cnf, nvars, in_dim: int):
    import torch
    import torch_geometric

    x, ei, _ = cnf_to_graph(cnf, nvars)
    if x.size(1) != in_dim:
        pad = torch.zeros((x.size(0), in_dim - x.size(1)), dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    data = torch_geometric.data.Data(
        x=x, edge_index=ei, batch=torch.zeros(x.size(0), dtype=torch.long))
    return data


def _load_batch(rows: list, in_dim: int, cap: int):
    """Build a batched PyG graph from formula rows (uses their paths)."""
    import torch
    import torch_geometric

    xs = []
    eis = []
    ys = []
    offsets = []
    offset = 0
    for row in rows:
        cnf = _load_cnf(row)
        n = cnf.nvars + cnf.nclauses
        if n > cap:
            continue
        x, ei, _ = cnf_to_graph(cnf, cnf.nvars)
        if x.size(1) < in_dim:
            pad = torch.zeros((x.size(0), in_dim - x.size(1)), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        xs.append(x)
        eis.append(ei + offset)
        offsets.append(offset)
        ys.append(int(row["best_solver_idx"]))
        offset += x.size(0)
    if not xs:
        raise ValueError("no formula fits within the node cap for this batch")
    data = torch_geometric.data.Data(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(eis, dim=1),
        batch=torch.cat([torch.full((x.size(0),), i, dtype=torch.long)
                         for i, x in enumerate(xs)], dim=0),
        y=torch.tensor(ys, dtype=torch.long),
    )
    return data


def _minibatches(train, cap: int, batch_size: int = 32):
    rows = []
    for _fid, r in train.iterrows():
        if r["nvars"] + r["nclauses"] <= cap:
            rows.append(r.to_dict())
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    for i in range(0, len(rows), batch_size):
        yield rows[i:i + batch_size]


def train_gnn(dataset, cfg: dict, seed: int, epochs: int = 40) -> dict:
    import torch

    cap = int(cfg.get("gnn_node_cap", 500000))
    train = dataset[(dataset["split"] == "train") & dataset["vbs_solved"] &
                    (dataset["tie_flag"] == 0)].copy()
    train = train[train["nvars"] + train["nclauses"] <= cap]
    if train.empty:
        raise ValueError("no training rows within the GNN node cap")

    solver_idx = {s: k for k, s in enumerate(SOLVERS)}
    train["best_solver_idx"] = train["best_solver"].map(solver_idx)

    val = dataset[(dataset["split"] == "val") & dataset["vbs_solved"]].copy()
    val = val[val["nvars"] + val["nclauses"] <= cap]

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    in_dim = 4
    n_layers_choices = [int(x) for x in cfg.get("gnn_layer_ablation", [1, 2, 4])]
    best_state = None
    best_par2 = float("inf")

    for n_layers in n_layers_choices:
        torch.manual_seed(seed)
        model = _gnn_model(in_dim, n_layers, len(SOLVERS))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _epoch in range(epochs):
            model.train()
            for batch_rows in _minibatches(train, cap):
                data = _load_batch(batch_rows, in_dim, cap)
                opt.zero_grad()
                logits = model(data.x, data.edge_index, data.batch)
                p = torch.softmax(logits, dim=1)
                loss = _expected_cost_loss(p, batch_rows)
                loss.backward()
                opt.step()
        vp = _val_par2(model, val, in_dim, cap)
        if vp < best_par2:
            best_par2 = vp
            best_state = (n_layers, model.state_dict())

    n_layers, state_dict = best_state
    out_dir = Path(cfg["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": state_dict,
        "n_layers": n_layers,
        "hidden": 64,
        "solver_list": list(SOLVERS),
        "in_dim": in_dim,
        "node_cap": cap,
        "val_par2": float(best_par2),
        "seed": int(seed),
    }, Path(cfg["paths"]["models"]) / "gnn.pt")
    return {"val_par2": float(best_par2), "n_layers": n_layers}


def _expected_cost_loss(p, batch_rows) -> object:
    import torch

    wb = [[row[f"{s}_weight"] for s in SOLVERS] for row in batch_rows]
    wt = torch.tensor(wb, dtype=torch.float)
    return (p * wt).sum(dim=1).mean()


def _val_par2(model, val, in_dim: int, cap: float) -> float:
    import torch

    if val.empty:
        return float("inf")
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for _fid, row in val.iterrows():
            if row["nvars"] + row["nclauses"] > cap:
                continue
            cnf = _load_cnf(row)
            data = _single_data(cnf, cnf.nvars, in_dim)
            logits = model(data.x, data.edge_index, data.batch)
            pred = SOLVERS[int(torch.argmax(logits, dim=1)[0])]
            t = float(row[f"{pred}_time_s"])
            total += t if bool(row[f"{pred}_solved"]) else 2.0 * t
            count += 1
    return total / max(count, 1)


def predict_gnn(model_state: dict, cnf, nvars: int, solver_list) -> tuple[str, float]:
    import torch

    in_dim = int(model_state.get("in_dim", 4))
    n_layers = int(model_state.get("n_layers", 2))
    model = _gnn_model(in_dim, n_layers, len(solver_list),
                       hidden=int(model_state.get("hidden", 64)))
    model.load_state_dict(model_state["model_state"])
    model.eval()
    data = _single_data(cnf, nvars, in_dim)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.batch)
        p = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(p))
    return solver_list[idx], float(p[idx])
