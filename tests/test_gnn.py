"""Tests for the GraSS-style heterogeneous GNN selector.

Skipped entirely when the optional `gnn` extra is not installed, so the base
test suite still runs on a machine without PyTorch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from satsel.core.dimacs import parse_cnf  # noqa: E402
from satsel.models.gnn import (  # noqa: E402
    CLAUSE_PE_DIM,
    IN_DIMS,
    NODE_TYPES,
    RELATIONS,
    _clause_positional_encoding,
    _training_rows,
    cnf_to_hetero,
    predict_gnn,
    resolve_device,
    train_gnn,
)

SOLVERS = ("minisat22", "glucose4", "cadical153", "maplesat", "lingeling",
           "maplecm", "kissat404")


class TestTripartiteGraph:
    def test_polarity_is_preserved(self):
        """The regression test for this module's original defect.

        The previous implementation indexed edges by abs(lit), so (x OR y) and
        (NOT x OR NOT y) produced byte-identical graphs and the model could not
        tell them apart. Literal nodes must keep them distinct.
        """
        pos = parse_cnf(b"p cnf 2 1\n1 2 0\n")
        neg = parse_cnf(b"p cnf 2 1\n-1 -2 0\n")
        g_pos = cnf_to_hetero(pos, pos.nvars)
        g_neg = cnf_to_hetero(neg, neg.nvars)

        lits_pos = set(g_pos["lit", "in", "clause"].edge_index[0].tolist())
        lits_neg = set(g_neg["lit", "in", "clause"].edge_index[0].tolist())
        assert lits_pos != lits_neg
        # positive literals occupy [0, nvars), negatives [nvars, 2*nvars)
        assert lits_pos == {0, 1}
        assert lits_neg == {2, 3}

    def test_node_types_and_shapes(self):
        cnf = parse_cnf(b"p cnf 3 2\n1 -2 3 0\n-1 2 0\n")
        g = cnf_to_hetero(cnf, cnf.nvars)
        assert g["var"].x.shape == (3, IN_DIMS["var"])
        assert g["lit"].x.shape == (6, IN_DIMS["lit"])
        assert g["clause"].x.shape == (2, IN_DIMS["clause"])
        for rel in RELATIONS:
            assert rel in g.edge_index_dict, f"missing relation {rel}"

    def test_every_literal_links_to_its_variable(self):
        cnf = parse_cnf(b"p cnf 4 2\n1 -2 0\n3 -4 0\n")
        g = cnf_to_hetero(cnf, cnf.nvars)
        src, dst = g["var", "owns", "lit"].edge_index
        assert src.numel() == 2 * cnf.nvars
        # each variable owns exactly two literal nodes
        counts = np.bincount(src.numpy(), minlength=cnf.nvars)
        assert (counts == 2).all()
        assert set(dst.tolist()) == set(range(2 * cnf.nvars))

    def test_clause_edges_match_occurrences(self):
        cnf = parse_cnf(b"p cnf 3 2\n1 -2 3 0\n-1 2 0\n")
        g = cnf_to_hetero(cnf, cnf.nvars)
        n_edges = g["lit", "in", "clause"].edge_index.shape[1]
        assert n_edges == 5  # 3 literals + 2 literals
        # the reverse relation mirrors it exactly
        fwd = g["lit", "in", "clause"].edge_index
        rev = g["clause", "has", "lit"].edge_index
        assert torch.equal(fwd[0], rev[1]) and torch.equal(fwd[1], rev[0])

    def test_empty_and_unit_clauses_do_not_crash(self):
        cnf = parse_cnf(b"p cnf 2 2\n1 0\n-2 0\n")
        g = cnf_to_hetero(cnf, cnf.nvars)
        assert g["clause"].x.shape[0] == 2
        assert torch.isfinite(g["clause"].x).all()
        assert torch.isfinite(g["var"].x).all()


class TestPositionalEncoding:
    def test_shape_and_range(self):
        pe = _clause_positional_encoding(16)
        assert pe.shape == (16, CLAUSE_PE_DIM)
        assert np.isfinite(pe).all()
        assert pe.min() >= -1.0001 and pe.max() <= 1.0001

    def test_distinguishes_positions(self):
        pe = _clause_positional_encoding(32)
        assert not np.allclose(pe[0], pe[-1])

    def test_degenerate_sizes(self):
        assert _clause_positional_encoding(0).shape == (0, CLAUSE_PE_DIM)
        assert _clause_positional_encoding(1).shape == (1, CLAUSE_PE_DIM)


def _synthetic_dataset(tmp_path, n=90):
    """Dataset where the fastest solver is a learnable function of structure."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        # Two regimes: wide clauses favour solver A, narrow favour solver B.
        wide = i % 2 == 0
        nvars = int(rng.integers(12, 20))
        nclauses = int(rng.integers(14, 24))
        clauses = []
        for _ in range(nclauses):
            k = 3 if wide else 2
            vs = rng.choice(np.arange(1, nvars + 1), size=k, replace=False)
            signs = rng.integers(0, 2, size=k) * 2 - 1
            clauses.append(" ".join(str(int(v * s)) for v, s in zip(vs, signs, strict=True)) + " 0")
        text = f"p cnf {nvars} {nclauses}\n" + "\n".join(clauses) + "\n"
        path = tmp_path / f"f{i:03d}.cnf"
        path.write_text(text)

        best = "minisat22" if wide else "glucose4"
        row = {
            "formula_id": f"f{i:03d}",
            "path": str(path),
            "family": "synthetic",
            "split": "train" if i < int(n * 0.75) else "val",
            "nvars": nvars,
            "nclauses": nclauses,
            "vbs_solved": True,
            "tie_flag": 0,
            "informative": True,
            "best_solver": best,
            "best_time_s": 1.0,
        }
        for s in SOLVERS:
            fast = s == best
            row[f"{s}_time_s"] = 1.0 if fast else 6.0
            row[f"{s}_solved"] = True
            row[f"{s}_weight"] = 1.0 if fast else 6.0
            row[f"{s}_regret_s"] = 0.0 if fast else 5.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("formula_id")


class TestTrainingFilter:
    def test_uses_informative_not_tie_flag(self, tmp_path):
        """The bug this fixes: tie_flag emptied the training set.

        Every row here is flagged as a tie but is informative. The old filter
        (tie_flag == 0) would have selected nothing.
        """
        ds = _synthetic_dataset(tmp_path, n=60)
        ds["tie_flag"] = 1
        ds["informative"] = True
        rows, note = _training_rows(ds, cap=10**9)
        assert len(rows) > 0
        assert note == ""

    def test_falls_back_and_says_so(self, tmp_path):
        ds = _synthetic_dataset(tmp_path, n=60)
        ds["informative"] = False
        rows, note = _training_rows(ds, cap=10**9)
        assert len(rows) > 0          # fell back rather than training on nothing
        assert "falling back" in note

    def test_node_cap_excludes_large_formulas(self, tmp_path):
        ds = _synthetic_dataset(tmp_path, n=30)
        rows, _ = _training_rows(ds, cap=1)
        assert rows.empty


class TestTrainAndPredict:
    def test_end_to_end(self, tmp_path):
        ds = _synthetic_dataset(tmp_path, n=90)
        cfg = {
            "paths": {"models": tmp_path / "models"},
            "gnn_layer_ablation": [1],
            "gnn_node_cap": 10**9,
            "gnn_batch_size": 8,
        }
        res = train_gnn(ds, cfg, seed=100, epochs=4)

        assert np.isfinite(res["val_par2"])
        assert res["n_train"] > 0
        assert (tmp_path / "models" / "gnn.pt").exists()
        assert len(res["layer_ablation"]) == 1

        state = torch.load(tmp_path / "models" / "gnn.pt",
                           map_location="cpu", weights_only=False)
        assert state["architecture"] == "grass-hetero-tripartite"
        assert state["in_dims"] == IN_DIMS

        cnf = parse_cnf(b"p cnf 3 2\n1 -2 3 0\n-1 2 0\n")
        solver, conf = predict_gnn(state, cnf, cnf.nvars, state["solver_list"])
        assert solver in SOLVERS
        assert 0.0 <= conf <= 1.0

    def test_collapse_is_reported(self, tmp_path):
        """A model that picks one solver everywhere must say so."""
        ds = _synthetic_dataset(tmp_path, n=60)
        # make one solver best everywhere -> the model should collapse onto it
        for s in SOLVERS:
            ds[f"{s}_weight"] = 1.0 if s == "minisat22" else 9.0
        ds["best_solver"] = "minisat22"
        cfg = {
            "paths": {"models": tmp_path / "models"},
            "gnn_layer_ablation": [1],
            "gnn_node_cap": 10**9,
            "gnn_batch_size": 8,
        }
        res = train_gnn(ds, cfg, seed=1, epochs=4)
        hist = res["layer_ablation"][0]["choice_hist"]
        assert sum(hist.values()) > 0
        if max(hist.values()) / sum(hist.values()) > 0.9:
            assert any("collapsed" in w for w in res["warnings"])


class TestDevice:
    def test_resolve_device_respects_env(self, monkeypatch):
        monkeypatch.setenv("SATSEL_DEVICE", "cpu")
        assert resolve_device(None).type == "cpu"

    def test_node_types_are_stable(self):
        # saved state dicts key off this order; changing it breaks old models
        assert NODE_TYPES == ("var", "lit", "clause")
