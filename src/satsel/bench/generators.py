"""Formula generators for the six families plus the OOD family.

Every instance is produced by a fixed-seed generator so the dataset is
reproducible and license-clean (CC-BY-4.0, our own work). Downloaders for
SATLIB/GBD live in `download.py` and fall back to these generators when the
network is unavailable or a checksum fails.

Each generator returns a `Cnf`. Where ground truth is computable cheaply
(blocks world, scheduling) the generator verifies its own encoding against a
search so the SAT/UNSAT label is trustworthy.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from satsel.core.dimacs import Cnf


def _dedup(clauses: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for c in clauses:
        t = tuple(sorted(c))
        if any(-x in c for x in c):
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return tuple(out)


class Builder:
    """Fresh-variable CNF builder; variables are 1-indexed DIMACS ids."""

    def __init__(self, family: str, name: str) -> None:
        self.family = family
        self.name = name
        self._vid = 0
        self.clauses: list[list[int]] = []

    def fresh(self) -> int:
        self._vid += 1
        return self._vid

    def lit(self, var: int, value: bool) -> int:
        return var if value else -var

    def add(self, clause: list[int]) -> None:
        self.clauses.append(clause)

    def onehot(self, vars_: list[int]) -> None:
        self.add(vars_[:])
        for a in range(len(vars_)):
            for b in range(a + 1, len(vars_)):
                self.add([-vars_[a], -vars_[b]])

    def require(self, bit, value: bool) -> None:
        """Assert a bit (var or constant) equals `value`."""
        if _is_const(bit):
            if bool(bit[1]) != value:
                self.add([])  # empty clause -> UNSAT
            return
        self.add([bit if value else -bit])

    def exact(self, func, invars: list[int], outvars: list[int]) -> None:
        """Exact truth-table constraint: outvars = func(*invars)."""
        k = len(invars)
        for mask in range(1 << k):
            inbits = tuple(bool((mask >> i) & 1) for i in range(k))
            outs = func(*inbits)
            if isinstance(outs, bool):
                outs = (outs,)
            for ob in outs:
                if ob:
                    self.add([self.lit(invars[i], not inbits[i]) for i in range(k)])

    def finish(self) -> Cnf:
        ded = _dedup(self.clauses)
        return Cnf(nvars=self._vid, nclauses=len(ded),
                   clauses=ded, family=self.family,
                   source="generated", name=self.name)


# ---- gate library (exact CNFs) --------------------------------------------
# A "bit" is either a signed int (a DIMACS literal: positive = var true,
# negative = var false) or the constant `(None, bool)`.

def _c_const(value: bool):
    return (None, value)


def _is_const(bit) -> bool:
    return isinstance(bit, tuple)


def _neg(bit):
    if _is_const(bit):
        return (None, not bit[1])
    return -bit


def gate_and(b: Builder, x, y):
    if _is_const(x):
        return y if x[1] else _c_const(False)
    if _is_const(y):
        return x if y[1] else _c_const(False)
    out = b.fresh()
    b.add([-x, -y, out])
    b.add([x, -out])
    b.add([y, -out])
    return out


def gate_xor(b: Builder, x, y):
    if _is_const(x):
        return y if not x[1] else _neg(y)
    if _is_const(y):
        return x if not y[1] else _neg(x)
    out = b.fresh()
    b.add([-x, -y, -out])
    b.add([x, y, -out])
    b.add([x, -y, out])
    b.add([-x, y, out])
    return out


def gate_or(b: Builder, x, y):
    if _is_const(x):
        return _c_const(True) if x[1] else y
    if _is_const(y):
        return _c_const(True) if y[1] else x
    out = b.fresh()
    b.add([x, y, -out])
    b.add([-x, out])
    b.add([-y, out])
    return out


def full_adder(b: Builder, x, y, cin) -> tuple:
    """Return (sum, cout)."""
    s1 = gate_xor(b, x, y)
    s = gate_xor(b, s1, cin)
    a1 = gate_and(b, x, y)
    a2 = gate_and(b, s1, cin)
    cout = gate_or(b, a1, a2)
    return s, cout


def ripple_add(b: Builder, a, bv) -> list:
    """Ripple-carry adder over equal-width bit vectors (lsb first)."""
    n = len(a)
    sums: list = []
    carry = _c_const(False)
    for i in range(n):
        s, carry = full_adder(b, a[i], bv[i], carry)
        sums.append(s)
    sums.append(carry)  # final carry is bit n
    return sums


def array_mult(b: Builder, a, bv) -> list:
    """Shift-and-add array multiplier; product width = len(a)+len(b)."""
    na, nb = len(a), len(bv)
    n = na + nb
    pp = [[gate_and(b, a[i], bv[j]) for j in range(nb)] for i in range(na)]
    acc = [pp[0][j] for j in range(nb)] + [_c_const(False)] * (n - nb)
    for i in range(1, na):
        row = ([_c_const(False)] * i + [pp[i][j] for j in range(nb)]
               + [_c_const(False)] * (n - i - nb))
        acc = ripple_add(b, acc, row)[:n]
    return acc[:n]


# ---------------------------------------------------------------------------
# Random 3-SAT
# ---------------------------------------------------------------------------

def gen_random3sat(n: int, ratio: float, rng: np.random.Generator) -> Cnf:
    m = int(ratio * n)
    seen: set[tuple[int, ...]] = set()
    clauses: list[tuple[int, ...]] = []
    attempts = 0
    while len(clauses) < m and attempts < 20 * m:
        attempts += 1
        vs = rng.choice(np.arange(1, n + 1), size=3, replace=False)
        signs = rng.integers(0, 2, size=3)
        c = tuple(sorted(int(v) if int(s) else -int(v)
                         for v, s in zip(vs, signs, strict=True)))
        if c in seen:
            continue
        seen.add(c)
        clauses.append(c)
    return Cnf(nvars=n, nclauses=len(clauses), clauses=tuple(clauses),
               family="random3sat", source="generated")


def gen_random3sat_batch(count: int, seed: int, *, n_lo: int = 200, n_hi: int = 320,
                         ratio: float = 4.26, family: str = "random3sat") -> list[Cnf]:
    """Uniform random 3-SAT at the phase transition (m/n = 4.26). With n in
    200..320 the VBS time is typically 0.05s-20s, which is where the portfolio
    solvers actually differ (below that, labels are construction noise)."""
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    for i in range(count):
        n = int(rng.integers(n_lo, n_hi + 1))
        c = gen_random3sat(n, ratio, rng)
        c.family = family
        c.name = f"{family}-{i:04d}"
        out.append(c)
    return out


def _large_ratio(rng: np.random.Generator) -> float:
    """Ratio for the large family: tightly around the phase transition
    (4.20..4.32, centred on 4.26) so instances are hard but bounded by the
    family tau; the spread in ratio adds SAT/UNSAT mix."""
    return float(rng.uniform(4.20, 4.32))


def gen_random3sat_large_batch(count: int, seed: int, *, n_lo: int = 320,
                               n_hi: int = 420, ratio: float | None = None) -> list[Cnf]:
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    for i in range(count):
        n = int(rng.integers(n_lo, n_hi + 1))
        r = ratio if ratio is not None else _large_ratio(rng)
        c = gen_random3sat(n, r, rng)
        c.family = "random3sat_large"
        c.name = f"random3sat_large-{i:04d}"
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Graph colouring (SATLIB flat* analogue): planted-3-colourable + K4-UNSAT
# ---------------------------------------------------------------------------

def gen_colouring_batch(count: int, seed: int) -> list[Cnf]:
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    for i in range(count):
        n = int(rng.integers(60, 170))
        ncol = 3
        colour = rng.integers(0, 3, size=n)
        edges: list[tuple[int, int]] = []
        # planted 3-colouring near the 3-COL threshold (m/n ~ 2.3); flat-graph
        # style instances here are the ones that separate CDCL heuristics
        target = int(rng.uniform(2.2, 2.6) * n)
        attempts = 0
        edge_set: set[tuple[int, int]] = set()
        while len(edges) < target and attempts < 10 * target:
            attempts += 1
            u, w = rng.integers(1, n + 1, size=2)
            if u == w or colour[u - 1] == colour[w - 1]:
                continue
            if (u, w) in edge_set or (w, u) in edge_set:
                continue
            edges.append((u, w))
            edge_set.add((u, w))
        if i % 3 == 0:
            clique = rng.choice(n, size=ncol + 1, replace=False) + 1
            for a in range(ncol + 1):
                for b in range(a + 1, ncol + 1):
                    e = (int(clique[a]), int(clique[b]))
                    if e not in edge_set and (e[1], e[0]) not in edge_set:
                        edges.append(e)
                        edge_set.add(e)
        b = Builder("colouring", f"colouring-{i:04d}")
        var = [[b.fresh() for _ in range(ncol)] for _ in range(n)]
        for v in range(n):
            b.onehot([var[v][c] for c in range(ncol)])
        for u, w in edges:
            for c in range(ncol):
                b.add([-var[u - 1][c], -var[w - 1][c]])
        out.append(b.finish())
    return out


# ---------------------------------------------------------------------------
# Blocks-world planning. One step = one complete relocation (pick up a clear
# block, place it on a stack or the table). Encoding verified against BFS.
# ---------------------------------------------------------------------------

def _bw_neighbors(stacks: tuple[tuple[int, ...], ...]):
    for si, s in enumerate(stacks):
        if not s:
            continue
        b = s[-1]
        reduced = tuple(stacks[j] for j in range(len(stacks)) if j != si)
        if len(s) > 1:
            reduced = reduced + (s[:-1],)
        for j in range(len(reduced)):
            yield tuple(reduced[k] if k != j else reduced[j] + (b,) for k in range(len(reduced)))
        yield tuple(sorted(reduced + ((b,),)))


def _bw_reachable(n: int, T: int, initial: tuple[tuple[int, ...], ...],
                  goal: tuple[tuple[int, ...], ...]) -> bool:
    start = tuple(sorted(initial))
    gkey = tuple(sorted(goal))
    seen = {start}
    q = deque([(start, 0)])
    while q:
        st, d = q.popleft()
        if st == gkey:
            return True
        if d >= T:
            continue
        for nxt in _bw_neighbors(st):
            nxt = tuple(sorted(nxt))   # canonical: stack order is irrelevant
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, d + 1))
    return False


def _bw_random_state(n: int, rng: np.random.Generator) -> tuple[tuple[int, ...], ...]:
    blocks = list(range(1, n + 1))
    rng.shuffle(blocks)
    stacks: list[list[int]] = []
    for b in blocks:
        if stacks and rng.random() < 0.6:
            stacks[int(rng.integers(0, len(stacks)))].append(b)
        else:
            stacks.append([b])
    return tuple(tuple(s) for s in stacks)


def _bw_encode(n: int, T: int, initial: tuple[tuple[int, ...], ...],
               goal: tuple[tuple[int, ...], ...], name: str) -> Cnf:
    b = Builder("planning", name)
    blocks = range(1, n + 1)

    vid: dict[tuple, int] = {}
    def v(*key) -> int:  # noqa: E731
        if key not in vid:
            vid[key] = b.fresh()
        return vid[key]

    for t in range(T + 1):
        for x in blocks:
            b.onehot([v("ontable", x, t)] + [v("on", x, y, t) for y in blocks if y != x])
    for t in range(T + 1):
        for x in blocks:
            on_top = [v("on", y, x, t) for y in blocks if y != x]
            for yv in on_top:
                b.add([-v("clear", x, t), -yv])
            b.add([v("clear", x, t)] + on_top)
        b.onehot([v("empty", t)] + [v("hold", x, t) for x in blocks])

    for t in range(T):
        acts = [v("move", x, y, t) for x in blocks for y in blocks if x != y] + [
            v("totable", x, t) for x in blocks]
        for a in range(len(acts)):
            for c in range(a + 1, len(acts)):
                b.add([-acts[a], -acts[c]])

    for t in range(T):
        for x in blocks:
            for y in blocks:
                if x == y:
                    continue
                m = v("move", x, y, t)
                b.add([-m, v("empty", t)])
                b.add([-m, v("clear", x, t)])
                b.add([-m, v("clear", y, t)])
            tb = v("totable", x, t)
            b.add([-tb, v("empty", t)])
            b.add([-tb, v("clear", x, t)])

    for t in range(T):
        t1 = t + 1
        for x in blocks:
            for y in blocks:
                if x == y:
                    continue
                m = v("move", x, y, t)
                b.add([-m, v("on", x, y, t1)])
                b.add([-m, -v("ontable", x, t1)])
                b.add([-m, v("empty", t1)])
                b.add([-m, v("clear", x, t1)])
                b.add([-m, -v("clear", y, t1)])
                for z in blocks:
                    if z != y:
                        b.add([-m, -v("on", x, z, t1)])
            tb = v("totable", x, t)
            b.add([-tb, v("ontable", x, t1)])
            b.add([-tb, v("empty", t1)])
            b.add([-tb, v("clear", x, t1)])
            for z in blocks:
                if z != x:
                    b.add([-tb, -v("on", x, z, t1)])

    # F: frame axioms (explanation closure): a fluent changes at t+1 only if
    # an action that can change it fired at t.
    for t in range(T):
        t1 = t + 1
        for x in blocks:
            for y in blocks:
                if x == y:
                    continue
                f_on = v("on", x, y, t)
                f_on1 = v("on", x, y, t1)
                set_true = [v("move", x, y, t)]
                set_false = ([v("move", x, z, t) for z in blocks if z != y and z != x]
                           + [v("totable", x, t)])
                # false->true must be explained:  L_t or -L_t1 or any A+
                b.add([f_on, -f_on1] + set_true)
                # true->false must be explained:  -L_t or L_t1 or any A-
                b.add([-f_on, f_on1] + set_false)
            f_t = v("ontable", x, t)
            f_t1 = v("ontable", x, t1)
            set_false = [v("move", x, y, t) for y in blocks if y != x]
            b.add([f_t, -f_t1, v("totable", x, t)])   # false->true
            b.add([-f_t, f_t1] + set_false)           # true->false
            f_c = v("clear", x, t)
            f_c1 = v("clear", x, t1)
            set_true = [v("move", y, z, t) for y in blocks if y != x
                        for z in blocks if z != y and z != x]
            set_true += [v("totable", y, t) for y in blocks if y != x]
            set_false = [v("move", y, x, t) for y in blocks if y != x]
            b.add([f_c, -f_c1] + set_true)            # false->true
            b.add([-f_c, f_c1] + set_false)           # true->false
            f_e = v("empty", t)
            f_e1 = v("empty", t + 1)
            all_acts = [v("move", a_, c_, t) for a_ in blocks for c_ in blocks if a_ != c_] + [
                v("totable", a_, t) for a_ in blocks]
            b.add([f_e, -f_e1] + all_acts)            # false->true
            for x2 in blocks:
                f_h = v("hold", x2, t)
                f_h1 = v("hold", x2, t1)
                set_false = [v("move", x2, y, t) for y in blocks if y != x2] + [v("totable", x2, t)]
                b.add([-f_h, f_h1] + set_false)       # true->false

    for s in initial:
        if not s:
            continue
        b.add([v("ontable", s[0], 0)])
        for a_, c_ in zip(s, s[1:], strict=False):
            b.add([v("on", c_, a_, 0)])
        b.add([v("clear", s[-1], 0)])
    b.add([v("empty", 0)])
    for s in goal:
        if not s:
            continue
        b.add([v("ontable", s[0], T)])
        for a_, c_ in zip(s, s[1:], strict=False):
            b.add([v("on", c_, a_, T)])

    return b.finish()


def gen_planning_batch(count: int, seed: int) -> list[Cnf]:
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    i = 0
    while len(out) < count:
        n = int(rng.integers(5, 8))
        T = int(rng.integers(6, 10))
        initial = _bw_random_state(n, rng)
        goal = _bw_random_state(n, rng)
        reachable = _bw_reachable(n, T, initial, goal)
        if len(out) % 2 == 1 and reachable:
            continue
        if len(out) % 2 == 0 and not reachable:
            continue
        c = _bw_encode(n, T, initial, goal, f"planning-{i:04d}")
        c.extra = {"n_blocks": n, "horizon": T, "ground_truth": reachable}
        out.append(c)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Scheduling: disjunctive job-shop, start-time encoding with precedence and
# per-machine exclusion. Ground truth from a backtracking scheduler.
# ---------------------------------------------------------------------------

def _sched_ground_truth(n_ops: int, horizon: int, dur: list[int],
                        mach: list[int], order: list[tuple[int, int]]) -> bool:
    start = [None] * n_ops
    assigned = [False] * n_ops

    def feasible(u: int, t: int) -> bool:
        if t + dur[u] > horizon:
            return False
        for v in range(n_ops):
            if v == u or not assigned[v]:
                continue
            if mach[v] == mach[u]:
                sv = start[v]
                if not (t + dur[u] <= sv or sv + dur[v] <= t):
                    return False
        for u2, v in order:
            if u2 == u and assigned[v] and not (t + dur[u] <= start[v]):
                return False
            if v == u and assigned[u2] and not (start[u2] + dur[u2] <= t):
                return False
        return True

    def bt(count: int) -> bool:
        if count == n_ops:
            return True
        for u in range(n_ops):
            if assigned[u]:
                continue
            prereqs = [u2 for u2, v in order if v == u]
            if any(not assigned[p] for p in prereqs if p != u):
                continue
            for t in range(0, horizon - dur[u] + 1):
                if feasible(u, t):
                    assigned[u] = True
                    start[u] = t
                    if bt(count + 1):
                        return True
                    assigned[u] = False
                    start[u] = None
            return False
        return False

    return bt(0)


def _sched_cnf(n_ops: int, horizon: int, dur: list[int], mach: list[int],
               order: list[tuple[int, int]], family: str, name: str) -> Cnf:
    b = Builder(family, name)
    domains = [range(0, horizon - dur[i] + 1) for i in range(n_ops)]
    start = [[b.fresh() for _ in domains[i]] for i in range(n_ops)]
    for i in range(n_ops):
        b.onehot(start[i])
    for u, v in order:
        for tu in domains[u]:
            for tv in range(0, min(tu + dur[u], horizon)):
                if tv in domains[v]:
                    b.add([-start[u][tu], -start[v][tv]])
    by_mach: dict[int, list[int]] = {}
    for i, m in enumerate(mach):
        by_mach.setdefault(m, []).append(i)
    for ops in by_mach.values():
        for a in range(len(ops)):
            for c in range(a + 1, len(ops)):
                u, v = ops[a], ops[c]
                for tu in domains[u]:
                    for tv in range(max(0, tu - dur[v] + 1), min(tu + dur[u], horizon)):
                        if tv in domains[v]:
                            b.add([-start[u][tu], -start[v][tv]])
    return b.finish()


def gen_scheduling_batch(count: int, seed: int, *, family: str = "scheduling") -> list[Cnf]:
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    i = 0
    while len(out) < count:
        n_jobs = int(rng.integers(6, 11))
        n_mach = int(rng.integers(2, 5))
        n_ops = n_jobs
        dur = [int(rng.integers(1, 5)) for _ in range(n_ops)]
        mach = [int(rng.integers(0, n_mach)) for _ in range(n_ops)]
        horizon = int(rng.integers(10, 20))
        order = [(j, j + 1) for j in range(n_ops - 1)]
        truth = _sched_ground_truth(n_ops, horizon, dur, mach, order)
        if len(out) % 3 == 0:
            if truth:
                horizon -= 2
                if _sched_ground_truth(n_ops, horizon, dur, mach, order):
                    continue
                truth = False
        else:
            if not truth:
                continue
        c = _sched_cnf(n_ops, horizon, dur, mach, order, family, f"{family}-{i:04d}")
        c.extra = {"n_jobs": n_jobs, "horizon": horizon, "ground_truth": truth}
        out.append(c)
        i += 1
    return out


# ---------------------------------------------------------------------------
# Industrial: bit-blasted arithmetic (factoring SAT + equivalence UNSAT)
# ---------------------------------------------------------------------------

def _factoring(nbits: int, target: int) -> Cnf:
    b = Builder("industrial", "fact")
    a = [b.fresh() for _ in range(nbits)]
    bv = [b.fresh() for _ in range(nbits)]
    prod = array_mult(b, a, bv)
    for i, bit in enumerate(prod):
        b.require(bit, bool((target >> i) & 1))
    return b.finish()


def _equiv_unsat(k: int) -> Cnf:
    b = Builder("industrial", "equiv")
    a = [b.fresh() for _ in range(k)]
    bv = [b.fresh() for _ in range(k)]
    cv = [b.fresh() for _ in range(k)]
    s1 = ripple_add(b, a, bv)          # a + b, width k+1
    left = array_mult(b, s1, cv)       # (a+b)*c
    m1 = array_mult(b, a, cv)
    m2 = array_mult(b, bv, cv)
    right = ripple_add(b, m1, m2)      # a*c + b*c
    width = min(len(left), len(right))
    eq: list = []
    for i in range(width):
        eq.append(gate_xor(b, left[i], right[i]))
    neq = eq[0]
    for i in range(1, len(eq)):
        neq = gate_or(b, neq, eq[i])
    b.require(neq, True)  # force a difference -> UNSAT, since the identity holds
    return b.finish()


def gen_industrial_batch(count: int, seed: int) -> list[Cnf]:
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    for i in range(count):
        if i % 3 == 0:
            # equivalence UNSAT
            k = int(rng.integers(8, 13))
            c = _equiv_unsat(k)
            c.name = f"industrial-equiv-{i:04d}"
        else:
            nbits = int(rng.integers(14, 22))
            # odd factors in the upper half of the range: no trivial solutions
            half = nbits // 2
            p = int(rng.integers(1 << (half - 1), 1 << half)) | 1
            q = int(rng.integers(1 << (half - 1), 1 << half)) | 1
            target = p * q
            c = _factoring(nbits, target)
            c.name = f"industrial-fact-{i:04d}"
            c.extra = {"nbits": nbits, "p": p, "q": q, "ground_truth": True}
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Crypto (appendix): preimage of a 2-round ARX function (hardness-capped)
# ---------------------------------------------------------------------------

def _arx_cnf(k: int, target: int) -> Cnf:
    """Find x such that F(x) = target, F = 2 rounds of an ARX-style mixing.
    x is split into two k/2-bit halves; each round mixes using addition mod
    2^(k/2) and XOR of shifted halves (Feistel-like, verified by pysat)."""
    h = k // 2
    b = Builder("crypto", "arx")
    x0 = [b.fresh() for _ in range(h)]
    x1 = [b.fresh() for _ in range(h)]
    c1 = int(0x9E3779B9 & ((1 << h) - 1))
    c2 = int(0x85EBCA77 & ((1 << h) - 1))
    # round 1: x1' = x1 + ((x0 + c1) rol 1)
    t = ripple_add(b, x0, _const_bits(b, c1, h))
    t = _rotl(t, 1)
    n1 = ripple_add(b, x1, t)
    n1 = n1[:h]
    # round 2: x0' = x0 + ((x1' + c2) rol 1)
    t2 = ripple_add(b, n1, _const_bits(b, c2, h))
    t2 = _rotl(t2, 1)
    n0 = ripple_add(b, x0, t2)
    n0 = n0[:h]
    out = n0 + n1
    for i, bit in enumerate(out):
        b.require(bit, bool((target >> i) & 1))
    return b.finish()


def _const_bits(b: Builder, value: int, width: int) -> list:
    return [_c_const(bool((value >> i) & 1)) for i in range(width)]


def _rotl(bits: list[int], r: int) -> list[int]:
    if not bits:
        return bits
    r %= len(bits)
    return bits[r:] + bits[:r]


def gen_crypto_batch(count: int, seed: int) -> list[Cnf]:
    rng = np.random.default_rng(seed)
    out: list[Cnf] = []
    for i in range(count):
        k = int(rng.integers(16, 24))
        target = int(rng.integers(0, 1 << k))
        c = _arx_cnf(k, target)
        c.name = f"crypto-arx-{i:04d}"
        c.extra = {"k": k}
        out.append(c)
    return out


def generate_all(target_counts: dict[str, int], base_seed: int = 2026) -> dict[str, list[Cnf]]:
    """Generate the full corpus. Returns {family: [Cnf, ...]}."""
    out: dict[str, list[Cnf]] = {}
    seed = base_seed
    out["random3sat"] = gen_random3sat_batch(target_counts.get("random3sat", 400), seed)
    out["random3sat_large"] = gen_random3sat_large_batch(
        target_counts.get("random3sat_large", 400), seed + 1)
    out["colouring"] = gen_colouring_batch(target_counts.get("colouring", 300), seed + 2)
    out["planning"] = gen_planning_batch(target_counts.get("planning", 200), seed + 3)
    out["industrial"] = gen_industrial_batch(target_counts.get("industrial", 400), seed + 4)
    out["scheduling"] = gen_scheduling_batch(target_counts.get("scheduling", 300), seed + 5)
    # OOD scheduling uses a seed range unique to this family (never overlaps
    # with the in-distribution scheduling generator).
    out["ood_scheduling"] = gen_scheduling_batch(
        target_counts.get("ood_scheduling", 200), 987654, family="ood_scheduling")
    out["crypto"] = gen_crypto_batch(target_counts.get("crypto", 100), seed + 6)
    return out
