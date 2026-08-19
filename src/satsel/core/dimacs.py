"""DIMACS CNF parser and lightweight formula container.

Handles standard CNF (conjunctive normal form: an AND of clauses, each clause
an OR of literals). Strict about header-vs-content consistency so the oracle can
reject malformed or oversized input before allocating solver memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256


class DimacsError(ValueError):
    """Raised when CNF text is malformed or headers do not match content."""


@dataclass
class Cnf:
    """A parsed CNF formula. Clauses are kept as a tuple of literal tuples."""

    nvars: int
    nclauses: int
    clauses: tuple[tuple[int, ...], ...]
    # Canonical hash: literals sorted within each clause, clauses sorted.
    canonical_hash: str = field(init=False)
    # Variable-renaming-invariant hash (best effort; sorted multiset of sizes
    # and sign patterns, not a true graph-isomorphism hash).
    renamesafe_hash: str = field(init=False)
    source: str = "memory"
    family: str = "unknown"
    name: str = ""
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.canonical_hash = canonical_hash(self.clauses, self.nvars)
        self.renamesafe_hash = renamesafe_hash(self.clauses)

    def to_dimacs(self) -> bytes:
        lines = [f"p cnf {self.nvars} {self.nclauses}"]
        lines.extend(" ".join(map(str, c)) + " 0" for c in self.clauses)
        return ("\n".join(lines) + "\n").encode()


def parse_cnf(
    text: str | bytes,
    *,
    max_vars: int | None = None,
    max_clauses: int | None = None,
    max_literals: int | None = None,
) -> Cnf:
    """Parse CNF text, enforcing optional size caps. Raises DimacsError on
    malformed input or header/content mismatch."""
    if isinstance(text, bytes):
        text = text.decode("ascii", errors="replace")
    elif not isinstance(text, str):
        raise DimacsError("CNF must be str or bytes")

    header: tuple[int, int] | None = None
    clauses: list[tuple[int, ...]] = []
    lit_count = 0

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("c", "p")):
            if line.startswith("p"):
                parts = line.split()
                if len(parts) != 4 or parts[1] != "cnf":
                    raise DimacsError(f"line {lineno}: bad header {line!r}")
                if header is not None:
                    raise DimacsError(f"line {lineno}: duplicate header")
                nvars, nclauses = int(parts[2]), int(parts[3])
                if nvars < 0 or nclauses < 0:
                    raise DimacsError(f"line {lineno}: negative header counts")
                header = (nvars, nclauses)
            continue
        # clause line
        nums = line.split()
        clause = []
        for tok in nums:
            if tok == "0":
                break
            lit = int(tok)
            if lit == 0:
                break
            clause.append(lit)
            lit_count += 1
            if lit_count > (max_literals if max_literals else 1 << 62):
                raise DimacsError("literal cap exceeded")
        clauses.append(tuple(clause))

    if header is None:
        raise DimacsError("missing p cnf header")

    nvars, nclauses = header
    if nvars > (max_vars if max_vars else 1 << 62):
        raise DimacsError("variable cap exceeded")
    if nclauses != len(clauses):
        raise DimacsError(
            f"header says {nclauses} clauses but content has {len(clauses)}"
        )
    if max_clauses and nclauses > max_clauses:
        raise DimacsError("clause cap exceeded")

    for c in clauses:
        for lit in c:
            if abs(lit) > nvars:
                raise DimacsError(f"literal {lit} out of range for {nvars} vars")

    return Cnf(nvars=nvars, nclauses=nclauses, clauses=tuple(clauses))


def canonical_hash(clauses: tuple[tuple[int, ...], ...], nvars: int) -> str:
    """Hash a formula canonically: literals sorted within clauses, clauses
    sorted, then the byte stream hashed. Two formulas that differ only in
    clause order or literal order collide."""
    normalized = [tuple(sorted(c)) for c in clauses]
    normalized.sort()
    blob = bytearray()
    blob.extend(nvars.to_bytes(4, "little"))
    for c in normalized:
        blob.extend(len(c).to_bytes(2, "little"))
        for lit in c:
            blob.extend(lit.to_bytes(4, "little", signed=True))
    return sha256(bytes(blob)).hexdigest()


def renamesafe_hash(clauses: tuple[tuple[int, ...], ...]) -> str:
    """A hash that is invariant to variable renaming (best effort): buckets by
    clause size, and within a size bucket records the number of positive and
    negative literals. Not a true isomorphism hash."""
    buckets: dict[int, tuple[int, int]] = {}
    for c in clauses:
        k = len(c)
        pos = sum(1 for lit in c if lit > 0)
        neg = k - pos
        p, n = buckets.get(k, (0, 0))
        buckets[k] = (p + pos, n + neg)
    blob = bytearray()
    for k in sorted(buckets):
        blob.extend(k.to_bytes(2, "little"))
        blob.extend(buckets[k][0].to_bytes(4, "little"))
        blob.extend(buckets[k][1].to_bytes(4, "little"))
    return sha256(bytes(blob)).hexdigest()
