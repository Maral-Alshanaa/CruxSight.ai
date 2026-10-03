"""
Analysis helpers for the learned causal structure (CausalInferenceLayer).

Pure numpy/scipy; no torch import, so every function is unit-testable on tiny
hand-made graphs (tests/test_notears_analysis.py).

Conventions
-----------
G is a (d, d) non-negative matrix: G[i, j] = weight of edge i -> j. The
diagonal is ignored (zeroed) everywhere. In CruxSight G is `causal_graph_mean`,
the batch-mean of sigmoid(W_raw) * sigmoid(encoder(h)) with zero diagonal.

h_impl      -- exactly what CausalInferenceLayer.acyclicity_constraint computes:
               tr(I + A/d + A^2/(2 d^2)) - d, with A = G*G (Hadamard). This is
               the order-2 truncation of tr(exp(A/d)) - d.
h_exact     -- same /d scaling, full series: tr(expm(A/d)) - d. The gap
               h_exact - h_impl is purely the truncation (cycles of length >= 3).
h_standard  -- Zheng et al. NOTEARS: tr(expm(A)) - d (no /d scaling).
"""
from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _offdiag(G) -> np.ndarray:
    M = np.array(G, dtype=np.float64, copy=True)
    np.fill_diagonal(M, 0.0)
    return M


# ----------------------------- acyclicity ----------------------------------

def h_impl(G) -> float:
    M = _offdiag(G)
    d = M.shape[0]
    A = M * M
    T = np.eye(d) + A / d + (A @ A) / (2 * d * d)
    return float(np.trace(T) - d)


def h_exact(G) -> float:
    M = _offdiag(G)
    d = M.shape[0]
    A = M * M
    return float(np.trace(expm(A / d)) - d)


def h_standard(G) -> float:
    M = _offdiag(G)
    d = M.shape[0]
    return float(np.trace(expm(M * M)) - d)


def binarize(G, tau: float) -> np.ndarray:
    return _offdiag(G) > tau


def is_dag(G, tau: float) -> bool:
    """True iff the binary graph {G > tau} has no directed cycle (Kahn)."""
    B = binarize(G, tau)
    indeg = B.sum(axis=0).astype(int)
    stack = [i for i in range(B.shape[0]) if indeg[i] == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in np.nonzero(B[u])[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return seen == B.shape[0]


def min_tau_for_dag(G) -> float:
    """
    Smallest threshold tau (taken from the off-diagonal weights themselves)
    such that {G > tau} is acyclic. Removing edges can never create a cycle,
    so acyclicity is monotone in tau and binary search is exact.
    """
    M = _offdiag(G)
    d = M.shape[0]
    vals = np.unique(M[~np.eye(d, dtype=bool)])
    lo, hi = 0, len(vals) - 1          # vals[hi] removes every edge -> DAG
    while lo < hi:
        mid = (lo + hi) // 2
        if is_dag(M, vals[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(vals[lo])


def n_two_cycles(G, tau: float) -> int:
    B = binarize(G, tau)
    return int(np.triu(B & B.T, k=1).sum())


def largest_scc(G, tau: float) -> int:
    """Size of the largest strongly connected component of {G > tau}."""
    B = binarize(G, tau)
    _, labels = connected_components(csr_matrix(B), directed=True,
                                     connection="strong")
    return int(np.bincount(labels).max())


# ----------------------------- sparsity ------------------------------------

def density(G, tau: float) -> float:
    d = np.shape(G)[0]
    return float(binarize(G, tau).sum() / (d * (d - 1)))


def weight_entropy(G) -> float:
    """Normalised Shannon entropy of off-diagonal mass; 1.0 = perfectly uniform."""
    M = _offdiag(G)
    d = M.shape[0]
    w = M[~np.eye(d, dtype=bool)]
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(d * (d - 1)))


def saturation_fraction(W, center: float = 0.5, eps: float = 0.05) -> float:
    """Fraction of off-diagonal entries within eps of `center` (no learning)."""
    M = np.array(W, dtype=np.float64, copy=True)
    d = M.shape[0]
    w = M[~np.eye(d, dtype=bool)]
    return float((np.abs(w - center) < eps).mean())


# ----------------------------- structure vs call-graph ---------------------

def true_edge_set(edge_index, d: int, directed: bool = True,
                  reverse: bool = False):
    """
    Edge set from a (2, E) array. directed -> set of (i, j); otherwise set of
    frozenset-like sorted tuples (i < j). Self-loops dropped, duplicates merged.
    `reverse=True` swaps (src, dst), because the orientation convention of
    graphs.pt must be verified from source before directed results are used.
    """
    ei = np.asarray(edge_index)
    src, dst = (ei[1], ei[0]) if reverse else (ei[0], ei[1])
    out: Set[Tuple[int, int]] = set()
    for a, b in zip(src.tolist(), dst.tolist()):
        if a == b:
            continue
        out.add((a, b) if directed else (min(a, b), max(a, b)))
    return out


def top_k_edges(G, k: int, directed: bool = True):
    """Deterministic top-k edges (stable tie-break by flat index)."""
    M = _offdiag(G)
    d = M.shape[0]
    if directed:
        idx = [(i, j) for i in range(d) for j in range(d) if i != j]
        sc = np.array([M[i, j] for i, j in idx])
    else:
        S = np.maximum(M, M.T)
        idx = [(i, j) for i in range(d) for j in range(i + 1, d)]
        sc = np.array([S[i, j] for i, j in idx])
    order = np.argsort(-sc, kind="stable")[:k]
    return {idx[o] for o in order}


def precision_at_k(G, true_edges, k: Optional[int] = None,
                   directed: bool = True) -> float:
    k = len(true_edges) if k is None else k
    top = top_k_edges(G, k, directed)
    return len(top & set(true_edges)) / k


def permutation_pvalue(G, true_edges, directed: bool = True,
                       n_perm: int = 10000, seed: int = 0) -> float:
    """
    One-sided p-value for precision@|E| against the null "node labels carry no
    information": relabel the true edge set with a random node permutation.
    p = (1 + #{null >= observed}) / (1 + n_perm).
    """
    d = np.shape(G)[0]
    k = len(true_edges)
    top = top_k_edges(G, k, directed)
    obs = len(top & set(true_edges)) / k
    rng = np.random.default_rng(seed)
    te = list(true_edges)
    ge = 0
    for _ in range(n_perm):
        pi = rng.permutation(d)
        if directed:
            perm = {(int(pi[a]), int(pi[b])) for a, b in te}
        else:
            perm = {(min(int(pi[a]), int(pi[b])), max(int(pi[a]), int(pi[b])))
                    for a, b in te}
        if len(top & perm) / k >= obs:
            ge += 1
    return (1 + ge) / (1 + n_perm)


def jaccard_top_k(G1, G2, k: int, directed: bool = True) -> float:
    a, b = top_k_edges(G1, k, directed), top_k_edges(G2, k, directed)
    return len(a & b) / len(a | b)


TAUS: Iterable[float] = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def summarize_graph(G, taus: Iterable[float] = TAUS) -> dict:
    """Scalar summary of one graph, JSON-serialisable."""
    M = _offdiag(G)
    d = M.shape[0]
    off = M[~np.eye(d, dtype=bool)]
    out = dict(
        h_impl=h_impl(M), h_exact=h_exact(M), h_standard=h_standard(M),
        w_mean=float(off.mean()), w_max=float(off.max()),
        w_std=float(off.std()), entropy=weight_entropy(M),
        min_tau_dag=min_tau_for_dag(M),
        per_tau={},
    )
    for t in taus:
        out["per_tau"][str(t)] = dict(
            density=density(M, t), is_dag=bool(is_dag(M, t)),
            n_two_cycles=n_two_cycles(M, t), largest_scc=largest_scc(M, t))
    return out
