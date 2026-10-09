# tests/test_f6_anomaly.py
"""
Verification for the F6 zero-shot anomaly study (PROTOCOL rule 1 equivalent).

Part A (this file, numpy/scipy only): metric functions on hand-checkable
synthetic cases. Part B (tests/test_f6_anomaly_torch.py, needs torch +
torch_geometric): the attention extraction path on a tiny synthetic CSTGNN:
  B1 extraction reproduces SpatialEncoder's own output (so the mirrored
     forward is faithful), for with-F6 and ablated (toc_cap = 0) inputs
  B2 aggregated attention rows sum to 1 for every layer
  B3 extraction mutates no parameter and leaves no gradient
  B4 ablation sensitivity: with toc_cap = 0 attention does not depend on
     toc_scale; with toc_cap > 0 it differs from the zero-cap attention
  B5 checked checkpoint loading accepts causal_30.* extras and rejects a
     genuinely missing key
  B6 rebuild_causal: same init seed -> identical zero-shot probabilities,
     different init seed -> different probabilities (the D1 premise)
No training is performed in this study, so there is no optimizer/gradient-flow
requirement to test; B3 is the corresponding "no side effects" guarantee.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import f6_anomaly as fa  # noqa: E402


# ---------------------------- Part A ----------------------------------

def test_cliffs_delta():
    assert fa.cliffs_delta([2, 3], [0, 1]) == 1.0
    assert fa.cliffs_delta([0, 1], [2, 3]) == -1.0
    assert fa.cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0


def test_f6_comparison_basic():
    cc = np.linspace(0.0, 1.0, 30)
    ch = np.array([0.0, 0.1, 0.2, 0.9, 1.0, 0.3, 0.4])
    r = fa.f6_comparison(cc, ch)
    assert r["compose"]["n"] == 30 and r["home"]["n"] == 7
    assert r["home_flagged_values"] == [0.9, 1.0]
    assert r["home_flagged_rank_desc"] == [2, 1]
    assert abs(r["capacity_weight_home"]["max"] - 3.0) < 1e-12   # 1 + 2*1*1
    assert r["wasserstein_1"] >= 0


def test_aggregate_rows_sum_to_one_and_values():
    N, T, B, H = 3, 2, 2, 2
    rng = np.random.RandomState(0)
    # complete graph with self loops: every target has 3 incoming edges per copy
    src, dst, alpha = [], [], []
    for g in range(B * T):
        for i in range(N):
            w = rng.rand(H, N)
            w = w / w.sum(1, keepdims=True)           # softmax-like per head
            for j in range(N):
                src.append(g * N + j); dst.append(g * N + i); alpha.append(w[:, j])
    A = fa.aggregate_attention(src, dst, np.array(alpha), N, T, B)
    assert A.shape == (B, N, N)
    assert np.allclose(A.sum(-1), 1.0)


def test_aggregate_matches_manual_single_window():
    # one window, T=2 copies, N=2, H=1; copy0 attention [[.5,.5],[.2,.8]], copy1 [[.1,.9],[.6,.4]]
    N, T = 2, 2
    m = [np.array([[.5, .5], [.2, .8]]), np.array([[.1, .9], [.6, .4]])]
    src, dst, alpha = [], [], []
    for g in range(2):
        for i in range(2):
            for j in range(2):
                src.append(g * N + j); dst.append(g * N + i); alpha.append([m[g][i, j]])
    A = fa.aggregate_attention(src, dst, np.array(alpha), N, T, 1)
    assert np.allclose(A[0], (m[0] + m[1]) / 2)


def test_entropy():
    uni = np.full((1, 4, 4), 0.25)
    assert abs(fa.mean_row_entropy(uni) - 1.0) < 1e-9
    peaked = np.zeros((1, 4, 4)); peaked[0, :, 0] = 0.97; peaked[0, :, 1:] = 0.01
    assert fa.mean_row_entropy(peaked) < 0.3
    onehot = np.zeros((1, 3, 3)); onehot[0, np.arange(3), np.arange(3)] = 1.0
    assert np.isnan(fa.mean_row_entropy(onehot))        # rows with k < 2 excluded


def test_critical_enrichment():
    N = 7
    uni = np.full((1, N, N), 1.0 / N)
    assert abs(fa.critical_enrichment(uni, (3, 4)) - 1.0) < 1e-9
    conc = np.full((1, N, N), 0.02); conc[0, :, 3] = 0.45; conc[0, :, 4] = 0.45
    conc = conc / conc.sum(-1, keepdims=True)
    assert fa.critical_enrichment(conc, (3, 4)) > 2.0
    away = np.full((1, N, N), 0.2); away[0, :, 3] = 0.01; away[0, :, 4] = 0.01
    away = away / away.sum(-1, keepdims=True)
    assert fa.critical_enrichment(away, (3, 4)) < 0.5


def test_js():
    p = np.full((1, 2, 4), 0.25)
    assert abs(fa.mean_row_js(p, p)) < 1e-12
    a = np.array([[[1.0, 0, 0, 0]]]); b = np.array([[[0, 0, 0, 1.0]]])
    assert abs(fa.mean_row_js(a, b) - 1.0) < 1e-12       # disjoint support, base 2


def test_spearman_constant_cap_is_nan():
    A = np.random.RandomState(1).rand(5, 7, 7)
    assert np.isnan(fa.received_vs_cap_spearman(A, np.zeros(7)))
    s = fa.received_vs_cap_spearman(A, np.arange(7.0))
    assert -1.0 <= s <= 1.0


def test_holm():
    adj, rej = fa.holm([0.01, 0.04])
    assert np.allclose(adj, [0.02, 0.04]) and rej == [True, True]
    adj, rej = fa.holm([0.03, 0.04])
    assert np.allclose(adj, [0.06, 0.06]) and rej == [False, False]


def test_prob_diagnostics_flags_constant_predictions():
    d = fa.prob_diagnostics(np.full(10, 0.7), np.array([0, 1] * 5))
    assert d["n_unique_probs"] == 1 and d["prob_sd"] == 0.0 and d["frac_pred_pos"] == 1.0
