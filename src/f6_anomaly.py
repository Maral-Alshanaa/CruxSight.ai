# src/f6_anomaly.py
"""
Metrics and helpers for the F6 zero-shot anomaly investigation (reviewer item 8).

Pure numpy/scipy part (testable without torch): F6 distribution comparison,
attention-matrix aggregation and attention summary metrics, Holm correction.
Torch part (imports torch lazily): checked checkpoint loading, attention
extraction from TOCGATLayer's GATConv, zero-shot probability evaluation, and
rebuilding causal_7 under a chosen RNG seed (init-noise diagnostic D1).

Nothing here trains anything: there is no new trainable component, so the
"gradient-flow" rule (PROTOCOL rule 1) does not apply as such. The relevant
guarantee is instead that the new extraction path (a) reproduces the model's
own spatial-encoder output exactly and (b) mutates no parameter and
accumulates no gradient. tests/test_f6_anomaly.py checks both.
"""
import numpy as np
from scipy import stats

CRITICAL_HOME = (3, 4)      # flagged bottleneck nodes used by RCS Top-1 (pipeline.finetune_home)
LARGE_DELTA = 0.474         # Romano et al. threshold for a "large" Cliff's delta
EPS = 1e-12


# ----------------------------------------------------------------------
# Analysis 1: F6 (toc_capacity) distributions
# ----------------------------------------------------------------------

def cliffs_delta(a, b):
    """P(a > b) - P(a < b) over all pairs. Positive: a tends to be larger."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def describe(v):
    v = np.asarray(v, float).ravel()
    return dict(n=int(v.size), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if v.size > 1 else float("nan"),
                min=float(v.min()), q25=float(np.percentile(v, 25)),
                median=float(np.median(v)), q75=float(np.percentile(v, 75)),
                max=float(v.max()))


def f6_comparison(cap_compose, cap_home, flagged=CRITICAL_HOME,
                  toc_lambda=2.0, toc_scale=1.0):
    """Descriptive node-level comparison of F6 between topologies.

    toc_cap is one value per node (constant across windows by construction),
    so the unit of comparison is the node (30 vs 7) and no p-value is
    reported: nodes are not exchangeable samples.
    """
    cc = np.asarray(cap_compose, float).ravel()
    ch = np.asarray(cap_home, float).ravel()
    delta = cliffs_delta(cc, ch)
    w = lambda c: 1.0 + toc_lambda * toc_scale * c   # TOCGATLayer capacity_weight at init scale
    order_desc = np.argsort(-ch)
    return dict(
        compose=describe(cc), home=describe(ch),
        cliffs_delta_compose_minus_home=delta,
        large_difference=bool(abs(delta) >= LARGE_DELTA),
        wasserstein_1=float(stats.wasserstein_distance(cc, ch)),
        home_flagged_nodes=list(flagged),
        home_flagged_values=[float(ch[i]) for i in flagged],
        home_unflagged_mean=float(np.mean([ch[i] for i in range(ch.size) if i not in flagged])),
        home_flagged_rank_desc=[int(np.where(order_desc == i)[0][0]) + 1 for i in flagged],
        capacity_weight_compose=describe(w(cc)), capacity_weight_home=describe(w(ch)),
    )


# ----------------------------------------------------------------------
# Analysis 2: attention aggregation and summary metrics (numpy)
# ----------------------------------------------------------------------

def aggregate_attention(src, dst, alpha, n_nodes, n_steps, n_windows):
    """Per-window attention matrices A[b, i, j] = attention target i pays to source j.

    src, dst: [E'] indices into the batched graph (B*T copies of N nodes, copy g
    holds nodes g*N .. g*N+N-1; windows are consecutive blocks of n_steps copies).
    alpha: [E', H] attention coefficients. Averaged over time steps and heads.
    Rows (over j) sum to 1 for every window.
    """
    src = np.asarray(src, np.int64)
    dst = np.asarray(dst, np.int64)
    alpha = np.asarray(alpha, float)
    H = alpha.shape[1]
    A = np.zeros((n_windows, n_nodes, n_nodes))
    win = (dst // n_nodes) // n_steps
    np.add.at(A, (win, dst % n_nodes, src % n_nodes), alpha.sum(axis=1) / (n_steps * H))
    return A


def mean_row_entropy(A):
    """Mean over windows/rows of entropy / log(k), k = number of incoming edges
    (rows with k >= 2 only). 1.0 = uniform attention, 0 = fully concentrated."""
    A = np.asarray(A, float)
    mask = A > EPS
    k = mask.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(mask, A * np.log(np.where(mask, A, 1.0)), 0.0).sum(-1)
        norm = ent / np.log(k)
    valid = k >= 2
    return float(norm[valid].mean()) if valid.any() else float("nan")


def critical_enrichment(A, crit=CRITICAL_HOME):
    """Share of attention paid TO the critical nodes, divided by the share a
    uniform-per-row attention over the same edges would pay. 1.0 = no enrichment."""
    A = np.asarray(A, float)
    mask = A > EPS
    k = np.maximum(mask.sum(-1, keepdims=True), 1)
    U = np.where(mask, 1.0 / k, 0.0)
    c = list(crit)
    share = A[..., c].sum() / A.sum()
    share_u = U[..., c].sum() / U.sum()
    return float(share / share_u) if share_u > 0 else float("nan")


def mean_row_js(A1, A2):
    """Mean Jensen-Shannon divergence (base 2, in [0,1]) between corresponding rows."""
    p = np.clip(np.asarray(A1, float), 0, None)
    q = np.clip(np.asarray(A2, float), 0, None)
    m = 0.5 * (p + q)

    def kl(a, b):
        ok = a > EPS
        return np.where(ok, a * np.log2(np.where(ok, a, 1.0) / np.where(ok, b, 1.0)), 0.0).sum(-1)

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    valid = (p.sum(-1) > 0) & (q.sum(-1) > 0)
    return float(js[valid].mean())


def received_vs_cap_spearman(A, cap):
    """Spearman between the attention mass each node SENDS (column sums, mean over
    windows) and its F6 value. NaN if F6 is constant (e.g. the ablated arm)."""
    cap = np.asarray(cap, float).ravel()
    if np.ptp(cap) == 0:
        return float("nan")
    sent = np.asarray(A, float).sum(1).mean(0)
    return float(stats.spearmanr(sent, cap)[0])


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. Returns (adjusted p-values, reject flags) in input order."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist(), (adj <= alpha).tolist()


def prob_diagnostics(probs, labels, thresh=0.5):
    """Artifact checks (PROTOCOL rule 4) on raw zero-shot probabilities."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels).astype(int)
    pred = probs >= thresh
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    return dict(
        n=int(probs.size), pos_rate=float(labels.mean()),
        prob_min=float(probs.min()), prob_max=float(probs.max()),
        prob_mean=float(probs.mean()), prob_sd=float(probs.std()),
        frac_pred_pos=float(pred.mean()),
        precision=float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        recall=float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        n_unique_probs=int(np.unique(np.round(probs, 6)).size),
    )


# ----------------------------------------------------------------------
# Torch helpers (lazy import)
# ----------------------------------------------------------------------

def load_model_checked(ckpt_path, cfg, device, seed=None):
    """Build CSTGNN and load a compose checkpoint.

    pipeline.finetune_home loads with strict=False, which would silently ignore
    a real key mismatch. Here the result is inspected: nothing may be missing,
    and the only unexpected keys allowed are the lazily built causal_30.* ones.
    If seed is given, the global RNG is seeded first (as in finetune_home) so
    that causal_7's later random init reproduces the original run.
    """
    import torch
    from src.core import CSTGNN
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model = CSTGNN(cfg).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    res = model.load_state_dict(ck["model_state"], strict=False)
    if res.missing_keys:
        raise RuntimeError(f"missing keys when loading {ckpt_path}: {res.missing_keys}")
    bad = [k for k in res.unexpected_keys if not k.startswith("causal_30.")]
    if bad:
        raise RuntimeError(f"unexpected keys when loading {ckpt_path}: {bad}")
    model.eval()
    return model


def extract_attention(model, x_seq, edge_index, toc_cap):
    """Run SpatialEncoder layer by layer, returning GATConv attention per layer.

    Mirrors TOCGATLayer.forward exactly (capacity weighting -> gat -> proj ->
    norm -> elu) with return_attention_weights=True. Returns
    ([(src, dst, alpha) per layer] as numpy, h_spatial [B,T,N,D]).
    """
    import torch
    import torch.nn.functional as F
    enc = model.spatial
    with torch.no_grad():
        B, T, N, Fd = x_seq.shape
        ei = enc.build_batched_edge_index(edge_index, N, B * T)
        cap_flat = toc_cap.repeat(B * T)
        h = x_seq.reshape(B * T * N, Fd)
        per_layer = []
        for layer in enc.layers:
            if layer.use_toc:
                w = 1.0 + layer.toc_lambda * layer.toc_scale * cap_flat
                x_toc = h * w.unsqueeze(-1)
            else:
                x_toc = h
            out, (ei_out, alpha) = layer.gat(x_toc, ei, return_attention_weights=True)
            per_layer.append((ei_out[0].cpu().numpy(), ei_out[1].cpu().numpy(),
                              alpha.cpu().numpy()))
            h = F.elu(layer.norm(layer.proj(out)))
        return per_layer, h.reshape(B, T, N, enc.out_dim)


def attention_matrices(model, x_all, edge_index, toc_cap, n_nodes, chunk=50):
    """Per-layer window-level attention matrices [n_windows, N, N] for x_all [W,T,N,F]."""
    import torch
    model.eval()
    mats = None
    for s in range(0, x_all.shape[0], chunk):
        xb = x_all[s:s + chunk]
        per_layer, _ = extract_attention(model, xb, edge_index, toc_cap)
        part = [aggregate_attention(src, dst, al, n_nodes, xb.shape[1], xb.shape[0])
                for (src, dst, al) in per_layer]
        mats = part if mats is None else [np.concatenate([m, p], 0) for m, p in zip(mats, part)]
    return mats


def rebuild_causal(model, n_nodes, init_seed, window_steps, n_features,
                   edge_index, toc_cap, device):
    """Delete causal_<n> (if any) and rebuild it with a fresh random init under
    torch.manual_seed(init_seed), exactly via the model's own lazy construction."""
    import torch
    name = f"causal_{n_nodes}"
    if name in model._modules:
        del model._modules[name]
    model._causal_cache.pop(n_nodes, None)
    torch.manual_seed(init_seed)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, window_steps, n_nodes, n_features, device=device),
              edge_index, toc_cap)


def predict_probs(model, loader, edge_index, toc_cap, device):
    """Sigmoid(bn_logit) and labels over a loader (same evaluation as finetune_home)."""
    import torch
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, label, _pattern, _ttb in loader:
            out = model(x.to(device), edge_index, toc_cap)
            probs.extend(torch.sigmoid(out["bn_logit"]).squeeze(-1).cpu().tolist())
            labels.extend(label.tolist())
    return np.array(probs), np.array(labels)
