# src/simple_baselines.py -- numpy/scikit-learn only (no torch), so it is unit-testable anywhere.
"""
Simple baselines for Section 5.2.7: random forest (per-node and flattened inputs) and a static
threshold rule on F0 = mean_latency. Design is frozen in PROTOCOL.md ("Simple baselines") and
configs/baseline_simple.py. Pattern information is passed in (patterns, idx_to_pattern) rather than
imported, so this module does not depend on src.core.

Inputs: X [n, 30, 7] = F0-F6 averaged over the 12 time steps of a window (same as the logreg baseline),
        y [n] window label, pat [n] pattern index.
Metric definitions follow experiments/baseline_run.py (logreg): RCS Top-1 is evaluated on positive
windows whose pattern has a non-empty, non-full flagged-node set (the rcs_top1 exclusion rule).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from configs import baseline_simple as C


def flagged_of(pat_idx, patterns, idx_to_pattern, n_nodes=C.N_NODES):
    f = [n for n in patterns.get(idx_to_pattern.get(int(pat_idx), "none"), frozenset()) if n < n_nodes]
    return f if (f and len(f) < n_nodes) else None


def eligible(y, pat, patterns, idx_to_pattern):
    """Indices of positive windows with a usable flagged set, and their flagged lists."""
    idx, fl = [], []
    for i in np.where(np.asarray(y) == 1)[0]:
        f = flagged_of(pat[i], patterns, idx_to_pattern)
        if f is not None:
            idx.append(int(i))
            fl.append(f)
    return np.array(idx, dtype=int), fl


def membership(fl_list, n_nodes=C.N_NODES):
    M = np.zeros((len(fl_list), n_nodes), dtype=int)
    for r, f in enumerate(fl_list):
        M[r, f] = 1
    return M


def _top1(score_rows, pat_rows, patterns, idx_to_pattern):
    hit = n = 0
    for s, p in zip(score_rows, pat_rows):
        f = flagged_of(p, patterns, idx_to_pattern)
        if f is not None:
            n += 1
            hit += int(np.argmax(s)) in f
    return 100.0 * hit / max(n, 1), n


# ------------------------------------------------------------------ static threshold rule on F0
def rule_fit(Xtr, ytr):
    """Per-node mean/SD of F0 from NORMAL (y = 0) TRAIN windows; tau = Youden-J optimum on TRAIN scores."""
    f0 = Xtr[:, :, C.F0_CHANNEL]
    normal = f0[np.asarray(ytr) == 0]
    mu, sd = normal.mean(0), normal.std(0)                  # population SD (ddof=0); epsilon guards SD = 0
    z = (f0 - mu) / (sd + C.EPS)
    score = z.max(1)
    fpr, tpr, thr = roc_curve(ytr, score)
    tau = float(thr[int(np.argmax(tpr - fpr))])             # first max = largest threshold among ties
    return dict(mu=mu, sd=sd, tau=tau, n_normal=int(normal.shape[0]))


def rule_scores(model, X):
    z = (X[:, :, C.F0_CHANNEL] - model["mu"]) / (model["sd"] + C.EPS)
    return z, z.max(1)


def rule_eval(model, Xva, yva, pva, patterns, idx_to_pattern):
    z, score = rule_scores(model, Xva)
    pos = np.where(yva == 1)[0]
    top1, n = _top1(z[pos], pva[pos], patterns, idx_to_pattern)
    pred = score > model["tau"]
    tp = int((pred & (yva == 1)).sum()); fp = int((pred & (yva == 0)).sum()); fn = int((~pred & (yva == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(auc=float(roc_auc_score(yva, score)), pat_acc=None, rcs_top1=top1, rcs_top1_n=n,
                tau=model["tau"], exploratory_at_tau=dict(precision=prec, recall=rec,
                f1=(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0, tp=tp, fp=fp, fn=fn))


# ------------------------------------------------------------------ random forests
def _rf(seed, **kw):
    return RandomForestClassifier(random_state=seed, **C.RF_PARAMS, **kw)


def _p1(clf, P):
    """Probability of class 1 (zeros if class 1 never seen)."""
    cl = list(clf.classes_)
    return P[:, cl.index(1)] if 1 in cl else np.zeros(P.shape[0])


def rf_pernode(Xtr, ytr, ptr, Xva, yva, pva, seed, patterns, idx_to_pattern):
    """Rows = (window, node), features = that node's time-averaged F0-F6 (7-dim).
    Detection: target = window label on every row; window score = max over nodes of P(y=1).
    PatAcc: target = window pattern on every row; window prediction = argmax of node-averaged class probs.
    Localization: eligible positive train windows only; target = node in flagged set; rank nodes by P(1)."""
    n_tr, n_va, N, d = len(Xtr), len(Xva), Xtr.shape[1], Xtr.shape[2]
    rows_tr, rows_va = Xtr.reshape(-1, d), Xva.reshape(-1, d)
    det = _rf(seed, class_weight=C.RF_DETECTION_CLASS_WEIGHT).fit(rows_tr, np.repeat(ytr, N))
    score = _p1(det, det.predict_proba(rows_va)).reshape(n_va, N).max(1)
    pat = _rf(seed).fit(rows_tr, np.repeat(ptr, N))
    Pm = pat.predict_proba(rows_va).reshape(n_va, N, -1).mean(1)
    pat_acc = float((pat.classes_[Pm.argmax(1)] == pva).mean())
    idx, fl = eligible(ytr, ptr, patterns, idx_to_pattern)
    loc = _rf(seed).fit(Xtr[idx].reshape(-1, d), membership(fl, N).reshape(-1))
    pos = np.where(yva == 1)[0]
    prob = _p1(loc, loc.predict_proba(Xva[pos].reshape(-1, d))).reshape(len(pos), N)
    top1, n = _top1(prob, pva[pos], patterns, idx_to_pattern)
    return dict(auc=float(roc_auc_score(yva, score)), pat_acc=pat_acc, rcs_top1=top1, rcs_top1_n=n)


def rf_flat(Xtr, ytr, ptr, Xva, yva, pva, seed, patterns, idx_to_pattern):
    """Rows = windows, features = all nodes' time-averaged F0-F6 flattened (210-dim).
    Detection / PatAcc: window-level RFs. Localization: ONE multi-output RF whose 30 binary outputs are
    node-in-flagged-set; per-node P(1) ranks nodes."""
    N = Xtr.shape[1]
    A, B = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1)
    det = _rf(seed, class_weight=C.RF_DETECTION_CLASS_WEIGHT).fit(A, ytr)
    auc = float(roc_auc_score(yva, _p1(det, det.predict_proba(B))))
    pat = _rf(seed).fit(A, ptr)
    pat_acc = float((pat.predict(B) == pva).mean())
    idx, fl = eligible(ytr, ptr, patterns, idx_to_pattern)
    loc = _rf(seed).fit(A[idx], membership(fl, N))
    pos = np.where(yva == 1)[0]
    P = loc.predict_proba(B[pos])
    prob = np.stack([(p[:, list(c).index(1)] if 1 in list(c) else np.zeros(len(pos)))
                     for p, c in zip(P, loc.classes_)], axis=1)
    top1, n = _top1(prob, pva[pos], patterns, idx_to_pattern)
    return dict(auc=auc, pat_acc=pat_acc, rcs_top1=top1, rcs_top1_n=n)
