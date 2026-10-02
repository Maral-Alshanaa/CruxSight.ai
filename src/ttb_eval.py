# src/ttb_eval.py
"""
Evaluation of the TTB (Time-to-Bottleneck) head and of detection-based alert
lead time. NumPy/SciPy only (no torch), so it is unit-testable anywhere.

Conventions (taken from the cache-building cell, Colab cell 7c):
  * windows are stride-1 inside a file: x covers steps start..end-1,
    y_fut = labels[end:end+HORIZON]  (starts right after the last x step)
  * ttb (minutes) = bn_steps * STEP_SEC / 60, bn_steps = argmax(y_fut) when the
    window label is 1, otherwise the sentinel -1 step (-1/6 min for STEP_SEC=10)
  * therefore ttb_i is exactly the time between the end of window i and the
    first bottleneck step o:  o - end_i = k_i steps.
  * with stride 1, end_i - end_a = i - a (window indices inside the file), so
    a lead time in steps is  (i + k_i) - a  for any positive window i that sees
    the same first onset; no knowledge of absolute offsets is needed.

Window-level TTB metrics are computed ONLY on windows whose TRUE label is 1
(the same mask the training loss uses); the negative sentinel is never scored.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

STEP_SEC = 10
HORIZON = 6
COLLAPSE_FRAC = 0.05        # pre-registered: pred SD < 5% of target SD => collapsed head
DETECT_THETA = 0.5          # pre-registered fixed threshold (not tuned on any split)


# ----------------------------------------------------------------------------
# Window-level TTB error
# ----------------------------------------------------------------------------
def _k_steps(ttb_min, step_sec=STEP_SEC):
    k = np.asarray(ttb_min, dtype=np.float64) * 60.0 / step_sec
    kr = np.rint(k)
    if not np.allclose(k, kr, atol=1e-3):
        raise ValueError("ttb is not an integer number of steps; unit/step_sec mismatch")
    return kr.astype(int)


def ttb_window_metrics(pred_min, true_min, label, step_sec=STEP_SEC, extra_mask=None):
    """MAE/RMSE (seconds), bias, Spearman, collapse flag and per-k breakdown on
    windows with TRUE label == 1 (optionally AND extra_mask, e.g. alerted windows)."""
    pred_min = np.asarray(pred_min, dtype=np.float64)
    true_min = np.asarray(true_min, dtype=np.float64)
    label = np.asarray(label).astype(int)
    sel = label == 1
    if extra_mask is not None:
        sel = sel & np.asarray(extra_mask, dtype=bool)
    p, t = pred_min[sel], true_min[sel]
    out = dict(n=int(sel.sum()), n_total=int(len(label)))
    if out["n"] == 0:
        out.update(mae_s=np.nan, rmse_s=np.nan, bias_s=np.nan, spearman=np.nan,
                   pred_sd_s=np.nan, true_sd_s=np.nan, collapsed=None, per_k={})
        return out
    if (t < 0).any():
        raise ValueError("negative TTB target inside label==1 windows: sentinel leaked into the scored set")
    err_s = (p - t) * 60.0
    out["mae_s"] = float(np.abs(err_s).mean())
    out["rmse_s"] = float(np.sqrt((err_s ** 2).mean()))
    out["bias_s"] = float(err_s.mean())
    out["pred_sd_s"] = float(p.std() * 60.0)
    out["true_sd_s"] = float(t.std() * 60.0)
    out["collapsed"] = bool(out["pred_sd_s"] < COLLAPSE_FRAC * out["true_sd_s"]) if out["true_sd_s"] > 0 else None
    if np.ptp(p) == 0 or np.ptp(t) == 0:
        out["spearman"] = float("nan")
    else:
        out["spearman"] = float(stats.spearmanr(p, t).statistic)
    ks = _k_steps(t, step_sec)
    per_k = {}
    for k in np.unique(ks):
        m = ks == k
        e = err_s[m]
        per_k[int(k)] = dict(n=int(m.sum()), mae_s=float(np.abs(e).mean()),
                             rmse_s=float(np.sqrt((e ** 2).mean())), bias_s=float(e.mean()))
    out["per_k"] = per_k
    return out


def constant_baseline_metrics(const_min, true_min, label):
    """MAE/RMSE (s) of a constant predictor on label==1 windows."""
    pred = np.full(len(np.asarray(label)), float(const_min))
    m = ttb_window_metrics(pred, true_min, label)
    return dict(n=m["n"], mae_s=m["mae_s"], rmse_s=m["rmse_s"])


def train_constants(train_true_min, train_label):
    """Constant baselines from the TRAIN split only: mean (RMSE-optimal), median (MAE-optimal)."""
    t = np.asarray(train_true_min, dtype=np.float64)[np.asarray(train_label).astype(int) == 1]
    if len(t) == 0:
        raise ValueError("no positive windows in train")
    return dict(mean_min=float(t.mean()), median_min=float(np.median(t)), n=int(len(t)))


# ----------------------------------------------------------------------------
# File structure recovery
# ----------------------------------------------------------------------------
def recover_file_ids(x):
    """x: array (N, W, ...) of consecutive stride-1 windows in cache order.
    Window b continues the same file as window a iff x[a][1:] == x[b][:-1]."""
    x = np.asarray(x)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    a, b = x[:-1, 1:], x[1:, :-1]
    eq = (a == b) | (np.isnan(a) & np.isnan(b)) if np.issubdtype(x.dtype, np.floating) else (a == b)
    same = eq.reshape(n - 1, -1).all(axis=1) if n > 1 else np.zeros(0, dtype=bool)
    return np.concatenate([[0], np.cumsum(~same)]).astype(int)


def file_onsets(file_id, label, ttb_min, step_sec=STEP_SEC):
    """One record per file with the FIRST bottleneck onset in window-index units.

    status:
      ok            onset identified and consistent
      no_bottleneck no positive window
      censored_left first positive window is window 0 (bottleneck may predate the file's
                    observable window; first onset not identifiable) -> excluded
      inconsistent  positive windows leading up to the onset do not satisfy k_j = onset - j
    onset_idx = i0 + k[i0]  (window index whose end is the onset step; may exceed n-1 when the
    onset falls in the last HORIZON steps of the file).
    """
    file_id = np.asarray(file_id)
    label = np.asarray(label).astype(int)
    k_all = np.where(label == 1, _k_steps(np.where(label == 1, ttb_min, 0.0), step_sec), -1)
    recs = []
    for f in np.unique(file_id):
        idx = np.where(file_id == f)[0]
        n = len(idx)
        lab, k = label[idx], k_all[idx]
        rec = dict(file=int(f), start=int(idx[0]), n=int(n), i0=None, onset_idx=None, status=None)
        pos = np.where(lab == 1)[0]
        if len(pos) == 0:
            rec["status"] = "no_bottleneck"
        else:
            i0 = int(pos[0])
            rec["i0"] = i0
            onset = i0 + int(k[i0])
            rec["onset_idx"] = onset
            if i0 == 0:
                rec["status"] = "censored_left"
            else:
                last = min(onset, n - 1)
                js = np.arange(i0, last + 1)
                ok = bool((lab[js] == 1).all() and (k[js] == onset - js).all())
                rec["status"] = "ok" if ok else "inconsistent"
        recs.append(rec)
    return recs


# ----------------------------------------------------------------------------
# Detection-based alert lead time (stable alert, fixed threshold)
# ----------------------------------------------------------------------------
def first_stable_alert(prob_file, onset_idx, i0, theta=DETECT_THETA, step_sec=STEP_SEC):
    """Earliest window a such that prob > theta for every window in [a, last_before_onset].
    Windows after the onset are ignored. Returns dict(missed, a, lead_steps, lead_s, premature).
    premature = alert issued in a label-negative window (a < i0), i.e. lead > (onset - i0) steps."""
    prob_file = np.asarray(prob_file, dtype=np.float64)
    n = len(prob_file)
    m = min(onset_idx, n - 1)
    above = prob_file[: m + 1] > theta
    if not above[m]:
        return dict(missed=True, a=None, lead_steps=None, lead_s=None, premature=None)
    a = m
    while a > 0 and above[a - 1]:
        a -= 1
    lead = onset_idx - a
    return dict(missed=False, a=int(a), lead_steps=int(lead), lead_s=float(lead * step_sec),
                premature=bool(a < i0))


def lead_time_table(prob, file_id, label, ttb_min, theta=DETECT_THETA, step_sec=STEP_SEC):
    prob = np.asarray(prob, dtype=np.float64)
    table = []
    for rec in file_onsets(file_id, label, ttb_min, step_sec):
        row = dict(rec)
        row.update(lead_s=None, missed=None, premature=None)
        if rec["status"] == "ok":
            sl = slice(rec["start"], rec["start"] + rec["n"])
            r = first_stable_alert(prob[sl], rec["onset_idx"], rec["i0"], theta, step_sec)
            row.update(lead_s=r["lead_s"], missed=r["missed"], premature=r["premature"])
        table.append(row)
    return table


def summarize_lead(table):
    ok = [r for r in table if r["status"] == "ok"]
    det = [r for r in ok if not r["missed"]]
    leads = np.array([r["lead_s"] for r in det], dtype=float)
    return dict(
        n_files=len(table),
        n_evaluable=len(ok),
        n_excluded=len(table) - len(ok),
        n_detected=len(det),
        n_missed=len(ok) - len(det),
        n_premature=int(sum(bool(r["premature"]) for r in det)),
        mean_lead_s=float(leads.mean()) if len(leads) else float("nan"),
        median_lead_s=float(np.median(leads)) if len(leads) else float("nan"),
        frac_lead_over_horizon=float((leads > (HORIZON - 1) * STEP_SEC).mean()) if len(leads) else float("nan"),
    )
