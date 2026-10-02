# experiments/ttb_stats.py
"""
Primary/secondary analysis for the TTB-head evaluation (tag: ttb-eval-prereg).
No torch. Reads OUT/ttb_val_seed{seed}.npz for the 10 pre-registered seeds.

PRIMARY (pre-registered, not to be changed after the tag):
  two-sided one-sample t-test (alpha=0.05) of the 10 per-seed validation RMSEs of the TTB
  head against the RMSE of the constant predictor = TRAIN mean of positive-window TTB
  (the constant has no seed variance, so this equals a paired test). Scored windows: val windows with TRUE label==1.
SECONDARY (reported, no multiplicity correction): MAE vs train-median constant, Spearman,
  per-k MAE/RMSE, TP-subset (label==1 & p>0.5), Wilcoxon (sensitivity), collapse count,
  file-level bootstrap CI of (RMSE_model - RMSE_const) over validation files (descriptive),
  detection-based alert lead time per validation file (fixed theta=0.5, stable alert).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from configs.runs import SEEDS
from src import ttb_eval as E

DEFAULT_OUT = Path(os.environ.get("CRUX_OUT_TTB", "/content/drive/MyDrive/CruxSight/ttb_eval"))
BOOT_SEED, N_BOOT = 0, 10000


def ms(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=float(np.nanmean(a)), sd=float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() > 1 else float("nan"))


def load(out):
    base = json.loads((out / "ttb_baselines.json").read_text())
    data = {}
    for s in SEEDS:
        z = np.load(out / f"ttb_val_seed{s}.npz")
        data[s] = {k: z[k] for k in z.files}
    ref = data[SEEDS[0]]
    for s in SEEDS[1:]:      # identical split/order across seeds is a precondition
        for k in ("label", "ttb_true", "file_id"):
            assert np.array_equal(data[s][k], ref[k]), f"{k} differs between seeds {SEEDS[0]} and {s}"
    return base, data


def main(out_dir=None, results_dir=None, make_figure=True):
    out = Path(out_dir) if out_dir else DEFAULT_OUT
    res_dir = Path(results_dir) if results_dir else ROOT / "results" / "ttb_eval"
    res_dir.mkdir(parents=True, exist_ok=True)
    base, data = load(out)
    ref = data[SEEDS[0]]
    label, true, fid = ref["label"], ref["ttb_true"], ref["file_id"]

    b_mean = E.constant_baseline_metrics(base["mean_min"], true, label)
    b_med = E.constant_baseline_metrics(base["median_min"], true, label)

    per_seed, leads_tab = {}, {}
    for s in SEEDS:
        d = data[s]
        w = E.ttb_window_metrics(d["ttb_pred"], d["ttb_true"], d["label"])
        tp = E.ttb_window_metrics(d["ttb_pred"], d["ttb_true"], d["label"], extra_mask=d["prob"] > E.DETECT_THETA)
        tab = E.lead_time_table(d["prob"], d["file_id"], d["label"], d["ttb_true"])
        per_seed[s] = dict(window=w, tp=tp, lead=E.summarize_lead(tab))
        leads_tab[s] = tab

    rmse = np.array([per_seed[s]["window"]["rmse_s"] for s in SEEDS])
    mae = np.array([per_seed[s]["window"]["mae_s"] for s in SEEDS])
    t_p = stats.ttest_1samp(rmse, b_mean["rmse_s"])
    t_s = stats.ttest_1samp(mae, b_med["mae_s"])
    try:
        w_p = stats.wilcoxon(rmse - b_mean["rmse_s"]).pvalue
    except ValueError:
        w_p = float("nan")

    summary = dict(
        n_seeds=len(SEEDS), n_scored_windows=b_mean["n"],
        baseline_mean_const=dict(const_min=base["mean_min"], **b_mean),
        baseline_median_const=dict(const_min=base["median_min"], **b_med),
        model_rmse_s=ms(rmse), model_mae_s=ms(mae),
        primary=dict(test="one-sample t (two-sided) of per-seed RMSE vs train-mean constant RMSE",
                     t=float(t_p.statistic), p=float(t_p.pvalue), wilcoxon_p=float(w_p),
                     alpha=0.05, significant=bool(t_p.pvalue < 0.05),
                     direction="model better" if rmse.mean() < b_mean["rmse_s"] else "model NOT better"),
        secondary_mae_vs_median=dict(t=float(t_s.statistic), p=float(t_s.pvalue)),
        spearman=ms([per_seed[s]["window"]["spearman"] for s in SEEDS]),
        pred_sd_s=ms([per_seed[s]["window"]["pred_sd_s"] for s in SEEDS]),
        true_sd_s=per_seed[SEEDS[0]]["window"]["true_sd_s"],
        n_collapsed_seeds=int(sum(bool(per_seed[s]["window"]["collapsed"]) for s in SEEDS)),
        tp_subset=dict(n=ms([per_seed[s]["tp"]["n"] for s in SEEDS]),
                       rmse_s=ms([per_seed[s]["tp"]["rmse_s"] for s in SEEDS]),
                       mae_s=ms([per_seed[s]["tp"]["mae_s"] for s in SEEDS])),
        per_k={}, bootstrap={}, lead={},
    )

    ks = sorted({k for s in SEEDS for k in per_seed[s]["window"]["per_k"]})
    for k in ks:
        rows = [per_seed[s]["window"]["per_k"].get(k) for s in SEEDS]
        summary["per_k"][int(k)] = dict(n=rows[0]["n"], mae_s=ms([r["mae_s"] for r in rows]),
                                        rmse_s=ms([r["rmse_s"] for r in rows]))

    # ---- file-level bootstrap of the RMSE difference (descriptive) ----
    files = np.unique(fid)
    pos = label == 1
    n_pf = np.array([(pos & (fid == f)).sum() for f in files], dtype=float)
    ss_base = np.array([(((base["mean_min"] - true[pos & (fid == f)]) * 60) ** 2).sum() for f in files])
    ss_mod = np.array([[(((data[s]["ttb_pred"][pos & (fid == f)] - true[pos & (fid == f)]) * 60) ** 2).sum()
                        for f in files] for s in SEEDS])
    rng = np.random.default_rng(BOOT_SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(files), len(files))
        n = n_pf[idx].sum()
        if n == 0:
            continue
        r_base = np.sqrt(ss_base[idx].sum() / n)
        r_mod = np.sqrt(ss_mod[:, idx].sum(axis=1) / n).mean()
        diffs.append(r_mod - r_base)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    summary["bootstrap"] = dict(n_files=int(len(files)), n_files_with_positives=int((n_pf > 0).sum()),
                                n_boot=N_BOOT, seed=BOOT_SEED, rmse_diff_ci95_s=[float(lo), float(hi)],
                                note="model minus constant; negative favours the model; files resampled with replacement")

    # ---- lead time ----
    ok_files = [r["file"] for r in leads_tab[SEEDS[0]] if r["status"] == "ok"]
    status = {int(r["file"]): r["status"] for r in leads_tab[SEEDS[0]]}
    L = np.full((len(SEEDS), len(ok_files)), np.nan)         # NaN = missed
    prem = np.zeros_like(L, dtype=bool)
    for i, s in enumerate(SEEDS):
        by = {r["file"]: r for r in leads_tab[s]}
        for j, f in enumerate(ok_files):
            r = by[f]
            if not r["missed"]:
                L[i, j] = r["lead_s"]; prem[i, j] = bool(r["premature"])
    summary["lead"] = dict(
        file_status=status, n_evaluable_files=len(ok_files), theta=E.DETECT_THETA,
        per_seed_mean_lead_s=ms([per_seed[s]["lead"]["mean_lead_s"] for s in SEEDS]),
        per_seed_n_missed=ms([per_seed[s]["lead"]["n_missed"] for s in SEEDS]),
        per_seed_n_premature=ms([per_seed[s]["lead"]["n_premature"] for s in SEEDS]),
        per_seed_frac_over_50s=ms([per_seed[s]["lead"]["frac_lead_over_horizon"] for s in SEEDS]),
        per_file_mean_lead_s={int(f): (float(np.nanmean(L[:, j])) if np.isfinite(L[:, j]).any() else None)
                              for j, f in enumerate(ok_files)},
        note="alerts earlier than 50 s are issued in label-negative windows by construction (premature)",
    )

    (res_dir / "ttb_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    if make_figure:
        _figure(L, prem, ok_files, res_dir)
    return summary


def _figure(L, prem, ok_files, res_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    vals = L[np.isfinite(L)]
    n_missed = int(np.isnan(L).sum())
    if len(vals):
        top = max(60, float(vals.max()) + 10)
        ax[0].hist(vals, bins=np.arange(-5, top + 10, 10), color="#4C72B0", edgecolor="white")
    ax[0].axvline(50, color="k", ls="--", lw=1)
    ax[0].text(48, ax[0].get_ylim()[1] * 0.92, "50 s: max lead visible to\nlabel-positive windows", fontsize=7, va="top", ha="right")
    ax[0].set_xlabel("Alert lead time (s)"); ax[0].set_ylabel("File x seed count")
    ax[0].set_title(f"Pooled lead time (detected: {len(vals)}, missed: {n_missed}, premature: {int(prem.sum())})", fontsize=8)
    means = np.array([np.nanmean(L[:, j]) if np.isfinite(L[:, j]).any() else np.nan for j in range(L.shape[1])])
    sds = np.array([np.nanstd(L[:, j], ddof=1) if np.isfinite(L[:, j]).sum() > 1 else 0.0 for j in range(L.shape[1])])
    xs = np.arange(len(ok_files))
    ax[1].errorbar(xs, means, yerr=sds, fmt="o", color="#C44E52", capsize=3)
    ax[1].axhline(50, color="k", ls="--", lw=1); ax[1].axhline(0, color="gray", lw=0.5)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"F{f}" for f in ok_files], fontsize=7)
    ax[1].set_xlabel("Held-out validation file"); ax[1].set_ylabel("Mean lead time over seeds (s) +/- SD")
    ax[1].set_title("Per-file lead time (10 seeds)", fontsize=8)
    fig.tight_layout()
    fig.savefig(res_dir / "ttb_lead_time_distribution.png", dpi=200)
    fig.savefig(res_dir / "ttb_lead_time_distribution.pdf")
    plt.close(fig)


if __name__ == "__main__":
    s = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(json.dumps({k: s[k] for k in ("n_scored_windows", "baseline_mean_const", "model_rmse_s", "primary", "n_collapsed_seeds")},
                     indent=2, default=float))
