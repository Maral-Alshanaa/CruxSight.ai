# experiments/analyze_causal_fix_all4.py
"""Unified analysis: all four Table 10 configurations under the causal-layer fix.
Reads results/causal_fix (40 files: run1-4 x 10 seeds, PREBUILD_CAUSAL=1)."""
import itertools
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import RUNS, SEEDS

RES = Path("results/causal_fix")
METRICS = [("auc", "AUC"), ("cp_recall", "CP-Recall"), ("pat_acc", "PatAcc")]
EXACT_BASELINE = 1 - comb(12, 3) / comb(30, 3)


def load(run_id):
    recs = []
    for seed in SEEDS:
        p = RES / f"{run_id}_seed{seed}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    assert all(r["causal_w_absmax"] > 0 for r in recs), f"{run_id}: fix not active in all seeds"
    return recs


def signflip_exact(d):
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    tot = r.sum()
    w_plus = r[d > 0].sum()
    dist = np.array([np.dot(s, r) for s in itertools.product([0, 1], repeat=len(d))])
    p = float(np.mean(np.abs(dist - tot / 2) >= abs(w_plus - tot / 2) - 1e-9))
    return float(min(w_plus, tot - w_plus)), p


def pm(m, s):
    return f"{m:.4f} +- {s:.4f}"


def paired(x, y, n_seeds):
    d = y - x
    n = len(d)
    assert n == n_seeds
    t, p = stats.ttest_rel(y, x)
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    w, p_exact = signflip_exact(d)
    return dict(mean_diff=float(d.mean()), ci95=[float(d.mean() - half), float(d.mean() + half)],
               t=float(t), p_ttest=float(p), wilcoxon_w=w, p_wilcoxon_exact=p_exact,
               cohen_dz=float(d.mean() / sd), n_pos=int((d > 0).sum()), n=n)


def main():
    data = {r: load(r) for r in RUNS}
    n = len(SEEDS)

    print(f"=== Table: all four runs under the causal-layer fix (n={n} seeds) ===\n")
    rows = []
    for r in sorted(data):
        cells = {}
        for m, _ in METRICS:
            vals = np.array([rec[m] for rec in data[r]])
            cells[m] = (vals.mean(), vals.std(ddof=1))
        above = sum(rec["cp_recall"] > EXACT_BASELINE for rec in data[r])
        rows.append((r, cells, above))
        print(f"{r}: AUC {pm(*cells['auc'])} | CP-Recall {pm(*cells['cp_recall'])} "
              f"| PatAcc {pm(*cells['pat_acc'])} | CP-Recall>baseline: {above}/{n}")

    print(f"\nExact random baseline: {EXACT_BASELINE:.4f}\n")

    # Primary test (still Run4 vs Run2, now under the fix -- matches causal-fix-v1 exactly)
    x2 = np.array([r["cp_recall"] for r in data["run2"]])
    y4 = np.array([r["cp_recall"] for r in data["run4"]])
    res42 = paired(x2, y4, n)
    print("=== Primary test: Run 4 vs Run 2 on CP-Recall (fixed causal layer) ===")
    print(f"mean diff {res42['mean_diff']:.4f} CI{res42['ci95']}, "
          f"t={res42['t']:.3f} p={res42['p_ttest']:.4f}, "
          f"Wilcoxon W={res42['wilcoxon_w']:.1f} p={res42['p_wilcoxon_exact']:.4f}, "
          f"d_z={res42['cohen_dz']:.2f}, Run4>Run2 in {res42['n_pos']}/{n}")

    # Exploratory: Run3 vs Run1 (mirrors the primary test under the OTHER loss-weight regime)
    x1 = np.array([r["cp_recall"] for r in data["run1"]])
    y3 = np.array([r["cp_recall"] for r in data["run3"]])
    res31 = paired(x1, y3, n)
    print("\n=== Exploratory: Run 3 vs Run 1 on CP-Recall (fixed causal layer) ===")
    print(f"mean diff {res31['mean_diff']:.4f} CI{res31['ci95']}, "
          f"t={res31['t']:.3f} p={res31['p_ttest']:.4f}, "
          f"Wilcoxon W={res31['wilcoxon_w']:.1f} p={res31['p_wilcoxon_exact']:.4f}, "
          f"d_z={res31['cohen_dz']:.2f}, Run3>Run1 in {res31['n_pos']}/{n}")

    out = {
        "n_seeds": n,
        "per_run": {r: {m: {"mean": float(cells[m][0]), "std": float(cells[m][1])}
                       for m, _ in METRICS} for r, cells, _ in rows},
        "baseline_exact": EXACT_BASELINE,
        "primary_run4_vs_run2": res42,
        "exploratory_run3_vs_run1": res31,
    }
    (RES / "causal_fix_all4_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RES/'causal_fix_all4_stats.json'}")


if __name__ == "__main__":
    main()