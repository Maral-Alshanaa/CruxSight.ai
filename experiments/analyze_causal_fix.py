# experiments/analyze_causal_fix.py
"""Standalone analysis for the causal-fix parallel study (Run2/Run4, PREBUILD_CAUSAL=1).
Does not touch stage2/3/4 or their outputs (multiseed-v1 stays untouched)."""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import SEEDS

RES = Path("results/causal_fix")
EXACT_BASELINE = 1 - comb(12, 3) / comb(30, 3)


def load(run_id):
    recs = []
    for seed in SEEDS:
        p = RES / f"{run_id}_seed{seed}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    return recs


def main():
    r2 = load("run2")
    r4 = load("run4")
    assert all(r["causal_w_absmax"] > 0 for r in r2 + r4), \
        "Fix did not take effect in one or more jobs (causal_w_absmax == 0)"

    x = np.array([r["cp_recall"] for r in r2])
    y = np.array([r["cp_recall"] for r in r4])
    d = y - x
    n = len(d)
    t, p = stats.ttest_rel(y, x)
    w, pw = stats.wilcoxon(y, x, alternative="two-sided")
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)

    print(f"n = {n}")
    print(f"Run 2 CP-Recall: {x.mean():.4f} +- {x.std(ddof=1):.4f}")
    print(f"Run 4 CP-Recall: {y.mean():.4f} +- {y.std(ddof=1):.4f}")
    print(f"Mean diff (Run4-Run2): {d.mean():.4f}  95% CI [{d.mean()-half:.4f}, {d.mean()+half:.4f}]")
    print(f"Paired t-test: t({n-1})={t:.3f}, p={p:.4f}")
    print(f"Wilcoxon: W={w:.1f}, p={pw:.4f}")
    print(f"Cohen's d_z: {d.mean()/sd:.2f}")
    print(f"Baseline (exact): {EXACT_BASELINE:.4f}")
    print(f"Run2 seeds above baseline: {sum(v > EXACT_BASELINE for v in x)}/{n}")
    print(f"Run4 seeds above baseline: {sum(v > EXACT_BASELINE for v in y)}/{n}")
    print(f"causal_w_absmax range: run2 [{min(r['causal_w_absmax'] for r in r2):.4f}, "
          f"{max(r['causal_w_absmax'] for r in r2):.4f}], "
          f"run4 [{min(r['causal_w_absmax'] for r in r4):.4f}, "
          f"{max(r['causal_w_absmax'] for r in r4):.4f}]")

    out = {"n": n, "run2_cp_recall": x.tolist(), "run4_cp_recall": y.tolist(),
           "mean_diff": float(d.mean()), "ci95": [float(d.mean()-half), float(d.mean()+half)],
           "t": float(t), "p_ttest": float(p), "wilcoxon_w": float(w), "p_wilcoxon": float(pw),
           "cohen_dz": float(d.mean()/sd), "baseline_exact": EXACT_BASELINE}
    (RES / "causal_fix_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RES/'causal_fix_stats.json'}")


if __name__ == "__main__":
    main()