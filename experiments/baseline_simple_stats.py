# experiments/baseline_simple_stats.py
"""
Statistics + Section 5.2.7 table for the simple baselines. Reads results/baseline/ (references) and
results/baseline_simple/ only. Pre-registered rules (PROTOCOL.md, "Simple baselines"):
  alpha = 0.05 two-sided; PRIMARY claim = unadjusted p of the paired t-test (RF) / one-sample t-test (rule);
  exact Wilcoxon = sensitivity; Holm-adjusted p shown alongside, within each comparison across the metrics
  that apply (RF: AUC, PatAcc, RCS Top-1; rule: AUC, RCS Top-1); 95 % t-CI of each difference.
  Direction: simple baseline MINUS reference. p >= alpha is reported as "no detectable difference".
Usage: python experiments/baseline_simple_stats.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from configs import baseline_simple as C                                 # noqa: E402
from configs.runs import SEEDS                                           # noqa: E402
from experiments.baseline_stats import paired, one_sample, holm          # noqa: E402

REF, SIM = REPO / "results" / "baseline", REPO / "results" / "baseline_simple"
LABEL = {"run4_retrain": "CruxSight Run 4", "no_toc": "GAT+TFT, no TOC", "rf_pernode": "RF, per-node F0-F6",
         "rf_flat": "RF, flattened all-node", "rule": "Static threshold rule (F0)"}
NAME = {"auc": "AUC", "pat_acc": "PatAcc", "rcs_top1": "RCS Top-1 (%)"}


def vec(arm, m, root):
    return np.array([json.loads((root / f"{arm}_seed{s}.json").read_text())[m] for s in SEEDS], float)


def wording(p, diff):
    return ("no detectable difference" if p >= C.ALPHA
            else f"difference detected ({'higher' if diff > 0 else 'lower'} than reference)")


def compare():
    out = {}
    for ref in C.REFERENCE_ARMS:
        for arm in C.RF_ARMS:
            rows = {m: paired(vec(arm, m, SIM), vec(ref, m, REF)) for m in C.PAIRED_METRICS}
            h = holm({m: r["p_t"] for m, r in rows.items()})
            out[f"{arm}__vs__{ref}"] = {m: dict(rows[m], p_holm=h[m], test="paired t (seed-wise)",
                                              wording=wording(rows[m]["p_t"], rows[m]["mean_diff"])) for m in rows}
        rule = json.loads((SIM / "rule.json").read_text())
        rows = {}
        for m in C.RULE_METRICS:
            r = one_sample(vec(ref, m, REF), float(rule[m]))
            # one_sample is reference-vs-rule; flip to rule MINUS reference
            # ci95_mean is the CI of the reference MEAN; the CI of (rule - reference mean) is mu - that interval (Amendment A2)
            lo, hi = r["ci95_mean"]
            r.update(mean_diff=-r["mean_diff"], ci95_mean=(r["mu"] - hi, r["mu"] - lo), dz=-r["dz"])
            assert r["ci95_mean"][0] <= r["mean_diff"] <= r["ci95_mean"][1]
            rows[m] = r
        h = holm({m: r["p_t"] for m, r in rows.items()})
        out[f"rule__vs__{ref}"] = {m: dict(rows[m], p_holm=h[m], test="one-sample t of reference seeds vs rule value",
                                         wording=wording(rows[m]["p_t"], rows[m]["mean_diff"])) for m in rows}
    return out


def table():
    f = lambda x, nd: f"{x:.{nd}f}"
    L = ["| Method | AUC | PatAcc | RCS Top-1 (%) |", "|---|---|---|---|"]
    for a in C.REFERENCE_ARMS + C.RF_ARMS:
        root = REF if a in C.REFERENCE_ARMS else SIM
        c = []
        for m, nd in (("auc", 4), ("pat_acc", 4), ("rcs_top1", 2)):
            v = vec(a, m, root)
            c.append(f"{f(v.mean(), nd)} ± {f(v.std(ddof=1), nd)}")
        L.append(f"| {LABEL[a]} (10 seeds) | " + " | ".join(c) + " |")
    lr = json.loads((REF / "logreg.json").read_text())
    L.append(f"| Logistic regression, F0-F6 (single fit) | {f(lr['auc'],4)} | {f(lr['pat_acc'],4)} | {f(lr['rcs_top1'],2)} |")
    r = json.loads((SIM / "rule.json").read_text())
    L.append(f"| {LABEL['rule']} (single fit) | {f(r['auc'],4)} | not applicable | {f(r['rcs_top1'],2)} |")
    return "\n".join(L)


def tests_md(res):
    L = ["| Comparison (baseline − reference) | Metric | Mean diff | 95 % CI | p (primary, unadjusted) | p Wilcoxon | p Holm | Reading |",
         "|---|---|---|---|---|---|---|---|"]
    for k, rows in res.items():
        for m, r in rows.items():
            ci = r.get("ci95", r.get("ci95_mean"))
            pw = r["p_wilcoxon"]
            L.append(f"| {k.replace('__vs__', ' vs ')} | {NAME[m]} | {r['mean_diff']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{r['p_t']:.4f} | {pw:.4f} | {r['p_holm']:.4f} | {r['wording']} |")
    return "\n".join(L)


def main():
    res = compare()
    (SIM / "baseline_simple_stats.json").write_text(json.dumps(res, indent=2, default=float))
    md = ("# Simple baselines (recomputed from results/baseline and results/baseline_simple)\n\n" + table() +
          "\n\nRCS Top-1 n per method is recorded in the result files; n = 10 seeds limits power, "
          "so a non-significant result is not evidence of equality.\n\n" + tests_md(res) + "\n")
    (SIM / "results_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
