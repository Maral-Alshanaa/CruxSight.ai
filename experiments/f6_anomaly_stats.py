# experiments/f6_anomaly_stats.py
"""
Aggregation and pre-registered secondary tests for the F6 anomaly study.
Locked by tag f6-anomaly-prereg: do not modify afterwards without a documented reason.

Reads results/f6_anomaly/seed{s}.json (10 seeds) and writes f6_anomaly_stats.json.

Secondary inferential tests (paired across the same 10 seeds, two-sided Wilcoxon
signed-rank, Holm-corrected over m = 2, alpha = 0.05), on layer-averaged metrics:
  (1) normalized attention entropy, with-F6 vs ablated
  (2) critical-node attention enrichment (positive windows), with-F6 vs ablated
Everything else is descriptive. There is NO confirmatory test of zero-shot AUC in
this study (that 10-seed comparison already exists and was post hoc; see PROTOCOL.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.f6_anomaly import holm  # noqa: E402

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
INIT_NOISE_MATERIAL = 0.05   # pre-registered: mean within-checkpoint SD >= this => "material"
REPRO_TOL = 0.005


def _ms(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(v.size),
                values=[float(x) for x in v])


def aggregate(res_dir):
    res_dir = Path(res_dir)
    runs = [json.load(open(res_dir / f"seed{s}.json")) for s in SEEDS]
    assert [r["seed"] for r in runs] == SEEDS

    def layer_mean(r, arm, key):
        return float(np.mean([l[key] for l in r["attention"][arm].values()]))

    out = dict(seeds=SEEDS, git_shas=sorted({r["git_sha"] for r in runs}))

    # gates
    out["gates"] = dict(
        repro_all_passed=all(r["gates"]["repro_passed"] for r in runs),
        repro_max_abs_diff=float(max(v["abs_diff"] for r in runs for v in r["repro"].values())),
        rows_sum_to_one_all=all(r["gates"]["rows_sum_to_one"] for r in runs),
        no_nan_all=all(r["gates"]["no_nan"] for r in runs))

    # attention: descriptive + secondary tests
    att, pv, names = {}, [], []
    for key in ("entropy", "enrichment_pos"):
        w = np.array([layer_mean(r, "with_f6", key) for r in runs])
        a = np.array([layer_mean(r, "ablated", key) for r in runs])
        d = a - w
        t = stats.wilcoxon(a, w)
        att[key] = dict(with_f6=_ms(w), ablated=_ms(a), diff_ablated_minus_with=_ms(d),
                        wilcoxon_W=float(t.statistic), wilcoxon_p=float(t.pvalue),
                        paired_t_p=float(stats.ttest_rel(a, w).pvalue),
                        ablated_higher_in=int((d > 0).sum()))
        pv.append(float(t.pvalue)); names.append(key)
    adj, rej = holm(pv)
    for k, ap, rj in zip(names, adj, rej):
        att[k]["wilcoxon_p_holm"] = ap
        att[k]["reject_at_0.05_holm"] = bool(rj)
    att["js_between_arms"] = _ms([np.mean(list(r["attention"]["js_between_arms"].values()))
                                  for r in runs])
    sp = [np.mean([l["sent_vs_f6_spearman"] for l in r["attention"]["with_f6"].values()])
          for r in runs]
    att["sent_vs_f6_spearman_with_f6"] = _ms(sp)
    out["attention"] = att

    # D1: init-noise floor
    d1 = {}
    for arm in ("with_f6", "ablated"):
        within_sd = np.array([r["d1"][arm]["sub_sd"] for r in runs])
        ref = np.array([r["d1"][arm]["reference_auc"] for r in runs])
        sub_means = np.array([np.mean(r["d1"][arm]["sub_aucs"]) for r in runs])
        within_var = float(np.mean(within_sd ** 2))
        between_var = float(np.var(sub_means, ddof=1))
        d1[arm] = dict(mean_within_checkpoint_sd=float(within_sd.mean()),
                       reference_auc=_ms(ref),
                       between_checkpoint_sd_of_means=float(np.sqrt(between_var)),
                       init_variance_share=within_var / (within_var + between_var),
                       material=bool(within_sd.mean() >= INIT_NOISE_MATERIAL))
    out["d1_init_noise"] = d1

    # D2: artifact checks
    d2 = {}
    for arm in ("with_f6", "ablated"):
        d2[arm] = dict(
            frac_pred_pos=_ms([r["d2"][arm]["frac_pred_pos"] for r in runs]),
            prob_sd=_ms([r["d2"][arm]["prob_sd"] for r in runs]),
            n_constant_prediction_seeds=int(sum(r["d2"][arm]["n_unique_probs"] <= 1 for r in runs)),
            seeds_all_pred_pos_or_all_pred_neg=[r["seed"] for r in runs
                                                if r["d2"][arm]["frac_pred_pos"] in (0.0, 1.0)])
    out["d2_prob_diagnostics"] = d2
    return out


if __name__ == "__main__":
    d = Path(__file__).resolve().parent.parent / "results" / "f6_anomaly"
    res = aggregate(d)
    json.dump(res, open(d / "f6_anomaly_stats.json", "w"), indent=1)
    print(json.dumps(res, indent=1)[:4000])
