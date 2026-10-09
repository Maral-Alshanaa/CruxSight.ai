# tests/test_f6_anomaly_stats.py
"""Synthetic-JSON test of experiments/f6_anomaly_stats.py (numpy/scipy only)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from experiments import f6_anomaly_stats as st  # noqa: E402


def _fake(seed, rng, shift):
    def arm(sh, init_sd):
        layers = {f"layer{i}": dict(entropy=0.8 + sh + 0.01 * rng.randn(),
                                    enrichment_pos=1.1 + 3 * sh + 0.01 * rng.randn(),
                                    sent_vs_f6_spearman=0.1, row_sum_min=1.0, row_sum_max=1.0)
                  for i in range(2)}
        return layers, dict(reference_auc=0.6, sub_aucs=list(0.6 + init_sd * rng.randn(5)),
                            sub_sd=init_sd)
    la, da = arm(0.0, 0.08); lb, db = arm(shift, 0.08)
    diag = dict(frac_pred_pos=0.5, prob_sd=0.1, n_unique_probs=100)
    rep = dict(logged=0.6, recomputed=0.6, abs_diff=0.0, passed=True)
    return dict(seed=seed, git_sha="abc", repro=dict(with_f6=rep, ablated=rep),
                attention=dict(with_f6=la, ablated=lb, js_between_arms=dict(layer0=0.1, layer1=0.2)),
                d1=dict(with_f6=da, ablated=db), d2=dict(with_f6=diag, ablated=diag),
                gates=dict(repro_passed=True, rows_sum_to_one=True, no_nan=True))


def test_aggregate_detects_shift_and_null(tmp_path):
    for shift, expect_reject in ((0.0, False), (0.2, True)):
        rng = np.random.RandomState(0)
        d = tmp_path / f"s{shift}"; d.mkdir()
        for s in st.SEEDS:
            json.dump(_fake(s, rng, shift), open(d / f"seed{s}.json", "w"))
        r = st.aggregate(d)
        assert r["gates"]["repro_all_passed"] and r["gates"]["no_nan_all"]
        assert r["attention"]["entropy"]["reject_at_0.05_holm"] is expect_reject
        assert r["d1_init_noise"]["with_f6"]["material"] is True      # SD 0.08 >= 0.05
        assert 0 < r["d1_init_noise"]["with_f6"]["init_variance_share"] <= 1
