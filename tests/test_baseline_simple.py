# tests/test_baseline_simple.py -- numpy/scipy/sklearn only (no torch). Run: python tests/test_baseline_simple.py
import json, sys, tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs import baseline_simple as C
from configs.runs import SEEDS
from src import simple_baselines as SB
import experiments.baseline_simple_stats as ss

PAT = {"A": frozenset({1, 2, 3}), "B": frozenset({10, 11}), "FULL": frozenset(range(30))}
IDX = {0: "A", 1: "B", 2: "FULL"}          # pattern 3 = "none"


def synth(n, seed, shift=6.0):
    r = np.random.default_rng(seed)
    X = r.normal(10, 2, (n, 30, 7)); y = r.integers(0, 2, n); p = np.where(y == 1, r.integers(0, 4, n), 3)
    for i in np.where(y == 1)[0]:
        f = list(PAT.get(IDX.get(int(p[i]), "none"), []))
        X[i, f, 0] += shift
        X[i, f, 1] += shift / 2
    return X, y, p


def test_eligibility_excludes_empty_and_full():
    X, y, p = synth(200, 0)
    idx, fl = SB.eligible(y, p, PAT, IDX)
    assert all(y[i] == 1 and p[i] in (0, 1) for i in idx) and all(len(f) in (2, 3) for f in fl)


def test_rule_uses_normal_train_windows_only_and_matches_manual():
    X, y, p = synth(300, 1)
    m = SB.rule_fit(X, y)
    f0 = X[:, :, 0]
    assert np.allclose(m["mu"], f0[y == 0].mean(0)) and np.allclose(m["sd"], f0[y == 0].std(0))
    assert m["n_normal"] == int((y == 0).sum())
    z, s = SB.rule_scores(m, X)
    assert np.allclose(s, ((f0 - m["mu"]) / (m["sd"] + C.EPS)).max(1))


def test_rule_tau_depends_on_train_only():
    Xtr, ytr, ptr = synth(300, 2); Xva, yva, pva = synth(150, 3)
    t1 = SB.rule_fit(Xtr, ytr)["tau"]
    Xva2, yva2 = Xva * 5 + 100, 1 - yva                       # arbitrary change of the validation split
    assert SB.rule_fit(Xtr, ytr)["tau"] == t1                 # fit signature has no validation input
    m = SB.rule_fit(Xtr, ytr)
    assert SB.rule_eval(m, Xva, yva, pva, PAT, IDX)["tau"] == SB.rule_eval(m, Xva2, yva2, pva, PAT, IDX)["tau"]


def test_rule_metrics_and_planted_signal():
    Xtr, ytr, ptr = synth(400, 4); Xva, yva, pva = synth(200, 5)
    m = SB.rule_fit(Xtr, ytr); r = SB.rule_eval(m, Xva, yva, pva, PAT, IDX)
    assert abs(r["auc"] - roc_auc_score(yva, SB.rule_scores(m, Xva)[1])) < 1e-12
    assert r["pat_acc"] is None and r["auc"] > 0.85 and r["rcs_top1"] > 80
    idx, _ = SB.eligible(yva, pva, PAT, IDX); assert r["rcs_top1_n"] == len(idx)


def test_rule_zero_sd_node_is_finite():
    Xtr, ytr, ptr = synth(100, 6); Xtr[:, 7, 0] = 3.0
    m = SB.rule_fit(Xtr, ytr); z, s = SB.rule_scores(m, Xtr)
    assert np.isfinite(z).all() and np.isfinite(s).all()


def _small_rf(monkey=True):
    C.RF_PARAMS.update(n_estimators=40, n_jobs=1)


def test_rf_arms_run_finite_deterministic_and_find_planted_signal():
    _small_rf()
    Xtr, ytr, ptr = synth(250, 7); Xva, yva, pva = synth(120, 8)
    idx, _ = SB.eligible(yva, pva, PAT, IDX)
    for fn in (SB.rf_pernode, SB.rf_flat):
        a = fn(Xtr, ytr, ptr, Xva, yva, pva, 42, PAT, IDX)
        b = fn(Xtr, ytr, ptr, Xva, yva, pva, 42, PAT, IDX)
        assert a == b, fn.__name__
        assert all(np.isfinite(a[k]) for k in ("auc", "pat_acc", "rcs_top1")) and a["rcs_top1_n"] == len(idx)
        assert a["auc"] > 0.8, (fn.__name__, a)


def test_decision_wording_and_holm_family():
    assert ss.wording(0.06, 0.1) == "no detectable difference" and "higher" in ss.wording(0.01, 0.1)
    assert C.RULE_METRICS == ("auc", "rcs_top1") and "pat_acc" in C.PAIRED_METRICS


def test_stats_end_to_end_on_fake_results():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); ref, sim = d / "ref", d / "sim"; ref.mkdir(); sim.mkdir()
        r = np.random.default_rng(0)
        for arm in C.REFERENCE_ARMS:
            for s in SEEDS:
                (ref / f"{arm}_seed{s}.json").write_text(json.dumps(dict(auc=.87 + r.normal(0, .01), pat_acc=.88 + r.normal(0, .01), rcs_top1=40 + r.normal(0, 2))))
        for arm in C.RF_ARMS:
            for s in SEEDS:
                (sim / f"{arm}_seed{s}.json").write_text(json.dumps(dict(auc=.70 + r.normal(0, .01), pat_acc=.80 + r.normal(0, .01), rcs_top1=30 + r.normal(0, 2))))
        (sim / "rule.json").write_text(json.dumps(dict(auc=.65, rcs_top1=35.0)))
        (ref / "logreg.json").write_text(json.dumps(dict(auc=.71, pat_acc=.88, rcs_top1=41.7)))
        ss.REF, ss.SIM = ref, sim
        res = ss.compare()
        assert set(res) == {f"{a}__vs__{b}" for a in ("rf_pernode", "rf_flat", "rule") for b in C.REFERENCE_ARMS}
        assert "pat_acc" not in res["rule__vs__run4_retrain"]
        x = res["rule__vs__run4_retrain"]["auc"]; assert x["mean_diff"] < 0 and x["ci95_mean"][0] < x["ci95_mean"][1]
        assert abs(sum(x["ci95_mean"]) / 2 - x["mean_diff"]) < 1e-12      # CI centred on the difference (A2)
        assert res["rf_flat__vs__no_toc"]["auc"]["wording"].startswith("difference detected")
        assert "not applicable" in ss.table() and "|" in ss.tests_md(res)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
