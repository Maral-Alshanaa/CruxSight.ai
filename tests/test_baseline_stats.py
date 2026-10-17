# tests/test_baseline_stats.py -- numpy/scipy only (no torch). Run: python tests/test_baseline_stats.py
import json, sys, tempfile
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.runs import SEEDS
import experiments.baseline_stats as bs


def test_wilcoxon_hand_cases():
    r = bs.wilcoxon_exact(np.arange(1, 11, dtype=float))          # all positive -> W=0, p=2/2^10
    assert r["W"] == 0 and abs(r["p"] - 2 / 1024) < 1e-12, r
    r = bs.wilcoxon_exact(np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10.]))
    assert r["p"] > 0.5, r                                         # symmetric-looking -> not significant
    r = bs.wilcoxon_exact(np.array([0., 0., 1., 2., 3.]))          # zeros dropped (wilcox), n_eff = 3
    assert r["n_eff"] == 3 and abs(r["p"] - 2 / 8) < 1e-12, r


def test_wilcoxon_matches_scipy_without_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = rng.normal(0.3, 1.0, 10)
        mine = bs.wilcoxon_exact(d)["p"]
        ref = stats.wilcoxon(d, method="exact").pvalue
        assert abs(mine - ref) < 1e-12, (mine, ref)


def test_paired_quantities():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.7, 0.05, 10), rng.normal(0.69, 0.05, 10)
    r = bs.paired(a, b)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    assert abs(r["t"] - t) < 1e-12 and abs(r["p_t"] - p) < 1e-12
    assert abs(r["dz"] - d.mean() / d.std(ddof=1)) < 1e-12
    assert abs(r["dz"] - r["t"] / np.sqrt(10)) < 1e-12               # d_z = t / sqrt(n)
    lo, hi = r["ci95"]
    assert lo < d.mean() < hi and abs((lo + hi) / 2 - d.mean()) < 1e-12
    assert r["higher_a"] + r["higher_b"] + r["tied"] == 10


def test_one_sample_and_holm():
    x = np.array([60, 70, 55, 65, 62, 58, 72, 66, 61, 64.])
    r = bs.one_sample(x, 38.2)
    ref = stats.ttest_1samp(x, 38.2)
    assert abs(r["t"] - ref.statistic) < 1e-12 and r["above"] == 10 and abs(r["p_wilcoxon"] - 2 / 1024) < 1e-12
    h = bs.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert abs(h["a"] - 0.03) < 1e-12 and abs(h["c"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12


def test_random_baseline_recompute():
    counts = {"A": 152, "C": 67, "B": 49, "D": 42, "E": 14, "F": 14}
    assert round(bs.weighted_random_top1(counts), 1) == 38.2


def _write(d: Path, arm, seed, auc, pat, rcs, cp=0.95):
    (d / f"{arm}_seed{seed}.json").write_text(json.dumps(dict(
        arm=arm, seed=seed, auc=auc, pat_acc=pat, rcs_top1=rcs, rcs_top1_n=338, cp_recall=cp,
        git_sha="x" * 40, prereg_commit="y" * 40,
        preflight=dict(val_pos_pattern_counts={"A": 152, "C": 67, "B": 49, "D": 42, "E": 14, "F": 14}))))


def test_end_to_end_and_mismatch_is_reported_not_adjusted():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for s in SEEDS:
            _write(d, "run4_retrain", s, rng.normal(.87, .01), rng.normal(.90, .02), rng.normal(62, 8))
            _write(d, "no_toc", s, rng.normal(.87, .01), rng.normal(.89, .02), rng.normal(63, 20))
        code = bs.run(d, d)
        assert code == 1                                            # synthetic data cannot match the paper numbers
        res = json.loads((d / "baseline_stats.json").read_text())
        assert any(v["status"] == "MISMATCH" for v in res["verification"])
        assert (d / "results_table.md").exists()
        (d / "no_toc_seed42.json").unlink()
        try:
            bs.run(d, d); raise AssertionError("missing seed must raise")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
