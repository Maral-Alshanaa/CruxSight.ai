# tests/test_ttb_stats.py
"""Runs experiments/ttb_stats.py end-to-end on synthetic per-seed exports (NumPy only)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "tests"))
from configs.runs import SEEDS
from experiments import ttb_stats as S          # experiments/ has no __init__: namespace import
from test_ttb_eval_metrics import make_file, concat


def build_export(dirpath, model="good"):
    rng = np.random.default_rng(0)
    files = [make_file(45, range(o, o + 4), seed=100 + i) for i, o in enumerate([24, 27, 30, 33, 36, 38, 26, 29])]
    files.append(make_file(45, [], seed=200))                   # a file without bottleneck
    X, L, T = concat(files)
    ids = np.concatenate([np.full(len(f[1]), i) for i, f in enumerate(files)])
    base = dict(mean_min=0.1055253632, median_min=0.0, n=1104)
    (dirpath / "ttb_baselines.json").write_text(json.dumps(base))
    for s in SEEDS:
        r = np.random.default_rng(s)
        if model == "good":
            pred = np.where(L == 1, T + r.normal(0, 0.02, len(T)), 0.1) .clip(min=0)
        else:                                                       # collapsed to the train mean
            pred = base["mean_min"] + r.normal(0, 1e-6, len(T))
        prob = np.where(L == 1, 0.9, 0.1)                           # alerts exactly on label-positive windows
        np.savez(dirpath / f"ttb_val_seed{s}.npz", prob=prob, ttb_pred=pred, label=L, ttb_true=T,
                 pattern=np.zeros(len(T), dtype=int), file_id=ids)
    return len(files)


class TestTTBStats(unittest.TestCase):
    def test_good_model_is_detected_as_better(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); build_export(d, "good")
            s = S.main(d, d / "res")
            self.assertTrue(s["primary"]["significant"])
            self.assertEqual(s["primary"]["direction"], "model better")
            self.assertEqual(s["n_collapsed_seeds"], 0)
            self.assertLess(s["bootstrap"]["rmse_diff_ci95_s"][1], 0.0)      # whole CI favours model
            self.assertEqual(s["lead"]["n_evaluable_files"], 8)               # 8 with onset, 1 without bottleneck
            self.assertEqual(s["lead"]["file_status"][8], "no_bottleneck")
            self.assertEqual(s["lead"]["per_seed_n_missed"]["mean"], 0.0)
            self.assertEqual(s["lead"]["per_seed_n_premature"]["mean"], 0.0)  # alerts start at first positive window
            self.assertLessEqual(s["lead"]["per_seed_mean_lead_s"]["mean"], 50.0)
            for f in ("ttb_eval_summary.json", "ttb_lead_time_distribution.png", "ttb_lead_time_distribution.pdf"):
                self.assertTrue((d / "res" / f).exists(), f)

    def test_collapsed_head_is_flagged_and_not_better(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); build_export(d, "collapsed")
            s = S.main(d, d / "res", make_figure=False)
            self.assertEqual(s["n_collapsed_seeds"], 10)
            self.assertFalse(s["primary"]["significant"] and s["primary"]["direction"] == "model better")


if __name__ == "__main__":
    unittest.main()
