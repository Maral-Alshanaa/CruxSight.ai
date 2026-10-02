# tests/test_ttb_eval_metrics.py
"""
Verifies src/ttb_eval.py on synthetic data with hand-computed answers.
Windows are generated with EXACTLY the cache-building formula (cell 7c):
    for start in range(T - WINDOW - HORIZON + 1):
        end = start + WINDOW; y_fut = labels[end:end+HORIZON]
        label = int(y_fut.max() > 0); bn_steps = argmax(y_fut) if label else -1
        ttb = bn_steps * STEP_SEC / 60
Pure NumPy: runs without torch.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import ttb_eval as E

WINDOW, HORIZON, STEP_SEC = 12, 6, 10


def make_file(T, on_steps, seed=0, feat=(3, 2)):
    """Windows of one synthetic file. on_steps: iterable of steps flagged as bottleneck."""
    rng = np.random.default_rng(seed)
    series = rng.normal(size=(T,) + feat)
    labels = np.zeros(T, dtype=int)
    labels[list(on_steps)] = 1
    X, L, TTB = [], [], []
    for start in range(T - WINDOW - HORIZON + 1):
        end = start + WINDOW
        y = labels[end:end + HORIZON]
        lab = int(y.max() > 0)
        bn = int(np.argmax(y)) if lab else -1
        X.append(series[start:end]); L.append(lab); TTB.append(bn * STEP_SEC / 60.0)
    return np.stack(X), np.array(L), np.array(TTB)


def concat(files):
    return (np.concatenate([f[0] for f in files]), np.concatenate([f[1] for f in files]),
            np.concatenate([f[2] for f in files]))


class TestWindowMetrics(unittest.TestCase):
    def test_known_values_and_sentinel_ignored(self):
        true = np.array([0, 1, 2, 4]) / 6.0           # minutes (k steps of 10 s)
        pred = np.array([0, 0, 2, 0]) / 6.0
        label = np.ones(4, dtype=int)
        # errors in seconds: 0, -10, 0, -40  -> MAE 12.5, RMSE sqrt((100+1600)/4)
        m = E.ttb_window_metrics(pred, true, label)
        self.assertAlmostEqual(m["mae_s"], 12.5, places=6)
        self.assertAlmostEqual(m["rmse_s"], float(np.sqrt(1700 / 4)), places=6)
        self.assertAlmostEqual(m["bias_s"], -12.5, places=6)
        # add negatives carrying the -1/6 sentinel and absurd predictions: must change nothing
        pred2 = np.concatenate([pred, [99.0, 99.0]])
        true2 = np.concatenate([true, [-1 / 6, -1 / 6]])
        label2 = np.concatenate([label, [0, 0]])
        m2 = E.ttb_window_metrics(pred2, true2, label2)
        self.assertEqual(m2["n"], 4)
        self.assertAlmostEqual(m2["mae_s"], m["mae_s"], places=9)
        self.assertAlmostEqual(m2["rmse_s"], m["rmse_s"], places=9)

    def test_sentinel_in_positive_windows_raises(self):
        with self.assertRaises(ValueError):
            E.ttb_window_metrics([0.1, 0.1], [0.1, -1 / 6], [1, 1])

    def test_empty_and_collapse(self):
        m = E.ttb_window_metrics([0.3, 0.2], [-1 / 6, -1 / 6], [0, 0])
        self.assertEqual(m["n"], 0)
        self.assertTrue(np.isnan(m["rmse_s"]))
        true = np.array([0, 1, 2, 3, 4, 5]) / 6.0
        const = np.full(6, 0.1)
        mc = E.ttb_window_metrics(const, true, np.ones(6, dtype=int))
        self.assertTrue(mc["collapsed"])
        self.assertTrue(np.isnan(mc["spearman"]))
        mg = E.ttb_window_metrics(true + 0.01, true, np.ones(6, dtype=int))
        self.assertFalse(mg["collapsed"])
        self.assertAlmostEqual(mg["spearman"], 1.0, places=9)

    def test_per_k_breakdown(self):
        true = np.array([0, 0, 3]) / 6.0
        pred = np.array([1, 0, 3]) / 6.0
        m = E.ttb_window_metrics(pred, true, [1, 1, 1])
        self.assertEqual(m["per_k"][0]["n"], 2)
        self.assertAlmostEqual(m["per_k"][0]["mae_s"], 5.0, places=6)
        self.assertEqual(m["per_k"][3]["n"], 1)

    def test_baselines_match_hand_computation_on_printed_val_counts(self):
        """Counts printed from the real cache (positives only, k=0..5)."""
        tr_counts = [849, 61, 60, 55, 42, 37]
        va_counts = [262, 17, 17, 16, 14, 12]
        tr = np.repeat(np.arange(6), tr_counts) / 6.0
        va = np.repeat(np.arange(6), va_counts) / 6.0
        c = E.train_constants(tr, np.ones(len(tr), dtype=int))
        self.assertAlmostEqual(c["mean_min"], 0.1055253632, places=6)   # matches printed train mean
        self.assertEqual(c["median_min"], 0.0)
        lab = np.ones(len(va), dtype=int)
        mean_b = E.constant_baseline_metrics(c["mean_min"], va, lab)
        zero_b = E.constant_baseline_metrics(c["median_min"], va, lab)
        self.assertEqual(mean_b["n"], 338)
        self.assertAlmostEqual(mean_b["rmse_s"], 13.50, delta=0.05)
        self.assertAlmostEqual(mean_b["mae_s"], 9.85, delta=0.05)
        self.assertAlmostEqual(zero_b["mae_s"], 6.36, delta=0.05)
        self.assertAlmostEqual(zero_b["rmse_s"], 14.93, delta=0.05)


class TestFileRecovery(unittest.TestCase):
    def test_ids_recovered_on_stride1_files(self):
        fa, fb, fc = make_file(30, [], seed=1), make_file(26, [], seed=2), make_file(19, [], seed=3)
        X, L, T = concat([fa, fb, fc])
        ids = E.recover_file_ids(X)
        expected = np.concatenate([np.full(len(fa[0]), 0), np.full(len(fb[0]), 1), np.full(len(fc[0]), 2)])
        self.assertTrue((ids == expected).all())
        self.assertEqual(len(fc[0]), 2)       # smallest case still splits correctly

    def test_single_window_file_between_two_files(self):
        fa, fb, fc = make_file(24, [], seed=4), make_file(18, [], seed=5), make_file(24, [], seed=6)
        self.assertEqual(len(fb[0]), 1)
        ids = E.recover_file_ids(concat([fa, fb, fc])[0])
        self.assertEqual(len(np.unique(ids)), 3)


class TestOnsetAndLead(unittest.TestCase):
    def setUp(self):
        # onset o=30, steps 30..33 flagged, T=45 -> windows 0..27
        self.X, self.L, self.TTB = make_file(45, range(30, 34), seed=7)
        self.n = len(self.L)
        self.ids = np.zeros(self.n, dtype=int)

    def test_onset_reconstruction(self):
        r = E.file_onsets(self.ids, self.L, self.TTB)[0]
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["i0"], 13)
        self.assertEqual(r["onset_idx"], 18)
        self.assertEqual(self.TTB[18], 0.0)          # window 18 ends exactly at the onset

    def test_stable_alert_and_lead(self):
        p = np.zeros(self.n); p[15:] = 0.9
        row = E.lead_time_table(p, self.ids, self.L, self.TTB)[0]
        self.assertFalse(row["missed"]); self.assertEqual(row["lead_s"], 30.0)
        self.assertFalse(row["premature"])

    def test_retraction_resets_alert(self):
        p = np.zeros(self.n); p[10:15] = 0.9; p[16:] = 0.9      # drop at window 15
        row = E.lead_time_table(p, self.ids, self.L, self.TTB)[0]
        self.assertEqual(row["lead_s"], 20.0)

    def test_premature_alert_flagged_and_exceeds_horizon(self):
        p = np.zeros(self.n); p[5:] = 0.9
        row = E.lead_time_table(p, self.ids, self.L, self.TTB)[0]
        self.assertEqual(row["lead_s"], 130.0)
        self.assertTrue(row["premature"])
        s = E.summarize_lead([row])
        self.assertEqual(s["n_premature"], 1); self.assertEqual(s["frac_lead_over_horizon"], 1.0)

    def test_missed_when_window_before_onset_not_alerting(self):
        p = np.zeros(self.n); p[10:18] = 0.9; p[18] = 0.4
        row = E.lead_time_table(p, self.ids, self.L, self.TTB)[0]
        self.assertTrue(row["missed"]); self.assertIsNone(row["lead_s"])
        s = E.summarize_lead([row])
        self.assertEqual(s["n_missed"], 1); self.assertTrue(np.isnan(s["mean_lead_s"]))

    def test_threshold_is_strict_greater(self):
        p = np.zeros(self.n); p[15:] = 0.5
        row = E.lead_time_table(p, self.ids, self.L, self.TTB)[0]
        self.assertTrue(row["missed"])

    def test_censored_left(self):
        # steps 5..11 are invisible to every window (first horizon starts at step 12); 5..14 reaches it
        X, L, T = make_file(45, range(5, 15), seed=8)
        r = E.file_onsets(np.zeros(len(L), dtype=int), L, T)[0]
        self.assertEqual(r["status"], "censored_left")

    def test_no_bottleneck_file(self):
        X, L, T = make_file(40, [], seed=9)
        r = E.file_onsets(np.zeros(len(L), dtype=int), L, T)[0]
        self.assertEqual(r["status"], "no_bottleneck")

    def test_onset_in_last_horizon_steps(self):
        X, L, T = make_file(45, [42, 43], seed=10)       # onset 42, windows 0..27
        n = len(L)
        r = E.file_onsets(np.zeros(n, dtype=int), L, T)[0]
        self.assertEqual((r["status"], r["i0"], r["onset_idx"]), ("ok", 25, 30))
        p = np.zeros(n); p[26:] = 0.9
        row = E.lead_time_table(p, np.zeros(n, dtype=int), L, T)[0]
        self.assertEqual(row["lead_s"], 40.0)             # 30 - 26 = 4 steps

    def test_tampered_ttb_is_inconsistent(self):
        T = self.TTB.copy(); T[16] += 1 / 6.0
        r = E.file_onsets(self.ids, self.L, T)[0]
        self.assertEqual(r["status"], "inconsistent")

    def test_second_episode_does_not_move_first_onset(self):
        X, L, T = make_file(60, list(range(30, 34)) + list(range(45, 49)), seed=11)
        r = E.file_onsets(np.zeros(len(L), dtype=int), L, T)[0]
        self.assertEqual((r["status"], r["onset_idx"]), ("ok", 18))

    def test_end_to_end_two_files_via_recovered_ids(self):
        fa = make_file(45, range(30, 34), seed=12)
        fb = make_file(40, range(28, 31), seed=13)
        X, L, T = concat([fa, fb])
        ids = E.recover_file_ids(X)
        self.assertEqual(len(np.unique(ids)), 2)
        p = np.zeros(len(L)); p[:] = 0.9
        tab = E.lead_time_table(p, ids, L, T)
        self.assertTrue(all(r["status"] == "ok" for r in tab))
        self.assertEqual(tab[0]["onset_idx"], 18)
        self.assertEqual(tab[1]["onset_idx"], 16)        # file B: o=28 -> i0+k = 11+5
        self.assertTrue(all(r["premature"] for r in tab))   # always-on detector alerts from window 0


if __name__ == "__main__":
    unittest.main()
