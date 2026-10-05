# tests/test_pf_resource_posthoc.py
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from experiments import pf_resource_posthoc as H


def rec(pre, bn, post, n_pre=5, n_bn=20, n_post=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n_pre, int), np.ones(n_bn, int), np.zeros(n_post, int)]
    v = np.r_[rng.normal(pre, 0.01, n_pre), rng.normal(bn, 0.01, n_bn), rng.normal(post, 0.01, n_post)]
    return dict(label=y, values={"cpu": v})


class TestPosthoc(unittest.TestCase):
    def test_segments(self):
        y = [0, 0, 1, 1, 0, 1, 0]
        m = H.segment_masks(y)
        self.assertEqual(list(np.where(m["pre"])[0]), [0, 1])
        self.assertEqual(list(np.where(m["bn"])[0]), [2, 3, 5])
        self.assertEqual(list(np.where(m["post"])[0]), [4, 6])
        self.assertIsNone(H.segment_masks([0, 0, 0]))

    def test_no_phase_effect_and_planted_bottleneck_effect(self):
        recs = [rec(10, 8, 10, seed=i) for i in range(12)]
        pre, bn, nb = H.seg_ratios(recs, "cpu")
        self.assertAlmostEqual(pre.mean(), 1.0, places=2)
        self.assertAlmostEqual(bn.mean(), 0.8, places=2)
        self.assertEqual(nb[0].tolist(), [5, 20, 5])

    def test_warmup_pattern_is_visible(self):
        recs = [rec(13, 10, 10, seed=i) for i in range(12)]
        pre, bn, _ = H.seg_ratios(recs, "cpu")
        self.assertAlmostEqual(pre.mean(), 1.3, places=2)
        self.assertAlmostEqual(bn.mean(), 1.0, places=2)

    def test_skips_files_with_short_segments_or_zero_baseline(self):
        recs = [rec(10, 8, 10, n_post=2), rec(10, 8, 0.0, seed=3)]
        recs[1]["values"]["cpu"][-5:] = 0.0
        pre, bn, nb = H.seg_ratios(recs, "cpu")
        self.assertEqual(len(pre), 0)

    def test_summarize_threshold_and_determinism(self):
        few = H.summarize([rec(10, 8, 10, seed=i) for i in range(5)], ["cpu"])
        self.assertTrue(few["cpu"]["insufficient"])
        recs = [rec(10, 8, 10, seed=i) for i in range(12)]
        a, b = H.summarize(recs, ["cpu"]), H.summarize(recs, ["cpu"])
        self.assertEqual(a, b)
        lo, hi = a["cpu"]["bn_ci90"]
        self.assertTrue(lo <= a["cpu"]["bn_over_post"] <= hi)


if __name__ == "__main__":
    unittest.main()
