# tests/test_ttb_figure.py
"""Smoke test + count consistency of experiments/ttb_figure.py (NumPy/matplotlib only)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "tests"))
from configs.runs import SEEDS
from experiments import ttb_figure as F
from experiments import ttb_stats as S
from test_ttb_eval_metrics import make_file, concat
from test_ttb_stats import build_export


class TestTTBFigure(unittest.TestCase):
    def test_counts_match_stats_script_and_files_written(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            build_export(d, "good")
            # perturb detection: seed-dependent misses and early alerts
            for i, s in enumerate(SEEDS):
                z = dict(np.load(d / f"ttb_val_seed{s}.npz"))
                r = np.random.default_rng(s)
                z["prob"] = np.clip(z["prob"] + r.normal(0, 0.45, len(z["prob"])), 0, 1)
                np.savez(d / f"ttb_val_seed{s}.npz", **z)
            summ = S.main(d, d / "stats", make_figure=False)
            out = F.main(d, d / "fig")
            for f in ("ttb_lead_time_distribution.png", "ttb_lead_time_distribution.pdf"):
                self.assertTrue((d / "fig" / f).exists())
            n_files = summ["lead"]["n_evaluable_files"]
            self.assertEqual(out["n_pairs"], n_files * len(SEEDS))
            self.assertEqual(out["n_missed"], round(summ["lead"]["per_seed_n_missed"]["mean"] * len(SEEDS)))
            self.assertEqual(out["n_premature"], round(summ["lead"]["per_seed_n_premature"]["mean"] * len(SEEDS)))


if __name__ == "__main__":
    unittest.main()
