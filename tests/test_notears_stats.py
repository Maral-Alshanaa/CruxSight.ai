# tests/test_notears_stats.py -- synthetic result tree for experiments/notears_stats.py
import json, sys, tempfile, shutil, unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import notears_stats as NS                      # noqa: E402
import configs.notears_runs as NR               # noqa: E402
from configs.runs import SEEDS                  # noqa: E402
from src import causal_analysis as ca           # noqa: E402


def fake_tree(out: Path, diverge=("lam0.5", 1011), chain_signal=True):
    d = 30
    ei = np.array([list(range(29)), list(range(1, 30))])
    (out / "true_edges.json").write_text(json.dumps(dict(edge_index=ei.tolist())))
    for arm in NR.ARMS:
        for si, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed)
            rd = out / arm / f"seed{seed}"; rd.mkdir(parents=True)
            G = rng.random((d, d)) * 0.2
            if chain_signal:
                for i in range(29): G[i, i + 1] = 0.9
            np.fill_diagonal(G, 0)
            div = (arm, seed) == tuple(diverge)
            lam = float(arm.split("lam")[1])
            hist = [dict(epoch=e, val_graph=ca.summarize_graph(G * (0.5 + 0.1 * e)),
                         val_auc=0.8) for e in (1, 2)]
            (rd / "causal_history.json").write_text(json.dumps(hist))
            if not div:
                np.savez(rd / "causal_analysis.npz", val_graph=G, W_sigmoid=G)
            rec = dict(arm=arm, seed=seed, git_sha="abc", best_epoch=2, diverged=div,
                       auc=None if div else 0.87 - 0.01 * lam + 0.001 * si,
                       pat_acc=0.9, cp_recall=0.9, rcs_top1=0.5, best_pos_rate=0.5)
            (out / f"{arm}_seed{seed}.json").write_text(json.dumps(rec))


class T(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp()); fake_tree(self.out)
    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)

    def test_holm(self):
        self.assertTrue(np.allclose(NS.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))

    def test_end_to_end_and_divergence_handling(self):
        res = NS.analyse(self.out)
        self.assertEqual(res["p1"]["n_seeds"], 10)
        self.assertEqual(res["p1_sensitivity_drop_diverged_seeds"]["n_seeds"], 9)
        self.assertEqual(res["arms"]["lam0.5"]["n_diverged"], 1)
        self.assertEqual(res["arms"]["lam0.5"]["auc"]["n"], 9)
        self.assertLess(res["p1"]["p"], 0.05)             # built-in lambda effect
        self.assertEqual(len(res["secondary_vs_reference"]), 5)
        self.assertEqual(len(res["secondary_fn_weight"]), 2)
        json.dumps(res)

    def test_p2_detects_planted_structure(self):
        res = NS.analyse(self.out)
        p2 = res["p2_reference_arm"]
        self.assertEqual(p2["p2_n"], 10)
        self.assertEqual(p2["p2_k_significant"], 10)
        self.assertLess(p2["p2_binom_p"], 1e-6)
        self.assertGreater(p2["mean_pairwise_jaccard"]["mean"],
                           5 * p2["random_jaccard_expectation"])

    def test_missing_result_is_loud(self):
        (self.out / "lam0.1_seed42.json").unlink()
        with self.assertRaises(FileNotFoundError):
            NS.analyse(self.out)


if __name__ == "__main__":
    unittest.main()
