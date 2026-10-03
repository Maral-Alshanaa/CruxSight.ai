# tests/test_notears_analysis.py
"""
Step (b) verification for the NOTEARS-layer analysis (pre-registration tag
notears-analysis-prereg). Everything runs on tiny synthetic data on CPU.

Run:  CRUX_PREBUILD_CAUSAL=1 PYTHONPATH=. python -m unittest discover -s tests -p "test_notears_analysis.py" -v

Three groups:
  1. Unit tests of src/causal_analysis.py on hand-made graphs, including the
     claim that the implemented acyclicity term is blind to 3-cycles.
  2. Gradient flow through the REAL pipeline for every pre-registered arm:
     (a) every parameter is in the optimizer, (b) every parameter gets a
     non-zero gradient on the first step, (c) every parameter value changes
     after optimizer.step(). Plus a negative control (lazy causal layer).
  3. The logging flag must not change training.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src import causal_analysis as ca                       # noqa: E402
from test_gradient_flow import build_fake_cache              # noqa: E402
import configs.notears_runs as nr                            # noqa: E402


def _cycle(d, nodes, w=0.9):
    G = np.zeros((d, d))
    for a, b in zip(nodes, nodes[1:] + nodes[:1]):
        G[a, b] = w
    return G


class TestAnalysisHelpers(unittest.TestCase):

    def test_h_impl_matches_layer_implementation(self):
        from src.core import CausalInferenceLayer
        layer = CausalInferenceLayer(d_model=8, n_nodes=6)
        rng = np.random.default_rng(0)
        G = rng.random((6, 6))
        np.fill_diagonal(G, 0.0)
        ref = float(layer.acyclicity_constraint(torch.tensor(G)))
        self.assertAlmostEqual(ca.h_impl(G), ref, places=12)

    def test_dag_has_zero_h_everywhere(self):
        G = np.zeros((5, 5))
        for i in range(4):
            G[i, i + 1] = 0.9
        self.assertEqual(ca.h_impl(G), 0.0)
        self.assertLess(ca.h_exact(G), 1e-12)
        self.assertLess(ca.h_standard(G), 1e-12)
        self.assertTrue(ca.is_dag(G, 0.5))
        self.assertEqual(ca.min_tau_for_dag(G), 0.0)

    def test_two_cycle_detected_by_both(self):
        G = np.zeros((4, 4))
        G[0, 1], G[1, 0] = 0.9, 0.6
        self.assertGreater(ca.h_impl(G), 0)
        self.assertGreater(ca.h_exact(G), 0)
        self.assertFalse(ca.is_dag(G, 0.5))
        self.assertEqual(ca.n_two_cycles(G, 0.5), 1)
        self.assertEqual(ca.min_tau_for_dag(G), 0.6)   # drop the 0.6 edge

    def test_three_cycle_invisible_to_implemented_penalty(self):
        """Key premise of the analysis: h_impl only sees 2-cycles."""
        G = _cycle(6, [0, 1, 2])
        self.assertEqual(ca.h_impl(G), 0.0)             # blind
        self.assertGreater(ca.h_exact(G), 0.0)          # exact sees it
        self.assertGreater(ca.h_standard(G), 0.0)
        self.assertFalse(ca.is_dag(G, 0.5))
        self.assertEqual(ca.n_two_cycles(G, 0.5), 0)
        self.assertEqual(ca.largest_scc(G, 0.5), 3)

    def test_h_can_be_small_by_shrinking_weights(self):
        """h -> 0 under scaling while the binary cycle persists at small tau."""
        G = _cycle(6, [0, 1, 2], w=1e-3)
        self.assertLess(ca.h_exact(G), 1e-12)
        self.assertFalse(ca.is_dag(G, 0.0))             # cycle still there
        self.assertTrue(ca.is_dag(G, 0.01))             # gone only above tau

    def test_density_entropy_saturation(self):
        d = 4
        G = np.zeros((d, d))
        for i, j in [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3)]:
            G[i, j] = 0.8
        self.assertAlmostEqual(ca.density(G, 0.5), 6 / 12)
        U = np.full((d, d), 0.5)
        self.assertAlmostEqual(ca.weight_entropy(U), 1.0, places=9)
        onehot = np.zeros((d, d))
        onehot[0, 1] = 1.0
        self.assertAlmostEqual(ca.weight_entropy(onehot), 0.0, places=9)
        self.assertEqual(ca.saturation_fraction(U), 1.0)
        self.assertEqual(ca.saturation_fraction(np.full((d, d), 0.9)), 0.0)

    def test_precision_and_orientation(self):
        d = 10
        ei = np.array([list(range(9)), list(range(1, 10))])   # chain i -> i+1
        te = ca.true_edge_set(ei, d, directed=True)
        G = np.zeros((d, d))
        for a, b in te:
            G[a, b] = 0.9
        self.assertEqual(ca.precision_at_k(G, te, directed=True), 1.0)
        rev = ca.true_edge_set(ei, d, directed=True, reverse=True)
        self.assertEqual(ca.precision_at_k(G, rev, directed=True), 0.0)
        und = ca.true_edge_set(ei, d, directed=False)
        self.assertEqual(ca.precision_at_k(G, und, directed=False), 1.0)

    def test_permutation_test_power_and_validity(self):
        d = 10
        ei = np.array([list(range(9)), list(range(1, 10))])
        te = ca.true_edge_set(ei, d, directed=True)
        G = np.zeros((d, d))
        for a, b in te:
            G[a, b] = 0.9
        self.assertLess(ca.permutation_pvalue(G, te, True, n_perm=2000, seed=1), 0.01)
        rng = np.random.default_rng(7)
        ps = [ca.permutation_pvalue(rng.random((d, d)), te, True,
                                    n_perm=300, seed=i) for i in range(40)]
        self.assertLessEqual(sum(p < 0.05 for p in ps), 8)   # null stays near 5%

    def test_jaccard(self):
        rng = np.random.default_rng(3)
        G = rng.random((8, 8))
        self.assertEqual(ca.jaccard_top_k(G, G, 10), 1.0)

    def test_summarize_graph_serialisable(self):
        rng = np.random.default_rng(4)
        out = ca.summarize_graph(rng.random((30, 30)))
        json.dumps(out)
        self.assertEqual(set(out["per_tau"]), {str(t) for t in ca.TAUS})


class _PipelineCase(unittest.TestCase):
    """Builds a fake cache and (re)imports the pipeline with chosen env."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="notears_test_"))
        self.model_dir = self.tmp / "cst_gnn"
        build_fake_cache(self.model_dir / "dataset_cache", self.model_dir)
        os.environ["CRUX_MODEL_DIR"] = str(self.model_dir.resolve())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def pipeline(self, prebuild: str):
        os.environ["CRUX_PREBUILD_CAUSAL"] = prebuild
        import src.pipeline as pl
        return importlib.reload(pl)

    @staticmethod
    def cfg(arm_name, **over):
        c = dict(nr.ARMS[arm_name], epochs=2, patience=2, lr=1e-3)
        c.update(over)
        return c

    def run_arm(self, pl, arm_name, name, log, seed=42, **over):
        run_dir = self.tmp / name
        run_dir.mkdir(parents=True, exist_ok=True)
        res = pl.train_and_evaluate(self.cfg(arm_name, **over), seed, run_dir,
                                    log_causal=log)
        return res, run_dir


class TestConfigs(unittest.TestCase):
    def test_arms_valid(self):
        self.assertEqual(nr.validate(), 80)


class TestGradientFlowAllArms(_PipelineCase):

    def test_every_parameter_every_arm(self):
        pl = self.pipeline("1")
        for arm in nr.ARMS:
            with self.subTest(arm=arm):
                res, rd = self.run_arm(pl, arm, f"gf_{arm}", log=True)
                self.assertEqual(res["n_params"], 209587)
                audit = json.loads((rd / "param_audit.json").read_text())
                self.assertEqual(sum(v["numel"] for v in audit.values()), 209587)
                bad_opt = [n for n, v in audit.items() if not v["in_optimizer"]]
                bad_grad = [n for n, v in audit.items() if not v["grad_norm"] > 0]
                bad_step = [n for n, v in audit.items() if not v["max_abs_delta"] > 0]
                self.assertEqual(bad_opt, [], "parameters outside the optimizer")
                self.assertEqual(bad_grad, [], "parameters with zero gradient")
                self.assertEqual(bad_step, [], "parameters unchanged by step()")
                self.assertIn("causal_30.W_raw", audit)
                self.assertGreater(res["causal_w_absmax"], 0.0)

    def test_negative_control_lazy_causal_layer(self):
        """PREBUILD=0 reproduces the original bug; the audit must expose it."""
        pl = self.pipeline("0")
        res, rd = self.run_arm(pl, "lam0.05", "lazy", log=True)
        audit = json.loads((rd / "param_audit.json").read_text())
        self.assertFalse(audit["causal_30.W_raw"]["in_optimizer"])
        self.assertEqual(audit["causal_30.W_raw"]["max_abs_delta"], 0.0)
        self.assertEqual(res["causal_w_absmax"], 0.0)


class TestLoggingIsSideEffectFree(_PipelineCase):

    def test_same_checkpoint_with_and_without_logging(self):
        pl = self.pipeline("1")
        # Bitwise equality is only a valid criterion on CPU: GATConv's scatter
        # uses atomic adds on CUDA, so two identical GPU runs already differ in
        # the last bits (observed on a Colab T4 in att_src). Force CPU here.
        with mock.patch("torch.cuda.is_available", return_value=False):
            r0, d0 = self.run_arm(pl, "lam0.05", "plain", log=False)
            r1, d1 = self.run_arm(pl, "lam0.05", "logged", log=True)
        s0 = torch.load(d0 / "best_model.pt", weights_only=False)["model_state"]
        s1 = torch.load(d1 / "best_model.pt", weights_only=False)["model_state"]
        self.assertEqual(set(s0), set(s1))
        for k in s0:
            self.assertTrue(torch.equal(s0[k], s1[k]), f"state differs: {k}")
        self.assertEqual(r0["auc"], r1["auc"])
        self.assertEqual(r0["best_epoch"], r1["best_epoch"])

    def test_artifacts_written_and_finite(self):
        pl = self.pipeline("1")
        res, rd = self.run_arm(pl, "lam0.05", "art", log=True)
        z = np.load(rd / "causal_analysis.npz")
        self.assertEqual(z["val_graph"].shape, (30, 30))
        self.assertEqual(z["W_sigmoid"].shape, (30, 30))
        self.assertTrue(np.allclose(np.diag(z["W_sigmoid"]), 0.0))
        hist = json.loads((rd / "causal_history.json").read_text())
        self.assertEqual(len(hist), res["n_epochs_run"])
        for rec in hist:
            g = rec["val_graph"]
            for key in ("h_impl", "h_exact", "h_standard", "min_tau_dag"):
                self.assertTrue(np.isfinite(g[key]), key)
            self.assertTrue(np.isfinite(rec["train_h_impl_batch_mean"]))
        self.assertIn("diverged", res)

    def test_nonfinite_loss_is_recorded_not_crashing(self):
        pl = self.pipeline("1")
        res, rd = self.run_arm(pl, "lam0.05", "nan", log=True, lambda_causal=1e39)
        self.assertTrue(res["diverged"])
        self.assertEqual(res["nonfinite_epoch"], 1)
        self.assertIsNone(res["auc"])


if __name__ == "__main__":
    unittest.main()
