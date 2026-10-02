# tests/test_ttb_gradient_flow.py
"""
PROTOCOL rule 1 for the TTB head, on tiny synthetic data (CPU).

 A. module level: every model parameter is inside the optimizer (after the causal layer is
    built), TTB-only loss gives non-zero grads to ALL head_ttb tensors and to the backbone,
    zero grads to the other heads, and head_ttb values change after optimizer.step().
    Sensitivity control: with no positive window the TTB loss carries no gradient, and the
    same detector reports "no gradient" (so the test can fail).
 B. real pipeline.train_and_evaluate (PREBUILD_CAUSAL=1) with a spy AdamW/CSTGNN: head_ttb
    params are in the optimizer, receive a non-zero grad during real training, and change
    between the first optimizer step and the end of training.
 C. src.ttb_export reproduces the checkpoint AUC, keeps cache order and recovers 2 val files.
Synthetic windows follow the cell-7c formula so ttb targets are consistent.
Run: python -m unittest tests.test_ttb_gradient_flow -v   (from the repo root)
"""
import importlib
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

WINDOW, HORIZON, STEP_SEC = 12, 6, 10
RUN_CFG = dict(fn_weight=1.5, fp_weight=1.0, lambda_causal=0.05, lambda_sub=0.05,
               lambda_rcs_sup=0.3, weight_decay=1e-3, lr=1e-3, patience=2, epochs=2,
               gat_hidden=32, tft_hidden=64, gat_dropout=0.2, tft_dropout=0.2,
               expected_params=205617)
CONFIG_YAML = """
data: {window_steps: 12, horizon_steps: 6, step_sec: 10, min_samples: 1,
       n_features: 7, train_ratio: 0.7, val_ratio: 0.15, random_seed: 42,
       batch_size: 4, num_workers: 0}
model: {gat_in_feats: 7, gat_hidden: 32, gat_heads: 4, gat_layers: 2,
        gat_dropout: 0.2, toc_lambda: 2.0, toc_gamma: 0.5, tft_hidden: 64,
        tft_heads: 4, lstm_layers: 2, tft_dropout: 0.2, causal_hidden: 64,
        dag_reg: 1.0, n_patterns: 8}
training: {epochs: 2, lr: 0.001, weight_decay: 0.001, grad_clip: 1.0,
           patience: 2, lambda_pattern: 0.5, lambda_ttb: 0.3,
           lambda_causal: 0.05, lambda_sub: 0.05, fn_weight: 1.5,
           fp_weight: 1.0, constraint_mult: 3.0}
device: cpu
"""


def synth_file(T, onset, seed):
    g = torch.Generator().manual_seed(seed)
    series = torch.randn(T, 30, 7, generator=g)
    labels = np.zeros(T, dtype=int)
    if onset is not None:
        labels[onset:onset + 4] = 1
    out = []
    for start in range(T - WINDOW - HORIZON + 1):
        end = start + WINDOW
        y = labels[end:end + HORIZON]
        lab = int(y.max() > 0)
        bn = int(np.argmax(y)) if lab else -1
        out.append({"x": series[start:end].clone(), "label": torch.tensor(lab),
                    "pattern_idx": torch.tensor(start % 8),
                    "ttb": torch.tensor(bn * STEP_SEC / 60.0, dtype=torch.float32)})
    return out


def build_fake_cache(model_dir: Path):
    cache = model_dir / "dataset_cache"
    cache.mkdir(parents=True, exist_ok=True)
    train = synth_file(30, 20, 1) + synth_file(30, 22, 2)
    val = synth_file(30, 20, 3) + synth_file(26, 18, 4)
    torch.save(train, cache / "train.pt")
    torch.save(val, cache / "val.pt")
    ei = torch.randint(0, 30, (2, 40))
    torch.save({30: {"edge_index": ei, "toc_cap": torch.ones(30)}}, cache / "graphs.pt")
    np.save(model_dir / "capacity_compose.npy", np.ones(30, dtype=np.float32))
    np.save(model_dir / "capacity_home.npy", np.ones(7, dtype=np.float32))
    (model_dir / "config.yaml").write_text(CONFIG_YAML)
    return train, val


def head_ttb_grad_report(model):
    """name -> summed |grad| for head_ttb tensors (0.0 when grad is None)."""
    return {n: (float(p.grad.abs().sum()) if p.grad is not None else 0.0)
            for n, p in model.named_parameters() if ".head_ttb." in n}


class TTBGradientFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = ROOT / "_ttb_gf_tmp"
        if cls.tmp.exists():
            shutil.rmtree(cls.tmp)
        cls.model_dir = cls.tmp / "cst_gnn"
        cls.model_dir.mkdir(parents=True)
        cls.train, cls.val = build_fake_cache(cls.model_dir)
        os.environ["CRUX_MODEL_DIR"] = str(cls.model_dir.resolve())
        os.environ["CRUX_PREBUILD_CAUSAL"] = "1"
        import src.pipeline as pipeline          # env vars are read at import time
        cls.pipeline = importlib.reload(pipeline)
        assert cls.pipeline.PREBUILD_CAUSAL, "PREBUILD_CAUSAL must be on for this test"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ A
    def _model_and_loss(self):
        from src.core import CSTGNN, TOCPriorLoader, TOCWeightedLoss
        P = self.pipeline
        torch.manual_seed(0)
        cfg = P._build_cfg(RUN_CFG)
        cfg.model.use_toc = True
        graphs = torch.load(P.CACHE_DIR / "graphs.pt", weights_only=False)
        ei, cap = graphs[30]["edge_index"], graphs[30]["toc_cap"]
        model = CSTGNN(cfg)
        n_before = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 12, 30, 7), ei, cap)
        n_after = sum(p.numel() for p in model.parameters())
        toc = TOCPriorLoader(np.load(P.MODEL_DIR / "capacity_compose.npy"), np.load(P.MODEL_DIR / "capacity_home.npy"))
        loss_fn = TOCWeightedLoss(cfg, toc, torch.ones(8))
        return model, loss_fn, ei, cap, n_before, n_after

    def _batch(self, positives=True):
        s = self.train
        pos = [x for x in s if int(x["label"]) == 1][:4]
        neg = [x for x in s if int(x["label"]) == 0][:4]
        chosen = (pos + neg) if positives else neg
        x = torch.stack([c["x"] for c in chosen])
        lab = torch.stack([c["label"] for c in chosen])
        ttb = torch.stack([c["ttb"] for c in chosen])
        return x, lab, ttb

    def test_A_module_level_gradient_flow(self):
        model, loss_fn, ei, cap, n_before, n_after = self._model_and_loss()
        self.assertGreater(n_after, n_before, "causal layer was not built by the prebuild forward")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        missing = [n for n, p in model.named_parameters() if id(p) not in opt_ids]
        self.assertEqual(missing, [], f"parameters outside the optimizer: {missing}")        # (a)

        x, lab, ttb = self._batch(True)
        self.assertGreater(int(lab.sum()), 0)
        model.train()
        out = model(x, ei, cap)
        loss = loss_fn.ttb_loss(out["ttb"], ttb, lab)
        self.assertTrue(loss.requires_grad)
        opt.zero_grad(); loss.backward()
        rep = head_ttb_grad_report(model)
        self.assertEqual(len(rep), 4, rep)                       # 2 Linear layers x (weight, bias)
        self.assertTrue(all(v > 0 for v in rep.values()), f"zero grad in head_ttb: {rep}")   # (b)
        named = dict(model.named_parameters())
        self.assertGreater(float(named["heads.pool.weight"].grad.abs().sum()), 0.0)
        backbone = [n for n, p in model.named_parameters()
                    if not n.startswith("heads.") and p.grad is not None and float(p.grad.abs().sum()) > 0]
        self.assertTrue(len(backbone) > 0, "TTB loss does not reach the backbone")
        for n, p in model.named_parameters():                    # isolation: other heads get exactly 0
            if n.startswith(("heads.head_bn.", "heads.head_pattern.")):
                self.assertTrue(p.grad is None or float(p.grad.abs().sum()) == 0.0, n)

        before = {n: p.detach().clone() for n, p in model.named_parameters() if ".head_ttb." in n}
        opt.step()                                               # (c)
        for n, p in model.named_parameters():
            if ".head_ttb." in n:
                self.assertFalse(torch.equal(before[n], p.detach()), f"{n} unchanged after step")

    def test_A_control_no_positive_window_gives_no_gradient(self):
        model, loss_fn, ei, cap, *_ = self._model_and_loss()
        x, lab, ttb = self._batch(False)
        self.assertEqual(int(lab.sum()), 0)
        model.train()
        loss = loss_fn.ttb_loss(model(x, ei, cap)["ttb"], ttb, lab)
        self.assertFalse(loss.requires_grad)                      # constant 0.0: nothing to backprop
        self.assertTrue(all(v == 0.0 for v in head_ttb_grad_report(model).values()))   # detector can say "no grad"

    # ------------------------------------------------------------------ B + C
    def test_B_C_real_pipeline_and_exporter(self):
        P = self.pipeline
        captured = {"nonzero": set()}
        orig_adamw, orig_model = torch.optim.AdamW, P.CSTGNN

        class SpyAdamW(orig_adamw):
            def step(self, *a, **k):
                if "pre" not in captured:
                    captured["pre"] = {id(p): p.detach().clone() for g in self.param_groups for p in g["params"]}
                for g in self.param_groups:
                    for p in g["params"]:
                        if p.grad is not None and float(p.grad.abs().sum()) > 0:
                            captured["nonzero"].add(id(p))
                return super().step(*a, **k)

        class SpyModel(orig_model):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                captured["model"] = self

        torch.optim.AdamW, P.CSTGNN = SpyAdamW, SpyModel
        run_dir = self.tmp / "run"
        run_dir.mkdir(exist_ok=True)
        try:
            torch.manual_seed(42); np.random.seed(42)
            result = P.train_and_evaluate(RUN_CFG, seed=42, run_dir=run_dir)
        finally:
            torch.optim.AdamW, P.CSTGNN = orig_adamw, orig_model

        model = captured["model"]
        opt_ids = set(captured["pre"])
        for n, p in model.named_parameters():
            self.assertIn(id(p), opt_ids, f"{n} not in optimizer at first step")           # (a) real run
        ttb_params = [(n, p) for n, p in model.named_parameters() if ".head_ttb." in n]
        self.assertEqual(len(ttb_params), 4)
        for n, p in ttb_params:
            self.assertIn(id(p), captured["nonzero"], f"{n} never received a non-zero grad")   # (b)
            self.assertFalse(torch.equal(captured["pre"][id(p)], p.detach()), f"{n} unchanged by training")  # (c)

        # ---- exporter (C)
        from src.ttb_export import export_val_predictions
        ex = export_val_predictions(run_dir, RUN_CFG, device=torch.device("cpu"))
        n_val = len(self.val)
        for k in ("prob", "ttb_pred", "label", "ttb_true", "pattern", "file_id"):
            self.assertEqual(len(ex[k]), n_val, k)
        self.assertTrue(np.isfinite(ex["ttb_pred"]).all())
        self.assertTrue(((ex["prob"] >= 0) & (ex["prob"] <= 1)).all())
        self.assertTrue(np.array_equal(ex["label"], np.array([int(s["label"]) for s in self.val])))   # cache order kept
        self.assertTrue(np.allclose(ex["ttb_true"], np.array([float(s["ttb"]) for s in self.val])))
        self.assertEqual(len(np.unique(ex["file_id"])), 2)
        self.assertAlmostEqual(ex["val_auc"], result["auc"], places=4)


if __name__ == "__main__":
    unittest.main()
