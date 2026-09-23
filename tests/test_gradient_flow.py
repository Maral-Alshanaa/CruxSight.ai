# tests/test_gradient_flow.py
"""
Runs the REAL pipeline.train_and_evaluate for a couple of epochs on tiny
synthetic data and checks that causal_30's parameters actually receive
gradients and get updated by the optimizer. This exercises the exact
build order used in real training runs (src/pipeline.py), unlike a
hand-built scenario.

Needs a tiny fake dataset_cache — build it once with build_fake_cache().
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))


def build_fake_cache(cache_dir: Path, model_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    def make_samples(n):
        return [{
            "x": torch.randn(12, 30, 7),
            "label": torch.tensor(int(i % 2)),
            "pattern_idx": torch.tensor(int(i % 8)),
            "ttb": torch.tensor(0.5),
        } for i in range(n)]

    torch.save(make_samples(16), cache_dir / "train.pt")
    torch.save(make_samples(8), cache_dir / "val.pt")

    edge_index = torch.randint(0, 30, (2, 40))
    torch.save({30: {"edge_index": edge_index, "toc_cap": torch.ones(30)}},
               cache_dir / "graphs.pt")

    np.save(model_dir / "capacity_compose.npy", np.ones(30, dtype=np.float32))
    np.save(model_dir / "capacity_home.npy", np.ones(7, dtype=np.float32))

    (model_dir / "config.yaml").write_text("""
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
""")


def test_real_pipeline_trains_causal_layer():
    tmp = Path("./_pf_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model_dir = tmp / "cst_gnn"
    cache_dir = model_dir / "dataset_cache"
    build_fake_cache(cache_dir, model_dir)

    os.environ["CRUX_MODEL_DIR"] = str(model_dir.resolve())
    os.environ["CRUX_PREBUILD_CAUSAL"] = os.environ.get("CRUX_PREBUILD_CAUSAL", "0")

    # Re-import AFTER setting env vars, since pipeline.py reads them at import time.
    import importlib
    import src.pipeline as pipeline
    importlib.reload(pipeline)

    run_cfg = dict(fn_weight=1.5, fp_weight=1.0, lambda_causal=0.05, lambda_sub=0.05,
                   lambda_rcs_sup=0.3, weight_decay=1e-3, lr=1e-3, patience=2, epochs=2,
                   gat_hidden=32, tft_hidden=64, gat_dropout=0.2, tft_dropout=0.2,
                   expected_params=209587)

    run_dir = tmp / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.train_and_evaluate(run_cfg, seed=42, run_dir=run_dir)

    print("PREBUILD_CAUSAL =", os.environ["CRUX_PREBUILD_CAUSAL"])
    print("causal_w_absmax =", result["causal_w_absmax"])

    shutil.rmtree(tmp)

    assert result["causal_w_absmax"] > 0.0, (
        "causal_30.W_raw did not move from its zero initialisation during real "
        "training. This is the exact silent bug found earlier: the lazy causal "
        "layer is built after the optimizer, so it never receives updates."
    )
    print("OK: causal_30.W_raw received real gradient updates.")


if __name__ == "__main__":
    test_real_pipeline_trains_causal_layer()