# tests/test_home_finetune_gradient_flow.py
"""
Runs the REAL src.pipeline.finetune_home on tiny synthetic data, with a
synthetic compose checkpoint as the starting point. Control experiment:
PREBUILD_CAUSAL=0 must show causal_w_absmax staying at (near-)zero;
PREBUILD_CAUSAL=1 must show it moving.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))


def build_fake_env(model_dir: Path):
    cache_dir = model_dir / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def home_samples(n):
        return [{"x": torch.randn(12, 7, 7), "label": torch.tensor(int(i % 2)),
                 "pattern_idx": torch.tensor(6), "ttb": torch.tensor(0.5)} for i in range(n)]

    torch.save(home_samples(40), cache_dir / "test.pt")
    edge_index_30 = torch.randint(0, 30, (2, 40))
    edge_index_7 = torch.randint(0, 7, (2, 10))
    torch.save({30: {"edge_index": edge_index_30, "toc_cap": torch.ones(30)},
                7: {"edge_index": edge_index_7, "toc_cap": torch.ones(7)}},
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


def make_fake_compose_ckpt(model_dir: Path) -> Path:
    from src.core import Config, CSTGNN
    cfg = Config.load(str(model_dir / "config.yaml"))
    model = CSTGNN(cfg)
    x30 = torch.randn(1, 12, 30, 7)
    ei30 = torch.randint(0, 30, (2, 40))
    model(x30, ei30, torch.ones(30))  # builds causal_30
    ckpt_path = model_dir / "fake_compose.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    return ckpt_path


def run_case(prebuild: str):
    tmp = Path("./_pf_home_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model_dir = tmp / "cst_gnn"
    build_fake_env(model_dir)
    ckpt = make_fake_compose_ckpt(model_dir)

    os.environ["CRUX_MODEL_DIR"] = str(model_dir.resolve())
    os.environ["CRUX_PREBUILD_CAUSAL"] = prebuild

    import importlib
    import src.pipeline as pipeline
    importlib.reload(pipeline)

    run_dir = tmp / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.finetune_home(ckpt, seed=42, run_dir=run_dir,
                                    epochs_detect=2, epochs_causal=2)
    shutil.rmtree(tmp)
    return result


def test_home_finetune_causal_gradient_flow():
    r0 = run_case("0")
    print("PREBUILD_CAUSAL=0 ->", r0["causal_w_absmax_before"], "->", r0["causal_w_absmax_after"])
    assert r0["causal_w_absmax_after"] == 0.0, "Expected bug to reproduce: causal_7 should stay at 0.0"

    r1 = run_case("1")
    print("PREBUILD_CAUSAL=1 ->", r1["causal_w_absmax_before"], "->", r1["causal_w_absmax_after"])
    assert r1["causal_w_absmax_after"] > 0.0, "Fix did not take effect: causal_7 should have moved"

    print("\nOK: finetune_home reproduces the bug at PREBUILD_CAUSAL=0 and fixes it at =1.")


if __name__ == "__main__":
    test_home_finetune_causal_gradient_flow()