# tests/test_gnn_baseline_gradient_flow.py
"""
Pre-flight checks for the "GNN baseline" (use_toc=False) required by
PROTOCOL.md rule 1, before any real GPU training for the reviewer-required
empirical baseline comparison (Table: GAT+TFT no-TOC vs Run 4).

use_toc=False is a DIFFERENT ablation from the already-closed F6-ablation
study (ablate_f6=True): it architecturally removes the F6 injection (no
toc_scale parameter at all) and the RCS multiplier (rcs = out_degree, not
out_degree * toc_capacity), and uses _TOCWeightedLossBase (no L_rcs term in
the graph at all, not merely zero-weighted). L_sub is zeroed via
lambda_sub=0.0 by configuration, matching the existing Run1-4 convention of
zeroing loss coefficients rather than branching the loss class. The causal
layer (L_cause) is deliberately untouched -- it is not a TOC component.

Four things are checked, all on tiny synthetic data:

1. check_baseline_params_receive_gradient(): with use_toc=False, verifies
   (a) no toc_scale parameter exists in the model at all (the ablation is
   architectural, not a dead/inert parameter left in the graph -- unlike
   ablate_f6's toc_scale, which stays in the graph and is provably inert;
   see test_f6_ablation_gradient_flow.py), (b) every parameter that DOES
   exist is in the optimizer, receives a non-zero gradient, and moves after
   one optimizer.step(), and (c) rcs is NOT identically zero (unlike
   ablate_f6) -- it is the raw causal out-degree, a real signal.

2. check_param_count_delta(): confirms the baseline has exactly
   gat_layers fewer parameters than the use_toc=True control with an
   otherwise identical config (one toc_scale removed per TOCGATLayer) --
   computed programmatically, not hardcoded, per protocol rule 7 ("no
   correction applied by memory or assumption -- verify against source").

3. Real pipeline.train_and_evaluate(use_toc=False) on a tiny fake compose
   cache: confirms end-to-end integration and that use_toc=False cannot be
   combined with ablate_f6=True (raises ValueError).

4. test_rcs_top1_metric_correctness(): a hand-crafted, from-scratch check
   of the NEW rcs_top1 metric added to TOCEvaluator (2026-09-27) for the
   reviewer's required "same metrics for every method" comparison. This
   metric never existed before for Compose (N=30) -- it must be verified
   against known-by-construction inputs, not trusted on first integration,
   per protocol rule 4 (any new/good-looking metric must be checked before
   being reported).
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))


def check_baseline_params_receive_gradient():
    from src.core import Config, TOCPriorLoader, CSTGNN, _TOCWeightedLossBase

    cfg = Config()
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64
    cfg.model.use_toc = False
    cfg.training.lambda_sub = 0.0

    toc = TOCPriorLoader(np.linspace(0.1, 1.0, 30).astype(np.float32),
                         np.linspace(0.1, 1.0, 7).astype(np.float32))
    model = CSTGNN(cfg)

    edge_index = torch.randint(0, 30, (2, 60))
    toc_cap = torch.linspace(0.1, 1.0, 30)

    x = torch.randn(4, cfg.data.window_steps, 30, cfg.data.n_features)
    label = torch.randint(0, 2, (4,))
    pattern_idx = torch.randint(0, cfg.model.n_patterns, (4,))
    ttb = torch.rand(4)

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, cfg.data.window_steps, 30, cfg.data.n_features),
              edge_index, toc_cap)
    model.train()

    model_params = list(model.named_parameters())

    toc_scale_names = [name for name, _ in model_params if name.endswith(".toc_scale")]
    assert not toc_scale_names, (
        f"use_toc=False must not create toc_scale parameters, found: {toc_scale_names}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [name for name, p in model_params if id(p) not in opt_param_ids]
    assert not missing, f"Parameters missing from optimizer: {missing}"

    pattern_class_weights = torch.ones(cfg.model.n_patterns)
    loss_fn = _TOCWeightedLossBase(cfg, toc, pattern_class_weights)

    before = {name: p.detach().clone() for name, p in model_params}

    out = model(x, edge_index, toc_cap)

    assert out["rcs"].abs().max().item() > 0.0, (
        "rcs is identically zero under use_toc=False -- expected the raw "
        "causal out-degree (no TOC multiplier), not a zeroed signal."
    )

    targets = {"label": label, "pattern_idx": pattern_idx, "ttb": ttb}
    losses = loss_fn(out, targets)
    assert "rcs_supervision" not in losses, (
        "_TOCWeightedLossBase must not produce an rcs_supervision term."
    )

    optimizer.zero_grad()
    losses["total"].backward()

    no_grad, zero_grad = [], []
    for name, p in model_params:
        if p.grad is None:
            no_grad.append(name)
        elif p.grad.abs().max().item() == 0.0:
            zero_grad.append(name)
    assert not no_grad, f"Parameters with no gradient at all: {no_grad}"
    assert not zero_grad, f"Parameters with unexpected all-zero gradient: {zero_grad}"

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    unmoved = [name for name, p in model_params
              if torch.equal(p.detach(), before[name])]
    assert not unmoved, f"Parameters that did not change after optimizer.step(): {unmoved}"

    print(f"OK: all {len(model_params)} parameters (no toc_scale present) are in "
          "the optimizer, received non-zero gradient, and moved after one step, "
          "under use_toc=False. rcs is a real (non-zero-forced) signal.")


def check_param_count_delta():
    from src.core import Config, CSTGNN

    cfg_ctl = Config()
    cfg_ctl.model.gat_hidden, cfg_ctl.model.tft_hidden = 32, 64
    cfg_ctl.model.use_toc = True

    cfg_base = Config()
    cfg_base.model.gat_hidden, cfg_base.model.tft_hidden = 32, 64
    cfg_base.model.use_toc = False

    n_ctl = sum(p.numel() for p in CSTGNN(cfg_ctl).parameters())
    n_base = sum(p.numel() for p in CSTGNN(cfg_base).parameters())

    expected_removed = cfg_ctl.model.gat_layers
    assert n_ctl - n_base == expected_removed, (
        f"Expected exactly {expected_removed} fewer parameters, "
        f"got n_ctl={n_ctl}, n_base={n_base}, delta={n_ctl - n_base}."
    )
    print(f"OK: baseline has {n_ctl - n_base} fewer parameters than the "
          f"use_toc=True control ({n_base} vs {n_ctl}).")
    return n_ctl, n_base


def build_fake_compose_cache(cache_dir: Path, model_dir: Path):
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

    edge_index_30 = torch.randint(0, 30, (2, 40))
    torch.save({30: {"edge_index": edge_index_30,
                     "toc_cap": torch.linspace(0.1, 1.0, 30)}},
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
           lambda_causal: 0.05, lambda_sub: 0.0, fn_weight: 1.5,
           fp_weight: 1.0, constraint_mult: 3.0}
device: cpu
""")


def test_baseline_pipeline_integration():
    tmp = Path("./_gnn_baseline_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model_dir = tmp / "cst_gnn"
    cache_dir = model_dir / "dataset_cache"
    build_fake_compose_cache(cache_dir, model_dir)

    os.environ["CRUX_MODEL_DIR"] = str(model_dir.resolve())
    os.environ["CRUX_PREBUILD_CAUSAL"] = "1"

    import importlib
    import src.pipeline as pipeline
    importlib.reload(pipeline)

    n_ctl, n_base = check_param_count_delta()

    run_cfg = dict(fn_weight=1.5, fp_weight=1.0, lambda_causal=0.05, lambda_sub=0.0,
                  lambda_rcs_sup=0.0, weight_decay=1e-3, lr=1e-3, patience=2, epochs=2,
                  gat_hidden=32, tft_hidden=64, gat_dropout=0.2, tft_dropout=0.2,
                  expected_params=n_base)

    run_dir = tmp / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.train_and_evaluate(run_cfg, seed=42, run_dir=run_dir, use_toc=False)

    print("baseline result:", result)
    assert result["use_toc"] is False
    assert result["causal_w_absmax"] > 0.0

    raised = False
    try:
        pipeline.train_and_evaluate(run_cfg, seed=42, run_dir=run_dir,
                                    use_toc=False, ablate_f6=True)
    except ValueError:
        raised = True
    assert raised, "ablate_f6=True + use_toc=False must raise ValueError, did not."

    shutil.rmtree(tmp)
    print("OK: pipeline integration passed; ablate_f6+use_toc=False guard works.")


def test_rcs_top1_metric_correctness():
    from src.core import TOCPriorLoader, TOCEvaluator

    toc = TOCPriorLoader(np.ones(30, dtype=np.float32), np.ones(7, dtype=np.float32))
    ev = TOCEvaluator(toc, n_nodes=30)

    g_idx = toc.PATTERN_TO_IDX['G']
    none_idx = toc.PATTERN_TO_IDX['none']

    def rcs_with_argmax(node, n=30, base=0.01):
        r = np.full(n, base)
        r[node] = 1.0
        return r.tolist()

    outputs = [
        dict(bn_logit=torch.tensor([[5.0]]), pattern_logit=torch.zeros(1, 8),
            ttb=torch.tensor([[0.1]]), rcs=torch.tensor([rcs_with_argmax(4)])),
        dict(bn_logit=torch.tensor([[5.0]]), pattern_logit=torch.zeros(1, 8),
            ttb=torch.tensor([[0.1]]), rcs=torch.tensor([rcs_with_argmax(0)])),
        dict(bn_logit=torch.tensor([[5.0]]), pattern_logit=torch.zeros(1, 8),
            ttb=torch.tensor([[0.1]]), rcs=torch.tensor([rcs_with_argmax(1)])),
        dict(bn_logit=torch.tensor([[-5.0]]), pattern_logit=torch.zeros(1, 8),
            ttb=torch.tensor([[0.1]]), rcs=torch.tensor([rcs_with_argmax(1)])),
    ]
    targets = [
        dict(label=torch.tensor([1]), pattern_idx=torch.tensor([g_idx]), ttb=torch.tensor([0.1])),
        dict(label=torch.tensor([1]), pattern_idx=torch.tensor([g_idx]), ttb=torch.tensor([0.1])),
        dict(label=torch.tensor([1]), pattern_idx=torch.tensor([none_idx]), ttb=torch.tensor([0.1])),
        dict(label=torch.tensor([0]), pattern_idx=torch.tensor([g_idx]), ttb=torch.tensor([0.1])),
    ]
    for out, tg in zip(outputs, targets):
        ev.update(out, tg)

    res = ev.compute()
    assert res['rcs_top1_n'] == 2, f"Expected 2 eligible samples, got {res['rcs_top1_n']}"
    assert abs(res['rcs_top1'] - 50.0) < 1e-6, f"Expected 50.0, got {res['rcs_top1']}"
    print(f"OK: rcs_top1 = {res['rcs_top1']}% on {res['rcs_top1_n']} eligible samples.")


if __name__ == "__main__":
    check_baseline_params_receive_gradient()
    test_baseline_pipeline_integration()
    test_rcs_top1_metric_correctness()
