# tests/test_f6_ablation_gradient_flow.py
"""
Pre-flight checks for the F6 (toc_capacity) ablation mechanism, required by
PROTOCOL.md rule 1 before any real GPU training for the F6-ablation
multi-seed study.

Three things are checked, all on tiny synthetic data:

1. check_all_params_receive_gradient(): a from-scratch, per-parameter check
   (not routed through train_and_evaluate) that with toc_cap forced to
   zero -- the exact ablation condition -- every parameter in a freshly
   built CSTGNN (a) is present in the optimizer, (b) receives a non-zero
   gradient after one real backward pass through TOCWeightedLoss, and
   (c) actually changes value after optimizer.step(). This specifically
   guards against the RCS branch being zeroed (rcs = out_degree * 0 == 0)
   silently cutting gradient to the causal encoder/W_raw -- it doesn't,
   because causal_loss (sparsity + dag_penalty) does not depend on
   toc_capacity, but this must be verified, not assumed.

2. Real pipeline.train_and_evaluate(ablate_f6=True) on a tiny fake compose
   cache: confirms end-to-end integration (toc_cap_max == 0.0, causal_30
   still trains under PREBUILD_CAUSAL=1) and, as a control, that the same
   call with ablate_f6=False leaves toc_cap_max > 0 -- i.e. the flag is not
   a silent no-op.

3. Real pipeline.finetune_home(ablate_f6=True) starting from an ablated
   fake compose checkpoint: confirms causal_7 still trains, and that
   rcs_absmax == 0.0 in every stage -- the functional mechanism behind the
   expected RCS Top-1 collapse to 0% -- while the non-ablated control shows
   rcs_absmax > 0.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ----------------------------------------------------------------------
# Part 1: low-level per-parameter gradient-flow check under ablation
# ----------------------------------------------------------------------

def check_all_params_receive_gradient():
    from src.core import Config, TOCPriorLoader, CSTGNN, TOCWeightedLoss

    cfg = Config()
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64
    cfg.training.lambda_rcs_sup = 0.3  # Run 4 config

    toc = TOCPriorLoader(np.ones(30, dtype=np.float32), np.ones(7, dtype=np.float32))
    model = CSTGNN(cfg)

    edge_index = torch.randint(0, 30, (2, 60))
    toc_cap_ablated = torch.zeros(30)  # <-- the ablation condition itself

    x = torch.randn(4, cfg.data.window_steps, 30, cfg.data.n_features)
    label = torch.randint(0, 2, (4,))
    pattern_idx = torch.randint(0, cfg.model.n_patterns, (4,))
    ttb = torch.rand(4)

    # Build causal_30 BEFORE the optimizer (PREBUILD_CAUSAL=1 convention),
    # exactly as the real ablation runs will do.
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, cfg.data.window_steps, 30, cfg.data.n_features),
              edge_index, toc_cap_ablated)
    model.train()

    # weight_decay=0 here deliberately: AdamW's decoupled weight decay would
    # otherwise move toc_scale (see EXPECTED_INERT_UNDER_ABLATION below) via
    # pure decay even with an exact-zero gradient, confounding this isolated
    # check. The real pipeline's Run4 weight_decay=1e-3 will let toc_scale
    # drift slightly that way -- harmless, since toc_scale has zero effect on
    # the ablated model's output regardless of its value -- but that drift is
    # not a gradient-flow signal and is deliberately excluded here.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)

    # (a) every parameter must be in the optimizer
    opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    model_params = list(model.named_parameters())
    missing = [name for name, p in model_params if id(p) not in opt_param_ids]
    assert not missing, f"Parameters missing from optimizer: {missing}"

    pattern_class_weights = torch.ones(cfg.model.n_patterns)
    loss_fn = TOCWeightedLoss(cfg, toc, pattern_class_weights)

    before = {name: p.detach().clone() for name, p in model_params}

    out = model(x, edge_index, toc_cap_ablated)
    # Sanity: the ablation must actually zero the RCS branch.
    assert out["rcs"].abs().max().item() == 0.0, (
        "toc_cap=0 did not zero rcs (out_degree * toc_capacity) -- ablation "
        "mechanism is not doing what it's supposed to."
    )
    targets = {"label": label, "pattern_idx": pattern_idx, "ttb": ttb}
    losses = loss_fn(out, targets)

    optimizer.zero_grad()
    losses["total"].backward()

    # toc_scale (one per TOCGATLayer) only ever appears multiplied by
    # toc_capacity in capacity_weight = 1 + toc_lambda*toc_scale*toc_capacity.
    # When toc_capacity is identically zero (the ablation condition), d(loss)/
    # d(toc_scale) is EXACTLY zero by the chain rule, for any input -- this is
    # not a build-order bug like the causal_30/causal_7 one, it's a provable
    # mathematical consequence of ablating F6, and toc_scale is expected to
    # stay frozen at its initial value for the entire ablation study. It is
    # the only exception allowed here; every other parameter must still
    # satisfy (b) and (c) normally.
    EXPECTED_INERT_UNDER_ABLATION = {
        name for name, _ in model_params if name.endswith(".toc_scale")
    }
    assert EXPECTED_INERT_UNDER_ABLATION, "expected to find toc_scale parameters"

    # (b) every parameter must receive a non-zero gradient, except the
    # provably-inert toc_scale parameters above
    no_grad, unexpected_zero_grad = [], []
    for name, p in model_params:
        if p.grad is None:
            no_grad.append(name)
        elif p.grad.abs().max().item() == 0.0 and name not in EXPECTED_INERT_UNDER_ABLATION:
            unexpected_zero_grad.append(name)
    assert not no_grad, f"Parameters with no gradient at all: {no_grad}"
    assert not unexpected_zero_grad, (
        f"Parameters with an unexpected all-zero gradient under F6 ablation: "
        f"{unexpected_zero_grad}. This would mean the RCS==0 branch is silently "
        "cutting gradient to components that should still train via "
        "causal_loss/detection_loss."
    )
    for name in EXPECTED_INERT_UNDER_ABLATION:
        p = dict(model_params)[name]
        assert p.grad is not None and p.grad.abs().max().item() == 0.0, (
            f"{name}: expected an exact-zero gradient under full F6 ablation, got "
            f"{None if p.grad is None else p.grad.abs().max().item()} -- the "
            "capacity_weight formula may have changed; re-derive this exception "
            "before trusting it."
        )

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # (c) every parameter must actually move, except the provably-inert ones
    unmoved = [name for name, p in model_params
               if torch.equal(p.detach(), before[name])
               and name not in EXPECTED_INERT_UNDER_ABLATION]
    assert not unmoved, f"Parameters that did not change after optimizer.step(): {unmoved}"
    for name in EXPECTED_INERT_UNDER_ABLATION:
        p = dict(model_params)[name]
        assert torch.equal(p.detach(), before[name]), (
            f"{name}: expected to stay frozen under full F6 ablation, but it moved."
        )

    n_checked = len(model_params) - len(EXPECTED_INERT_UNDER_ABLATION)
    print(f"OK: {n_checked} of {len(model_params)} parameters are in the optimizer, "
          "received non-zero gradient, and moved after one step, under F6 ablation "
          f"(toc_cap=0). The remaining {len(EXPECTED_INERT_UNDER_ABLATION)} "
          f"({sorted(EXPECTED_INERT_UNDER_ABLATION)}) are provably inert by "
          "construction under this ablation and correctly stayed frozen.")


# ----------------------------------------------------------------------
# Part 2 & 3: real pipeline, end-to-end, on tiny fake caches
# ----------------------------------------------------------------------

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
    edge_index_7 = torch.randint(0, 7, (2, 10))
    # Real (non-ablated) toc_cap is non-trivial so the "with-F6" control
    # branch is meaningfully different from the ablated one.
    torch.save({30: {"edge_index": edge_index_30,
                     "toc_cap": torch.linspace(0.1, 1.0, 30)},
                7: {"edge_index": edge_index_7,
                    "toc_cap": torch.linspace(0.1, 1.0, 7)}},
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


def build_fake_home_samples(cache_dir: Path):
    def home_samples(n):
        return [{"x": torch.randn(12, 7, 7), "label": torch.tensor(int(i % 2)),
                 "pattern_idx": torch.tensor(6), "ttb": torch.tensor(0.5)} for i in range(n)]
    torch.save(home_samples(40), cache_dir / "test.pt")


def run_compose(ablate_f6: bool):
    tmp = Path(f"./_f6_compose_tmp_{ablate_f6}")
    if tmp.exists():
        shutil.rmtree(tmp)
    model_dir = tmp / "cst_gnn"
    cache_dir = model_dir / "dataset_cache"
    build_fake_compose_cache(cache_dir, model_dir)
    build_fake_home_samples(cache_dir)

    os.environ["CRUX_MODEL_DIR"] = str(model_dir.resolve())
    os.environ["CRUX_PREBUILD_CAUSAL"] = "1"

    import importlib
    import src.pipeline as pipeline
    importlib.reload(pipeline)

    run_cfg = dict(fn_weight=1.5, fp_weight=1.0, lambda_causal=0.05, lambda_sub=0.05,
                   lambda_rcs_sup=0.3, weight_decay=1e-3, lr=1e-3, patience=2, epochs=2,
                   gat_hidden=32, tft_hidden=64, gat_dropout=0.2, tft_dropout=0.2,
                   expected_params=205617)

    run_dir = tmp / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = pipeline.train_and_evaluate(run_cfg, seed=42, run_dir=run_dir, ablate_f6=ablate_f6)
    ckpt_path = run_dir / "best_model.pt"
    return result, ckpt_path, tmp


def test_compose_ablation_vs_control():
    res_abl, ckpt_abl, tmp_abl = run_compose(ablate_f6=True)
    res_ctl, ckpt_ctl, tmp_ctl = run_compose(ablate_f6=False)

    print("ablated:  toc_cap_max =", res_abl["toc_cap_max"],
          " causal_w_absmax =", res_abl["causal_w_absmax"])
    print("control:  toc_cap_max =", res_ctl["toc_cap_max"],
          " causal_w_absmax =", res_ctl["causal_w_absmax"])

    assert res_abl["toc_cap_max"] == 0.0, "ablate_f6=True did not zero toc_cap_30"
    assert res_ctl["toc_cap_max"] > 0.0, "control run unexpectedly has toc_cap_30 == 0"
    assert res_abl["causal_w_absmax"] > 0.0, (
        "causal_30.W_raw did not move under F6 ablation -- gradient to the causal "
        "layer must survive via causal_loss even though the RCS branch is zeroed."
    )

    # Fine-tune Home from each arm's own compose checkpoint, ablation-consistent.
    import src.pipeline as pipeline

    fh_abl = pipeline.finetune_home(ckpt_abl, seed=42, run_dir=tmp_abl / "run",
                                    epochs_detect=2, epochs_causal=2, ablate_f6=True)
    fh_ctl = pipeline.finetune_home(ckpt_ctl, seed=42, run_dir=tmp_ctl / "run",
                                    epochs_detect=2, epochs_causal=2, ablate_f6=False)

    print("home ablated stages:", fh_abl["stages"])
    print("home control stages:", fh_ctl["stages"])

    for stage_name, m in fh_abl["stages"].items():
        assert m["rcs_absmax"] == 0.0, (
            f"Home stage {stage_name}: rcs_absmax != 0 under F6 ablation -- "
            "RCS did not collapse as the ablation mechanism requires."
        )
    assert any(m["rcs_absmax"] > 0.0 for m in fh_ctl["stages"].values()), (
        "Control (non-ablated) Home run unexpectedly has rcs_absmax == 0 in every "
        "stage -- control is not a meaningful contrast to the ablated arm."
    )

    # Unlike train_and_evaluate (whose TOCWeightedLoss trains causal_30 via
    # causal_loss -- sparsity + dag_penalty -- independently of toc_capacity),
    # finetune_home's loss reaches causal_7 ONLY through rcs
    # (rcs_sup_loss_g in the causal stage; the detection stage never touches
    # causal_graph at all). Under full F6 ablation, rcs == out_degree * 0 == 0
    # identically, so causal_7 (W_raw and encoder) gets an exact-zero
    # gradient in every fine-tune stage and must stay frozen at whatever it
    # inherited from the (ablated) compose checkpoint. This is a real,
    # provable consequence of the ablation, not a bug -- and it means
    # RCS Top-1 in the ablated arm is not merely expected to collapse, it is
    # deterministic exactly 0.0% for every seed (rcs is identically zero
    # regardless of training), unlike the with-F6 arm's already-documented
    # high seed-to-seed variance.
    assert fh_abl["causal_w_absmax_after"] == fh_abl["causal_w_absmax_before"], (
        f"Expected causal_7.W_raw to stay exactly frozen during Home fine-tuning "
        f"under F6 ablation (before={fh_abl['causal_w_absmax_before']}, "
        f"after={fh_abl['causal_w_absmax_after']}) -- if this fails, the "
        "no-gradient-via-rcs argument above no longer holds and needs re-checking."
    )
    assert fh_ctl["causal_w_absmax_after"] != fh_ctl["causal_w_absmax_before"], (
        "Control (non-ablated) causal_7.W_raw unexpectedly stayed frozen too -- "
        "control is not a meaningful contrast to the ablated arm."
    )

    shutil.rmtree(tmp_abl)
    shutil.rmtree(tmp_ctl)
    print("\nOK: F6 ablation zeroes toc_cap and collapses rcs to exactly 0 end-to-end. "
          "causal_30 still trains during ablated compose training (via causal_loss, "
          "independent of toc_capacity); causal_7 correctly stays frozen during "
          "ablated Home fine-tuning (its only loss path is via rcs). The "
          "non-ablated control differs on every count above as expected.")


if __name__ == "__main__":
    check_all_params_receive_gradient()
    test_compose_ablation_vs_control()
