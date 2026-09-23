# tests/test_gradient_flow.py
"""
Run this BEFORE any real training (new code, new architecture change, or
after any refactor). It costs seconds and catches whole classes of silent
bugs: parameters that never get gradients, parameters missing from the
optimizer, shape mismatches, and NaN propagation.
"""
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.core import Config, TOCPriorLoader, CSTGNN, TOCWeightedLoss
import numpy as np


def make_dummy_cfg():
    cfg = Config()
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64
    cfg.training.lambda_rcs_sup = 0.3
    return cfg


def test_all_params_get_gradients():
    """Every parameter that exists in model.parameters() at optimizer
    creation time must receive a non-None, non-zero gradient after one
    backward pass. This is the check that would have caught W_raw."""
    cfg = make_dummy_cfg()
    toc = TOCPriorLoader(np.ones(30), np.ones(7))
    model = CSTGNN(cfg)

    # Force lazy submodules (causal_30) to be built BEFORE grabbing param names,
    # exactly as the real pipeline should do if PREBUILD_CAUSAL=1.
    x = torch.randn(2, 12, 30, 7)
    edge_index = torch.randint(0, 30, (2, 40))
    cap = torch.ones(30)
    model(x, edge_index, cap)   # build lazy layers

    names_before = {n for n, _ in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optim_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    model_param_ids = {id(p) for _, p in model.named_parameters()}

    # 1. Every model parameter must be inside the optimizer.
    missing_from_optim = model_param_ids - optim_param_ids
    assert not missing_from_optim, (
        f"{len(missing_from_optim)} parameters exist in the model but are "
        "NOT in the optimizer (built after optimizer creation -> never trained)"
    )

    # 2. One forward+backward, then check every parameter got a gradient.
    weights = torch.ones(30)
    loss_fn = TOCWeightedLoss(cfg, toc, weights.repeat(8)[:8] / 8)
    out = model(x, edge_index, cap)
    targets = {
        "label": torch.randint(0, 2, (2,)),
        "pattern_idx": torch.randint(0, 8, (2,)),
        "ttb": torch.rand(2),
    }
    loss = loss_fn(out, targets)["total"]
    loss.backward()

    dead = []
    for n, p in model.named_parameters():
        if p.grad is None:
            dead.append((n, "grad is None"))
        elif torch.all(p.grad == 0):
            dead.append((n, "grad is all zero"))
    assert not dead, f"Parameters with no effective gradient: {dead}"

    # 3. No new parameters silently appeared after backward (e.g. another lazy layer).
    names_after = {n for n, _ in model.named_parameters()}
    assert names_before == names_after, f"New params appeared after forward: {names_after - names_before}"

    print("OK: every parameter is in the optimizer and received a gradient.")


def test_overfit_one_batch():
    """A model that cannot drive loss down on a single repeated batch
    within ~50 steps almost certainly has a wiring bug (dead layer,
    frozen weights, wrong loss target, etc.), regardless of how good
    the real dataset is."""
    cfg = make_dummy_cfg()
    toc = TOCPriorLoader(np.ones(30), np.ones(7))
    model = CSTGNN(cfg)
    x = torch.randn(4, 12, 30, 7)
    edge_index = torch.randint(0, 30, (2, 40))
    cap = torch.ones(30)
    model(x, edge_index, cap)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = TOCWeightedLoss(cfg, toc, torch.ones(8) / 8)
    targets = {
        "label": torch.tensor([1, 0, 1, 0]),
        "pattern_idx": torch.tensor([0, 1, 0, 1]),
        "ttb": torch.tensor([0.5, 0.5, 0.5, 0.5]),
    }

    losses = []
    for _ in range(50):
        out = model(x, edge_index, cap)
        loss = loss_fn(out, targets)["total"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"Loss did not drop on a single overfit batch: {losses[0]:.4f} -> {losses[-1]:.4f}. "
        "Likely a wiring bug (dead parameters, wrong target, frozen layer)."
    )
    print(f"OK: loss dropped {losses[0]:.4f} -> {losses[-1]:.4f} on overfit test.")


if __name__ == "__main__":
    test_all_params_get_gradients()
    test_overfit_one_batch()
    print("\nAll pre-flight checks passed.")