# tests/test_baseline_residual_signal.py
"""
Autograd check for the residual-signal finding (PROTOCOL.md, "Baseline comparison: residual-signal
finding, autograd check, fix"). Needs torch (+ src.core imports); synthetic tensors only, CPU.
Run: python tests/test_baseline_residual_signal.py        (before any GPU time, protocol rule 1)

Claims checked
  1. LEAK (legacy behaviour): with CRITICAL_PATH node_weights (3.0 vs 1.0), the missed-bottleneck
     amplification term m * r_crit gives d(detection_loss)/d(rcs) that differs from the uniform-weight
     value -> the term carries TOC-derived training signal.
  2. FIX: _TOCWeightedLossBase(use_toc=False).node_weights == 1, and its gradient w.r.t. rcs equals the
     manually neutralised computation EXACTLY (max abs diff == 0.0).
  3. Run 4 unchanged: TOCWeightedLoss(use_toc=True) keeps CRITICAL_PATH weights (3.0) and its gradient
     differs from the neutral one.
  4. Gate: if every positive is detected (missed == 0) the weights have no effect (amp == 1) -> the term
     is only active on false negatives.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core import Config, TOCPriorLoader, TOCWeightedLoss, _TOCWeightedLossBase   # noqa: E402


def _setup(seed=0, B=16):
    g = torch.Generator().manual_seed(seed)
    cfg = Config()
    toc = TOCPriorLoader(np.ones(30, dtype=np.float32), np.ones(7, dtype=np.float32))
    pcw = torch.ones(cfg.model.n_patterns)
    rcs0 = torch.rand(B, 30, generator=g) + 0.1
    label = torch.tensor([1, 0] * (B // 2))
    return cfg, toc, pcw, rcs0, label


def _grad(loss_fn, logit, label, rcs0):
    rcs = rcs0.clone().requires_grad_(True)
    loss = loss_fn.detection_loss(logit, label, rcs)
    (g,) = torch.autograd.grad(loss, rcs)
    return float(loss), g


def test_leak_fix_and_gate():
    cfg, toc, pcw, rcs0, label = _setup()
    miss_logit = torch.full((label.shape[0], 1), -3.0)                        # every positive is missed

    neutral = _TOCWeightedLossBase(cfg, toc, pcw, use_toc=False)
    assert torch.all(neutral.node_weights == 1.0)
    legacy = _TOCWeightedLossBase(cfg, toc, pcw, use_toc=False)          # emulate pre-fix behaviour
    legacy.node_weights = torch.ones(30)
    legacy.node_weights[list(toc.CRITICAL_PATH)] = cfg.training.constraint_mult
    toc_loss = TOCWeightedLoss(cfg, toc, pcw, use_toc=True)
    crit = torch.zeros(30, dtype=torch.bool)
    crit[list(toc.CRITICAL_PATH)] = True
    assert torch.all(toc_loss.node_weights[crit] == cfg.training.constraint_mult)   # claim 3 (weights)

    l_n, g_n = _grad(neutral, miss_logit, label, rcs0)
    l_l, g_l = _grad(legacy, miss_logit, label, rcs0)
    l_t, g_t = _grad(toc_loss, miss_logit, label, rcs0)

    leak = (g_l - g_n).abs().max().item()
    assert leak > 1e-6, f"legacy weights should change d loss/d rcs, got {leak}"      # claim 1
    assert (g_t - g_n).abs().max().item() > 1e-6                                       # claim 3 (gradient)
    assert torch.equal(g_t, g_l) and l_t == l_l                                        # Run 4 == legacy weights

    # claim 2: fixed no-TOC == manual neutral computation, bit-exact
    manual = _TOCWeightedLossBase(cfg, toc, pcw, use_toc=False)
    manual.node_weights = torch.ones(30)
    _, g_m = _grad(manual, miss_logit, label, rcs0)
    assert (g_n - g_m).abs().max().item() == 0.0

    # claim 4: gate. all positives detected -> weights irrelevant
    hit_logit = torch.where(label == 1, torch.tensor(5.0), torch.tensor(-5.0)).unsqueeze(-1)
    _, g_hit_n = _grad(neutral, hit_logit, label, rcs0)
    _, g_hit_l = _grad(legacy, hit_logit, label, rcs0)
    assert (g_hit_n - g_hit_l).abs().max().item() == 0.0
    print(f"OK: leak |dgrad|max={leak:.6f}; fixed vs manual-neutral diff=0.0; gate (no misses) diff=0.0")


if __name__ == "__main__":
    test_leak_fix_and_gate()
