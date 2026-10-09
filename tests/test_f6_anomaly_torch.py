# tests/test_f6_anomaly_torch.py
"""
Part B of the F6 anomaly verification: torch-dependent checks (see the list B1-B6
in tests/test_f6_anomaly.py's docstring). Skipped automatically when torch or
torch_geometric cannot be imported (e.g. a sandbox without them); run in Colab.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    import torch
    import torch_geometric  # noqa: F401
except Exception as e:  # ImportError or a broken install (OSError)
    pytest.skip(f"torch/torch_geometric unavailable: {e}", allow_module_level=True)

from src import f6_anomaly as fa  # noqa: E402

def _tiny_model(seed=0):
    from src.core import Config, CSTGNN
    torch.manual_seed(seed)
    cfg = Config()
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64
    return cfg, CSTGNN(cfg)


def _edges(n, e=14, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, n, (2, e), generator=g)


def _x(cfg, n, b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, cfg.data.window_steps, n, cfg.data.n_features, generator=g)


def test_B1_extraction_matches_encoder():
    cfg, model = _tiny_model(); model.eval()
    N = 7; ei = _edges(N); x = _x(cfg, N)
    for cap in (torch.zeros(N), torch.rand(N)):
        with torch.no_grad():
            ref = model.spatial(x, ei, cap)
        _, h = fa.extract_attention(model, x, ei, cap)
        assert torch.allclose(ref, h, atol=1e-5), (ref - h).abs().max()


def test_B2_rows_sum_to_one():
    cfg, model = _tiny_model(); model.eval()
    N = 7; ei = _edges(N); x = _x(cfg, N, b=4)
    mats = fa.attention_matrices(model, x, ei, torch.rand(N), N, chunk=3)
    assert len(mats) == cfg.model.gat_layers
    for A in mats:
        assert A.shape == (4, N, N) and np.allclose(A.sum(-1), 1.0, atol=1e-5)


def test_B3_no_side_effects():
    cfg, model = _tiny_model(); model.eval()
    N = 7; ei = _edges(N); x = _x(cfg, N)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    fa.attention_matrices(model, x, ei, torch.rand(N), N)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in model.parameters())


def test_B4_ablation_sensitivity():
    cfg, model = _tiny_model(); model.eval()
    N = 7; ei = _edges(N); x = _x(cfg, N)
    zero = torch.zeros(N); cap = torch.rand(N) + 0.2
    base = fa.attention_matrices(model, x, ei, zero, N)
    for layer in model.spatial.layers:
        layer.toc_scale.data.fill_(5.0)                      # irrelevant when cap == 0
    scaled0 = fa.attention_matrices(model, x, ei, zero, N)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(base, scaled0))
    withcap = fa.attention_matrices(model, x, ei, cap, N)
    assert any(not np.allclose(a, b, atol=1e-4) for a, b in zip(base, withcap))


def test_B5_checked_loading(tmp_path):
    cfg, model = _tiny_model()
    N = 30; ei = _edges(N, 60)
    model.eval()
    with torch.no_grad():
        model(_x(cfg, N, b=1), ei, torch.rand(N))            # builds causal_30
    good = tmp_path / "good.pt"
    torch.save({"model_state": model.state_dict()}, good)
    m = fa.load_model_checked(good, cfg, "cpu")              # causal_30.* extras tolerated
    assert m is not None
    sd = model.state_dict()
    drop = next(k for k in sd if k.startswith("spatial.layers.0.proj"))
    bad = tmp_path / "bad.pt"
    torch.save({"model_state": {k: v for k, v in sd.items() if k != drop}}, bad)
    with pytest.raises(RuntimeError):
        fa.load_model_checked(bad, cfg, "cpu")


def test_B6_rebuild_causal_seed_dependence():
    cfg, model = _tiny_model(); model.eval()
    N = 7; ei = _edges(N); cap = torch.rand(N)
    x = _x(cfg, N, b=8)

    def probs(init_seed):
        fa.rebuild_causal(model, N, init_seed, cfg.data.window_steps,
                          cfg.data.n_features, ei, cap, "cpu")
        with torch.no_grad():
            return torch.sigmoid(model(x, ei, cap)["bn_logit"]).squeeze(-1).numpy()

    p1, p1b, p2 = probs(1), probs(1), probs(2)
    assert np.allclose(p1, p1b, atol=1e-7)
    assert not np.allclose(p1, p2, atol=1e-6)
