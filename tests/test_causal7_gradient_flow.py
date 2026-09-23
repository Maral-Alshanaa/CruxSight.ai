# tests/test_causal7_gradient_flow.py
"""
Checks whether causal_7's parameters (built lazily on first forward over the
7-node Home graph, during fine-tuning) receive gradient updates -- mirroring
the exact sequence in notebook Cell 10: load a compose-trained checkpoint,
build ft_optimizer from model.parameters() BEFORE any Home-graph forward pass,
then fine-tune.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.core import Config, CSTGNN


def make_cfg():
    cfg = Config()
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64
    return cfg


def test_causal7_receives_gradients_in_finetuning_order():
    cfg = make_cfg()

    # Step 1: build+"pretrain" on compose (N=30), as in Cell 7 -- this is what
    # final_model_v1.pt corresponds to. causal_30 exists at this point.
    model = CSTGNN(cfg)
    x30 = torch.randn(2, 12, 30, 7)
    ei30 = torch.randint(0, 30, (2, 40))
    cap30 = torch.ones(30)
    model(x30, ei30, cap30)  # builds causal_30

    # Step 2: exact Cell 10 sequence -- optimizer built from model.parameters()
    # BEFORE the first Home-graph (N=7) forward pass.
    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Step 3: first Home-graph forward -- this is what lazily builds causal_7.
    x7 = torch.randn(4, 12, 7, 7)
    ei7 = torch.randint(0, 7, (2, 10))
    cap7 = torch.ones(7)
    names_before = {n for n, _ in model.named_parameters()}
    out = model(x7, ei7, cap7)  # builds causal_7 here, AFTER ft_optimizer exists
    names_after = {n for n, _ in model.named_parameters()}
    new_params = names_after - names_before
    print("Parameters created by the first Home-graph forward:", sorted(new_params))

    optim_ids = {id(p) for g in ft_optimizer.param_groups for p in g["params"]}
    causal7 = model.get_submodule("causal_7")
    causal7_ids = {id(p) for p in causal7.parameters()}

    missing = causal7_ids - optim_ids
    print("causal_7 params NOT in ft_optimizer:", len(missing), "of", len(causal7_ids))

    # Step 4: run the actual Stage-1 fine-tuning loss (detection-only BCE, as in Cell 10)
    label = torch.tensor([1, 0, 1, 0])
    bn_prob = torch.sigmoid(out["bn_logit"].squeeze(-1))
    loss = F.binary_cross_entropy(bn_prob, label.float())
    ft_optimizer.zero_grad()
    loss.backward()

    w_grad = causal7.W_raw.grad
    print("causal_7.W_raw.grad is None:", w_grad is None)
    if w_grad is not None:
        print("causal_7.W_raw.grad abs max:", w_grad.abs().max().item())

    before = causal7.W_raw.detach().clone()
    ft_optimizer.step()
    after = causal7.W_raw.detach().clone()
    moved = (before - after).abs().max().item()
    print("causal_7.W_raw change after optimizer.step():", moved)

    assert missing == causal7_ids, (
        "Unexpected: some causal_7 params ARE in ft_optimizer. "
        "The bug may not apply here the way it applies to causal_30 -- investigate."
    ) if not missing else None

    if missing:
        assert moved == 0.0, "missing from optimizer but weights moved anyway -- inconsistent, investigate"
        print("\nCONFIRMED: causal_7 has the same bug as causal_30 -- built after "
              "ft_optimizer, so its parameters never receive updates during Home fine-tuning.")
    else:
        print("\nNOT AFFECTED: causal_7 parameters ARE inside ft_optimizer.")


if __name__ == "__main__":
    test_causal7_receives_gradients_in_finetuning_order()