import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.core import (Config, TOCPriorLoader, CSTGNN, TOCWeightedLoss,
                      TOCEvaluator, make_loader)

MODEL_DIR = Path(os.environ.get("CRUX_MODEL_DIR",
                                "/content/drive/MyDrive/bottleneck_project/cst_gnn"))
CACHE_DIR = MODEL_DIR / "dataset_cache"
# "0" reproduces the notebook exactly (causal layer created lazily AFTER the optimizer).
# "1" builds the causal layer BEFORE the optimizer so its parameters are trained.
PREBUILD_CAUSAL = os.environ.get("CRUX_PREBUILD_CAUSAL", "0") == "1"


def _build_cfg(r: dict) -> Config:
    cfg = Config.load(str(MODEL_DIR / "config.yaml"))
    t, m = cfg.training, cfg.model
    t.fn_weight, t.fp_weight = r["fn_weight"], r["fp_weight"]
    t.lambda_causal, t.lambda_sub = r["lambda_causal"], r["lambda_sub"]
    t.lambda_rcs_sup = r["lambda_rcs_sup"]
    t.weight_decay, t.lr = r["weight_decay"], r["lr"]
    t.patience, t.epochs = r["patience"], r["epochs"]
    m.gat_hidden, m.tft_hidden = r["gat_hidden"], r["tft_hidden"]
    m.gat_dropout, m.tft_dropout = r["gat_dropout"], r["tft_dropout"]
    return cfg


def train_and_evaluate(run_cfg: dict, seed: int, run_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _build_cfg(run_cfg)

    toc = TOCPriorLoader(np.load(MODEL_DIR / "capacity_compose.npy"),
                         np.load(MODEL_DIR / "capacity_home.npy"))
    graphs = torch.load(CACHE_DIR / "graphs.pt", weights_only=False)
    edge_index_30 = graphs[30]["edge_index"].to(device)
    toc_cap_30 = graphs[30]["toc_cap"].to(device)

    train_samples = torch.load(CACHE_DIR / "train.pt", weights_only=False)
    cnt = Counter(int(s["pattern_idx"]) for s in train_samples)
    counts = np.array([cnt.get(i, 0) for i in range(cfg.model.n_patterns)], dtype=np.float32)
    w = 1.0 / (counts + 1.0)
    w = w / w.sum() * cfg.model.n_patterns
    pattern_class_weights = torch.tensor(w, dtype=torch.float32)

    model = CSTGNN(cfg).to(device)
    n_shared = sum(p.numel() for p in model.parameters())
    assert n_shared == run_cfg["expected_params"], (n_shared, run_cfg["expected_params"])

    if PREBUILD_CAUSAL:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, cfg.data.window_steps, 30, cfg.data.n_features, device=device),
                  edge_index_30, toc_cap_30)

    loss_fn = TOCWeightedLoss(cfg, toc, pattern_class_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    g = torch.Generator().manual_seed(seed)
    train_loader = make_loader("train", cfg.data.batch_size, True, CACHE_DIR, generator=g)
    val_loader = make_loader("val", cfg.data.batch_size, False, CACHE_DIR)

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        evaluator = TOCEvaluator(toc, n_nodes=30)
        with (torch.enable_grad() if train else torch.no_grad()):
            for x, label, pattern_idx, ttb in loader:
                x, label = x.to(device), label.to(device)
                pattern_idx, ttb = pattern_idx.to(device), ttb.to(device)
                out = model(x, edge_index_30, toc_cap_30)
                tg = {"label": label, "pattern_idx": pattern_idx, "ttb": ttb}
                losses = loss_fn(out, tg)
                if train:
                    optimizer.zero_grad()
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                evaluator.update(out, tg)
        return evaluator.compute()

    best_auc, best_epoch, best_metrics, bad = -1.0, 0, None, 0
    for epoch in range(1, cfg.training.epochs + 1):
        run_epoch(train_loader, train=True)
        val = run_epoch(val_loader, train=False)
        scheduler.step()
        if val["auc"] > best_auc:
            best_auc, best_epoch, best_metrics, bad = val["auc"], epoch, val, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val}, run_dir / "best_model.pt")
        else:
            bad += 1
            if bad >= cfg.training.patience:
                break

    assert best_metrics is not None and best_metrics["cp_recall"] is not None
    causal = model.get_submodule("causal_30")
    return dict(
        auc=best_metrics["auc"],
        cp_recall=best_metrics["cp_recall"],
        pat_acc=best_metrics["pattern_accuracy"],
        n_params=sum(p.numel() for p in model.parameters()),
        best_epoch=best_epoch,
        causal_w_absmax=float(causal.W_raw.abs().max().item()),
    )