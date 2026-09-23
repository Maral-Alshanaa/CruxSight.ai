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

# ----------------------------- Home fine-tuning -----------------------------

def finetune_home(compose_ckpt_path, seed: int, run_dir: Path,
                  epochs_detect: int = 8, epochs_causal: int = 8) -> dict:
    """
    Mirrors notebook Cell 10: load a compose-trained checkpoint, split the
    Home (N=7) samples into a fine-tune set and a held-out test set, then
    fine-tune in two stages (detection-only BCE, then +RCS supervision on
    Pattern G = nodes {3,4}).

    PREBUILD_CAUSAL controls when causal_7 is instantiated relative to
    ft_optimizer, exactly as for causal_30 in train_and_evaluate:
      "0" -> lazy build AFTER ft_optimizer (reproduces the original bug)
      "1" -> forced build BEFORE ft_optimizer (fix: causal_7 actually trains)
    """
    import torch.nn.functional as F
    from torch.utils.data import random_split
    from sklearn.metrics import roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = Config.load(str(MODEL_DIR / "config.yaml"))
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64  # Run 4 architecture

    toc = TOCPriorLoader(np.load(MODEL_DIR / "capacity_compose.npy"),
                         np.load(MODEL_DIR / "capacity_home.npy"))
    graphs = torch.load(CACHE_DIR / "graphs.pt", weights_only=False)
    edge_index_7 = graphs[7]["edge_index"].to(device)
    toc_cap_7 = graphs[7]["toc_cap"].to(device)

    model = CSTGNN(cfg).to(device)
    ckpt = torch.load(compose_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)  # causal_7 missing -> fine

    home_samples = torch.load(CACHE_DIR / "test.pt", weights_only=False)
    n_total = len(home_samples)
    n_ft = int(0.2 * n_total)
    g = torch.Generator().manual_seed(seed)
    ft_set, holdout_set = random_split(home_samples, [n_ft, n_total - n_ft], generator=g)
    ft_loader = torch.utils.data.DataLoader(list(ft_set), batch_size=32, shuffle=True,
                                            generator=torch.Generator().manual_seed(seed))
    holdout_loader = torch.utils.data.DataLoader(list(holdout_set), batch_size=32, shuffle=False)

    if PREBUILD_CAUSAL:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, cfg.data.window_steps, 7, cfg.data.n_features, device=device),
                  edge_index_7, toc_cap_7)  # forces causal_7 to exist before ft_optimizer

    ft_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    def rcs_sup_loss_g(rcs, margin=0.1):
        flagged = [3, 4]
        unflagged = [n for n in range(rcs.shape[-1]) if n not in flagged]
        return F.relu(margin - (rcs[:, flagged].mean(-1) - rcs[:, unflagged].mean(-1))).mean()

    def evaluate():
        model.eval()
        probs, labels, rcs_all = [], [], []
        with torch.no_grad():
            for x, label, pattern_idx, ttb in holdout_loader:
                x = x.to(device)
                out = model(x, edge_index_7, toc_cap_7)
                probs.extend(torch.sigmoid(out["bn_logit"]).squeeze(-1).cpu().tolist())
                labels.extend(label.tolist())
                rcs_all.extend(out["rcs"].cpu().tolist())
        probs, labels, rcs_all = np.array(probs), np.array(labels), np.array(rcs_all)
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        pos = labels == 1
        top1 = (sum(1 for i in np.where(pos)[0] if int(np.argmax(rcs_all[i])) in [3, 4])
                / max(pos.sum(), 1) * 100)
        return float(auc), float(top1)

    stages = {}
    auc0, top1_0 = evaluate()
    stages["zero_shot"] = dict(auc=auc0, rcs_top1=top1_0)

    model.train()
    for _ in range(epochs_detect):
        for x, label, pattern_idx, ttb in ft_loader:
            x = x.to(device)
            out = model(x, edge_index_7, toc_cap_7)
            loss = F.binary_cross_entropy(torch.sigmoid(out["bn_logit"].squeeze(-1)),
                                          label.to(device).float())
            ft_optimizer.zero_grad(); loss.backward(); ft_optimizer.step()
    auc1, top1_1 = evaluate()
    stages["finetune_detection"] = dict(auc=auc1, rcs_top1=top1_1)

    causal7 = model.get_submodule("causal_7")
    w_before = causal7.W_raw.detach().abs().max().item()

    model.train()
    for _ in range(epochs_causal):
        for x, label, pattern_idx, ttb in ft_loader:
            x = x.to(device)
            out = model(x, edge_index_7, toc_cap_7)
            loss = (F.binary_cross_entropy(torch.sigmoid(out["bn_logit"].squeeze(-1)),
                                           label.to(device).float())
                    + 0.3 * rcs_sup_loss_g(out["rcs"]))
            ft_optimizer.zero_grad(); loss.backward(); ft_optimizer.step()
    auc2, top1_2 = evaluate()
    stages["finetune_causal"] = dict(auc=auc2, rcs_top1=top1_2)

    return dict(
        seed=seed, stages=stages,
        causal_w_absmax_before=w_before,
        causal_w_absmax_after=float(causal7.W_raw.detach().abs().max().item()),
    )