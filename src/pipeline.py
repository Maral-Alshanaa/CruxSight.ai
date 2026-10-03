import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.core import (Config, TOCPriorLoader, CSTGNN, TOCWeightedLoss,
                      _TOCWeightedLossBase, TOCEvaluator, make_loader,
                      CachedWindowDataset)

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


def train_and_evaluate(run_cfg: dict, seed: int, run_dir: Path,
                       ablate_f6: bool = False, use_toc: bool = True,
                       log_causal: bool = False) -> dict:
    """
    ablate_f6=True reproduces the F6 (toc_capacity) ablation: toc_cap_30 is
    replaced with a constant zero tensor before being threaded through the
    model. This neutralises the capacity_weight injection in every
    TOCGATLayer (1 + toc_lambda*toc_scale*0 == 1.0, i.e. F6's contribution
    there becomes the constant 1.0) and zeroes the RCS multiplier
    (rcs = out_degree * toc_capacity == 0 for every sample/node), which is
    the mechanism behind the RCS Top-1 collapse. Nothing else (x_seq,
    edge_index, labels, hyperparameters) changes relative to a normal run
    with the same run_cfg/seed, so the with-F6 and F6-ablated arms are
    otherwise identical.

    use_toc=False is a DIFFERENT, architectural ablation for the GNN
    baseline required by the reviewer report: no F6 injection in TOCGATLayer
    (no toc_scale parameter at all, not merely zeroed), no RCS multiplier
    (rcs = out_degree, not out_degree * toc_capacity), no L_sub, no L_rcs
    (loss_fn falls back to _TOCWeightedLossBase, which never computes an
    RCS-supervision term at all). The causal layer (L_cause) is NOT touched
    -- it is a separate structural-correlation mechanism, not a TOC
    component. ablate_f6 and use_toc=False must not be combined in the same
    run: they are two distinct ablations with overlapping but different
    mechanisms (soft input-zeroing vs. architectural removal) and combining
    them would conflate two separate research questions.

    log_causal=True (NOTEARS-layer analysis) only ADDS read-only logging:
    a first-step parameter audit (in optimizer / grad norm / value change),
    per-epoch h and graph summaries, the best-epoch graph, and divergence
    flags. It must not change training; tests/test_notears_analysis.py checks
    this by comparing checkpoints with and without the flag. The one
    behavioural exception is deliberate: a non-finite training loss stops the
    run and is recorded as divergence instead of crashing later in the AUC.
    """
    if ablate_f6 and not use_toc:
        raise ValueError("ablate_f6 and use_toc=False are distinct ablations "
                         "and must not be combined in one run.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = _build_cfg(run_cfg)
    cfg.model.use_toc = use_toc

    toc = TOCPriorLoader(np.load(MODEL_DIR / "capacity_compose.npy"),
                         np.load(MODEL_DIR / "capacity_home.npy"))
    graphs = torch.load(CACHE_DIR / "graphs.pt", weights_only=False)
    edge_index_30 = graphs[30]["edge_index"].to(device)
    toc_cap_30 = graphs[30]["toc_cap"].to(device)
    if ablate_f6:
        toc_cap_30 = torch.zeros_like(toc_cap_30)

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

    # use_toc=False -> no L_rcs term exists in the graph at all (not merely
    # zero-weighted); L_sub is zeroed via run_cfg["lambda_sub"]=0.0 by
    # convention, matching how Run1-4 already zero lambda_rcs_sup for
    # ablation instead of branching the loss class.
    loss_cls = TOCWeightedLoss if use_toc else _TOCWeightedLossBase
    loss_fn = loss_cls(cfg, toc, pattern_class_weights, use_toc=use_toc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    audit, first_step, causal_hist = None, {"done": False}, []
    if log_causal:
        from src import causal_analysis as ca
        opt_ids = {id(q) for g_ in optimizer.param_groups for q in g_["params"]}
        audit = {n: dict(numel=q.numel(), requires_grad=q.requires_grad,
                         in_optimizer=id(q) in opt_ids)
                 for n, q in model.named_parameters()}

    g = torch.Generator().manual_seed(seed)
    train_loader = make_loader("train", cfg.data.batch_size, True, CACHE_DIR, generator=g)
    val_loader = make_loader("val", cfg.data.batch_size, False, CACHE_DIR)

    def run_epoch(loader, train: bool, col=None):
        model.train() if train else model.eval()
        evaluator = TOCEvaluator(toc, n_nodes=30)
        with (torch.enable_grad() if train else torch.no_grad()):
            for x, label, pattern_idx, ttb in loader:
                x, label = x.to(device), label.to(device)
                pattern_idx, ttb = pattern_idx.to(device), ttb.to(device)
                out = model(x, edge_index_30, toc_cap_30)
                tg = {"label": label, "pattern_idx": pattern_idx, "ttb": ttb}
                losses = loss_fn(out, tg)
                if col is not None:
                    n_b = x.shape[0]
                    col["n"] += n_b
                    col["graph"] += out["causal_graph"].detach().double().cpu() * n_b
                    col["dag"].append(float(out["dag_penalty"].detach()))
                    col["pos"] += int((out["bn_logit"].detach() > 0).sum())
                    col["finite"] &= bool(torch.isfinite(losses["total"]).item())
                    if not col["finite"]:
                        return None      # divergence: stop before backward/step
                if train:
                    optimizer.zero_grad()
                    losses["total"].backward()
                    if audit is not None and not first_step["done"]:
                        before = {n: q.detach().clone()
                                  for n, q in model.named_parameters()}
                        for n, q in model.named_parameters():
                            # setdefault: under PREBUILD_CAUSAL=0 the lazy causal
                            # layer appears only now, outside the optimizer --
                            # recording that is the point of the audit.
                            a_ = audit.setdefault(n, dict(
                                numel=q.numel(), requires_grad=q.requires_grad,
                                in_optimizer=id(q) in opt_ids))
                            a_["grad_norm"] = (0.0 if q.grad is None
                                               else float(q.grad.norm()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    optimizer.step()
                    if audit is not None and not first_step["done"]:
                        for n, q in model.named_parameters():
                            audit[n]["max_abs_delta"] = float(
                                (q.detach() - before[n]).abs().max())
                        first_step["done"] = True
                evaluator.update(out, tg)
        return evaluator.compute()

    def _new_col():
        return dict(n=0, graph=torch.zeros(30, 30, dtype=torch.float64),
                    dag=[], pos=0, finite=True)

    best_auc, best_epoch, best_metrics, bad = -1.0, 0, None, 0
    nonfinite_epoch, best_pos_rate = None, None
    for epoch in range(1, cfg.training.epochs + 1):
        tr_col = _new_col() if log_causal else None
        try:
            run_epoch(train_loader, train=True, col=tr_col)
        except RuntimeError:
            # Only swallow errors caused by NaN/inf parameters (e.g. BCE input
            # outside [0, 1] after a blown-up step); anything else is a real bug.
            if not log_causal or all(torch.isfinite(q).all()
                                     for q in model.parameters()):
                raise
            nonfinite_epoch = epoch
            break
        if tr_col is not None and (not tr_col["finite"] or not all(
                torch.isfinite(q).all() for q in model.parameters())):
            nonfinite_epoch = epoch
            break
        va_col = _new_col() if log_causal else None
        val = run_epoch(val_loader, train=False, col=va_col)
        scheduler.step()
        if log_causal:
            g_val = (va_col["graph"] / va_col["n"]).numpy()
            rec = dict(epoch=epoch, val_auc=float(val["auc"]),
                       train_h_impl_batch_mean=float(np.mean(tr_col["dag"])),
                       val_h_impl_batch_mean=float(np.mean(va_col["dag"])),
                       val_pos_rate=va_col["pos"] / va_col["n"],
                       val_graph=ca.summarize_graph(g_val))
            causal_hist.append(rec)
        if val["auc"] > best_auc:
            best_auc, best_epoch, best_metrics, bad = val["auc"], epoch, val, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": val}, run_dir / "best_model.pt")
            if log_causal:
                best_pos_rate = rec["val_pos_rate"]
                cz = model.get_submodule("causal_30")
                w_sig = (torch.sigmoid(cz.W_raw.detach()).cpu().numpy()
                         * (1 - np.eye(30)))
                np.savez(run_dir / "causal_analysis.npz", epoch=epoch,
                         val_graph=g_val, W_sigmoid=w_sig,
                         W_raw=cz.W_raw.detach().cpu().numpy())
        else:
            bad += 1
            if bad >= cfg.training.patience:
                break

    if log_causal:
        import json
        (run_dir / "param_audit.json").write_text(json.dumps(audit, indent=1))
        (run_dir / "causal_history.json").write_text(json.dumps(causal_hist))
        if best_metrics is None:      # diverged before any usable epoch
            return dict(auc=None, diverged=True, nonfinite_epoch=nonfinite_epoch,
                        best_epoch=0, n_params=sum(q.numel() for q in model.parameters()))
    assert best_metrics is not None and best_metrics["cp_recall"] is not None
    causal = model.get_submodule("causal_30")
    extra = {}
    if log_causal:
        extra = dict(
            nonfinite_epoch=nonfinite_epoch, best_pos_rate=best_pos_rate,
            diverged=bool(nonfinite_epoch is not None
                          or (best_pos_rate is not None and best_pos_rate >= 0.99)),
            n_epochs_run=len(causal_hist))
    return dict(**extra,
        auc=best_metrics["auc"],
        cp_recall=best_metrics["cp_recall"],
        pat_acc=best_metrics["pattern_accuracy"],
        rcs_top1=best_metrics["rcs_top1"],
        rcs_top1_n=best_metrics["rcs_top1_n"],
        n_params=sum(p.numel() for p in model.parameters()),
        best_epoch=best_epoch,
        causal_w_absmax=float(causal.W_raw.abs().max().item()),
        toc_cap_max=float(toc_cap_30.abs().max().item()),
        ablate_f6=ablate_f6,
        use_toc=use_toc,
    )

# ----------------------------- Home fine-tuning -----------------------------

def finetune_home(compose_ckpt_path, seed: int, run_dir: Path,
                  epochs_detect: int = 8, epochs_causal: int = 8,
                  ablate_f6: bool = False) -> dict:
    """
    Mirrors notebook Cell 10: load a compose-trained checkpoint, split the
    Home (N=7) samples into a fine-tune set and a held-out test set, then
    fine-tune in two stages (detection-only BCE, then +RCS supervision on
    Pattern G = nodes {3,4}).

    PREBUILD_CAUSAL controls when causal_7 is instantiated relative to
    ft_optimizer, exactly as for causal_30 in train_and_evaluate:
      "0" -> lazy build AFTER ft_optimizer (reproduces the original bug)
      "1" -> forced build BEFORE ft_optimizer (fix: causal_7 actually trains)

    ablate_f6=True zeroes toc_cap_7 (see train_and_evaluate docstring for the
    mechanism). compose_ckpt_path should point at a checkpoint that was
    itself trained with ablate_f6=True, so the ablation is consistent across
    the whole compose-to-home pipeline for that seed.
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
    if ablate_f6:
        toc_cap_7 = torch.zeros_like(toc_cap_7)

    model = CSTGNN(cfg).to(device)
    ckpt = torch.load(compose_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)  # causal_7 missing -> fine

    home_samples = torch.load(CACHE_DIR / "test.pt", weights_only=False)
    home_ds = CachedWindowDataset(home_samples)
    n_total = len(home_ds)
    n_ft = int(0.2 * n_total)
    g = torch.Generator().manual_seed(seed)
    ft_set, holdout_set = random_split(home_ds, [n_ft, n_total - n_ft], generator=g)
    ft_loader = torch.utils.data.DataLoader(ft_set, batch_size=32, shuffle=True,
                                            generator=torch.Generator().manual_seed(seed))
    holdout_loader = torch.utils.data.DataLoader(holdout_set, batch_size=32, shuffle=False)

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
        rcs_absmax = float(np.abs(rcs_all).max()) if len(rcs_all) else 0.0
        return float(auc), float(top1), rcs_absmax

    stages = {}
    auc0, top1_0, rcsmax0 = evaluate()
    stages["zero_shot"] = dict(auc=auc0, rcs_top1=top1_0, rcs_absmax=rcsmax0)

    model.train()
    for _ in range(epochs_detect):
        for x, label, pattern_idx, ttb in ft_loader:
            x = x.to(device)
            out = model(x, edge_index_7, toc_cap_7)
            loss = F.binary_cross_entropy(torch.sigmoid(out["bn_logit"].squeeze(-1)),
                                          label.to(device).float())
            ft_optimizer.zero_grad(); loss.backward(); ft_optimizer.step()
    auc1, top1_1, rcsmax1 = evaluate()
    stages["finetune_detection"] = dict(auc=auc1, rcs_top1=top1_1, rcs_absmax=rcsmax1)

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
    auc2, top1_2, rcsmax2 = evaluate()
    stages["finetune_causal"] = dict(auc=auc2, rcs_top1=top1_2, rcs_absmax=rcsmax2)

    return dict(
        seed=seed, stages=stages,
        causal_w_absmax_before=w_before,
        causal_w_absmax_after=float(causal7.W_raw.detach().abs().max().item()),
        toc_cap_max=float(toc_cap_7.abs().max().item()),
        ablate_f6=ablate_f6,
    )