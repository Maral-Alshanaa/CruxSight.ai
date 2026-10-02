# src/ttb_export.py
"""
Re-evaluates the best checkpoint written by pipeline.train_and_evaluate on the
validation split (in cache order, eval mode) and returns raw per-window
predictions. It does NOT modify training: src/pipeline.py is untouched.

Integrity check: the AUC recomputed here must equal the AUC stored in the
checkpoint (the one pipeline.train_and_evaluate selected the epoch with);
otherwise the export is not the same evaluation and we abort.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src import pipeline as P
from src.core import CSTGNN, make_loader
from src.ttb_eval import recover_file_ids


def export_val_predictions(run_dir, run_cfg, device=None, auc_tol=1e-4):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = P._build_cfg(run_cfg)
    cfg.model.use_toc = True
    graphs = torch.load(P.CACHE_DIR / "graphs.pt", weights_only=False)
    edge_index = graphs[30]["edge_index"].to(device)
    toc_cap = graphs[30]["toc_cap"].to(device)

    model = CSTGNN(cfg).to(device)
    model.eval()
    with torch.no_grad():   # build the lazy causal layer so state_dict keys match the checkpoint
        model(torch.zeros(1, cfg.data.window_steps, 30, cfg.data.n_features, device=device),
              edge_index, toc_cap)
    ckpt = torch.load(Path(run_dir) / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = make_loader("val", cfg.data.batch_size, False, P.CACHE_DIR)   # shuffle=False: cache order
    prob, ttb_pred, label, ttb_true, pattern = [], [], [], [], []
    with torch.no_grad():
        for x, lab, pat, ttb in loader:
            out = model(x.to(device), edge_index, toc_cap)
            prob.append(torch.sigmoid(out["bn_logit"]).reshape(-1).cpu())
            ttb_pred.append(out["ttb"].reshape(-1).cpu())
            label.append(lab.reshape(-1)); ttb_true.append(ttb.reshape(-1)); pattern.append(pat.reshape(-1))
    cat = lambda xs: torch.cat(xs).numpy()
    res = dict(prob=cat(prob).astype(np.float64), ttb_pred=cat(ttb_pred).astype(np.float64),
               label=cat(label).astype(int), ttb_true=cat(ttb_true).astype(np.float64),
               pattern=cat(pattern).astype(int))

    samples = loader.dataset.samples
    x_all = np.stack([s["x"].numpy() for s in samples])
    res["file_id"] = recover_file_ids(x_all)

    auc = float(roc_auc_score(res["label"], res["prob"])) if len(set(res["label"])) > 1 else 0.0
    stored = float(ckpt["val_metrics"]["auc"])
    if abs(auc - stored) > auc_tol:
        raise RuntimeError(f"export AUC {auc:.6f} != checkpoint AUC {stored:.6f}: not the same evaluation")
    res["val_auc"] = auc
    res["best_epoch"] = int(ckpt["epoch"])
    return res
