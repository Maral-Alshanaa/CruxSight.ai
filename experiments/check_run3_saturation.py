# experiments/check_run3_saturation.py
"""Quick check: is Run 3's high CP-Recall driven by probability saturation
(predicting positive too often)? Reloads each seed's checkpoint and computes
the positive-prediction rate and precision/recall on validation."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import RUNS, SEEDS
from src.core import Config, TOCPriorLoader, CSTGNN, TOCEvaluator, make_loader

MODEL_DIR = Path("/content/drive/MyDrive/bottleneck_project/cst_gnn")  # adjust if needed
CACHE_DIR = MODEL_DIR / "dataset_cache"
CKPT_ROOT = Path("/content/drive/MyDrive/CruxSight/causal_fix")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toc = TOCPriorLoader(np.load(MODEL_DIR / "capacity_compose.npy"),
                         np.load(MODEL_DIR / "capacity_home.npy"))
    graphs = torch.load(CACHE_DIR / "graphs.pt", weights_only=False)
    ei30, cap30 = graphs[30]["edge_index"].to(device), graphs[30]["toc_cap"].to(device)
    val_loader = make_loader("val", 64, False, CACHE_DIR)

    cfg = Config.load(str(MODEL_DIR / "config.yaml"))
    cfg.model.gat_hidden, cfg.model.tft_hidden = RUNS["run3"]["gat_hidden"], RUNS["run3"]["tft_hidden"]

    print(f"{'seed':>6} {'pos_pred_rate':>14} {'precision':>10} {'recall':>8} {'cp_recall':>10}")
    for seed in SEEDS:
        ckpt_path = CKPT_ROOT / "run3" / f"seed{seed}" / "best_model.pt"
        model = CSTGNN(cfg).to(device)
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 12, 30, 7, device=device), ei30, cap30)  # build causal_30
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        ev = TOCEvaluator(toc, n_nodes=30)
        with torch.no_grad():
            for x, label, pattern_idx, ttb in val_loader:
                x, label = x.to(device), label.to(device)
                pattern_idx, ttb = pattern_idx.to(device), ttb.to(device)
                out = model(x, ei30, cap30)
                ev.update(out, {"label": label, "pattern_idx": pattern_idx, "ttb": ttb})
        m = ev.compute()
        pos_pred_rate = (np.array(ev.probs) > 0.5).mean()
        print(f"{seed:>6} {pos_pred_rate:>14.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['cp_recall']:>10.4f}")


if __name__ == "__main__":
    main()