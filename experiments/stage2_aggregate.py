# experiments/stage2_aggregate.py

import json
import os
import sys
from pathlib import Path
import numpy as np

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.runs import RUNS, SEEDS, validate

RESULTS_DIR = Path("results/multiseed")
AGGREGATED_FILE = RESULTS_DIR / "aggregated_summary.json"

def aggregate_results():
    validate()
    summary = {}

    for run_id in RUNS.keys():
        cp_recalls = []
        aucs = []

        for seed in SEEDS:
            history_path = RESULTS_DIR / f"{run_id}_seed{seed}" / "history.json"
            if not history_path.exists():
                raise FileNotFoundError(f"Missing results file for {run_id} seed {seed} at {history_path}")

            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                metrics = data.get("metrics", {})
                cp_recalls.append(metrics["cp_recall"])
                aucs.append(metrics["auc"])

        summary[run_id] = {
            "cp_recall": {
                "mean": float(np.mean(cp_recalls)),
                "std": float(np.std(cp_recalls, ddof=1)),
                "raw_values": cp_recalls
            },
            "auc": {
                "mean": float(np.mean(aucs)),
                "std": float(np.std(aucs, ddof=1)),
                "raw_values": aucs
            }
        }

    with open(AGGREGATED_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Aggregation complete. Summary saved strictly to {AGGREGATED_FILE}")

if __name__ == "__main__":
    aggregate_results()