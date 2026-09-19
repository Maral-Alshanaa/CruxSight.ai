# experiments/stage3_stats.py

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.runs import SEEDS

RESULTS_DIR = Path("results/multiseed")
AGGREGATED_FILE = RESULTS_DIR / "aggregated_summary.json"
STATS_FILE = RESULTS_DIR / "statistical_analysis.json"
RANDOM_BASELINE_CP_RECALL = 0.9436

def perform_statistical_analysis():
    if not AGGREGATED_FILE.exists():
        raise FileNotFoundError(f"Missing aggregated summary at {AGGREGATED_FILE}. Run stage2 first.")

    with open(AGGREGATED_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Extract raw CP-Recall values across 10 pre-registered seeds
    run2_recalls = np.array(summary["run2"]["cp_recall"]["raw_values"])
    run4_recalls = np.array(summary["run4"]["cp_recall"]["raw_values"])

    # 1. Primary Test: Two-tailed Paired t-test (Run 2 vs Run 4)
    t_stat, p_val_ttest = stats.ttest_rel(run4_recalls, run2_recalls)

    # 2. Sensitivity Test: Wilcoxon signed-rank test
    # Min possible p-value for N=10 is 2 / (2**10) = 0.00195
    wilcoxon_stat, p_val_wilcoxon = stats.wilcoxon(run4_recalls, run2_recalls, alternative="two-sided")

    # 3. Random Baseline Comparisons (CP-Recall vs 0.9436)
    baseline_comparisons = {}
    for run_id, metrics in summary.items():
        recalls = np.array(metrics["cp_recall"]["raw_values"])
        mean_recall = metrics["cp_recall"]["mean"]
        diff_from_baseline = mean_recall - RANDOM_BASELINE_CP_RECALL
        baseline_comparisons[run_id] = {
            "mean_cp_recall": mean_recall,
            "random_baseline": RANDOM_BASELINE_CP_RECALL,
            "diff_from_baseline": float(diff_from_baseline),
            "outperforms_random": bool(mean_recall > RANDOM_BASELINE_CP_RECALL)
        }

    results = {
        "primary_test_paired_ttest": {
            "comparison": "Run 4 vs Run 2 (CP-Recall)",
            "n_seeds": len(SEEDS),
            "t_statistic": float(t_stat),
            "p_value": float(p_val_ttest),
            "alpha": 0.05,
            "significant": bool(p_val_ttest < 0.05)
        },
        "sensitivity_test_wilcoxon": {
            "comparison": "Run 4 vs Run 2 (CP-Recall)",
            "n_seeds": len(SEEDS),
            "wilcoxon_statistic": float(wilcoxon_stat),
            "p_value": float(p_val_wilcoxon),
            "min_possible_p_value": 2.0 / (2 ** len(SEEDS))
        },
        "baseline_comparison": baseline_comparisons
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Statistical analysis completed successfully. Results written to {STATS_FILE}")

if __name__ == "__main__":
    perform_statistical_analysis()