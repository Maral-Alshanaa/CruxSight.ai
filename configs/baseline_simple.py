# configs/baseline_simple.py
"""
Frozen design constants for the simple baselines (random forest, static threshold rule).
Pre-registered under tag `baseline-simple-prereg`; do not edit afterwards
(experiments/baseline_simple_run.py aborts if this file differs from the tag).
No hyperparameter is tuned on any data: every value below is fixed before the first run.
"""
from configs.runs import SEEDS

F0_CHANNEL = 0                  # F0 = mean_latency (ms), channel 0 of the cached x [12, 30, 7] (paper Table 2)
N_NODES = 30                    # Compose graph; Home (N = 7) is not evaluated here
EPS = 1e-8                      # added to the per-node SD in the z-score (nodes with SD = 0)
RULE_FIT_ON = "normal_train_windows"     # per-node mean/SD of F0 from train windows with y = 0 only
RULE_TAU_CRITERION = "youden_j_on_train"  # tau = argmax_t (TPR - FPR) over train window scores; ties -> largest t

RF_PARAMS = dict(n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt", n_jobs=-1)
RF_DETECTION_CLASS_WEIGHT = "balanced_subsample"   # detection only; pattern/localization unweighted (as logreg)
RF_ARMS = ("rf_pernode", "rf_flat")
ALPHA = 0.05
REFERENCE_ARMS = ("run4_retrain", "no_toc")        # results/baseline/{arm}_seed{seed}.json
PAIRED_METRICS = ("auc", "pat_acc", "rcs_top1")     # RF vs reference (paired by seed)
RULE_METRICS = ("auc", "rcs_top1")                  # PatAcc not applicable to the rule


def validate() -> None:
    assert SEEDS == [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021], SEEDS
    assert F0_CHANNEL == 0 and N_NODES == 30
