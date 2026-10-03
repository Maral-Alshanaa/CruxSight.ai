"""
NOTEARS / causal-layer analysis arms (pre-registered, tag notears-analysis-prereg).

Base = RUNS["run4"] from configs/runs.py, imported (not copied) so any drift
in the Run 4 definition is impossible. Only lambda_causal (and, in the
deconfounding arms, fn_weight) differ between arms. All arms: same 10 seeds,
CRUX_PREBUILD_CAUSAL=1, train_and_evaluate(..., log_causal=True).

Arm names are stable identifiers used for result directories.
"""
from configs.runs import RUNS, SEEDS

BASE = RUNS["run4"]

LAMBDA_GRID = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5)

ARMS = {}
for lam in LAMBDA_GRID:
    ARMS[f"lam{lam:g}"] = dict(BASE, lambda_causal=lam)

# Deconfounding arms: Run 1/3 used fn_weight=5.0 together with lambda_causal=0.20
# (and a larger architecture, which stays unseparated). Together with the
# fn_weight=1.5 cells lam0.05 / lam0.2 above they form a 2x2.
ARMS["fn5_lam0.05"] = dict(BASE, fn_weight=5.0, lambda_causal=0.05)
ARMS["fn5_lam0.2"] = dict(BASE, fn_weight=5.0, lambda_causal=0.2)

REFERENCE_ARM = "lam0.05"
FRIEDMAN_ARMS = tuple(f"lam{l:g}" for l in LAMBDA_GRID)   # primary test set


def validate():
    assert len(SEEDS) == 10, len(SEEDS)
    assert len(ARMS) == 8, len(ARMS)
    assert ARMS[REFERENCE_ARM] == BASE, "reference arm must equal Run 4 exactly"
    for name, cfg in ARMS.items():
        assert cfg["expected_params"] == 205617, name
        for k in ("lambda_sub", "lambda_rcs_sup", "fp_weight", "lr", "weight_decay",
                  "gat_hidden", "tft_hidden", "patience", "epochs"):
            assert cfg[k] == BASE[k], (name, k)
    return len(ARMS) * len(SEEDS)   # 80 jobs
