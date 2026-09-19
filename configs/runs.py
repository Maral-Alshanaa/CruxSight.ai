# configs/runs.py

# List of 10 seeds fixed prior to execution (pre-registered)
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Hyperparameters mapped explicitly from Table 10 & Ablation configurations
RUNS = {
    # Run 1: High FN penalty, no causal/sub/RCS constraints
    "run1": dict(
        fn_weight=5.0,
        lambda_causal=0.0,
        lambda_sub=0.0,
        lambda_rcs_sup=0.0,
    ),
    # Run 2: Balanced FN penalty, mild causal/sub constraints, no RCS supervision
    "run2": dict(
        fn_weight=1.5,
        lambda_causal=0.05,
        lambda_sub=0.05,
        lambda_rcs_sup=0.0,
    ),
    # Run 3: Higher causal weight divergence baseline
    "run3": dict(
        fn_weight=1.5,
        lambda_causal=0.10,
        lambda_sub=0.05,
        lambda_rcs_sup=0.0,
    ),
    # Run 4: Canonical architecture (Full model with RCS constraint)
    "run4": dict(
        fn_weight=1.5,
        lambda_causal=0.05,
        lambda_sub=0.05,
        lambda_rcs_sup=0.3,
    ),
}

def validate():
    """Verify all run configurations and seeds count strictly."""
    missing = [k for k, v in RUNS.items() if v is None]
    assert not missing, f"Configuration incomplete for keys: {missing}"
    assert len(SEEDS) == 10, f"Expected exactly 10 seeds, got {len(SEEDS)}"
    print(f"Successfully validated {len(RUNS)} runs across {len(SEEDS)} pre-registered seeds.")

if __name__ == "__main__":
    validate()