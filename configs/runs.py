SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# Run 1/3 non-loss settings are ASSUMED from notebook Cell 2 defaults (no logs exist for them).
_OLD = dict(gat_hidden=64, tft_hidden=128, gat_dropout=0.1, tft_dropout=0.1,
            weight_decay=1e-4, lr=1e-3, patience=10, epochs=60, fp_weight=1.0)
_NEW = dict(gat_hidden=32, tft_hidden=64, gat_dropout=0.2, tft_dropout=0.2,
            weight_decay=1e-3, lr=5e-4, patience=12, epochs=60, fp_weight=1.0)

RUNS = {
    "run1": dict(_OLD, fn_weight=5.0, lambda_causal=0.20, lambda_sub=0.10,
                 lambda_rcs_sup=0.0, expected_params=779921),
    "run2": dict(_NEW, fn_weight=1.5, lambda_causal=0.05, lambda_sub=0.05,
                 lambda_rcs_sup=0.0, expected_params=205617),
    "run3": dict(_OLD, fn_weight=5.0, lambda_causal=0.20, lambda_sub=0.10,
                 lambda_rcs_sup=0.3, expected_params=779921),
    "run4": dict(_NEW, fn_weight=1.5, lambda_causal=0.05, lambda_sub=0.05,
                 lambda_rcs_sup=0.3, expected_params=205617),
}

def validate():
    assert all(v is not None for v in RUNS.values())
    assert len(SEEDS) == 10, len(SEEDS)