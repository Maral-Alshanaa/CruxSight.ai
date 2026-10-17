# configs/baseline_runs.py
"""
Frozen arm definitions for the baseline-comparison reproduction (Section 5.2.7).
Pre-registered under tag `baseline-corrected-prereg`; do not edit afterwards
(experiments/baseline_run.py aborts if this file differs from the tag).

Arms
  run4_retrain : CruxSight Run 4, identical to configs.runs.RUNS["run4"], use_toc=True.
  no_toc       : same sizing/coefficients, use_toc=False. TOC-derived training signal
                 removed from architecture (no toc_scale, no RCS multiplier, no L_rcs)
                 AND from the detection loss (node_weights == 1, see PROTOCOL.md,
                 "Residual-signal finding"). lambda_sub = 0.0 by configuration.
                 lambda_rcs_sup is irrelevant (the base loss class never reads it)
                 and set to 0.0 so the config cannot be misread.
  logreg       : deterministic, no seed sweep (see baseline_run.py).
"""
from configs.runs import RUNS, SEEDS

RANDOM_RCS_TOP1_PCT = 38.2          # weighted random baseline, PROTOCOL.md (RCS Top-1 random-baseline check)
RANDOM_CP_RECALL = 0.9458           # exact hypergeometric, 18/30 critical-path nodes, top-3

ARMS = {
    "run4_retrain": dict(use_toc=True, run_cfg=dict(RUNS["run4"])),
    "no_toc": dict(
        use_toc=False,
        run_cfg=dict(RUNS["run4"], lambda_sub=0.0, lambda_rcs_sup=0.0,
                     expected_params=205615),   # 205,617 - 2 toc_scale scalars (PROTOCOL.md, verified in tests)
    ),
}


def validate() -> None:
    assert len(SEEDS) == 10 and SEEDS == [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021], SEEDS
    r4, nt = ARMS["run4_retrain"]["run_cfg"], ARMS["no_toc"]["run_cfg"]
    assert r4 == RUNS["run4"], "run4_retrain must equal configs.runs.RUNS['run4']"
    changed = {k for k in r4 if r4[k] != nt[k]}
    assert changed == {"lambda_sub", "lambda_rcs_sup", "expected_params"}, changed
    assert nt["lambda_sub"] == 0.0 and nt["expected_params"] == 205615
    assert ARMS["run4_retrain"]["use_toc"] is True and ARMS["no_toc"]["use_toc"] is False
