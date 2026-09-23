*# Multi-seed protocol (pre-registered, v2)*

*- Seeds: 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021 (10 per run, Runs 1-4)*

*- Metrics: AUC, CP-Recall, PatAcc on the compose VALIDATION set at the best-val-AUC epoch*

&#x20; *(same protocol as Table 10). Checkpoint selection uses AUC on the same set, so values are optimistic.*

*- Primary test: paired two-sided t-test, Run 2 vs Run 4, CP-Recall, alpha=0.05. Exact Wilcoxon = sensitivity.*

*- Seeds vary init and batch order only; the split is fixed.*

*- CRUX\_PREBUILD\_CAUSAL=0 (faithful replication: causal layer built lazily after the optimizer, as in the original notebook).*

*- The independent compose holdout (cpu\_sept9, single run) is reported separately in the thesis and is NOT part of this study.*

*- Assumed, not verified from logs: Run 1/3 use lr=1e-3, weight\_decay=1e-4, dropout 0.1, patience=10, epochs=60.*

*- Correction note: the v1 tag had wrong Run 1/3 loss weights; found before any real run executed.*



*## Deviations and notes (recorded before the results commit)*

*- experiments/stage2\_aggregate.py and experiments/stage4\_tables.py were replaced after tag multiseed-prereg-v2:*

&#x20; *the tagged stage2 read a per-seed history.json layout that stage1 does not produce and did not aggregate PatAcc.*

&#x20; *configs/runs.py, experiments/stage3\_stats.py, the seeds, the metrics and the primary test are unchanged from the tag.*

*- The training adapter (src/core.py, src/pipeline.py) and an import fix in stage1 were added after the tag.*

&#x20; *Code version that produced all 40 result files: git\_sha bde5938740c8651f359328c2e13cece75d304de0.*

*- stage3 uses the thesis baseline 0.9436 (Monte Carlo); the exact value is 0.9458. Both are reported.*

*- Metrics are compose validation metrics at the best-val-AUC epoch; the independent compose holdout is excluded.*

*- Observed: causal\_30.W\_raw stayed exactly 0.0 in all 40 jobs (the causal layer was created after the optimizer, as in the original notebook).*



*## Mandatory pre-flight check (before any real training run)*

*Run `python tests/test\_gradient\_flow.py` and confirm both checks pass.*

*This check must pass after any change to src/core.py or src/pipeline.py,*

*before spending any GPU time. It was added after discovering causal\_30.W\_raw*

*was never trained in the original notebook and in the first multiseed run.*

## Confirmed bug and fix (pre-flight test result)
tests/test_gradient_flow.py::test_real_pipeline_trains_causal_layer, run on the
real src/pipeline.py with a tiny synthetic dataset:
  CRUX_PREBUILD_CAUSAL=0 (original notebook order): causal_w_absmax = 0.0 -> FAILS
  CRUX_PREBUILD_CAUSAL=1 (causal_30 built before optimizer creation): causal_w_absmax = 0.006 -> PASSES
This confirms causal_30 (and by the same mechanism causal_7 in home fine-tuning)
never received gradient updates in the original notebook, in checkpoints
final_model_v1.pt / best_model.pt, and in all 40 multiseed-v1 result files
(all logged causal_w_absmax = 0.0). Going forward, CRUX_PREBUILD_CAUSAL=1 is used
for any new training run. This pre-flight test is mandatory before any real
training and must pass before spending GPU time.

## Parallel study: causal layer fix (pre-registered before running)
- Motivation: tests/test_gradient_flow.py proved causal_30.W_raw never receives
  gradients when the causal layer is built after optimizer creation (as in the
  original notebook and in all multiseed-v1 results). Building it before the
  optimizer (CRUX_PREBUILD_CAUSAL=1) fixes this (confirmed: causal_w_absmax
  0.0 -> 0.006 on synthetic data).
- Scope: Run 2 and Run 4 ONLY (the pair in the primary test), 10 seeds each
  (same seeds as multiseed-v1: 42,123,456,789,1011,1213,1415,1617,1819,2021).
  Run 1 and Run 3 are not part of this study.
- Setting: CRUX_PREBUILD_CAUSAL=1, CRUX_RUNS=run2,run4, output directory
  separate from multiseed-v1 (CruxSight/causal_fix on Drive), so the original
  40 faithful-replication results are never touched or overwritten.
- Primary test: identical to multiseed-prereg-v2 -- two-sided paired t-test on
  CP-Recall, Run 4 vs Run 2, alpha=0.05, exact Wilcoxon as sensitivity.
- This is a SEPARATE, additional result. It does not replace or invalidate
  multiseed-v1; both are reported. Whether the fixed-causal-layer numbers or
  the faithful-replication numbers are used as the thesis's primary claim is
  a decision for the supervisor, not resolved here.
- Success/failure of the fix itself is verified by causal_w_absmax > 0 in
  every one of these 20 result files (checked in analysis, not assumed).

## causal_7 bug confirmation (Home fine-tuning)
tests/test_causal7_gradient_flow.py, mirroring the exact Cell 10 sequence
(ft_optimizer built from model.parameters() BEFORE the first Home-graph forward
pass, which lazily builds causal_7):
  - causal_7's 5 parameters (W_raw + encoder) are NOT in ft_optimizer's param groups.
  - causal_7.W_raw.grad IS computed (abs max 0.00106 on synthetic data) but is
    never applied: W_raw changes by exactly 0.0 after optimizer.step().
CONFIRMED: causal_7 has the identical bug as causal_30, via the identical
mechanism (lazy instantiation after optimizer construction). This means the
Home-workflow RCS Top-1 result (0% -> 48.7% after 16 fine-tuning epochs,
reported in final_results.json and the thesis) cannot be attributed to learned
causal structure on the Home graph -- causal_7 never trained. The source of that
improvement (likely the shared temporal representation, which IS inside
ft_optimizer) is not yet identified. This finding, and its implications for the
Generalization Study section of the thesis, require supervisor review before
any re-run or reinterpretation of Section 9-10 results.

## Home fine-tuning fix (item 1 of the full remediation plan)
Added src.pipeline.finetune_home(), mirroring notebook Cell 10, with
CRUX_PREBUILD_CAUSAL applied to causal_7 exactly as for causal_30. Verified
with tests/test_home_finetune_gradient_flow.py as a control experiment on
synthetic data: PREBUILD_CAUSAL=0 reproduces the bug (causal_w_absmax stays
0.0), PREBUILD_CAUSAL=1 fixes it (causal_w_absmax > 0.0 after fine-tuning).
Real Home fine-tuning re-runs (item 2 of the remediation plan: zero-shot ->
detection-only -> +RCS, on real Home data, using a fixed compose checkpoint
from causal-fix-v1) are pre-registered separately before execution -- not yet
run as of this entry.

