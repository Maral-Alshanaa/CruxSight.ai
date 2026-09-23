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

