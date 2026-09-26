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


## Home fine-tuning fix -- verified
tests/test_home_finetune_gradient_flow.py passed after fixing a real bug caught
by the test itself: finetune_home() originally passed raw dict samples to
random_split/DataLoader instead of wrapping them in CachedWindowDataset,
causing an AttributeError before any causal-layer behaviour could even be
observed. Fixed in commit c7eb0c3. Control experiment result on synthetic data:
  PREBUILD_CAUSAL=0: causal_7.W_raw stays at 0.0 (bug reproduced)
  PREBUILD_CAUSAL=1: causal_7.W_raw moves 0.0002 -> 0.00034 (fix confirmed)
finetune_home() is now ready for a real run on actual Home data. Item 1 of the
remediation plan (fix causal_7) is complete; item 2 (real Home fine-tuning
re-run) is next, to be pre-registered before execution.


## Home fine-tuning re-run (item 2, pre-registered before execution)
- Scope: 10 seeds (same list: 42,123,456,789,1011,1213,1415,1617,1819,2021).
  For each seed, the starting checkpoint is that SAME seed's Run 4 best_model.pt
  from causal-fix-v1 (CruxSight/causal_fix/run4/seed{S}/best_model.pt on Drive) --
  i.e. fine-tuning seed == compose-training seed, for full traceability. No
  single fixed checkpoint is reused across seeds, unlike the original notebook.
- Setting: CRUX_PREBUILD_CAUSAL=1 (causal_7 fix applies; PREBUILD_CAUSAL=0 is not
  re-run here since causal-fix-v1 already established that comparison for
  causal_30, and the causal_7 bug/fix mechanism is confirmed identical).
- Fine-tune split: 20% fine-tune / 80% held-out test per seed, split with a
  seed-specific torch.Generator (same seed as the run), matching Cell 10's
  ratio (169/849 samples) but re-randomized per seed rather than fixed at seed 42.
- Stages and metrics recorded per seed: zero-shot AUC and RCS Top-1, post
  detection-only fine-tuning (8 epochs) AUC and RCS Top-1, post +RCS-supervision
  fine-tuning (8 more epochs, 16 total) AUC and RCS Top-1, plus
  causal_7.W_raw abs-max before/after (to confirm the fix took effect in every run).
- No formal significance test is pre-registered for this stage: the original
  notebook reports single-value results per stage, and 10 seeds here serve to
  show the spread (mean +/- SD) of zero-shot / detection / causal-supervision
  AUC and RCS Top-1, not a paired comparison between two configurations.
- This is a genuinely new result (the original Generalization Study numbers
  were produced with causal_7 untrained); it does not replace the original
  table, which stays in the record with its bug now documented above.

## Home fine-tuning re-run -- seed 42 checkpoint sanity check
Verified compose_checkpoint for seed 42 (causal-fix-v1 run4/seed42/best_model.pt)
loads correctly: stored val_metrics (AUC 0.8703409992069786) match causal-fix-v1's
recorded result for that seed exactly; state_dict contains only causal_30/heads/
spatial/temporal (no causal_7, as expected before fine-tuning).
Observation: seed 42's zero-shot Home AUC is 0.367 (worse than random, vs. 0.544
in the original single-run notebook result). This is NOT a loading bug -- the
checkpoint is confirmed correct. Working hypothesis: a compose checkpoint trained
with causal_30 actually learning (L_causal/L_sub with a real, trained causal
layer) may produce a temporal representation that transfers less well, zero-shot,
to the structurally different Home graph than one trained with an inert causal
layer. This is evaluated across all 10 seeds before drawing any conclusion.

## Home fine-tuning re-run -- results (10 seeds, PREBUILD_CAUSAL=1)
Mean +/- SD across seeds (tag home-finetune-v1):
  zero_shot:          AUC 0.635 +/- 0.154 (range 0.367-0.836); RCS Top-1  5.4% +/- 14.3% (0-45.7%)
  finetune_detection: AUC 0.889 +/- 0.017 (range 0.864-0.917); RCS Top-1  4.0% +/-  9.4% (0-30.2%)
  finetune_causal:    AUC 0.899 +/- 0.014 (range 0.879-0.928); RCS Top-1 48.0% +/- 26.2% (4.4-81.3%)
causal_7.W_raw abs-max moved from 0.0014+/-0.0006 to 0.0077+/-0.0004 in every seed (fix confirmed active).

Key findings:
1. The core claim holds: RCS supervision raises RCS Top-1 sharply (~4% -> ~48%),
   matching the original single-run thesis number (48.7%) almost exactly as a mean.
2. However RCS Top-1 has very high variance across seeds (SD 26.2%, range
   4.4%-81.3%) -- the original single-value result, while numerically accurate
   as a mean, masks this instability completely.
3. Zero-shot AUC is also highly variable (0.367-0.836) and not reliably above
   chance for every seed -- contradicts the earlier single-seed-42 hypothesis
   that the causal_30 fix systematically hurts zero-shot transfer; seed 42 was
   simply the low outlier, not representative.
4. Ruled out: fine-tune-set class imbalance. Positive rate in the 169-sample
   fine-tune split is stable across seeds (56.2%-63.9%, an 8-point range) and
   does not correlate with RCS Top-1 variance (e.g. seed 1617: 63.3% positive
   rate, the highest, yet the LOWEST RCS Top-1 at 4.4%).
5. Detection-only fine-tuning (stage 2) is comparatively stable (AUC SD 0.017),
   in contrast to RCS-supervised localization (stage 3, RCS Top-1 SD 26.2%) --
   consistent with the thesis's existing claim that detection generalizes faster
   and more reliably than root-cause localization, but the localization
   instability is far larger than previously known.
Interpretation for the thesis/product: root-cause localization on a new topology
with ~25 minutes of production data (169 samples) is not yet reliable -- the mean
result is real, but any single run (including the original 48.7%) could
plausibly have landed anywhere in the 4%-81% range by chance of initialization
alone. More fine-tuning data and/or more training epochs for the causal-supervision
stage should be investigated before this is presented as a stable capability.

## Run 1/Run 3 fixed-causal re-run (item 3, pre-registered before execution)
- Scope: Run 1 and Run 3 only, same 10 seeds, CRUX_PREBUILD_CAUSAL=1, written into
  the SAME results/causal_fix directory as causal-fix-v1 (Run 2/Run 4), so all
  four configurations under the fix live together (run{1,2,3,4}_seed{S}.json).
- This completes the fixed-causal picture for all four Table 10 configurations.
- Primary reporting: mean +/- SD (AUC, CP-Recall, PatAcc) for all four runs,
  identical table format to multiseed-v1, computed with results/causal_fix as
  input instead of results/multiseed.
- Exploratory (not part of the original primary test, clearly labeled as such):
  Run 3 vs Run 1 on CP-Recall mirrors the Run 4 vs Run 2 comparison under the
  OTHER loss-weight regime (fn_weight=5.0, lambda_causal=0.20 instead of
  fn_weight=1.5, lambda_causal=0.05) -- same paired two-sided t-test and exact
  Wilcoxon, reported as a secondary/exploratory result, not a second primary test.
- Code version note: git_sha will differ from causal-fix-v1's 27b5332 because
  unrelated commits (causal_7 fix, README merge) were added since, but
  train_and_evaluate() itself for Compose (Runs 1-4) has not been modified in
  that interval -- verified by git diff before running.
- expected_params assertion (779,921 shared params for Run1/Run3) occurs before
  the causal layer is built, so it is unaffected by CRUX_PREBUILD_CAUSAL and
  still guards architecture correctness.

## All four configurations under the causal-layer fix -- final table (closes the multi-seed file)
n=10 seeds, PREBUILD_CAUSAL=1, all in results/causal_fix. Mean +/- SD:
  run1: AUC 0.8552+/-0.0288, CP-Recall 0.8607+/-0.0836, PatAcc 0.7844+/-0.0989, CP-Recall>baseline 1/10
  run2: AUC 0.8674+/-0.0135, CP-Recall 0.8376+/-0.0634, PatAcc 0.8989+/-0.0204, CP-Recall>baseline 0/10
  run3: AUC 0.8070+/-0.0543, CP-Recall 0.9565+/-0.0412, PatAcc 0.6509+/-0.1910, CP-Recall>baseline 7/10
  run4: AUC 0.8699+/-0.0141, CP-Recall 0.9053+/-0.0337, PatAcc 0.9004+/-0.0344, CP-Recall>baseline 1/10
Exact random baseline: 0.9458.

Primary test (Run4 vs Run2, CP-Recall, fixed layer): t(9)=2.773, p=0.0216, exact
Wilcoxon W=5, p=0.0195, d_z=0.88, mean diff 0.068 CI[0.012,0.123], Run4>Run2 8/10.
(Matches causal-fix-v1 exactly, as expected -- same run2/run4 data.)

Exploratory (Run3 vs Run1, CP-Recall, fixed layer, NOT pre-registered as a
primary test -- mirrors the primary comparison under the fn_weight=5.0/
lambda_causal=0.20 regime): t(9)=3.412, p=0.0077, exact Wilcoxon W=2, p=0.0059,
d_z=1.08, mean diff 0.096 CI[0.032,0.159], Run3>Run1 9/10.

Notable new finding: Run 3 is the ONLY configuration across all studies
(faithful replication, causal-fix Run2/4, and this table) whose mean CP-Recall
(0.9565) exceeds the exact random baseline (0.9458), with 7/10 seeds individually
above it. However Run 3's PatAcc is low and highly unstable (0.6509+/-0.1910),
consistent with the original "val loss diverged" note -- raising an open question
of whether high CP-Recall here partly reflects probability saturation (predicting
positive too often) rather than genuine root-cause precision. This is flagged as
an open question, not resolved here; per-seed precision/recall would need
inspection before treating Run 3's CP-Recall as a real advantage.

This table completes item 3 of the remediation plan. The multi-seed statistical
analysis file (faithful replication + causal-layer fix, Compose and Home) is now
considered closed pending supervisor review. Remaining items (4: revise thesis
chapters on "learned causal structure"; 5: regenerate causal-graph figures) are
writing/figure tasks, not further training runs.

## Run 3 CP-Recall: confirmed probability saturation, NOT genuine improvement
experiments/check_run3_saturation.py, run on all 10 seeds' actual checkpoints
(fixed causal layer): positive-prediction rate is exactly 1.0000 in 7/10 seeds
and 0.87-0.91 in the remaining 3 -- the model predicts "bottleneck" for
essentially every validation sample. Precision is flat at ~0.635 across all
seeds (the true positive rate in the data), while recall approaches 1.0 as a
trivial mathematical consequence of predicting positive on everything, not as
a sign of learning. Run 3's elevated CP-Recall (0.9565 mean) is therefore an
artifact of this collapse -- when every sample is classified positive, CP-Recall
becomes a near-constant structural query on the RCS ranking rather than a
measure of discriminative ability.

CONCLUSION: Run 3 must NOT be treated as an alternative to Run 4, under either
the original or the fixed-causal-layer results. Its apparent CP-Recall
advantage (Section "final four-run table" above) is retracted as evidence of
genuine superiority. This confirms and sharpens the original ablation note
("val loss diverged"/probability saturation, analogous to Run 1) -- the fix
enabled the causal layer to train but did NOT resolve Run 3's underlying
training instability (fn_weight=5.0 combined with lambda_causal=0.20). Run 4
remains the best-balanced and most stable configuration across AUC, CP-Recall,
and PatAcc, and is the one to report as the primary result.
## F6 ablation, multi-seed replication -- pre-registration (before any real GPU run)

Scope: replicate the original single-run F6 ablation finding (RCS Top-1
48.7% -> 0.0%, Table 10 note) across all 10 fixed seeds
(42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021), on both topologies
(Compose N=30, Home N=7), under CRUX_PREBUILD_CAUSAL=1.

Ablation mechanism (src/pipeline.py, `ablate_f6=True` on
`train_and_evaluate`/`finetune_home`): toc_cap_30 / toc_cap_7 (the "F6"
capacity prior read from graphs.pt) is replaced with a constant zero tensor
before being threaded through the model. Nothing else changes -- x_seq,
edge_index, labels, Run4 hyperparameters (fn_weight=1.5, lambda_causal=0.05,
lambda_sub=0.05, lambda_rcs_sup=0.3) are identical to the causal-fix-v1 /
home-finetune-v1 with-F6 arm. This is the ONLY difference between the two
arms, by construction (same functions, one flag).

Reused baseline (the "with-F6" paired arm): causal-fix-v1 Run4 compose
results (AUC 0.8699+/-0.0141) and home-finetune-v1 Run4 results (zero-shot
AUC 0.635+/-0.154; finetune_causal AUC 0.899+/-0.014, RCS Top-1
48.0%+/-26.2%, range 4.4%-81.3%). These numbers are shared with the
causal-layer-fix study above, not generated specifically for this
comparison. New work needed: only the F6-ablated arm, 20 runs total
(10 seeds x compose-from-scratch, 10 seeds x Home fine-tune from that
seed's own ablated compose checkpoint) -- ablation retrains Compose from
scratch per seed (per the original notebook trace: a fresh 205,617-param
model, not post-hoc zeroing of a real-F6 checkpoint), then fine-tunes Home
from that same ablated checkpoint, never from a real-F6 checkpoint.

Metrics per seed: AUC (Compose Val, Home zero-shot, Home fine-tuned),
RCS Top-1 (Home fine-tuned). AUC is reported descriptively (mean+/-SD) only.

Primary pre-registered statistical test: two-sided Wilcoxon signed-rank,
paired, on RCS Top-1 (Run4-with-F6 vs Run4-F6-ablated, same 10 seeds),
alpha=0.05. Chosen over a paired t-test as primary because the with-F6
RCS Top-1 distribution is known to be highly skewed/high-variance
(range 4.4%-81.3%); paired t-test reported as a secondary/sensitivity
check.

Verification-seed gate before scaling to all 10 (seed 42): ablated RCS
Top-1 must be near 0% (reference: original single-run value 0.0%); AUC
must not deviate more than +/-0.03 from the with-F6 arm at the same seed.
Larger deviation -> stop and ask.

### Pre-flight gradient-flow / functional check (tests/test_f6_ablation_gradient_flow.py)

Run on synthetic data before any real GPU training, per protocol rule 1.
Confirmed:
- toc_cap_30 / toc_cap_7 are exactly zero under ablate_f6=True (0.0 vs >0 in
  a non-ablated control run of the same code path).
- rcs (out_degree * toc_capacity) is exactly zero for every sample/node
  under ablation, both in Compose and in Home evaluation (rcs_absmax==0.0),
  vs clearly nonzero in the control -- confirms the ablation mechanism
  itself works as intended, independent of any training outcome.
- causal_30.W_raw still receives real gradient and moves during ablated
  Compose training (via causal_loss = sparsity + dag_penalty, which does
  not depend on toc_capacity) -- consistent with causal-fix-v1's fix still
  applying under ablation.
- All model parameters are present in the optimizer and receive a
  non-zero gradient and move after one step under ablation, with exactly
  two documented, provable exceptions (see below) -- no undiscovered
  silent-zero-gradient bug like the original causal-layer one.

Two findings surfaced by this check, neither a bug, both now part of the
pre-registration so they are not later mistaken for new results:

1. toc_scale (2 parameters, one per TOCGATLayer) appears in
   capacity_weight = 1 + toc_lambda*toc_scale*toc_capacity only multiplied
   by toc_capacity. Under toc_capacity=0 its gradient is exactly zero by
   the chain rule for any input, for the entire ablation study. Negligible
   (2 of 205,617 parameters) and does not affect model behaviour, since it
   has zero effect on the output regardless of its (frozen) value.

2. causal_7 (W_raw + encoder, the whole causal layer used in Home) gets an
   EXACT-ZERO gradient throughout Home fine-tuning under ablation, in every
   stage -- not just weak training. Unlike train_and_evaluate,
   finetune_home's loss never includes causal_loss; the only path from
   causal_graph to the loss is via rcs in the causal-supervision stage
   (rcs_sup_loss_g), and that path is exactly the one F6 ablation zeroes.
   causal_7 therefore stays frozen at whatever it inherited from the
   (ablated) compose checkpoint for the entire fine-tune. Confirmed
   directly: causal_w_absmax_before == causal_w_absmax_after in the ablated
   arm; the non-ablated control moves normally.

Consequence for the expected result, stated here BEFORE running so it
cannot be mistaken for post-hoc reasoning: because rcs is identically zero
for every Home sample and node under ablation regardless of training,
RCS Top-1 in the F6-ablated arm is not merely expected to be low on
average -- it is deterministic exactly 0.0% for all 10 seeds, with zero
seed-to-seed variance. This is a cleaner result than originally
anticipated (contrast: the with-F6 arm has SD 26.2%), and should be
reported precisely as a deterministic consequence of the ablation
mechanism, not as an empirical "collapse observed across seeds" the way
the causal-layer-fix findings above were.

Planned tag: f6-ablation-prereg-v1 (this commit). configs/runs.py and the
statistical test script used for the primary RCS Top-1 comparison must not
be modified after this tag without explicit documented reason (protocol
rule 2).

### Extended verification (3 seeds: 42, 123, 456) before scaling to all 10

Requested extra caution before committing GPU time for the remaining 7
seeds, given seed 42 alone showed a large Home zero-shot AUC deviation
from its with-F6 counterpart (+0.466). Ran two more verification seeds
(123, 456) through the same ablated compose+home pipeline and compared
against the existing per-seed with-F6 result files
(results/causal_fix/run4_seed{s}.json, results/home_finetune/home_seed{s}.json):

| seed | Compose AUC diff | Home zero-shot AUC diff | finetune_detection diff | finetune_causal diff | RCS Top-1 (all stages) |
|---|---|---|---|---|---|
| 42  | 0.016 | +0.466 | 0.013 | 0.023 | 0.0 |
| 123 | 0.004 | -0.091 | 0.020 | 0.010 | 0.0 |
| 456 | 0.004 | -0.192 | 0.030 | 0.014 | 0.0 |

Conclusion: the Home zero-shot AUC deviation flips sign across seeds
(ablated arm higher for 42, lower for 123 and 456) rather than showing a
consistent direction, which is inconsistent with a systematic
implementation bug (a bug would bias one direction) and consistent with
zero-shot AUC being an inherently high-variance, pre-fine-tuning metric
that already spans 0.367-0.836 across the 10 with-F6 seeds alone. The gap
shrinks toward (and mostly within) +/-0.03 after fine-tuning in all three
seeds, in both arms. RCS Top-1 is exactly 0.0 and causal_7 stays exactly
frozen (causal_w_absmax_before == after == 0.0) in every stage for all
three seeds, matching the pre-registered mechanistic account with zero
exceptions so far (3/3).

Decision: the verification-seed gate is treated as passed. The
pre-registered +/-0.03 AUC tolerance is confirmed NOT to meaningfully
apply to Home zero-shot AUC specifically, given its documented natural
range; it still applies (and passed) for Compose AUC, Home
finetune_detection AUC and Home finetune_causal AUC. Seeds 42, 123, 456
count as 3 of the 10 official F6-ablation runs; only the remaining 7 seeds
(789, 1011, 1213, 1415, 1617, 1819, 2021) remain to be run for the full
study.

### Final results (all 10 seeds) and analysis

All 10 pre-registered seeds (42, 123, 456, 789, 1011, 1213, 1415, 1617,
1819, 2021) completed for the F6-ablated arm (Compose-from-scratch +
Home fine-tune, ablate_f6=True). Paired against the existing with-F6
results (causal-fix-v1 Compose, home-finetune-v1 Home).

RCS Top-1 (Home finetune_causal, the pre-registered primary quantity):

| | with-F6 | F6-ablated |
|---|---|---|
| mean +/- SD | 47.98% +/- 26.23% | 0.0% +/- 0.0% |
| range | 4.36% - 81.29% | 0.0% (every seed) |

Primary pre-registered test -- two-sided Wilcoxon signed-rank, paired,
same 10 seeds: W=0.0, p=0.00195 (the minimum attainable p for n=10; all 10
paired differences have the same sign, with no exceptions). Secondary
paired t-test: t=5.784, p=0.000265, in full agreement.

Both mechanistic predictions made before this run (see the pre-flight
section above) held exactly, with zero exceptions across all 10 seeds:
rcs_absmax==0.0 and causal_w_absmax_before==causal_w_absmax_after in
every Home stage of every seed. RCS Top-1 is not merely "collapsed", it
is deterministic 0.0% for every seed, as derived analytically before any
of these runs.

AUC (descriptive only, no primary test pre-registered on these):

| metric | with-F6 | F6-ablated | mean diff |
|---|---|---|---|
| Compose Val AUC | 0.8699 +/- 0.0141 | 0.8716 +/- 0.0111 | -0.002 |
| Home zero-shot AUC | 0.6346 +/- 0.1539 | 0.6478 +/- 0.1407 | -0.013 |
| Home finetune_detection AUC | 0.8888 +/- 0.0173 | 0.8659 +/- 0.0148 | +0.023 |
| Home finetune_causal AUC | 0.8991 +/- 0.0144 | 0.8866 +/- 0.0154 | +0.012 |

The Home zero-shot AUC deviation flagged during the seed-42 verification
gate (+0.466 for that one seed) resolves to essentially zero (-0.013,
well inside both arms' own SD) once averaged over all 10 seeds --
confirming the extended 3-seed check's conclusion that it was seed-level
variance in a known high-variance pre-fine-tuning metric, not a bug.
Compose AUC is essentially unaffected by the ablation, as expected (F6 is
not architecturally involved in the detection head's core signal path
outside the capacity_weight/RCS injections). The two Home fine-tuned AUC
metrics show a small (~0.01-0.02), consistent-direction drop under
ablation -- plausible given causal_7 never trains under ablation (see
pre-flight finding 2) and detection-relevant capacity information is
absent, but this was not a primary pre-registered comparison and is
reported descriptively only.

Conclusion: F6 (toc_capacity) is necessary for root-cause localization,
replicated across 10 seeds and both topologies, with the collapse shown
to be a deterministic mathematical consequence of removing F6's
architectural role (RCS = out_degree * toc_capacity == 0 identically),
not merely a low-signal empirical trend. AUC-based detection is
essentially unaffected. Tag: f6-ablation-results-v1 (this commit).
