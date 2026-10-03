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

## GNN baseline (use_toc=False), reviewer-required comparison -- pre-registration (before any real GPU run)

Scope: implement the "standard GNN baseline" required by the reviewer
report item 2 ("Missing Baseline Comparisons") -- GAT+TFT with no TOC
components, trained on the same Compose (52-file) dataset and features as
Run 4, evaluated with the same metrics as every other method in the
comparison table: AUC, CP-Recall, RCS Top-1, PatAcc.

Ablation mechanism (src/core.py, src/pipeline.py `use_toc=False` on
`ModelConfig`/`train_and_evaluate`): architectural removal, not soft
input-zeroing --
- No F6 injection in TOCGATLayer: no `toc_scale` parameter exists at all
  (not created, not merely inert), `capacity_weight` is never computed,
  `x_toc = x` unchanged.
- No RCS multiplier in CausalInferenceLayer: `rcs = out_degree` (the raw
  causal out-degree), not `out_degree * toc_capacity`.
- No L_rcs: `loss_fn = _TOCWeightedLossBase` directly -- the RCS-
  supervision term is architecturally absent from the loss graph, not
  zero-weighted.
- L_sub zeroed via `lambda_sub=0.0` by configuration, matching the
  existing Run1-4 convention of zeroing loss coefficients for ablation
  rather than branching the loss class for this term specifically.
- L_cause (the causal layer: W_raw, encoder, causal_loss) is deliberately
  UNTOUCHED -- it is a separate structural-correlation-discovery
  mechanism, not one of TOC's Five Focusing Steps, and was not named in
  the reviewer's or the requested ablation list.

Explicitly distinct from the already-closed F6-ablation study
(`ablate_f6=True`): that ablation zeroes the toc_capacity *input* while
keeping the architecture (and its `toc_scale` parameters) intact. This
ablation removes the architecture itself and has a different (smaller)
parameter count as a direct, verified consequence. `train_and_evaluate`
now raises `ValueError` if both flags are passed together, so the two
research questions cannot be silently conflated in one run.

Config (matches Run4 causal-fix-v1 sizing/coefficients except where the
ablation itself changes something): fn_weight=1.5, fp_weight=1.0,
gat_hidden=32, tft_hidden=64, lambda_causal=0.05, lambda_sub=0.0,
CRUX_PREBUILD_CAUSAL=1 (same causal-fix convention as the Run4 numbers
this baseline is compared against).

expected_params = 205,615 -- verified programmatically
(tests/test_gnn_baseline_gradient_flow.py::check_param_count_delta:
205,617 [use_toc=True control] - 2 [one toc_scale scalar per TOCGATLayer,
gat_layers=2] = 205,615), not assumed or copied from Run4's count.

New metric -- RCS Top-1 for Compose (N=30): added to
`TOCEvaluator.compute()` on 2026-09-27. This metric never existed for
Compose in any prior run (Run1-4 only ever reported AUC, CP-Recall,
PatAcc for Compose; RCS Top-1 previously existed only for Home, with a
fixed flagged-node set {3,4}). Definition mirrors
`rcs_supervision_loss`'s exclusion rule exactly: for each positive
(bottleneck) sample, look up its pattern's flagged nodes via
`pattern_idx -> IDX_TO_PATTERN -> PATTERNS`, and check whether the single
argmax-RCS node falls in that set; samples whose pattern has no flagged
nodes ('none') or whose flagged set covers every node are excluded from
both numerator and denominator. Verified against hand-constructed
synthetic inputs with a known-by-construction answer (2 eligible samples,
1 correct -> 50.0%) in
`tests/test_gnn_baseline_gradient_flow.py::test_rcs_top1_metric_correctness`,
since a brand-new metric must be checked before being trusted, not only
integration-tested.

Required follow-up (separate action item, NOT done yet): RCS Top-1 must
also be computed retroactively on the existing Run4 (causal-fix-v1)
checkpoints/logs for a like-for-like comparison -- Run4 was never
evaluated with this metric at the time it was originally run.

Two findings surfaced while designing this ablation, neither invalidating
prior results, both logged here so they are not later mistaken for new
findings:
1. The closed F6-ablation study's gradient-flow test checks
   `p.grad.abs().max()` per whole parameter tensor, not per element. Under
   `ablate_f6=True`, rcs is identically zero for every sample, so the
   `rcs_summary` (6 dims) feeding every prediction head is a constant
   zero vector -- meaning the corresponding 6 columns of
   `head_bn.0.weight`, `head_pattern.0.weight`, `head_ttb.0.weight` are
   provably dead (exact-zero gradient) under that ablation, but this
   passed undetected because the rest of each weight tensor is nonzero.
   This does not affect the F6-ablation study's reported conclusions
   (the RCS Top-1 collapse mechanism and finding are independent of it,
   and dead weight columns cannot help or hurt detection performance
   since their contribution is exactly zero regardless of value) -- it
   is a documentation gap in test granularity, not a result-invalidating
   bug. Not re-opened as active work.
2. This baseline's own architectural ablation does not reproduce that
   issue: rcs = out_degree here is a real, sample-varying signal (see
   pre-flight check below), so no head weight columns are constant-zero-
   starved in this run.

Seeds: same 10 fixed seeds (42, 123, 456, 789, 1011, 1213, 1415, 1617,
1819, 2021).

Primary pre-registered statistical test: two-sided paired t-test,
GNN-baseline vs Run4 (causal-fix-v1), on CP-Recall, alpha=0.05 -- same
convention as the original Run2-vs-Run4 primary test. Wilcoxon
signed-rank as secondary/sensitivity, given CP-Recall's already-documented
non-normal, high-variance behaviour across these studies.

Verification-seed gate (seed 42) before scaling to all 10: no specific
per-seed causal-fix-v1 Run4 number is on file to compare against (only
the 10-seed aggregate, CP-Recall 0.9053+/-0.0337, AUC 0.8699+/-0.0141) --
so the gate is: seed-42 AUC and CP-Recall must be plausible (AUC clearly
above the ~0.5 random-detector floor, CP-Recall not identically 0 or 1
across all validation samples, no NaNs) and within roughly 3 SD of the
Run4 aggregate on each metric. Larger or unexplained deviation -> stop
and ask, per protocol rule 3.

### Pre-flight gradient-flow / functional / metric check (tests/test_gnn_baseline_gradient_flow.py)

Run on synthetic data before any real GPU training, per protocol rule 1.
All four parts passed:
1. With use_toc=False: no `toc_scale` parameter exists at all (0 found,
   confirmed by name suffix); all 103 remaining parameters are in the
   optimizer, receive non-zero gradient, and move after one
   `optimizer.step()` -- no exceptions, unlike the F6-ablation study's
   provably-inert `toc_scale`. `rcs.abs().max() > 0` confirmed (a real
   signal, not zeroed) and `_TOCWeightedLossBase`'s loss dict has no
   `rcs_supervision` key (L_rcs architecturally absent).
2. Parameter-count delta computed programmatically: 205,617 (use_toc=True
   control) - 205,615 (use_toc=False) = 2, exactly matching
   `gat_layers=2` toc_scale scalars.
3. Real `pipeline.train_and_evaluate(use_toc=False)` on a tiny fake
   Compose cache: `causal_w_absmax > 0` confirmed (L_cause trains
   normally, as expected since it is untouched by this ablation); the
   `ablate_f6=True` + `use_toc=False` combination correctly raises
   `ValueError`.
4. `rcs_top1` metric verified against hand-constructed synthetic inputs
   with a known answer (2 eligible samples of 4, 1 correct -> exactly
   50.0%), confirming both the correctness of the metric and its
   exclusion rule for 'none'-pattern and non-bottleneck samples.

Planned tag: gnn-baseline-prereg (this commit). configs/runs.py and the
primary statistical test script must not be modified after this tag
without explicit documented reason (protocol rule 2).

## CP-Recall saturation finding -- documented limitation (no re-opening of prior statistical work)

While comparing the GNN baseline (use_toc=False, all 10 seeds) against
Run4 (causal-fix-v1) on CP-Recall, the primary pre-registered metric for
that comparison, the following was discovered and verified analytically:

`CRITICAL_PATH` covers 18 of 30 Compose nodes (60%). A uniformly random
top-3 node selection therefore intersects `CRITICAL_PATH` with probability
1 - C(12,3)/C(30,3) = 0.9458 by the hypergeometric distribution -- this
exact value matches `baseline_exact` already recorded in
`results/causal_fix/causal_fix_all4_stats.json`, confirming the
calculation, not merely estimating it.

Consequence: CP-Recall as currently defined (top-3 intersection with an
18/30-node set) is near-saturated. Three of the four original thesis
configurations score AT OR BELOW this random baseline (run1: 0.8607,
run2: 0.8376, run4: 0.9053; only run3: 0.9565 exceeds it), and the new
GNN baseline (0.9615+/-0.0242, all 10 seeds) also exceeds it and exceeds
Run4's mean (paired t(9)=-6.04, p=0.000194, Wilcoxon p=0.00195, Run4
higher in 0/10 seeds).

This comparison (GNN-baseline vs Run4 on CP-Recall) is therefore NOT
treated as evidence that TOC components fail to help, or that the
GNN baseline is architecturally superior. With a near-ceiling random
baseline, a large fraction of the achievable range on this metric is
attributable to chance regardless of model quality, so a difference of
this size cannot be read as a clean measure of TOC's contribution. The
CP-Recall column in any comparison table involving this baseline MUST be
reported alongside the random-baseline value (0.9458) and this caveat, not
as a standalone superiority claim in either direction.

Scope of this finding, explicitly bounded: this note documents a property
of the CP-Recall metric surfaced during the GNN-baseline comparison. It
does NOT reopen, retest, or reinterpret the previously closed
`multiseed-v1` / `causal-fix-v1` statistical work (the Run4-vs-Run2
primary test on CP-Recall, p=0.0007 faithful / p=0.0216 causal-fix). That
work remains closed as recorded. Revisiting it would require its own
separate, deliberate decision and is out of scope here.

Forward-looking implication for the reviewer-required baseline comparison
table specifically: RCS Top-1 (added 2026-09-27, see the GNN-baseline
pre-registration entry above) is NOT subject to this saturation -- it
varies meaningfully across the 10 baseline seeds (range 34-78%, roughly),
giving it real discriminative power. RCS Top-1 and AUC should therefore
carry more interpretive weight than CP-Recall in that specific
GNN-baseline-vs-Run4 comparison; CP-Recall is included for completeness
and comparability with the reviewer's requested metric set, not as the
deciding metric.

## Logistic regression baseline (F0-F6) -- GAMMA [4] infeasibility + design + results

### GAMMA [4] infeasibility (pre-registration for the fallback)

Attempted first, per reviewer report item 2: GAMMA (Somashekar et al.,
WWW'24), reimplementation or published code. Findings, in order:

1. GAMMA's published code (github.com/PACELab/GAMMA, model/models.py)
   implements 5 fixed binary localizer heads, while the paper (Section
   3.3, Appendix B) describes a per-service mixture-of-experts localizer
   -- the two do not match.
2. The published training script (model/train.py) re-instantiates
   `BaseModel` immediately before `evaluate()`, discarding the trained
   weights and evaluating a freshly-initialized (untrained) model.
   `collate()` also passes the reshaped latency tensor twice, feeding it
   into the CPU-usage branch. `localizer3..5` train against `local1`/
   `local2` targets, and `localizer5` uses `localizer2`'s loss criterion.
3. GAMMA requires per-node CPU, memory, and network tx/rx (5 raw
   modalities), not our engineered F0-F6. Storage constraints in this
   environment previously blocked uploading the full 52-file Compose
   feature set at this granularity (this is why our own F0-F6 pipeline
   processes one file at a time); only 3 of 52 Compose files exist with
   the required multi-modal features
   (processed_dataset/compose/multi-modal-data-separate/).
4. Inspecting those 3 files: the per-node `{n}_label_RPC` target is
   static across every temporal window within a file -- file 1 has
   exactly one unique (node-set) pattern across all 257 windows, file 2
   is all-zero (no bottleneck ever labeled), file 3 has 2 unique
   patterns. This indicates the available multi-modal labels encode a
   per-file/per-injection-point property, not a per-window localization
   signal, independent of the file-count shortfall -- so extracting the
   remaining 49 files would not by itself resolve the localization-target
   problem for GAMMA's architecture.

Conclusion: full reimplementation is infeasible here for reasons
independent of effort invested (data availability and target validity,
not merely time). Per the reviewer's own stated fallback ("If
reimplementation is infeasible, provide a detailed justification and
compare against a simple statistical baseline"), we substitute logistic
regression on F0-F6.

### Design (pre-registered before running; deterministic, no seed sweep)

- Detection & PatAcc: one row per window. Features = F0-F6 for all 30
  nodes, time-averaged over the 12-step window (not last-timestep), then
  flattened (210-dim). Detection: binary logistic regression,
  class_weight="balanced", target = label. PatAcc: multinomial logistic
  regression, target = pattern_idx (8 classes).
- Localization (CP-Recall, RCS Top-1): one row per (node, window) pair,
  restricted to positive (label=1) windows whose pattern has a non-empty,
  non-full-coverage flagged-node set -- the same exclusion rule as
  `rcs_supervision_loss` and the `rcs_top1` metric. Features = F0-F6 for
  that node only (time-averaged). Target = 1 if that node is in the
  window's pattern's flagged set. A single shared classifier (not one per
  node) is trained across all eligible (node, window) rows. At
  evaluation, per-node probabilities for a window rank nodes for CP-Recall
  (top-3 vs CRITICAL_PATH) and RCS Top-1 (argmax vs flagged set) exactly
  as for the GNN-based methods.
- Train/eval split: existing dataset_cache/{train,val}.pt (same split as
  Run4 and the GNN baseline). No held-out selection is performed (no
  hyperparameters to tune), so val is used directly for the reported
  numbers, matching how other rows in this comparison report best-val
  metrics.
- All three logistic regressions use StandardScaler -> LogisticRegression
  (max_iter=5000) in a Pipeline; an earlier unscaled run produced
  documented lbfgs non-convergence warnings and is discarded in favor of
  this scaled version.
- Deterministic (convex optimization, no random seed), so this row is
  reported as a single number, not mean+/-SD, unlike the seeded rows in
  this table -- documented explicitly as a methodological difference, not
  an omission.

### CP-Recall diagnostic (checked before reporting, per protocol rule 4)

CP-Recall (0.453) is below the 0.9458 random baseline. Two hypotheses
were checked and ruled out/in before accepting the number:
- Hypothesis A (pattern-flagged-set vs CRITICAL_PATH mismatch): checked
  and REJECTED -- overlap between each pattern's flagged set and
  CRITICAL_PATH is 75-100% for 6 of 7 patterns (only G at 50%), so a
  classifier correctly solving its actual training target would not be
  structurally penalized on CP-Recall.
- Hypothesis B (classifier degenerately over-selects a small fixed node
  subset never flagged in any pattern): CONFIRMED. Per-pattern CP-Recall
  breakdown: A 21/152 (0.138), B 0/49 (0.000), C 62/67 (0.925), D 42/42
  (1.000), E 14/14 (1.000), F 14/14 (1.000) -- patterns A and B (60% of
  positive val windows combined) drive the low aggregate; the rest are
  near-perfect. Nodes 16 and 17 alone account for 199 and 198 of the
  top-3 selections across <=338 eligible windows, are never in ANY
  pattern's flagged set, and have low, low-variance raw F0/F5 values
  (mean ~5.5-7.6) versus other nodes (e.g. node 0: mean ~206, max ~3912).
  This is a plausible, non-buggy consequence of training independent
  per-node rows with no shared context across a window's other nodes and
  no explicit pattern input -- a structural limitation of this simple
  baseline design, not an artifact requiring correction. Not investigated
  further, consistent with this being the reviewer-suggested minimal
  fallback baseline, not a method to be optimized.

### Results (val set, single deterministic run)

AUC=0.7120, CP-Recall=0.4527 (n=338, see diagnostic above), PatAcc=0.8816,
RCS Top-1=41.72% (n=338).

## RCS Top-1 random-baseline check + Run4 retraining pre-registration (to close the RCS Top-1 gap)

### RCS Top-1 random-baseline check (GNN baseline, use_toc=False)

Unlike CP-Recall, RCS Top-1's chance level is pattern-dependent (each
pattern's flagged-node set size k gives random-argmax odds k/30, ranging
6.7% for G to 63% for E). The random baseline weighted by the actual
positive-val pattern distribution (A:152, C:67, B:49, D:42, E:14, F:14) is
38.2%. One-sample t-test, GNN-baseline's 10 per-seed RCS Top-1 values vs
38.2%: t(9)=4.969, p=0.00077, mean difference +28.2 points. RCS Top-1 is
therefore NOT saturated for this baseline -- the metric has real
discriminative power here, unlike CP-Recall. This confirms the tool is
valid; it does not by itself demonstrate any TOC-specific advantage,
since no Run4 RCS Top-1 value exists yet to compare against.

### Run4 retraining pre-registration (to obtain the missing RCS Top-1 value)

Scope: Run4's original causal-fix-v1 training (10 seeds) predates the
rcs_top1 metric (added 2026-09-27 in the GNN-baseline work) and no
per-seed checkpoints were retained, so RCS Top-1 cannot be computed
retroactively. This retrains Run4 from scratch, seed-for-seed identical
to causal-fix-v1 in every respect (config, PREBUILD_CAUSAL=1), for the
sole purpose of obtaining RCS Top-1 alongside AUC/CP-Recall/PatAcc as a
built-in side effect of the already-merged rcs_top1 code in
TOCEvaluator.compute() (no code changes needed for this run).

Config (identical to configs/runs.py RUNS["run4"], use_toc=True default):
gat_hidden=32, tft_hidden=64, gat_dropout=0.2, tft_dropout=0.2,
weight_decay=1e-3, lr=5e-4, patience=12, epochs=60, fn_weight=1.5,
fp_weight=1.0, lambda_causal=0.05, lambda_sub=0.05, lambda_rcs_sup=0.3,
expected_params=205617, CRUX_PREBUILD_CAUSAL=1.

Seeds: same 10 fixed seeds (42, 123, 456, 789, 1011, 1213, 1415, 1617,
1819, 2021).

Verification-seed gate (seed 42) before scaling to 10: since this is a
seed-for-seed reproduction of an already-completed experiment (not a new
ablation), the bar is stricter than the 3-SD band used for the
GNN-baseline: AUC and CP-Recall should closely match the recorded
run4_seed42.json values (auc=0.8703, cp_recall=0.8757) -- large deviation
here would indicate a training-pipeline regression since causal-fix-v1,
not sampling noise, and must stop and be investigated before scaling up.

Primary pre-registered tests once both arms exist: (1) paired t-test,
Run4-retrain vs GNN-baseline, on RCS Top-1, alpha=0.05, same 10 seeds,
Wilcoxon as sensitivity; (2) one-sample t-test, Run4-retrain's RCS Top-1
vs the 38.2% weighted random baseline, to confirm Run4 is not itself
saturated on this metric before any TOC-specific claim is drawn from
test (1).

## Run4 retraining complete -- closing result: no TOC advantage on RCS Top-1 either

Run4 retrained seed-for-seed identical to causal-fix-v1 (10 seeds,
use_toc=True, same config), for the sole purpose of obtaining RCS Top-1
(unavailable retroactively). Seed-42 gate passed (AUC/CP-Recall closely
matched the original run4_seed42.json; n_params identical at 209,587).

### Run4-retrain results (10 seeds, mean +/- SD)

AUC=0.8705+/-0.0133, CP-Recall=0.9056+/-0.0389, PatAcc=0.9047+/-0.0281,
RCS Top-1=62.81+/-8.00% (n=338 eligible positive-val samples per seed).

### Pre-registered tests (both now computable; both run as planned)

1. One-sample t-test, Run4-retrain RCS Top-1 vs the 38.2% weighted random
   baseline: t(9)=9.733, p<0.00001, mean diff +24.6 points. Confirms RCS
   Top-1 is not saturated for Run4 either -- the metric is valid for both
   arms.
2. PRIMARY TEST for this comparison -- paired t-test, GNN-baseline
   (use_toc=False) vs Run4-retrain, on RCS Top-1, same 10 seeds: t(9)=0.918,
   p=0.38235 (NOT significant). Wilcoxon signed-rank W=18.0, p=0.375
   (NOT significant). Cohen's d_z=0.290 (small). GNN-baseline higher in
   7/10 seeds, Run4-retrain higher in 3/10.

### Combined conclusion across all baseline-comparison work in this section

On every metric with a validated, non-saturated chance level (AUC,
PatAcc, RCS Top-1), there is no statistically significant difference
between CruxSight (with TOC: L_sub, L_rcs, F6 injection, RCS multiplier)
and an architecturally identical GNN with all TOC components removed.
CP-Recall remains the sole excluded metric (saturated, see the
cp-recall-saturation-note entry above) and was never a valid basis for a
directional claim in either direction. This is the complete empirical
picture from this baseline-comparison effort, not a partial or
preliminary result awaiting a missing measurement -- the RCS Top-1 gap
that motivated this retraining was the last remaining metric that could
have shown a TOC-specific advantage, and it does not.

## Residual TOC signal found in the "no-TOC" GNN baseline -- fix and re-verification required

### Finding (credited: identified independently, verified jointly)

The GNN-baseline comparison (tags gnn-baseline-prereg, cp-recall-saturation-note,
run4-retrain-results-v1) ablated TOC at the architectural level (F6
injection, RCS multiplier, loss-class selection for L_rcs, lambda_sub=0)
but MISSED that `_TOCWeightedLossBase.detection_loss` -- shared by BOTH
`TOCWeightedLoss` (Run4) and `_TOCWeightedLossBase` (the "no-TOC"
baseline) -- independently builds `node_weights` from `toc.CRITICAL_PATH`
(weight 3.0 vs 1.0) and uses it to amplify the detection BCE loss for
missed (false-negative) bottlenecks via `amp = 1.0 + missed * rcs_crit`.
This path is NOT gated by the architectural use_toc flag at all, so the
"no-TOC" baseline was still receiving TOC-derived training signal through
its detection loss.

Verified in two stages before any code change, per protocol rule 4:
1. Synthetic check (NumPy, hand-computed): same predictions, only
   weights varied (CRITICAL_PATH vs uniform) -> loss changed
   (2.3365 vs 2.6864), gradient w.r.t. a tested service's score changed
   (0.6872 vs 0.0000); confirmed zero effect when the bottleneck IS
   detected (amp=1.0 regardless of weights, matching the code's `missed`
   gate exactly).
2. Real-batch check (16 real training-set samples, fresh CSTGNN(use_toc=False),
   single forward pass, torch.autograd.grad w.r.t. rcs, no optimizer step,
   no repo changes): loss 1.717820 (TOC-weighted) vs 1.717586 (neutral);
   max |grad difference| w.r.t. rcs = 0.099051 (nonzero) with 8/8 positive
   samples missed in that batch. Confirms the leak is real on actual data,
   not merely a theoretical code-reading concern.

### Fix

`_TOCWeightedLossBase.__init__` and `TOCWeightedLoss.__init__` now take a
`use_toc: bool = True` parameter; node_weights is all-ones when
use_toc=False (CRITICAL_PATH is never consulted), and unchanged (3.0 on
CRITICAL_PATH nodes) when use_toc=True, preserving Run4's exact prior
behavior. `pipeline.train_and_evaluate` passes its own `use_toc` argument
into the loss class constructor (previously only used it to pick the
class, not to configure it).

Verified post-fix: `_TOCWeightedLossBase(..., use_toc=False).node_weights`
is all 1.0 (assert passes); its `detection_loss` on the same real batch
and rcs tensor as the check above now gives a gradient w.r.t. rcs
IDENTICAL to the manually-neutralized computation to 8 decimal places
(max diff = 0.00000000). `TOCWeightedLoss(..., use_toc=True).node_weights`
still differs from all-ones (Run4's behavior is unchanged; assert passes).

### Consequence for prior results

All GNN-baseline results reported under tags gnn-baseline-prereg,
cp-recall-saturation-note, and the GNN-baseline side of
run4-retrain-results-v1 were trained against a baseline that still
received TOC-derived detection-loss guidance on missed bottlenecks. These
results are NOT retracted (the fix doesn't change what was measured, only
reveals a confound in what it was compared against), but the "no
significant difference" conclusion in those results cannot yet be
attributed to TOC's components being unnecessary -- it may equally be
explained by this residual signal. A verification seed (42) with the
corrected, fully-neutral baseline is pre-registered next; if it deviates
materially from the previously reported seed-42 baseline result
(auc=0.8756, cp_recall=0.9793, rcs_top1=65.09), full 10-seed retraining
of the GNN baseline is required before any of this section's conclusions
or the drafted paper text can be finalized.

## Fully-clean GNN baseline (10 seeds, loss leak fixed) -- final re-verification

All 10 seeds retrained with the use_toc=False fix (node_weights fully
neutralized, no CRITICAL_PATH influence anywhere in architecture or
loss). Compared against Run4-retrain (run4-retrain-results-v1), same 10
seeds, paired tests:

AUC:       clean 0.8737+/-0.0150 vs Run4 0.8705+/-0.0133; t(9)=0.600, p=0.5636 (NS)
CP-Recall: clean 0.9675+/-0.0576 vs Run4 0.9056+/-0.0389; t(9)=3.538, p=0.0063
           (significant, but EXCLUDED per cp-recall-saturation-note -- the
           clean baseline's CP-Recall is, if anything, higher than the
           already-contaminated version, reinforcing that this metric
           carries no reliable TOC-attributable signal)
RCS Top-1: clean 63.49+/-22.35 vs Run4 62.81+/-8.00; t(9)=0.127, p=0.9018 (NS)
PatAcc:    clean 0.8865+/-0.0237 vs Run4 0.9047+/-0.0281; t(9)=-1.742, p=0.1156 (NS)

Conclusion: fixing the loss leak does NOT change the study's conclusion
on any valid metric -- AUC, RCS Top-1, and PatAcc remain not
significantly different between CruxSight and a fully TOC-free GNN
baseline (RCS Top-1's p-value and effect size are in fact closer to null
than before the fix: p=0.38->0.90, d_z=0.29->0.04). This rules out the
possibility that the original "no difference" finding was an artifact of
the now-fixed residual TOC signal. The negative result for TOC's
loss-derived components (L_sub, L_rcs, F6/RCS mechanisms) stands,
re-verified under a fully clean ablation. gnn-baseline-prereg's and
run4-retrain-results-v1's substantive conclusions are CONFIRMED, not
superseded, by this re-verification.

## TTB-head evaluation (reviewer item: "Missing Evaluation of the TTB Head") -- pre-registration (before any real GPU run)

Date: 2026-10-02. Scope: the TTB (Time-to-Bottleneck) head is listed as a
contribution (Table 4) but no TTB result exists anywhere: `TOCEvaluator`
stores `ttb_pred`/`ttb_true` but computes no error, and
`train_and_evaluate` returns no TTB value. This study evaluates the head on
the validation split and reports detection-based alert lead time on the
validation files, with a figure of the lead-time distribution.

### Facts established from the code and the cache before pre-registering

- Target definition (Colab cell 7c): `ttb = bn_steps * STEP_SEC / 60`
  minutes, STEP_SEC=10, HORIZON=6, `y_fut = labels[end:end+HORIZON]`
  (starts at the first step after the window). `ttb = 0` means the
  bottleneck is at the first step after the window. Representable targets
  are therefore 0, 10, 20, 30, 40, 50 s. Windows without a bottleneck in
  the horizon carry the sentinel -1 step (-1/6 min).
- Training: `ttb_loss` = Huber(delta=1) on `ttb_pred*mask` vs
  `ttb*mask`, mask = TRUE label, lambda_ttb=0.3. On targets <= 0.84 min the
  Huber is in its quadratic region, i.e. the head is trained for squared
  error. `head_ttb` is built in `__init__` (not lazily).
- Cache (printed by the pre-flight inspection): train 1673 windows / 1104
  positive, positives by k=0..5: 849/61/60/55/42/37; val 532 / 338
  positive, k=0..5: 262/17/17/16/14/12 (77% of positives have k=0). Window
  stride is 1 and cache order preserves it; file boundaries recovered from
  window overlap give 35 train files and 9 val files (consistent with 44
  compose files and the `idx % 5 == 0` rule). Val files are disjoint from
  train at file level. There are no exact duplicate windows between train
  and val.
- The earlier 164 s +/- 55.5 figure (110, 240, 110, 170, 190 s on five
  Pattern A files) is NOT a TTB-head result: it was measured manually
  from `bn_logit` (sigmoid > 0.5, stable detection until onset), on a
  separate held-out set, from an unseeded single run. It is a preliminary
  estimate (rule 5), is not reproducible from this cache, and must not be
  reported as TTB-head lead time. It also exceeds the TTB head's
  representable range (50 s).
- Any alert issued more than 50 s ((HORIZON-1)*STEP_SEC) before onset is,
  by the label definition, an alert in a label-negative window
  ("premature"). Lead times above 50 s are therefore by construction
  detections the training labels do not ask for.

### Design

Config: identical to `configs/runs.py` RUNS["run4"] (use_toc=True,
fn_weight=1.5, fp_weight=1.0, lambda_causal=0.05, lambda_sub=0.05,
lambda_rcs_sup=0.3, gat_hidden=32, tft_hidden=64, dropouts 0.2,
weight_decay=1e-3, lr=5e-4, patience=12, epochs=60, expected_params=205617),
`CRUX_PREBUILD_CAUSAL=1`, retrained from scratch, training code untouched.
Seeds (10): 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021. The best
checkpoint (epoch with best val AUC, as in every other table) is exported
post hoc by `src/ttb_export.py`, which re-evaluates `best_model.pt` on val in
cache order and aborts unless its AUC equals the checkpoint AUC.

Metric 1 -- window-level TTB error (the reviewer's MAE/RMSE). Scored set:
val windows with TRUE label==1 (n=338 per seed; sentinel never scored; the
code raises if a negative target appears in the scored set). Reported in
seconds as mean +/- SD over seeds: MAE, RMSE, bias, Spearman(pred, true),
prediction SD, per-k MAE/RMSE, and the same on the alerted subset
(label==1 AND p>0.5).

Constant baselines (train positives only): mean = 0.1055 min (6.33 s,
RMSE-optimal) and median = 0 (MAE-optimal). Values on val computed from the
printed counts: mean constant RMSE 13.50 s, MAE 9.85 s; median constant
MAE 6.36 s, RMSE 14.93 s.

PRIMARY test (not to be changed after the tag): two-sided one-sample t-test,
alpha=0.05, of the 10 per-seed val RMSEs vs the RMSE of the train-mean
constant (the constant has no seed variance, so this equals a paired test).
Wilcoxon signed-rank as sensitivity. RMSE is primary because it is the
metric the head is trained for.

SECONDARY (reported, no multiplicity correction): MAE vs the train-median
constant (one-sample t); Spearman (note 77% ties at k=0); per-k breakdown;
alerted subset; file-level bootstrap (9 val files resampled with
replacement, 10,000 draws, seed 0) of RMSE_model - RMSE_constant,
descriptive only. Limitation declared in advance: the primary test measures
seed-to-seed variance only; the val set has 9 files and 338 strongly
autocorrelated stride-1 windows, so seed significance does not establish
generalisation across files.

Artifact checks (rule 4), always reported: head collapse (prediction SD <
5% of target SD, per seed, count of collapsed seeds); Spearman next to MAE;
comparison with both constants. A model that is not better than the constant
is reported as such.

Metric 2 -- detection-based alert lead time per val file (separate from
the TTB head and described as such). Per file: first bottleneck onset from
the window labels (`onset_idx = i0 + k[i0]`, verified consistent over all
positive windows up to the onset); stable alert = earliest window a such that
P(bottleneck) > 0.5 for every window from a to the last window before onset;
lead = (onset_idx - a) * 10 s. Threshold fixed at 0.5, never tuned. Files are
excluded and counted when: no bottleneck, first positive window is window 0
(onset not identifiable), or inconsistent labels. Always reported: files
excluded, files missed (no stable alert), premature alerts (a < i0, i.e.
lead > (onset_idx - i0) steps, always including every lead above 50 s),
share of lead times above 50 s. No hypothesis test; mean +/- SD over seeds and
per-file means. Figure: `results/ttb_eval/ttb_lead_time_distribution.{png,pdf}`
(pooled file x seed histogram with the 50 s line, per-file mean +/- SD).
Declared caveat: val also selected the best epoch, so the numbers carry mild
optimism; this is not corrected statistically.

Wording constraint for the paper/thesis: metric 1 is "window-level
time-to-bottleneck error", not file-level warning time; metric 2 is
"detection-based alert lead time", with the premature/over-50 s share
stated. The two are never merged and 164 s is not cited as a result of this
study.

Pre-registered interpretation:
- Primary significant AND model RMSE < constant: report the RMSE difference
  in seconds with the bootstrap interval; the claim is restricted to
  window-level TTB error on validation.
- Primary not significant, or any collapsed seed pattern that explains it:
  report as is; Table 4's TTB entry is downgraded to "auxiliary output with
  no demonstrated advantage over a constant predictor". No re-analysis,
  re-definition of the target, or change of test is permitted to rescue it.
- MAE vs the median constant may be unfavourable even if RMSE is favourable
  (the head is trained for squared error); both are reported without spin.

### Pre-flight (rule 1) -- passed

Run on CPU in Colab on 2026-10-02: 24 tests, all OK
(`tests/test_ttb_eval_metrics.py` 19, `tests/test_ttb_stats.py` 2,
`tests/test_ttb_gradient_flow.py` 3): every model parameter is in the
optimizer (module level and in the real `train_and_evaluate` with a spy
optimizer); all four `head_ttb` tensors receive non-zero gradient and
change under real training; the other heads receive exactly zero gradient from
a TTB-only loss; a no-positive-window control reports "no gradient" (the
detector can fail); the exporter reproduces the checkpoint AUC and keeps
cache order; metrics, file recovery, onset reconstruction and lead-time
edge cases match hand-computed answers.

### Verification-seed gate (seed 42) before the full run

Run only `CRUX_ONLY=42`. The reference is the already recorded Run4 seed-42
result (`results/causal_fix/run4_seed42.json`: AUC=0.8703, best_epoch=44,
n_params=209,587); Run4 retrain over 10 seeds: AUC=0.8705+/-0.0133. Stop
and ask before scaling if ANY of: (1) |AUC - 0.8703| > 0.005; (2) n_val != 532,
positives != 338 or n_val_files != 9; (3) the TTB head is collapsed;
(4) the exporter AUC check fails; (5) no val file is evaluable for lead time
(lead time would not be computable). Best_epoch is recorded for information
only.

Planned tags: `ttb-eval-prereg` (this commit) and `ttb-eval-v1` (results).
Locked after the tag unless the reason is documented here: `configs/runs.py`,
`experiments/ttb_stats.py`, `src/ttb_eval.py`, `src/ttb_export.py`;
`experiments/ttb_run.py` refuses to run if they differ from the tag.

### Base commit and training-code status

Pre-registered on top of master `aba8723`. Commits 5b9f7a1/aba8723 (loss-leak
fix: `use_toc` threaded into the loss class) were reviewed by code diff (not
re-run): `use_toc` defaults to True and the CRITICAL_PATH node weights are
unchanged on that path, so Run 4 training (use_toc=True) keeps the same loss
computation as causal-fix-v1 and the seed-42 reference above remains valid.
The TTB head and `ttb_loss` are unchanged. This study modifies no existing
file under src/ or configs/: it adds src/ttb_eval.py, src/ttb_export.py,
experiments/ttb_run.py, experiments/ttb_stats.py and tests/test_ttb_*.py,
plus this section.

## TTB-head evaluation -- results (results tag: ttb-eval-v1)

Run: Run 4 configuration, `CRUX_PREBUILD_CAUSAL=1`, 10 seeds (42, 123, 456, 789,
1011, 1213, 1415, 1617, 1819, 2021), T4, git_sha
`0b9f54b1a963f1a2b2b3b256d5a7c93ebdf3f743` for every seed (pre-registration tag
`ttb-eval-prereg`). Before the analysis, the diff of the four locked files
against the tag was empty. No deviation from the pre-registration. Seed 42 was
run first as the verification gate (all five checks passed, AUC 0.87034 vs the
recorded 0.8703, best_epoch 44) and is one of the 10 seeds: same code, same
configuration, nothing changed after seeing it. Integrity: val AUC over the 10
seeds = 0.8699 +/- 0.0141, identical to causal-fix-v1; every seed has
n_val=532, 338 scored positive windows, 9 val files, no failed run.

### Window-level TTB error (n = 338 label==1 val windows per seed; mean +/- SD over seeds, seconds)

| predictor | RMSE | MAE |
|---|---|---|
| TTB head (10 seeds) | 11.36 +/- 1.54 | 7.54 +/- 1.41 |
| constant = train mean (6.33 s) | 13.50 | 9.85 |
| constant = train median (0 s) | 14.93 | 6.36 |

Per-seed RMSE: 42: 10.99, 123: 10.40, 456: 10.10, 789: 10.88, 1011: 15.30,
1213: 11.27, 1415: 10.41, 1617: 10.81, 1819: 10.86, 2021: 12.58 (only seed 1011
is above the constant).

PRIMARY (pre-registered): two-sided one-sample t-test of the 10 per-seed RMSEs
vs the train-mean constant RMSE: t = -4.396, p = 0.0017 (Wilcoxon, sensitivity:
p = 0.0059). Direction: head better; mean difference -2.14 s (-15.9% RMSE,
about 29% lower mean squared error).

Secondary (no multiplicity correction): MAE vs the train-median constant:
t = +2.652, p = 0.026, i.e. the head is significantly WORSE than the zero
predictor in MAE (7.54 vs 6.36 s); it is better than the mean constant in MAE
(7.54 vs 9.85 s). Spearman(pred, true) = 0.481 +/- 0.052 (77% ties at k=0).
Prediction SD 9.33 +/- 2.27 s vs target SD 13.50 s; collapsed seeds: 0 of 10.
Alerted subset (label==1 and p>0.5; n = 291.6 +/- 20.5): RMSE 9.55 +/- 0.54,
MAE 6.13 +/- 0.65 s (descriptive, no baseline on this subset).
File-level bootstrap (9 val files, 10,000 draws, seed 0, descriptive) of
RMSE_head - RMSE_constant: 95% interval [-3.42, +0.20] s, which includes 0.

Per-k (true time to bottleneck = 10k s), MAE / RMSE of the head in seconds:
k=0 (n=262) 5.11 / 7.47; k=1 (17) 10.28 / 12.22; k=2 (17) 10.45 / 12.50;
k=3 (16) 15.34 / 18.18; k=4 (14) 20.61 / 23.21; k=5 (12) 26.90 / 28.84. The
error grows with the remaining time (predictions are shrunk toward the mean).
Arithmetic from the constant (6.33 s): its MAE is 6.33, 3.67, 13.67, 23.67,
33.67, 43.67 s for k=0..5, so the head is worse than the mean constant only at
k=1.

### Detection-based alert lead time (separate from the TTB head; fixed threshold 0.5, stable alert)

Only 4 of the 9 val files are evaluable (files 0-3); the other 5 are short
(14 windows) and left-censored (first positive window is window 0). Over the 40
file x seed pairs: 26 detected, 14 missed (35%), 10 of the 26 detections
premature (alert in a label-negative window, lead > 50 s). Per seed: missed
1.4 +/- 1.1 files, premature 1.0 +/- 1.1, share of lead times above 50 s
31 +/- 29%, mean lead of detected files 46.7 +/- 33.6 s (range over seeds
10.0-93.3 s). Per-file mean lead over the seeds in which the file was detected:
file 0: 68.3 s, file 1: 15.7 s, file 2: 116.7 s, file 3: 55.0 s. No
hypothesis test (pre-registered). Figure:
`results/ttb_eval/ttb_lead_time_distribution.{png,pdf}`.

### Interpretation (pre-registered rules applied)

The primary test is significant and the head is better, so the TTB head is
reported as reducing window-level time-to-bottleneck RMSE by 2.14 s relative to
a train-mean constant (10 initialisations). The claim is restricted to
window-level TTB error on validation, and is qualified by: (i) MAE is
significantly worse than a zero predictor; (ii) the file-level bootstrap
interval includes 0, so a robust advantage across files is not established
(the primary test measures seed variance only; 9 files, strongly
autocorrelated windows); (iii) strong shrinkage toward the mean and moderate
rank correlation (0.48): the head is a coarse indicator, not an accurate
time-to-event predictor, on a target whose whole range is 0-50 s; (iv) val also
selected the best epoch (mild optimism). The pre-registered downgrade
condition (not better than the constant, or collapse) was NOT triggered.
Lead time is descriptive only: 4 evaluable files, large dependence on the
seed, 35% missed, and a large share of premature alerts. The earlier 164 s
+/- 55.5 manual figure (different files, detection head, unseeded) is not
cited as a result of this study.

Raw per-seed predictions, per-seed completion records and the train-only
baselines are stored in `results/ttb_eval/` (`ttb_val_seed*.npz`,
`ttb_seed*.json`, `ttb_baselines.json`), together with `ttb_eval_summary.json`.

### Figure note (presentation only; no number affected)

The figure written by `experiments/ttb_stats.py` (pre-registered description:
pooled histogram plus per-file mean +/- SD) had an annotation overlapping the
bars and SD whiskers reaching below 0 s that hid how many seeds contribute to
each file (4 evaluable files, 26 detections). `experiments/ttb_figure.py` (new,
not locked) redraws it from the same exports and the same `src/ttb_eval`
functions: pooled histogram split into alerts in label-positive vs premature
windows, and every (file, seed) outcome (dot = lead, x = missed, diamond = mean
of detected, k/10 seeds per file). `tests/test_ttb_figure.py` asserts that its
counts equal those of the stats script. `experiments/ttb_stats.py` was not
modified; the figure it wrote is replaced by the one in `results/ttb_eval/`.

## NOTEARS-layer analysis (reviewer item: "NOTEARS layer behavior under-analyzed") -- pre-registration (before any real GPU run)

Reviewer asks for: (1) DAG validity -- does the acyclicity term converge to 0; (2) sparsity (density) of the
learned graph; (3) visualization / interpretation of the learned structure; (4) systematic sensitivity to the
causal-loss coefficient lambda_causal (a value that caused divergence is mentioned but not analysed). The symbols
were lost in the pasted reviewer text; h, W-density and lambda_causal are the assumed referents.

### Facts established from the code before pre-registering (src/core.py at base commit 85f00c7)

1. `CausalInferenceLayer.acyclicity_constraint` computes tr(I + A/d + A^2/(2 d^2)) - d with A = G*G (Hadamard),
   i.e. the ORDER-2 truncation of tr(exp(A/d)) - d. With a zero diagonal tr(A) = 0, so only tr(A^2) survives:
   the implemented term sees 2-cycles only; cycles of length >= 3 are not penalised. tests/test_notears_analysis.py
   confirms this on a hand-made 3-cycle (h_impl = 0.0 exactly, h_exact > 0).
2. The penalty is applied to `causal_graph_mean` (batch mean of sigmoid(W_raw) * input-dependent contribution),
   not to W alone, and enters the loss as h^2. L_cause = abs().mean() of the graph + h^2.
3. sigmoid(W_raw) starts at 0.5 off the diagonal and is never exactly 0, so the graph is dense by construction;
   "density" is only defined through a threshold tau. The L1 term is a MEAN over B*N*N entries (per-entry
   gradient ~ lambda/(B*N^2)); whether this is too weak to sparsify is a hypothesis tested here, not assumed.
4. h can reach ~0 by shrinking all weights without any acyclic structure; therefore h alone is not accepted as
   evidence of validity (rule 4). Validity is judged on the binarised graph {G > tau} (topological sort),
   on h_exact, and on min_tau_dag (smallest threshold making the graph acyclic).
5. Runs 1/3 (lambda_causal = 0.20) also used fn_weight = 5.0 and a 779,921-parameter architecture, so their
   reported divergence is confounded; the fn_weight = 5.0 arms below separate fn_weight from lambda_causal.
   Architecture size stays confounded and is a stated limitation.
6. The orientation of graphs.pt `edge_index` ((src,dst) vs (dst,src)) is built in the notebook cache cell and is
   NOT verifiable from this repository. Hence the primary interpretability metric is UNDIRECTED; directed
   precision is reported descriptively in both orientations.

### Design

- Base: `RUNS["run4"]` imported (not copied) from configs/runs.py; `CRUX_PREBUILD_CAUSAL=1`; each job trained from
  scratch; `pipeline.train_and_evaluate(..., log_causal=True)`.
- Seeds (fixed): 42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021.
- Arms (configs/notears_runs.py), 8 arms x 10 seeds = 80 jobs: lambda_causal in {0, 0.01, 0.05, 0.1, 0.2, 0.5}
  (fn_weight 1.5; lam0.05 is the reference and equals Run 4 exactly, re-run from this commit rather than reusing
  earlier results because logging requires the modified pipeline); plus fn_weight = 5.0 with lambda_causal in
  {0.05, 0.2} (2x2 with the fn_weight = 1.5 cells).
- Logged per job (read-only; verified not to change training): first-step parameter audit (in optimizer / grad
  norm / change after step), per-epoch h_impl (train batch mean, val batch mean), val-graph summary (h_impl,
  h_exact, h_standard, density / is_dag / n_two_cycles / largest SCC at tau in {0.01, 0.05, 0.1, 0.2, 0.3, 0.5,
  0.7}, min_tau_dag, entropy), val positive-prediction rate; best-epoch val graph and sigmoid(W_raw).
- Metrics from the best-val-AUC epoch (Table 10 protocol): AUC, PatAcc, RCS Top-1, CP-Recall (reported but not
  interpreted: saturated, random baseline 0.9458).
- Divergence (defined now): non-finite training loss / non-finite parameters, or val positive-prediction rate
  >= 0.99 at the best epoch (the Run 3 saturation signature).
- "Uninformative graph" flag (rule 4): normalised weight entropy > 0.99 or > 90% of sigmoid(W_raw) entries within
  0.05 of 0.5. Flagged runs are reported but no structural claim is made from them.

### Pre-registered tests (alpha = 0.05; fixed here, implemented in experiments/notears_stats.py)

- P1 (sensitivity): Friedman test on val AUC across the six lambda_causal arms, blocks = seeds; a diverged run
  enters with AUC = 0.5; sensitivity analysis drops every seed having a diverged run in any arm. Non-significant
  P1 is reported as "no detectable sensitivity at n = 10", never as equivalence.
- P2 (interpretability), reference arm only: per seed a permutation p-value (node-label permutation of the
  call-graph edge set, 10,000 draws, seed = run seed) for undirected precision@|E|; then a one-sided exact
  binomial test that the fraction of seeds with p < 0.05 exceeds 0.05. Seeds share the same data, so this speaks to
  training stability, not data variation (LOFO covers that).
- Secondary: Holm-corrected paired Wilcoxon of each other lambda arm vs lam0.05 on AUC (5 tests); Holm-corrected
  paired Wilcoxon fn5 vs fn1.5 at lambda 0.05 and 0.2 on AUC (2 tests).
- Descriptive only (mean +/- SD, n = 10): h_impl / h_exact / h_standard at best epoch and epoch 1, density and
  fraction-DAG per tau, min_tau_dag, entropy, saturation, two-cycle counts, divergence counts, pairwise Jaccard
  of top-|E| edges across seeds (vs a random-subset expectation), directed precision in both orientations.
- Reading rules: "h converged" is described by the numbers (best epoch vs epoch 1), never by a threshold;
  a "valid DAG" is claimed only if {G > 0.5} is acyclic in every non-diverged seed, otherwise the fraction and
  min_tau_dag are reported and the gap between h_impl and h_exact is stated. Text must say "structural correlation
  discovery under acyclicity regularization", never causal inference.

### Pre-flight (rule 1) -- passed in Colab-independent CPU tests

tests/test_notears_analysis.py (16 tests) and tests/test_notears_stats.py (4 tests): all 209,587 parameters are in
the optimizer, have non-zero gradient and change after optimizer.step() in the real pipeline for ALL eight arms
(including lambda_causal = 0, where W_raw is trained via the rcs path); negative control with PREBUILD_CAUSAL=0
exposes the original bug; logging flag leaves best_model.pt bit-identical; NaN divergence is recorded, not a crash.
The runner (experiments/notears_run.py) re-checks the audit on every real job and FAILS the job if it does not hold.

### Verification-seed gate (seed 42, arm lam0.05) before the full run (rule 3)

Run: `CRUX_ONLY_ARM=lam0.05 CRUX_ONLY=42`. All must hold: val AUC within 0.005 of 0.8703 (Run 4, seed 42,
PREBUILD_CAUSAL=1; causal-fix-v1 / ttb-eval-v1), 338 scored positive val windows (rcs_top1_n), n_params 209,587,
causal_w_absmax > 0, parameter audit passes, h_impl / h_exact finite at every epoch, not diverged. Before
continuing, the raw sigmoid(W_raw) and val-graph value distributions are inspected by hand (rule 4). If the AUC
deviates, stop and ask. Seed 42 is one of the 10 seeds; nothing is changed after seeing it.

### Deviations from the plan shown for approval (step a)

- The plan's "untrained initial graph as a second null" is dropped: the initial graph is not stored. The null is
  the node-label permutation; epoch-1 h / density / entropy are in causal_history.json and are reported.
- Directed precision is descriptive, undirected is primary (fact 6).

### Tags, locks, base

Planned tags: `notears-analysis-prereg` (this commit) and `notears-analysis-v1` (results). Locked after the tag
unless the reason is documented here: configs/runs.py, configs/notears_runs.py, src/causal_analysis.py,
src/pipeline.py, experiments/notears_run.py, experiments/notears_stats.py; the runner refuses to run if they
differ from the tag. Base commit: master `85f00c7`. Cost estimate: 80 jobs x ~2.5 min on a T4 (~3.5 h).

### Amendment after tag notears-analysis-prereg (test-only; no locked file touched)

On a Colab T4, `test_same_checkpoint_with_and_without_logging` failed with a bitwise difference in
`spatial.layers.0.gat.att_src`. Cause: CUDA atomic adds in GATConv's scatter make two identical GPU runs differ
in the last bits; it is not an effect of logging. The test now forces CPU (`mock.patch` of
`torch.cuda.is_available`), where it passes bitwise; the other 19 tests passed on the T4 unchanged.
No locked file (configs/runs.py, configs/notears_runs.py, src/causal_analysis.py, src/pipeline.py,
experiments/notears_run.py, experiments/notears_stats.py) and no pre-registered quantity was modified.
