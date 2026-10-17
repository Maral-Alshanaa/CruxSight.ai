# Baseline comparison (Section 5.2.7), 10 seeds, recomputed from results/baseline

git_sha(s): 47c4f49071963feb92cd561f5f7371d4fa432c13; seeds: [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]; RCS Top-1 n per seed: [338]

## Table A: per arm (mean ± sample SD)

| arm | AUC | PatAcc | RCS Top-1 (%) | CP-Recall (random top-3 = 0.9458; saturated) |
|---|---|---|---|---|
| run4_retrain | 0.8699 ± 0.0141 | 0.9004 ± 0.0344 | 62.49 ± 6.48 | 0.9053 ± 0.0337 |
| no_toc | 0.8688 ± 0.0127 | 0.8816 ± 0.0293 | 60.36 ± 23.10 | 0.9003 ± 0.2659 |
| logreg F0-F6 (single deterministic fit) | 0.7121 | 0.8816 | 41.72 | 0.4467 |

## Table B: paired tests, no_toc − run4_retrain (two-sided, df = 9)

| metric | mean diff [95% CI] | t | p (t) | p Holm (3 tests) | exact Wilcoxon W / p | d_z | seeds no_toc higher / Run4 higher / tied |
|---|---|---|---|---|---|---|---|
| auc | -0.0011 [-0.0120, 0.0098] | -0.225 | 0.8269 | 1.0000 | 25.0 / 0.8457 | -0.071 | 4 / 6 / 0 |
| pat_acc | -0.0188 [-0.0501, 0.0125] | -1.357 | 0.2077 | 0.6231 | 15.0 / 0.2324 | -0.429 | 3 / 7 / 0 |
| rcs_top1 | -2.13 [-16.42, 12.16] | -0.337 | 0.7436 | 1.0000 | 20.0 / 0.4922 | -0.107 | 7 / 3 / 0 |

## Table C: one-sample tests, RCS Top-1 vs 38.2% (weighted random baseline), df = 9

| arm | mean − 38.2 [95% CI of mean] | t | p (t) | exact Wilcoxon p | d_z | seeds above |
|---|---|---|---|---|---|---|
| run4_retrain | 24.29 [57.85, 67.12] | 11.854 | 0.00000 | 0.0020 | 3.749 | 10/10 |
| no_toc | 22.16 [43.83, 76.88] | 3.033 | 0.01417 | 0.0195 | 0.959 | 9/10 |

Weighted random RCS Top-1 recomputed from val_pos_pattern_counts in the result files and src/core.py PATTERNS: 38.1755% (pre-registered value used: 38.2%).

### Verification 1: paper numbers vs recomputed

| quantity | paper / recorded | recomputed from results/baseline | status |
|---|---|---|---|
| no_toc.auc.mean | 0.8737 | 0.868834 | MISMATCH |
| no_toc.auc.sd | 0.0150 | 0.012681 | MISMATCH |
| no_toc.pat_acc.mean | 0.8865 | 0.881579 | MISMATCH |
| no_toc.pat_acc.sd | 0.0237 | 0.029335 | MISMATCH |
| no_toc.rcs_top1.mean | 63.49 | 60.3550 | MISMATCH |
| no_toc.rcs_top1.sd | 22.35 | 23.0984 | MISMATCH |
| run4_retrain.auc.mean | 0.8705 | 0.869916 | MISMATCH |
| run4_retrain.auc.sd | 0.0133 | 0.014079 | MISMATCH |
| run4_retrain.pat_acc.mean | 0.9047 | 0.900376 | MISMATCH |
| run4_retrain.pat_acc.sd | 0.0281 | 0.034433 | MISMATCH |
| run4_retrain.rcs_top1.mean | 62.81 | 62.4852 | MISMATCH |
| run4_retrain.rcs_top1.sd | 8.00 | 6.4786 | MISMATCH |
| paired.rcs_top1.p_t | 0.902 | 0.74363 | MISMATCH |
| paired.rcs_top1.p_wilcoxon | 0.625 | 0.49219 | MISMATCH |

### Verification 2: values recorded in PROTOCOL.md vs recomputed

| quantity | paper / recorded | recomputed from results/baseline | status |
|---|---|---|---|
| paired.auc.t | 0.600 | -0.22520 | MISMATCH |
| paired.auc.p_t | 0.5636 | 0.826856 | MISMATCH |
| paired.pat_acc.t | -1.742 | -1.35748 | MISMATCH |
| paired.pat_acc.p_t | 0.1156 | 0.207688 | MISMATCH |
| paired.rcs_top1.t | 0.127 | -0.33730 | MISMATCH |
| paired.rcs_top1.p_t | 0.9018 | 0.743627 | MISMATCH |
| paired.rcs_top1.dz | 0.04 | -0.1067 | MISMATCH |
| run4_retrain.cp_recall.mean | 0.9056 | 0.905325 | MISMATCH |
| run4_retrain.cp_recall.sd | 0.0389 | 0.033704 | MISMATCH |
| no_toc.cp_recall.mean | 0.9675 | 0.900296 | MISMATCH |
| no_toc.cp_recall.sd | 0.0576 | 0.265947 | MISMATCH |
| one_sample.run4_retrain.rcs_top1.t | 9.733 | 11.85384 | MISMATCH |
| one_sample.run4_retrain.rcs_top1.mean_diff | 24.6 | 24.285 | MISMATCH |
| logreg.auc | 0.7120 | 0.712087 | MISMATCH |
| logreg.cp_recall | 0.4527 | 0.446746 | MISMATCH |
| logreg.pat_acc | 0.8816 | 0.881579 | match |
| logreg.rcs_top1 | 41.72 | 41.7160 | match |

29 row(s) not matching (reported, nothing adjusted).
