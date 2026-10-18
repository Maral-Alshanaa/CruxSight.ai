# Simple baselines (recomputed from results/baseline and results/baseline_simple)

| Method | AUC | PatAcc | RCS Top-1 (%) |
|---|---|---|---|
| CruxSight Run 4 (10 seeds) | 0.8699 ± 0.0141 | 0.9004 ± 0.0344 | 62.49 ± 6.48 |
| GAT+TFT, no TOC (10 seeds) | 0.8688 ± 0.0127 | 0.8816 ± 0.0293 | 60.36 ± 23.10 |
| RF, per-node F0-F6 (10 seeds) | 0.7963 ± 0.0053 | 0.9648 ± 0.0015 | 96.80 ± 0.36 |
| RF, flattened all-node (10 seeds) | 0.8751 ± 0.0018 | 0.9720 ± 0.0011 | 100.00 ± 0.00 |
| Logistic regression, F0-F6 (single fit) | 0.7121 | 0.8816 | 41.72 |
| Static threshold rule (F0) (single fit) | 0.6362 | not applicable | 52.37 |

RCS Top-1 n per method is recorded in the result files; n = 10 seeds limits power, so a non-significant result is not evidence of equality.

| Comparison (baseline − reference) | Metric | Mean diff | 95 % CI | p (primary, unadjusted) | p Wilcoxon | p Holm | Reading |
|---|---|---|---|---|---|---|---|
| rf_pernode vs run4_retrain | AUC | -0.0736 | [-0.0851, -0.0621] | 0.0000 | 0.0020 | 0.0000 | difference detected (lower than reference) |
| rf_pernode vs run4_retrain | PatAcc | +0.0645 | [+0.0395, +0.0894] | 0.0002 | 0.0020 | 0.0002 | difference detected (higher than reference) |
| rf_pernode vs run4_retrain | RCS Top-1 (%) | +34.3195 | [+29.4943, +39.1447] | 0.0000 | 0.0020 | 0.0000 | difference detected (higher than reference) |
| rf_flat vs run4_retrain | AUC | +0.0052 | [-0.0053, +0.0156] | 0.2937 | 0.3223 | 0.2937 | no detectable difference |
| rf_flat vs run4_retrain | PatAcc | +0.0716 | [+0.0472, +0.0961] | 0.0001 | 0.0020 | 0.0002 | difference detected (higher than reference) |
| rf_flat vs run4_retrain | RCS Top-1 (%) | +37.5148 | [+32.8803, +42.1493] | 0.0000 | 0.0020 | 0.0000 | difference detected (higher than reference) |
| rule vs run4_retrain | AUC | -0.2338 | [-0.2438, -0.2237] | 0.0000 | 0.0020 | 0.0000 | difference detected (lower than reference) |
| rule vs run4_retrain | RCS Top-1 (%) | -10.1183 | [-14.7529, -5.4838] | 0.0008 | 0.0039 | 0.0008 | difference detected (lower than reference) |
| rf_pernode vs no_toc | AUC | -0.0725 | [-0.0830, -0.0620] | 0.0000 | 0.0020 | 0.0000 | difference detected (lower than reference) |
| rf_pernode vs no_toc | PatAcc | +0.0833 | [+0.0621, +0.1045] | 0.0000 | 0.0020 | 0.0000 | difference detected (higher than reference) |
| rf_pernode vs no_toc | RCS Top-1 (%) | +36.4497 | [+19.8684, +53.0310] | 0.0008 | 0.0020 | 0.0008 | difference detected (higher than reference) |
| rf_flat vs no_toc | AUC | +0.0062 | [-0.0029, +0.0154] | 0.1582 | 0.1934 | 0.1582 | no detectable difference |
| rf_flat vs no_toc | PatAcc | +0.0904 | [+0.0691, +0.1117] | 0.0000 | 0.0020 | 0.0000 | difference detected (higher than reference) |
| rf_flat vs no_toc | RCS Top-1 (%) | +39.6450 | [+23.1214, +56.1686] | 0.0004 | 0.0020 | 0.0008 | difference detected (higher than reference) |
| rule vs no_toc | AUC | -0.2327 | [-0.2418, -0.2236] | 0.0000 | 0.0020 | 0.0000 | difference detected (lower than reference) |
| rule vs no_toc | RCS Top-1 (%) | -7.9882 | [-24.5118, +8.5354] | 0.3025 | 0.1934 | 0.3025 | no detectable difference |
