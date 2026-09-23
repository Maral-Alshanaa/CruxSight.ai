# CruxSight â€” TOC-Guided Spatio-Temporal GNN for Microservice Bottleneck Localization

Doctoral thesis project (CruxSight / CST-GNN): a graph neural network that applies
Theory of Constraints principles to detect and localize performance bottlenecks in
microservice architectures, evaluated on the DeathStarBench Social Network benchmark.

## Repository structure

- `src/core.py` â€” Model architecture, loss, dataset, evaluator (extracted from the training notebook)
- `src/pipeline.py` â€” train_and_evaluate() adapter used by the multi-seed runner
- `configs/runs.py` â€” Frozen hyperparameters for Runs 1-4 (Table 10) and the 10 pre-registered seeds
- `experiments/` â€” Multi-seed pipeline: run, aggregate, stats, tables
- `tests/` â€” Pre-flight checks (gradient flow) and statistics unit tests
- `results/multiseed/` â€” Raw per-seed results, faithful replication (40 runs)
- `results/causal_fix/` â€” Raw per-seed results, causal-layer-fix study (20 runs)
- `PROTOCOL.md` â€” Pre-registration record, deviations, and the causal-layer bug findings

## Multi-seed statistical analysis â€” summary

Each of the four training configurations (Table 10) was re-trained with 10
pre-registered random seeds to quantify training stochasticity and test whether
Run 4 (RCS-supervised) significantly outperforms Run 2 on CP-Recall.

During this work we found that the causal-inference layer's parameters
(`causal_30.W_raw` and its scoring encoder) never received gradient updates in the
original training procedure, because the layer is instantiated lazily on first
forward pass, after the optimizer is built. We report both the faithful
replication of the original protocol (bug present, matching Table 10's setup) and
a fixed-layer study (Run 2 / Run 4 only). Full results, code versions and
git tags below.

| | Faithful replication (`multiseed-v1`) | Fixed causal layer (`causal-fix-v1`) |
| --- | --- | --- |
| Run 2 CP-Recall | 0.652 +/- 0.186 | 0.838 +/- 0.063 |
| Run 4 CP-Recall | 0.910 +/- 0.112 | 0.905 +/- 0.034 |
| Paired t-test | t(9)=4.99, p=0.0007 | t(9)=2.77, p=0.0216 |
| Cohen's d_z | 1.58 | 0.88 |

Both results are statistically significant; the effect roughly halves once the
causal layer actually trains, and most of the original gap reflects an
artificially weak Run 2 baseline rather than a larger true RCS-supervision effect.
See `PROTOCOL.md` for the full write-up, and `results/` for raw numbers and tables.

## Reproducing the results

```bash
# 1. Pre-flight check (mandatory before any real training)
python tests/test_gradient_flow.py

# 2. Faithful replication (40 runs, ~2-3h on a T4 GPU)
CRUX_OUT=<drive_path>/multiseed CRUX_PREBUILD_CAUSAL=0 python experiments/stage1_multiseed.py
python experiments/stage2_aggregate.py
python experiments/stage3_stats.py
python experiments/stage4_tables.py

# 3. Causal-layer-fix study (Run 2 / Run 4 only, 20 runs, ~40min)
CRUX_OUT=<drive_path>/causal_fix CRUX_RUNS=run2,run4 CRUX_PREBUILD_CAUSAL=1 python experiments/stage1_multiseed.py
python experiments/analyze_causal_fix.py
```

## Provenance (git tags)

| Tag | Meaning |
| --- | --- |
| `multiseed-prereg` | First pre-registration (superseded, incorrect Run 1/3 configs) |
| `multiseed-prereg-v2` | Corrected pre-registration: 10 seeds, Run 1-4 configs, primary test |
| `multiseed-v1` | Faithful replication results (40 runs) plus statistical analysis tables |
| `causal-fix-prereg` | Pre-registration of the causal-layer-fix parallel study |
| `causal-fix-v1` | Causal-layer-fix results (20 runs) plus analysis |

All raw per-seed JSON results, the exact code version (`git_sha`) that produced
each one, and the statistical analysis scripts are versioned in this repository.

## Author

Maral Alshanaa â€” Academic thesis, supervised by Kadan Aljoumaa.