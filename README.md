*# CruxSight — TOC-Guided Spatio-Temporal GNN for Microservice Bottleneck Localization*



*Doctoral thesis project (CruxSight / CST-GNN): a graph neural network that applies*

*Theory of Constraints principles to detect and localize performance bottlenecks in*

*microservice architectures, evaluated on the DeathStarBench Social Network benchmark.*



*## Repository structure*



*src/core.py Model architecture, loss, dataset, evaluator (extracted from the training notebook)*

*src/pipeline.py train\_and\_evaluate() adapter used by the multi-seed runner*

*configs/runs.py Frozen hyperparameters for Runs 1-4 (Table 10) and the 10 pre-registered seeds*

*experiments/ Multi-seed pipeline: run -> aggregate -> stats -> tables*

*tests/ Pre-flight checks (gradient flow) and statistics unit tests*

*results/multiseed/ Raw per-seed results, faithful replication (40 runs)*

*results/causal\_fix/ Raw per-seed results, causal-layer-fix study (20 runs)*

*PROTOCOL.md Pre-registration record, deviations, and the causal-layer bug findings*





*## Multi-seed statistical analysis — summary*



*Each of the four training configurations (Table 10) was re-trained with 10*

*pre-registered random seeds to quantify training stochasticity and test whether*

*Run 4 (RCS-supervised) significantly outperforms Run 2 on CP-Recall.*



*During this work we found that the causal-inference layer's parameters*

*(`causal\_30.W\_raw` and its scoring encoder) never received gradient updates in the*

*original training procedure, because the layer is instantiated lazily on first*

*forward pass, after the optimizer is built. We report both the faithful*

*replication of the original protocol (bug present, matching Table 10's setup) and*

*a fixed-layer study (Run 2 / Run 4 only). Full results, code versions and*

*git tags below.*



*| | Faithful replication (`multiseed-v1`) | Fixed causal layer (`causal-fix-v1`) |*

*|---|---|---|*

*| Run 2 CP-Recall | 0.652 ± 0.186 | 0.838 ± 0.063 |*

*| Run 4 CP-Recall | 0.910 ± 0.112 | 0.905 ± 0.034 |*

*| Paired t-test | t(9)=4.99, p=0.0007 | t(9)=2.77, p=0.0216 |*

*| Cohen's d\_z | 1.58 | 0.88 |*



*Both results are statistically significant; the effect roughly halves once the*

*causal layer actually trains, and most of the original gap reflects an*

*artificially weak Run 2 baseline rather than a larger true RCS-supervision effect.*

*See `PROTOCOL.md` for the full write-up, and `results/\*/\[stats|tables]\*` for*

*raw numbers.*



*## Reproducing the results*



*```bash*

*# 1. Pre-flight check (mandatory before any real training)*

*python tests/test\_gradient\_flow.py*



*# 2. Faithful replication (40 runs, \~2-3h on a T4 GPU)*

*CRUX\_OUT=<drive\_path>/multiseed CRUX\_PREBUILD\_CAUSAL=0 python experiments/stage1\_multiseed.py*

*python experiments/stage2\_aggregate.py*

*python experiments/stage3\_stats.py*

*python experiments/stage4\_tables.py*



*# 3. Causal-layer-fix study (Run 2 / Run 4 only, 20 runs, \~40min)*

*CRUX\_OUT=<drive\_path>/causal\_fix CRUX\_RUNS=run2,run4 CRUX\_PREBUILD\_CAUSAL=1 python experiments/stage1\_multiseed.py*

*python experiments/analyze\_causal\_fix.py*

*```*



*## Provenance (git tags)*



*| Tag | Meaning |*

*|---|---|*

*| `multiseed-prereg` | First pre-registration (superseded — had incorrect Run 1/3 configs) |*

*| `multiseed-prereg-v2` | Corrected pre-registration: 10 seeds, Run 1-4 configs, primary test |*

*| `multiseed-v1` | Faithful replication results (40 runs) + statistical analysis tables |*

*| `causal-fix-prereg` | Pre-registration of the causal-layer-fix parallel study |*

*| `causal-fix-v1` | Causal-layer-fix results (20 runs) + analysis |*



*All raw per-seed JSON results, the exact code version (`git\_sha`) that produced*

*each one, and the statistical analysis scripts are versioned in this repository.*



*## Author*



*Maral Alshanaa — Academic thesis, supervised by Kadan Aljoumaa.*

