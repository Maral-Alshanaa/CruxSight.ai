# CruxSight.ai

> **Causal Bottleneck Prediction for Microservices**
> Applying Theory of Constraints + Causal Graph Neural Networks to predict *and* explain system bottlenecks before they cause outages.

---

## The Problem

Modern microservices systems can contain dozens of interdependent services. When a bottleneck forms, engineers face two questions simultaneously:

1. **Is a failure coming?** (detection)
2. **Which service is actually causing it?** (root-cause localization)

Existing tools answer these reactively -- *after* the SLO breach. Existing AI approaches either detect anomalies without explaining them, or explain past failures without predicting future ones.

**CruxSight.ai aims to do both, 3-5 minutes in advance, using only latency traces.**

---

## The Approach

CruxSight.ai combines two frameworks:

- **Theory of Constraints (ToC)** -- Goldratt's framework for identifying the single limiting factor in any system. We formalize this as a graph-theoretic constraint, compute a *Resource Constraint Score (RCS)* per node, and use it to supervise the causal layer.
- **CST-GNN** (Causal Spatio-Temporal Graph Neural Network) -- a three-stage architecture that processes latency time-series over the service call graph:
Raw Traces (Jaeger/Zipkin)
|
v
+---------------------+
| Spatial Encoder | GAT layers, ToC-capacity-biased attention
| (per time step) | -> which services are under stress right now?
+----------+-----------+
|
v
+---------------------+
| Temporal Encoder | LSTM + Multi-head Attention (TFT-style)
| (12-step window) | -> is the stress pattern escalating over time?
+----------+-----------+
|
v
+---------------------+
| Causal Inference | NOTEARS-inspired DAG learning
| Layer | -> which service is CAUSING the others to slow down?
+----------+-----------+
|
v
4 Prediction Heads:

Bottleneck probability (is a failure coming?)
Structural pattern (Pattern A-G, which tier?)
Time-to-breach estimate (how many minutes?)
Root-cause ranking (which service to fix first?)

**Key design principle:** The model is *N-agnostic* -- the same trained weights work across graphs of different sizes (30 nodes and 7 nodes tested), with fine-tuning required only for the causal layer on new topologies.

> **Known limitation (found during multi-seed verification, see below):** in the
> original implementation the causal layer's parameters are built lazily on
> first forward pass, after the optimizer is constructed, so they never receive
> gradient updates. See [Multi-Seed Statistical Analysis](#multi-seed-statistical-analysis-and-causal-layer-fix)
> for the fix and its effect on results.

---

## Dataset

**DeathStarBench Social Network** -- PACE Lab, Stony Brook University
Collected July-October 2023, published with the GAMMA paper (WWW 2024).

| Property | Value |
|----------|-------|
| Total trace records | ~3.3 million |
| Processed CSV files | 196 |
| Microservices (compose workflow) | 30 nodes, 17 servers |
| Microservices (home workflow) | 7 nodes |
| Bottleneck types | CPU stress, Memory stress, Network throttle, Combined |
| Kubernetes workload levels | 200 / 400 / 800 RPS |

**Central finding from the 196-file analysis:**
Constraints in this system appear **architectural, not resource-specific** -- the same irreducible storage core (nodes {13, 14, 20, 21, 26, 27, 28}) emerges across CPU, memory, and combined CPU+memory stress, spanning 92+ files under Pattern D alone. The clearest evidence is the control case: **network throttling is the one stress type that does *not* activate the storage core** -- it instead activates a disjoint middle-chain subgraph (Pattern F), consistent with the system's call-graph topology (network delays propagate through inter-service RPCs, not through storage). Latency is the reliable detection signal throughout; resource metrics (CPU%, RAM%) are noisy and inconsistent across identical experimental conditions.

---

## The 7 Structural Patterns

Analysis of 195 valid files (188 compose + 7 home, ~3.31M traces) identified 7 bottleneck structures, each defined by a fixed set of flagged nodes:

| Pattern | Name | Nodes | Flagged Set | Bottleneck Type | Files | AUC Range |
|---------|------|-------|-------------|------------------|-------|-----------|
| **A** | Wide storage | 15 | {4,5,7,8,11,12,13,14,18,19,20,21,26,27,28} | CPU only | ~31 | 0.864-0.981 |
| **B** | Core storage | 9 | {4,5,13,14,20,21,26,27,28} | CPU only | ~10 | 0.635-0.847 |
| **C** | Entry layer | 4 | {0,1,2,22} | CPU only | ~17 | 0.790-0.958 |
| **D** | Entry + Core hybrid | 11 | {0,1,2,13,14,20,21,22,26,27,28} | CPU, Memory, CPU+Memory | ~92 | 0.635-0.959 |
| **E** | Full system | 19 | {0,1,2,4,5,7,8,11,12,13,14,18,19,20,21,22,26,27,28} | CPU+Memory (rare) | 2 | 0.700-0.927 |
| **F** | Entry + Middle chain | 11 | {0,1,2,4,5,7,8,11,12,18,19} | Network throttle only | 30 | 0.593-0.883 |
| **G** | Home minimal | 2 | {3,4} (7-node graph) | CPU only | 7 | 0.933-0.938 |

**Pattern D dominates**, appearing across three different stress types (CPU, Memory, combined) -- the strongest evidence that the constraint is architectural rather than resource-specific.

**Pattern F is the critical control case.** Network throttling is the only stress type that does not touch the storage core, instead activating the middle RPC chain -- diverging in exactly the way the call-graph topology predicts.

**Pattern G reveals a latency inversion.** In the 7-node home workflow, the entry node is consistently faster (ratio 0.96-0.98x) during bottleneck traces, not slower -- the constraint is downstream.

**Secondary finding:** detection difficulty (AUC) correlates more strongly with pre-injection baseline latency than with bottleneck type or pattern.

**The memory-ratio paradox:** under memory stress, the memory utilization metric itself frequently sits at or below 1.0x (0.871x-1.099x across all 40 memory-stress files) -- it does not reliably rise even under genuine memory pressure. OS-level swapping surfaces instead as elevated CPU and latency.

---

## Results (Table 10, single-run ablation)

4 ablation configurations were tested. Best single-run result (Run 4):

| Metric | Value | Notes |
|--------|-------|-------|
| Val AUC | 0.869 | single run, best epoch |
| Pattern Accuracy | 88.9% | 8-class taxonomy |
| CP Recall | 0.91-1.00 | single run across epochs; see caveat below |
| Parameters | 205,617 | gat_hidden=32, tft_hidden=64 |

| Run | fn_weight | lambda_causal | RCS_sup | Best AUC | CP Recall | Issue |
|-----|-----------|---------------|---------|----------|-----------|-------|
| 1 | 5.0 | 0.20 | -- | 0.827 | 0.84 | Probabilities saturated |
| 2 | 1.5 | 0.05 | -- | 0.877 | 0.825 | RCS below random baseline |
| 3 | 5.0 | 0.20 | 0.3 | 0.824 | 0.94-0.99 | Val loss diverged |
| 4 | 1.5 | 0.05 | 0.3 | 0.869 | 0.91-1.00 | Best in this single run |

> **Caveat added after multi-seed verification:** these are single-run numbers
> from one (unseeded) training pass each. They should not be read as "beats
> random baseline" without qualification -- see the section below for mean +/- SD
> across 10 seeds, where CP-Recall for every configuration falls at or below the
> random-ranking baseline on average.

### Generalization Study (Home Workflow, N=7)

| Stage | AUC | RCS Top-1 | Notes |
|-------|-----|-----------|-------|
| Zero-shot | 0.544 | 0.0% | Total failure |
| Fine-tune: detection only (169 samples, 8 ep) | 0.900 | 0.0% | Detection recovers fast |
| Fine-tune: + RCS supervision (16 ep total) | 0.906 | 48.7% | 1.7x random baseline (28.6%) |

169 samples ~= 25 minutes of production traffic. Detection generalizes faster than root-cause localization.

> **Open question:** the causal_7 layer used in this fine-tuning is built by the
> same lazy-instantiation mechanism as causal_30 (see below) and is presumed
> affected by the same gradient bug, pending separate verification.

---

## Multi-Seed Statistical Analysis and Causal-Layer Fix

To quantify training stochasticity, Runs 1-4 were each re-trained with 10
pre-registered random seeds. During this work we found that the causal-inference
layer's parameters never received gradient updates in the original procedure
(lazy instantiation after optimizer construction, confirmed with
`tests/test_gradient_flow.py`). Two studies are reported:

| | Faithful replication (`multiseed-v1`, 40 runs) | Fixed causal layer (`causal-fix-v1`, 20 runs, Run 2/4 only) |
|---|---|---|
| Run 2 CP-Recall | 0.652 +/- 0.186 | 0.838 +/- 0.063 |
| Run 4 CP-Recall | 0.910 +/- 0.112 | 0.905 +/- 0.034 |
| Paired t-test (Run4 vs Run2) | t(9)=4.99, p=0.0007 | t(9)=2.77, p=0.0216 |
| Cohen's d_z | 1.58 | 0.88 |

Both are statistically significant; the effect size roughly halves once the
causal layer actually trains, and most of the originally reported gap reflects
an artificially weak Run 2 baseline rather than a larger true RCS-supervision
effect. Neither configuration's mean CP-Recall clearly exceeds the random
baseline (exact value 0.9458). See `PROTOCOL.md` for full details and
`results/multiseed/` and `results/causal_fix/` for raw per-seed data.

### Reproducing

```bash
python tests/test_gradient_flow.py        # mandatory pre-flight check

CRUX_OUT=<drive_path>/multiseed CRUX_PREBUILD_CAUSAL=0 python experiments/stage1_multiseed.py
python experiments/stage2_aggregate.py
python experiments/stage3_stats.py
python experiments/stage4_tables.py

CRUX_OUT=<drive_path>/causal_fix CRUX_RUNS=run2,run4 CRUX_PREBUILD_CAUSAL=1 python experiments/stage1_multiseed.py
python experiments/analyze_causal_fix.py
```

### Provenance (git tags)

| Tag | Meaning |
|---|---|
| `multiseed-prereg` | First pre-registration (superseded) |
| `multiseed-prereg-v2` | Corrected pre-registration: 10 seeds, Run 1-4 configs |
| `multiseed-v1` | Faithful replication results (40 runs) |
| `causal-fix-prereg` | Pre-registration of the causal-layer-fix study |
| `causal-fix-v1` | Causal-layer-fix results (20 runs) |

---

## Repository Structure

CruxSight.ai/
+-- README.md
+-- PROTOCOL.md <- pre-registration record, deviations, bug findings
+-- notebooks/ <- original training + ablation + generalization notebook
+-- src/
| +-- core.py <- model, loss, dataset, evaluator (multi-seed study)
| +-- pipeline.py <- train_and_evaluate() adapter
+-- configs/runs.py <- frozen Run 1-4 hyperparameters + 10 pre-registered seeds
+-- experiments/ <- multi-seed pipeline: run -> aggregate -> stats -> tables
+-- tests/ <- pre-flight gradient-flow checks, stats unit tests
+-- results/
| +-- multiseed/ <- faithful replication (40 runs)
| +-- causal_fix/ <- causal-layer-fix study (20 runs)
+-- docs/ <- figures, concept slides


---

## Limitations & Future Work

- **Dataset scope:** results are validated on DeathStarBench only.
- **Root-cause localization at N=7:** RCS top-1 reaches 48.7% after 16 fine-tuning epochs, above the 28.6% random baseline but below detection-level performance; the causal_7 layer's own gradient flow is not yet separately verified (see above).
- **Causal layer:** in the original implementation, the causal-inference layer's own parameters did not train (see Multi-Seed section). Claims of "learned causal structure" in the thesis text are being revised accordingly.
- **No Jaeger/Zipkin integration yet.**
- **Pattern taxonomy is system-specific** to DeathStarBench.

---

## Paper

**"Theory of Constraints-Driven Causal GNNs for Microservices Bottleneck Detection and Management"**
Maral Alshanaa -- doctoral thesis, supervised by Kadan Aljoumaa.

---

## Dataset Citation

```bibtex
@dataset{deathstarbench2023,
  title   = {DeathStarBench Microservices Bottleneck Localization Dataset},
  author  = {PACE Lab, Stony Brook University},
  year    = {2023},
  note    = {Collected July-October 2023, published with GAMMA (WWW 2024)}
}
```


*Dataset: DeathStarBench is property of PACE Lab, Stony Brook University.
Model weights trained on this dataset are provided for research purposes only.*