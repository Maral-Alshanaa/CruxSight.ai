# CruxSight.ai

> **Causal Bottleneck Prediction for Microservices**
> Applying Theory of Constraints + Causal Graph Neural Networks to predict *and* explain system bottlenecks before they cause outages.

<p align="center">
  <img src="docs/carousel/slide_1.png" width="340"/>
  &nbsp;&nbsp;
  <img src="docs/carousel/slide_2.png" width="340"/>
</p>

---

## The Problem

Modern microservices systems can contain dozens of interdependent services. When a bottleneck forms, engineers face two questions simultaneously:

1. **Is a failure coming?** (detection)
2. **Which service is actually causing it?** (root-cause localization)

Existing tools answer these reactively — *after* the SLO breach. Existing AI approaches either detect anomalies without explaining them, or explain past failures without predicting future ones.

**CruxSight.ai does both, 3–5 minutes in advance, using only latency traces.**

---

## The Approach

CruxSight.ai combines two frameworks that have never been formally integrated before:

- **Theory of Constraints (ToC)** — Goldratt's framework for identifying the single limiting factor in any system. We formalize this as a graph-theoretic constraint, compute a *Resource Constraint Score (RCS)* per node, and use it to supervise the causal layer.
- **CST-GNN** (Causal Spatio-Temporal Graph Neural Network) — a three-stage architecture that processes latency time-series over the service call graph:

```
Raw Traces (Jaeger/Zipkin)
        │
        ▼
┌──────────────────────┐
│   Spatial Encoder    │  GAT layers, ToC-capacity-biased attention
│   (per time step)    │  → which services are under stress right now?
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Temporal Encoder    │  LSTM + Multi-head Attention (TFT-style)
│  (12-step window)    │  → is the stress pattern escalating over time?
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Causal Inference    │  NOTEARS-inspired DAG learning
│  Layer               │  → which service is CAUSING the others to slow down?
└──────────┬───────────┘
           │
           ▼
   4 Prediction Heads:
   • Bottleneck probability  (is a failure coming?)
   • Structural pattern      (Pattern A–G, which tier?)
   • Time-to-breach estimate (how many minutes?)
   • Root-cause ranking      (which service to fix first?)
```

**Key design principle:** The model is *N-agnostic* — the same trained weights work across graphs of different sizes (30 nodes and 7 nodes tested), with fine-tuning required only for the causal layer on new topologies.

---

## Dataset

**DeathStarBench Social Network** — PACE Lab, Stony Brook University
Collected July–October 2023, published with the GAMMA paper (WWW 2024).

| Property | Value |
|----------|-------|
| Total trace records | ~3.3 million |
| Processed CSV files | 196 |
| Microservices (compose workflow) | 30 nodes, 17 servers |
| Microservices (home workflow) | 7 nodes |
| Bottleneck types | CPU stress, Memory stress, Network throttle, Combined |
| Kubernetes workload levels | 200 / 400 / 800 RPS |

**Central finding from the 196-file analysis:**
Constraints in this system are **architectural, not resource-specific** — the same irreducible storage core (nodes {13, 14, 20, 21, 26, 27, 28}) emerges across CPU, memory, and combined CPU+memory stress, spanning 92+ files under Pattern D alone. The clearest evidence is the control case: **network throttling is the one stress type that does *not* activate the storage core** — it instead activates a disjoint middle-chain subgraph (Pattern F), exactly as predicted by the system's call-graph topology (network delays propagate through inter-service RPCs, not through storage). Latency is the reliable detection signal throughout; resource metrics (CPU%, RAM%) are noisy and inconsistent across identical experimental conditions.

---

## The 7 Structural Patterns

Analysis of 195 valid files (188 compose + 7 home, ~3.31M traces) revealed 7 deterministic bottleneck structures. Each pattern is defined by an *exact, rigid* set of flagged nodes — confirmed identical across every file sharing that injection configuration, with zero variation observed within a pattern:

| Pattern | Name | Nodes | Flagged Set | Bottleneck Type | Files | AUC Range |
|---------|------|-------|-------------|------------------|-------|-----------|
| **A** | Wide storage | 15 | {4,5,7,8,11,12,13,14,18,19,20,21,26,27,28} | CPU only | ~31 | 0.864–0.981 |
| **B** | Core storage | 9 | {4,5,13,14,20,21,26,27,28} | CPU only | ~10 | 0.635–0.847 |
| **C** | Entry layer | 4 | {0,1,2,22} | CPU only | ~17 | 0.790–0.958 |
| **D** | Entry + Core hybrid | 11 | {0,1,2,13,14,20,21,22,26,27,28} | CPU, Memory, CPU+Memory | ~92 | 0.635–0.959 |
| **E** | Full system | 19 | {0,1,2,4,5,7,8,11,12,13,14,18,19,20,21,22,26,27,28} | CPU+Memory (rare) | 2 | 0.700–0.927 |
| **F** | Entry + Middle chain | 11 | {0,1,2,4,5,7,8,11,12,18,19} | Network throttle only | 30 | 0.593–0.883 |
| **G** | Home minimal | 2 | {3,4} (7-node graph) | CPU only | 7 | 0.933–0.938 |

**Pattern D dominates** — it is the single most common structural signature, appearing across three *different* stress types (CPU, Memory, and combined CPU+Memory), not just one. This cross-type recurrence is the strongest evidence that the constraint is architectural rather than resource-specific (see Dataset section above).

**Pattern F is the critical control case.** Network throttling is the *only* stress type that does **not** touch the irreducible storage core {13,14,20,21,26,27,28} — instead activating the middle RPC chain. This is what makes the "architectural, not resource-specific" claim falsifiable rather than just a correlation: a stress type exists that *doesn't* converge on the same subgraph, and it diverges in exactly the way the system's call-graph structure predicts (network delays propagate through inter-service calls, not through the storage layer itself).

**Pattern G reveals a latency inversion.** In the 7-node home workflow, the entry node is *consistently faster* (ratio 0.96–0.98×) during bottleneck traces, not slower. The constraint is downstream, throttling the volume of requests that reach the entry point — the same "lighter load upstream of a deep constraint" effect.

**A secondary finding across the full dataset:** detection difficulty (AUC) correlates more strongly with the *pre-injection baseline latency* of the system than with bottleneck type or pattern. Across the CPU+Memory batches, runs with an elevated baseline (system already under load before injection) consistently showed lower AUC regardless of which pattern was flagged — suggesting baseline system health is a stronger predictor of detection difficulty than the constraint type itself.

**The memory-ratio paradox:** under memory stress, the memory utilization *metric itself* frequently sits at or below 1.0× (0.871×–1.099× across all 40 memory-stress files) — it does not reliably rise even while the system is genuinely under memory pressure. This metric measures instantaneous usage, not pressure; OS-level swapping and throttling surface instead as elevated CPU and latency. It is a concrete mechanism for why resource metrics mislead: the metric most directly tied to the injected stress type was, in this dataset, the least informative signal of all.

---

## Results

### Model Training (Compose Workflow, N=30)

4 ablation configurations were tested. Best result (Run 4):

| Metric | Value | Notes |
|--------|-------|-------|
| Val AUC | **0.869** | 5-fold validation |
| Pattern Accuracy | **88.9%** | 8-class taxonomy |
| CP Recall | **0.91–1.00** | vs 0.94 random baseline |
| Parameters | 205,617 | gat_hidden=32, tft_hidden=64 |

**Ablation summary:**

| Run | fn_weight | λ_causal | RCS_sup | Best AUC | CP Recall | Issue |
|-----|-----------|----------|---------|----------|-----------|-------|
| 1 | 5.0 | 0.20 | — | 0.827 | 0.84 | Probabilities saturated |
| 2 | 1.5 | 0.05 | — | 0.877 | 0.825 | RCS below random baseline |
| 3 | 5.0 | 0.20 | 0.3 | 0.824 | 0.94–0.99 | Val loss diverged |
| **4** | **1.5** | **0.05** | **0.3** | **0.869** | **0.91–1.00** | **Best — stable** |

The key insight from ablation: **calibration fixes and RCS supervision solve different problems** — combining them (Run 4) achieves both simultaneously.

### Generalization Study (Home Workflow, N=7)

Testing zero-shot transfer to a structurally different, unseen graph:

| Stage | AUC | RCS Top-1 | Notes |
|-------|-----|-----------|-------|
| Zero-shot | 0.544 | 0.0% | Total failure |
| Fine-tune: detection only (169 samples, 8 ep) | **0.900** | 0.0% | Detection recovers fast |
| Fine-tune: + RCS supervision (16 ep total) | **0.906** | **48.7%** | 1.7× random baseline (28.6%) |

**169 samples ≈ ~25 minutes of production traffic.**

Detection generalizes faster than root-cause localization — a practical deployment finding: systems can alert reliably almost immediately after fine-tuning, while root-cause ranking needs explicit pattern supervision and more epochs.

---

## Repository Structure

```
CruxSight.ai/
├── README.md
├── notebooks/
│   ├── analysis/              ← 196-file bottleneck analysis pipeline
│   └── cruxsight.ipynb        ← full training + ablation + generalization
├── src/
│   └── cst_gnn/
│       ├── model.py           ← CST-GNN architecture (GAT + TFT + NOTEARS)
│       ├── loss.py            ← TOCWeightedLoss + RCS supervision
│       ├── dataset.py         ← windowed trace loader
│       ├── toc_priors.py      ← ToC patterns A-G, critical path, capacity priors
│       ├── evaluate.py        ← AUC, PatAcc, CPRecall, subordination score
│       └── config.py          ← Config dataclass
├── results/
│   ├── ablation_table.json
│   ├── final_results.json
│   └── checkpoints/
│       ├── final_model_v1.pt           ← compose-trained (N=30)
│       └── final_model_v1_finetuned_home_v2.pt  ← fine-tuned (N=7)
├── docs/
│   ├── carousel/              ← LinkedIn hackathon slides (PNG + PDF)
│   └── figures/               ← architecture diagram, pattern visualizations
└── carousel/
    └── generate_carousel.py   ← reproduces the LinkedIn slides from real data
```

---

## Quick Start

```bash
git clone https://github.com/Maral-Alshanaa/CruxSight.ai
cd CruxSight.ai
pip install torch torch-geometric networkx scikit-learn numpy
```

**Run inference on a trace file:**
```python
# coming soon — scripts/predict.py
```

**Reproduce the carousel:**
```bash
python carousel/generate_carousel.py
# → outputs CruxSight_LinkedIn_Carousel.pdf + slide_*.png
```

**Training notebook:**
Open `notebooks/cruxsight.ipynb` in Google Colab.
The dataset cache (~27MB) downloads automatically from the PACE Lab source on first run.

---

## Product Vision — Concept Dashboard (Mockups)

> ⚠️ **The screens below are early UI/UX concept mockups for the planned
> CruxSight.ai dashboard product. They illustrate the intended user
> experience and are *not* generated from live model output. All
> verified model results are in the [Results](#results) section above.**

<p align="center">
  <img src="docs/vision/Vision 4 toc topology.png" width="420"/>
  &nbsp;
  <img src="docs/vision/Vision 5 causal pattern_anatomy.png" width="420"/>
</p>
<p align="center">
  <img src="docs/vision/Vision 3 pattern anatomy_full.png" width="420"/>
  &nbsp;
  <img src="docs/vision/Vision 2 cascading prediction.png" width="420"/>
</p>

The intended product flow:
1. **Topology view** — a capacity-biased graph of the live service mesh, with ToC constraint nodes highlighted by real-time GAT attention weights.
2. **Causal pattern anatomy** — when a bottleneck is detected, the dashboard highlights the causal subgraph (Pattern A–G) and the DAG learned by the causal inference layer.
3. **Cascading bottleneck view** — a full-system map showing which services are currently healthy, degrading, or critical, with a proactive alert panel.

These mockups guided the architecture's output design (the `bn_logit`, `pattern_logit`, `rcs`, and `ttb` heads map directly onto these UI elements) but represent a future productization target beyond the current research scope.

---

## Limitations & Future Work

- **Dataset scope:** Results are validated on DeathStarBench only. Generalization to production systems with different service topologies requires per-deployment fine-tuning (validated at ~25 min of data).
- **Root-cause localization at N=7:** RCS top-1 reaches 48.7% after 16 fine-tuning epochs — above the 28.6% random baseline but not yet at the detection-level performance (AUC 0.906). This gap between detection and localization convergence rates is an open question.
- **No Jaeger/Zipkin integration yet:** The prediction pipeline currently requires pre-processed CSV traces. A live streaming adapter (`scripts/predict.py`) is planned.
- **Pattern taxonomy is system-specific:** The 7 patterns (A–G) were derived from DeathStarBench. Other systems may exhibit different structural patterns requiring re-analysis.

---

## Paper

**"Theory of Constraints-Driven Causal GNNs for Microservices Bottleneck Detection and Management"**
*Maral Alshanaa — 4th Year Graduation Project, 2026*

Full paper and academic write-up: coming soon.

---

## Dataset Citation

```bibtex
@dataset{deathstarbench2023,
  title   = {DeathStarBench Microservices Bottleneck Localization Dataset},
  author  = {PACE Lab, Stony Brook University},
  year    = {2023},
  note    = {Collected July–October 2023, published with GAMMA (WWW 2024)}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

*Dataset: DeathStarBench is property of PACE Lab, Stony Brook University.
Model weights trained on this dataset are provided for research purposes only.*

---

<p align="center">
  <b>CruxSight.ai</b> — Find the constraint. Fix the system.<br/>
  Theory of Constraints-Driven Causal GNNs for Microservices Bottleneck Detection and Management<br/>
  Maral Alshanaa · 4th Year Graduation Project · 2026<br/>
  <a href="https://github.com/Maral-Alshanaa/CruxSight.ai">GitHub</a>
</p>
