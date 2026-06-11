# CruxSight.ai
Causal Spatio-Temporal GNN for Bottleneck Prediction in Microservices using Theory of Constraints

> Predicting multi-layered bottlenecks in microservices
> before SLO violations — guided by Theory of Constraints

🚧 **Active research project — full release coming soon**

## What This Is

A novel hybrid deep learning framework that:
- Detects bottleneck patterns **before** they affect users
- Identifies root cause vs cascading symptoms using causal inference
- Applies Theory of Constraints as mathematical constraints in training
- Achieves AUC up to 0.981 on the DeathStarBench dataset

## Key Findings (Analysis Phase)

| Finding | Detail |
|---------|--------|
| Patterns discovered | 7 structural patterns (A–G) |
| Files analyzed | 196 CSV files |
| Traces processed | ~3.3 million |
| Best detection AUC | 0.981 (Pattern A, CPU stress) |
| Resource signals | Proven unreliable across all 196 experiments |
| Constraint type | Architectural, not resource-specific |

## Dataset

DeathStarBench Social Network · PACE Lab, Stony Brook University
Collected July–October 2023 · Published with GAMMA (WWW 2024)

- 30 microservices · 17 servers · 3 workflows
- 4 bottleneck types: CPU · Memory · CPU+Memory · Network
- [Kaggle Dataset](https://www.kaggle.com/datasets/gagansomashekar/microservices-bottleneck-detection-dataset)

## Architecture

## Status

- [x] Dataset analysis (196 files)
- [x] Pattern taxonomy (7 patterns)
- [x] TOC application framework
- [ ] CST-GNN training
- [ ] Evaluation vs GAMMA baseline
- [ ] Paper submission
- [ ] Public release

## Citation

If you use this work, please also cite the original dataset paper:

```bibtex
@inproceedings{somashekar2024gamma,
  title={GAMMA: Graph Neural Network-Based Multi-Bottleneck
         Localization for Microservices Applications},
  author={Somashekar, Gagan and Dutt, Anurag and Adak, Mainak
          and Lorido Botran, Tania and Gandhi, Anshul},
  booktitle={Proceedings of the ACM Web Conference 2024},
  year={2024}
}
```
