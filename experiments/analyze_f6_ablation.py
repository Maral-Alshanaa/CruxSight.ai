"""
Analysis for the F6 ablation multi-seed study (PROTOCOL.md, "F6 ablation,
multi-seed replication"). Pairs the F6-ablated results in
results/f6_ablation/ against the existing with-F6 results in
results/causal_fix/ (Run4 compose) and results/home_finetune/ (Run4 home),
same 10 seeds, and runs the pre-registered primary test (Wilcoxon
signed-rank, paired, on RCS Top-1) plus the secondary paired t-test.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
ROOT = Path(__file__).resolve().parent.parent

def load():
    with_f6, ablated = {}, {}
    for s in SEEDS:
        wc = json.load(open(ROOT / f"results/causal_fix/run4_seed{s}.json"))
        wh = json.load(open(ROOT / f"results/home_finetune/home_seed{s}.json"))
        with_f6[s] = dict(
            compose_auc=wc["auc"],
            zs=wh["stages"]["zero_shot"]["auc"],
            fd=wh["stages"]["finetune_detection"]["auc"],
            fc=wh["stages"]["finetune_causal"]["auc"],
            rcs_top1=wh["stages"]["finetune_causal"]["rcs_top1"],
        )
        ac = json.load(open(ROOT / f"results/f6_ablation/compose_seed{s}.json"))
        ah = json.load(open(ROOT / f"results/f6_ablation/home_seed{s}.json"))
        ablated[s] = dict(
            compose_auc=ac["auc"],
            zs=ah["stages"]["zero_shot"]["auc"],
            fd=ah["stages"]["finetune_detection"]["auc"],
            fc=ah["stages"]["finetune_causal"]["auc"],
            rcs_top1=ah["stages"]["finetune_causal"]["rcs_top1"],
            causal_w_before=ah["causal_w_absmax_before"],
            causal_w_after=ah["causal_w_absmax_after"],
        )
    return with_f6, ablated

def main():
    with_f6, ablated = load()

    rcs_f6 = np.array([with_f6[s]["rcs_top1"] for s in SEEDS])
    rcs_abl = np.array([ablated[s]["rcs_top1"] for s in SEEDS])

    assert (rcs_abl == 0.0).all(), "expected RCS Top-1 == 0.0 for every ablated seed"
    assert all(ablated[s]["causal_w_before"] == ablated[s]["causal_w_after"] for s in SEEDS), (
        "expected causal_7.W_raw to stay exactly frozen in every ablated Home run"
    )

    w_stat, w_p = stats.wilcoxon(rcs_f6, rcs_abl, alternative="two-sided")
    t_stat, t_p = stats.ttest_rel(rcs_f6, rcs_abl)

    print("=== RCS Top-1 (Home finetune_causal), with-F6 vs F6-ablated, n=10 seeds ===")
    print(f"with-F6:  mean={rcs_f6.mean():.2f}%  sd={rcs_f6.std(ddof=1):.2f}%  "
          f"range=[{rcs_f6.min():.2f}, {rcs_f6.max():.2f}]")
    print(f"ablated:  mean={rcs_abl.mean():.2f}%  sd={rcs_abl.std(ddof=1) if len(set(rcs_abl))>1 else 0.0:.2f}%  "
          f"(0.0% in every seed)")
    print(f"\nPRIMARY (pre-registered): Wilcoxon signed-rank, paired, two-sided: "
          f"W={w_stat}, p={w_p:.6g}")
    print(f"secondary: paired t-test: t={t_stat:.4f}, p={t_p:.6g}")

    print("\n=== AUC (descriptive only) ===")
    for label, key in [("Compose Val", "compose_auc"), ("Home zero-shot", "zs"),
                       ("Home finetune_detection", "fd"), ("Home finetune_causal", "fc")]:
        f6v = np.array([with_f6[s][key] for s in SEEDS])
        abv = np.array([ablated[s][key] for s in SEEDS])
        print(f"{label:28s} with-F6 {f6v.mean():.4f}+/-{f6v.std(ddof=1):.4f}  "
              f"ablated {abv.mean():.4f}+/-{abv.std(ddof=1):.4f}  "
              f"diff {(f6v-abv).mean():+.4f}")

    stats_out = dict(
        seeds=SEEDS,
        rcs_top1=dict(
            with_f6=dict(mean=float(rcs_f6.mean()), sd=float(rcs_f6.std(ddof=1)),
                        min=float(rcs_f6.min()), max=float(rcs_f6.max()), values=rcs_f6.tolist()),
            ablated=dict(mean=float(rcs_abl.mean()), sd=0.0, values=rcs_abl.tolist()),
        ),
        primary_test=dict(name="wilcoxon_signed_rank_paired_two_sided", statistic=float(w_stat), p=float(w_p)),
        secondary_test=dict(name="paired_ttest", statistic=float(t_stat), p=float(t_p)),
        auc=dict(),
    )
    for label, key in [("compose_val", "compose_auc"), ("home_zero_shot", "zs"),
                       ("home_finetune_detection", "fd"), ("home_finetune_causal", "fc")]:
        f6v = np.array([with_f6[s][key] for s in SEEDS])
        abv = np.array([ablated[s][key] for s in SEEDS])
        stats_out["auc"][label] = dict(
            with_f6_mean=float(f6v.mean()), with_f6_sd=float(f6v.std(ddof=1)),
            ablated_mean=float(abv.mean()), ablated_sd=float(abv.std(ddof=1)),
            mean_diff=float((f6v - abv).mean()),
        )
    out_path = ROOT / "results/f6_ablation/f6_ablation_stats.json"
    out_path.write_text(json.dumps(stats_out, indent=2))
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
