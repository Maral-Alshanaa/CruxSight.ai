# experiments/f6_anomaly_run.py
"""
F6 zero-shot anomaly study (reviewer item 8). No training: loads the existing
per-seed compose checkpoints of both arms and evaluates them on Home.

  with-F6 : $CRUX_RESULTS_ROOT/causal_fix/run4/seed{s}/best_model.pt, toc_cap_7 as cached
  ablated : $CRUX_RESULTS_ROOT/f6_ablation/compose/seed{s}/best_model.pt, toc_cap_7 = 0

Usage (from the repo root, in Colab):
  python experiments/f6_anomaly_run.py --analysis1
  python experiments/f6_anomaly_run.py --seeds 42          # step (d) validation seed
  python experiments/f6_anomaly_run.py --seeds all          # step (e)

Outputs are written to results/f6_anomaly/ (analysis1_f6.json, seed{s}.json).
Per seed:
  repro   : zero-shot AUC recomputed with the original finetune_home RNG order,
            versus the logged value (gate: |diff| <= 0.005)
  attention: per arm and GAT layer: normalized entropy (all 200 windows),
            critical-node enrichment (100 positive windows), received-vs-F6
            Spearman, mean attention matrix; and JS divergence between arms
  d1      : zero-shot AUC on the same holdout with causal_7 re-initialised under
            5 fixed sub-seeds (init-noise floor)
  d2      : raw-probability diagnostics of the reference zero-shot evaluation
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
INIT_SUBSEEDS = [1, 2, 3, 4, 5]           # D1, fixed in the pre-registration
SAMPLE_SEED, N_POS, N_NEG = 0, 100, 100   # Analysis 2 window sample, fixed in the pre-registration
REPRO_TOL = 0.005
ROOT = Path(os.environ.get("CRUX_RESULTS_ROOT", "/content/drive/MyDrive/CruxSight"))
OUT = Path(__file__).resolve().parent.parent / "results" / "f6_anomaly"
REPO = Path(__file__).resolve().parent.parent


def _git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _ckpt(arm, seed):
    return (ROOT / "causal_fix" / "run4" / f"seed{seed}" / "best_model.pt" if arm == "with_f6"
            else ROOT / "f6_ablation" / "compose" / f"seed{seed}" / "best_model.pt")


def _logged_zero_shot(arm, seed):
    p = (REPO / "results" / ("home_finetune" if arm == "with_f6" else "f6_ablation")
         / f"home_seed{seed}.json")
    return json.load(open(p))["stages"]["zero_shot"]["auc"]


def run_analysis1():
    import torch
    from src import f6_anomaly as fa
    from src.pipeline import CACHE_DIR
    graphs = torch.load(CACHE_DIR / "graphs.pt", weights_only=False)
    cc = graphs[30]["toc_cap"].cpu().numpy()
    ch = graphs[7]["toc_cap"].cpu().numpy()
    res = fa.f6_comparison(cc, ch)
    scales = {}
    for s in SEEDS:                                       # toc_scale actually learned (with-F6 arm)
        sd = torch.load(_ckpt("with_f6", s), map_location="cpu", weights_only=False)["model_state"]
        scales[s] = [float(sd[k]) for k in sorted(sd) if k.endswith(".toc_scale")]
    res["toc_scale_with_f6_by_seed"] = scales
    res["git_sha"] = _git_sha()
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "analysis1_f6.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("cliffs_delta_compose_minus_home", "large_difference",
                                         "wasserstein_1", "home_flagged_values",
                                         "home_flagged_rank_desc")}, indent=1))


def run_seed(seed):
    import torch
    from torch.utils.data import DataLoader, random_split
    from src import f6_anomaly as fa
    from src.core import Config, CachedWindowDataset
    from src.pipeline import MODEL_DIR, CACHE_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config.load(str(MODEL_DIR / "config.yaml"))
    cfg.model.gat_hidden, cfg.model.tft_hidden = 32, 64          # Run 4 architecture
    graphs = torch.load(CACHE_DIR / "graphs.pt", weights_only=False)
    ei7 = graphs[7]["edge_index"].to(device)
    cap7 = {"with_f6": graphs[7]["toc_cap"].to(device)}
    cap7["ablated"] = torch.zeros_like(cap7["with_f6"])
    N = 7

    samples = torch.load(CACHE_DIR / "test.pt", weights_only=False)
    ds = CachedWindowDataset(samples)
    n = len(ds)
    n_ft = int(0.2 * n)                                           # as in pipeline.finetune_home
    _, holdout = random_split(ds, [n_ft, n - n_ft], generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(holdout, batch_size=32, shuffle=False)

    # fixed Analysis-2 window sample (independent of seed)
    labels_all = np.array([int(s["label"]) for s in samples])
    rs = np.random.RandomState(SAMPLE_SEED)
    pos_idx = rs.choice(np.where(labels_all == 1)[0], N_POS, replace=False)
    neg_idx = rs.choice(np.where(labels_all == 0)[0], N_NEG, replace=False)
    x_pos = torch.stack([samples[i]["x"] for i in pos_idx]).to(device)
    x_neg = torch.stack([samples[i]["x"] for i in neg_idx]).to(device)
    x_all = torch.cat([x_pos, x_neg])

    out = dict(seed=seed, git_sha=_git_sha(), n_home_windows=n,
               sample_pos_idx=pos_idx.tolist(), sample_neg_idx=neg_idx.tolist(),
               repro={}, attention={}, d1={}, d2={})
    mats = {}
    for arm in ("with_f6", "ablated"):
        # --- reproduction of the logged zero-shot AUC (original RNG order) ---
        model = fa.load_model_checked(_ckpt(arm, seed), cfg, device, seed=seed)
        # PREBUILD exactly as finetune_home: forward on zeros right after loading, global RNG as left by
        # manual_seed(seed) + CSTGNN init (load_model_checked reproduces that order).
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, cfg.data.window_steps, N, cfg.data.n_features, device=device),
                  ei7, cap7[arm])
        probs, labels = fa.predict_probs(model, loader, ei7, cap7[arm], device)
        auc = float(roc_auc_score(labels, probs))
        logged = _logged_zero_shot(arm, seed)
        out["repro"][arm] = dict(logged=logged, recomputed=auc, abs_diff=abs(auc - logged),
                                 passed=bool(abs(auc - logged) <= REPRO_TOL))
        out["d2"][arm] = fa.prob_diagnostics(probs, labels)

        # --- D1: re-initialise causal_7 under fixed sub-seeds, same holdout ---
        sub = []
        for k in INIT_SUBSEEDS:
            fa.rebuild_causal(model, N, k, cfg.data.window_steps, cfg.data.n_features,
                              ei7, cap7[arm], device)
            p_k, l_k = fa.predict_probs(model, loader, ei7, cap7[arm], device)
            sub.append(float(roc_auc_score(l_k, p_k)))
        out["d1"][arm] = dict(reference_auc=auc, sub_aucs=sub,
                              sub_sd=float(np.std(sub, ddof=1)))

        # --- Analysis 2: attention on the fixed window sample (eval mode, zero-shot) ---
        mats[arm] = fa.attention_matrices(model, x_all, ei7, cap7[arm], N)
        cap_np = cap7[arm].cpu().numpy()
        layers = {}
        for li, A in enumerate(mats[arm]):
            layers[f"layer{li}"] = dict(
                entropy=fa.mean_row_entropy(A),
                enrichment_pos=fa.critical_enrichment(A[:N_POS]),
                sent_vs_f6_spearman=fa.received_vs_cap_spearman(A, cap_np),
                row_sum_min=float(A.sum(-1).min()), row_sum_max=float(A.sum(-1).max()),
                A_mean=A.mean(0).tolist())
        out["attention"][arm] = layers
        del model

    out["attention"]["js_between_arms"] = {
        f"layer{li}": fa.mean_row_js(mats["with_f6"][li], mats["ablated"][li])
        for li in range(len(mats["with_f6"]))}
    out["gates"] = dict(
        repro_passed=all(v["passed"] for v in out["repro"].values()),
        rows_sum_to_one=all(abs(l["row_sum_min"] - 1) < 1e-4 and abs(l["row_sum_max"] - 1) < 1e-4
                            for arm in ("with_f6", "ablated")
                            for l in out["attention"][arm].values()),
        no_nan=not any(np.isnan(l["entropy"]) or np.isnan(l["enrichment_pos"])
                       for arm in ("with_f6", "ablated") for l in out["attention"][arm].values()))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"seed{seed}.json", "w"), indent=1)
    print(f"seed {seed}: gates = {out['gates']}")
    for arm in ("with_f6", "ablated"):
        r = out["repro"][arm]
        print(f"  {arm}: logged {r['logged']:.4f} recomputed {r['recomputed']:.4f} "
              f"| D1 sub-seed SD {out['d1'][arm]['sub_sd']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis1", action="store_true")
    ap.add_argument("--seeds", default="")
    a = ap.parse_args()
    if a.analysis1:
        run_analysis1()
    if a.seeds:
        for s in (SEEDS if a.seeds == "all" else [int(v) for v in a.seeds.split(",")]):
            run_seed(s)
