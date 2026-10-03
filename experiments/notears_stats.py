# experiments/notears_stats.py
"""
Analysis of the NOTEARS / causal-layer study (pre-registration tag
notears-analysis-prereg). LOCKED after the tag. Reads what experiments/
notears_run.py wrote under OUT and prints / saves everything defined in the
PROTOCOL.md section "NOTEARS-layer analysis -- pre-registration".

Usage:  python experiments/notears_stats.py [OUT]     (default: CRUX_OUT_NT)

Primary tests (declared before any real run, alpha = 0.05, no modification
after seeing results):
  P1  Friedman test on validation AUC across the six lambda_causal arms
      (blocks = seeds). A run with no usable epoch (divergence) enters with
      AUC = 0.5; sensitivity: drop every seed with a diverged run in any arm.
  P2  Interpretability, reference arm lam0.05 only: per seed, permutation
      p-value of undirected precision@|E| against the call-graph edge set
      (node-label permutation, 10000 draws, seed = run seed); then a one-sided
      exact binomial test that the fraction of seeds with p < 0.05 exceeds 0.05.
Secondary: Holm-corrected paired Wilcoxon of each other lambda arm vs lam0.05
on AUC (5 tests); Holm-corrected paired Wilcoxon fn5 vs fn1.5 at lambda 0.05 and
0.20 on AUC (2 tests). Everything else is descriptive (mean +/- SD, n=10).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import SEEDS
import configs.notears_runs as NR
from src import causal_analysis as ca

DIVERGED_AUC = 0.5
N_PERM = 10000
SAT_ENTROPY, SAT_FRAC = 0.99, 0.90     # "uninformative graph" flags (rule 4)
OUT_DEFAULT = Path(os.environ.get("CRUX_OUT_NT", "/content/drive/MyDrive/CruxSight/notears"))


# ----------------------------- loading --------------------------------------

def load_run(out: Path, arm: str, seed: int) -> dict:
    marker = out / f"{arm}_seed{seed}.json"
    if not marker.exists():
        raise FileNotFoundError(f"missing result {marker}")
    rec = json.loads(marker.read_text())
    rd = out / arm / f"seed{seed}"
    rec["history"] = json.loads((rd / "causal_history.json").read_text())
    npz = rd / "causal_analysis.npz"
    if npz.exists():
        z = np.load(npz)
        rec["val_graph"], rec["W_sigmoid"] = z["val_graph"], z["W_sigmoid"]
    return rec


def load_all(out: Path, arms=None, seeds=None) -> dict:
    arms = list(arms or NR.ARMS)
    seeds = list(seeds or SEEDS)
    return {a: {s: load_run(out, a, s) for s in seeds} for a in arms}


# ----------------------------- generic stats --------------------------------

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def paired_wilcoxon(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.all(d == 0):
        return dict(statistic=0.0, p=1.0, n=len(d))
    r = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return dict(statistic=float(r.statistic), p=float(r.pvalue), n=len(d))


def mean_sd(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if len(v) == 0:
        return dict(mean=None, sd=None, n=0)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                n=int(len(v)))


def auc_value(rec):
    return DIVERGED_AUC if rec.get("auc") is None else float(rec["auc"])


# ----------------------------- P1 / secondary --------------------------------

def p1_friedman(runs, seeds, drop_diverged_seeds=False):
    arms = list(NR.FRIEDMAN_ARMS)
    use = [s for s in seeds
           if not (drop_diverged_seeds and any(runs[a][s].get("diverged") for a in NR.ARMS))]
    M = np.array([[auc_value(runs[a][s]) for a in arms] for s in use])
    r = stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    n, k = M.shape
    return dict(arms=arms, n_seeds=n, chi2=float(r.statistic), p=float(r.pvalue),
                kendall_w=float(r.statistic / (n * (k - 1))),
                mean_auc={a: float(M[:, j].mean()) for j, a in enumerate(arms)},
                seeds_used=use)


def secondary_vs_reference(runs, seeds):
    ref = [auc_value(runs[NR.REFERENCE_ARM][s]) for s in seeds]
    names = [a for a in NR.FRIEDMAN_ARMS if a != NR.REFERENCE_ARM]
    res = [paired_wilcoxon([auc_value(runs[a][s]) for s in seeds], ref) for a in names]
    adj = holm([r["p"] for r in res])
    return {a: dict(r, p_holm=p, mean_diff=float(np.mean(
        [auc_value(runs[a][s]) - auc_value(runs[NR.REFERENCE_ARM][s]) for s in seeds])))
        for a, r, p in zip(names, res, adj)}


def secondary_fn_weight(runs, seeds):
    pairs = [("lam0.05", "fn5_lam0.05"), ("lam0.2", "fn5_lam0.2")]
    res = [paired_wilcoxon([auc_value(runs[b][s]) for s in seeds],
                           [auc_value(runs[a][s]) for s in seeds]) for a, b in pairs]
    adj = holm([r["p"] for r in res])
    return {f"{b}_vs_{a}": dict(r, p_holm=p) for (a, b), r, p in zip(pairs, res, adj)}


# ----------------------------- P2 / descriptives -----------------------------

def best_summary(rec):
    """val_graph summary at the best epoch, from causal_history.json."""
    for h in rec["history"]:
        if h["epoch"] == rec["best_epoch"]:
            return h["val_graph"]
    raise KeyError(f"best epoch {rec['best_epoch']} not in history")


def run_descriptives(rec):
    if rec.get("auc") is None:
        return dict(diverged=True)
    g = best_summary(rec)
    first = rec["history"][0]["val_graph"]
    ent = g["entropy"]
    sat = ca.saturation_fraction(rec["W_sigmoid"])
    return dict(
        diverged=bool(rec.get("diverged")), auc=rec["auc"], pat_acc=rec["pat_acc"],
        cp_recall=rec["cp_recall"], rcs_top1=rec["rcs_top1"],
        h_impl=g["h_impl"], h_exact=g["h_exact"], h_standard=g["h_standard"],
        h_impl_epoch1=first["h_impl"], h_exact_epoch1=first["h_exact"],
        min_tau_dag=g["min_tau_dag"], entropy=ent, w_mean=g["w_mean"], w_max=g["w_max"],
        w_sigmoid_saturation=sat,
        uninformative=bool(ent > SAT_ENTROPY or sat > SAT_FRAC),
        per_tau=g["per_tau"],
        best_epoch=rec["best_epoch"], best_pos_rate=rec.get("best_pos_rate"))


def arm_table(runs, arm, seeds):
    d = [run_descriptives(runs[arm][s]) for s in seeds]
    ok = [x for x in d if not x.get("diverged")]
    out = dict(n_runs=len(d), n_diverged=len(d) - len(ok),
               n_uninformative=sum(x.get("uninformative", False) for x in ok))
    for k in ("auc", "pat_acc", "cp_recall", "rcs_top1", "h_impl", "h_exact", "h_standard",
              "h_impl_epoch1", "h_exact_epoch1", "min_tau_dag", "entropy",
              "w_sigmoid_saturation", "w_mean", "w_max"):
        out[k] = mean_sd([x[k] for x in ok])
    out["per_tau"] = {}
    for t in map(str, ca.TAUS):
        out["per_tau"][t] = dict(
            density=mean_sd([x["per_tau"][t]["density"] for x in ok]),
            frac_dag=float(np.mean([x["per_tau"][t]["is_dag"] for x in ok])) if ok else None,
            n_two_cycles=mean_sd([x["per_tau"][t]["n_two_cycles"] for x in ok]),
            largest_scc=mean_sd([x["per_tau"][t]["largest_scc"] for x in ok]))
    return out


def structure_vs_callgraph(runs, arm, seeds, edge_index, d=30):
    te_u = ca.true_edge_set(edge_index, d, directed=False)
    te_d = ca.true_edge_set(edge_index, d, directed=True)
    te_r = ca.true_edge_set(edge_index, d, directed=True, reverse=True)
    rows = []
    for s in seeds:
        rec = runs[arm][s]
        if rec.get("auc") is None:
            continue
        G = rec["val_graph"]
        rows.append(dict(
            seed=s, uninformative=run_descriptives(rec)["uninformative"],
            prec_undirected=ca.precision_at_k(G, te_u, directed=False),
            p_undirected=ca.permutation_pvalue(G, te_u, False, N_PERM, seed=s),
            prec_directed=ca.precision_at_k(G, te_d, directed=True),
            prec_directed_reversed=ca.precision_at_k(G, te_r, directed=True)))
    k = len(te_u)
    k_sig = sum(r["p_undirected"] < 0.05 for r in rows)
    binom = stats.binomtest(k_sig, len(rows), 0.05, alternative="greater") if rows else None
    rng = np.random.default_rng(0)
    pairs = d * (d - 1) // 2
    rand = []
    for _ in range(500):
        a = set(rng.choice(pairs, k, replace=False).tolist())
        b = set(rng.choice(pairs, k, replace=False).tolist())
        rand.append(len(a & b) / len(a | b))
    rand_j = float(np.mean(rand))
    graphs = [runs[arm][s]["val_graph"] for s in seeds if runs[arm][s].get("auc") is not None]
    jac = [ca.jaccard_top_k(graphs[i], graphs[j], k, directed=False)
           for i in range(len(graphs)) for j in range(i + 1, len(graphs))]
    return dict(n_true_undirected=k, n_true_directed=len(te_d),
                chance_precision_undirected=k / pairs, per_seed=rows,
                p2_k_significant=k_sig, p2_n=len(rows),
                p2_binom_p=None if binom is None else float(binom.pvalue),
                mean_pairwise_jaccard=mean_sd(jac),
                random_jaccard_expectation=float(rand_j))


# ----------------------------- main ------------------------------------------

def analyse(out: Path, seeds=None):
    seeds = list(seeds or SEEDS)
    runs = load_all(out, seeds=seeds)
    ei = np.array(json.loads((out / "true_edges.json").read_text())["edge_index"])
    res = dict(
        design=dict(arms=list(NR.ARMS), seeds=seeds, diverged_auc=DIVERGED_AUC, n_perm=N_PERM),
        p1=p1_friedman(runs, seeds),
        p1_sensitivity_drop_diverged_seeds=p1_friedman(runs, seeds, drop_diverged_seeds=True),
        secondary_vs_reference=secondary_vs_reference(runs, seeds),
        secondary_fn_weight=secondary_fn_weight(runs, seeds),
        arms={a: arm_table(runs, a, seeds) for a in NR.ARMS},
        p2_reference_arm=structure_vs_callgraph(runs, NR.REFERENCE_ARM, seeds, ei),
        structure_other_arms={a: structure_vs_callgraph(runs, a, seeds, ei)
                              for a in NR.FRIEDMAN_ARMS if a != NR.REFERENCE_ARM},
        git_shas=sorted({runs[a][s]["git_sha"] for a in runs for s in seeds}),
    )
    return res


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DEFAULT
    res = analyse(out)
    (out / "notears_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("p1", "p1_sensitivity_drop_diverged_seeds",
                                          "secondary_vs_reference", "secondary_fn_weight")},
                     indent=2))
    print("P2:", {k: res["p2_reference_arm"][k] for k in
                  ("p2_k_significant", "p2_n", "p2_binom_p", "n_true_undirected",
                   "chance_precision_undirected", "mean_pairwise_jaccard",
                   "random_jaccard_expectation")})
    print(f"full output: {out / 'notears_analysis.json'}")
