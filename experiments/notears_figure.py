# experiments/notears_figure.py
"""
Descriptive overview figure for the NOTEARS-layer analysis (tag notears-analysis-v1).
Reads only stored artifacts (results/notears); no test, no new statistic. Labels in English.

Usage: python experiments/notears_figure.py [RESULTS_DIR] [OUT_PREFIX]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/notears")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else R / "figures" / "notears_overview"
OUT.parent.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
LAMS = ["lam0", "lam0.01", "lam0.05", "lam0.1", "lam0.2", "lam0.5"]
LAB = {a: a.replace("lam", "") for a in LAMS}
TAUS = ["0.01", "0.05", "0.1", "0.2", "0.3", "0.5", "0.7"]
cols = dict(zip(LAMS, plt.cm.viridis(np.linspace(0.05, 0.9, len(LAMS)))))


def rec(arm, s):
    return json.loads((R / f"{arm}_seed{s}.json").read_text())


def hist(arm, s):
    return json.loads((R / arm / f"seed{s}" / "causal_history.json").read_text())


def ok_seeds(arm):
    return [s for s in SEEDS if not rec(arm, s).get("diverged")]


from scipy.stats import spearmanr

M = ~np.eye(30, dtype=bool)
IDX = np.argwhere(M)
XR = np.zeros((len(IDX), 30)); XC = np.zeros((len(IDX), 30))
for n, (i, j) in enumerate(IDX):
    XR[n, i] = 1; XC[n, j] = 1


def variance_explained(Gm):
    """R^2 of additive models on the off-diagonal entries: source-node effect, target-node effect, both."""
    y = Gm[M]
    def r2(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - (y - X @ b).var() / y.var()
    return r2(XR), r2(XC), r2(np.hstack([XR, XC[:, 1:]]))


ei = np.array(json.loads((R / "true_edges.json").read_text())["edge_index"])
A = np.zeros((30, 30)); A[ei[0], ei[1]] = 1
Au = np.maximum(A, A.T); np.fill_diagonal(Au, 0)
deg = Au.sum(1)

fig, ax = plt.subplots(2, 4, figsize=(19, 8.6))

# A: seed-mean learned graph (reference arm)
okref = ok_seeds("lam0.05")
Gs = [np.load(R / "lam0.05" / f"seed{s}" / "causal_analysis.npz")["val_graph"] for s in okref]
G = np.mean(Gs, axis=0)
im = ax[0, 0].imshow(G, cmap="magma", vmin=0)
ax[0, 0].set_title("A. Learned graph, seed-mean\n(lambda_causal=0.05, n=10)", fontsize=10)
ax[0, 0].set_xlabel("target node j"); ax[0, 0].set_ylabel("source node i")
fig.colorbar(im, ax=ax[0, 0], fraction=0.046, label="edge weight G[i,j]")

# B: call-graph adjacency
ax[0, 1].imshow(A, cmap="Greys", vmin=0, vmax=1)
ax[0, 1].set_title(f"B. Call graph\n(34 undirected edges, {int(A.sum())} directed entries)", fontsize=10)
ax[0, 1].set_xlabel("target node j"); ax[0, 1].set_ylabel("source node i")

# C: node-level view (exploratory)
rowm = np.mean([[g[i][np.arange(30) != i].mean() for i in range(30)] for g in Gs], axis=0)
rho_seed = [spearmanr([g[i][np.arange(30) != i].mean() for i in range(30)], deg).statistic for g in Gs]
jit = np.random.default_rng(1).uniform(-.12, .12, 30)
ax[0, 2].scatter(deg + jit, rowm, s=22, color="tab:purple")
ax[0, 2].set_xlabel("call-graph degree of the node (undirected)")
ax[0, 2].set_ylabel("mean outgoing weight of the node in G")
ax[0, 2].set_title(f"C. Node-level view (exploratory)\nper-seed Spearman {np.mean(rho_seed):.2f} +/- {np.std(rho_seed, ddof=1):.2f}", fontsize=10)

# D: variance explained by node effects (exploratory)
w = 0.26
for k, (lab, c) in enumerate((("source-node effect", "tab:purple"), ("target-node effect", "tab:orange"),
                              ("both (additive)", "tab:gray"))):
    vals = [np.mean([variance_explained(np.load(R / a / f"seed{s}" / "causal_analysis.npz")["val_graph"])[k]
                     for s in ok_seeds(a)]) for a in LAMS]
    ax[0, 3].bar(np.arange(len(LAMS)) + (k - 1) * w, vals, w, label=lab, color=c)
ax[0, 3].set_xticks(range(len(LAMS))); ax[0, 3].set_xticklabels([LAB[a] for a in LAMS])
ax[0, 3].set_xlabel("lambda_causal"); ax[0, 3].set_ylabel("share of variance of G explained (R^2)")
ax[0, 3].set_ylim(0, 1.05); ax[0, 3].legend(fontsize=8, loc="center right")
ax[0, 3].set_title("D. G is almost a per-source-node score\n(exploratory)", fontsize=10)

# E: h_exact versus epoch
for a in LAMS:
    hs = [[h["val_graph"]["h_exact"] for h in hist(a, s)] for s in ok_seeds(a)]
    L = max(len(x) for x in hs)
    m = [np.mean([x[e] for x in hs if len(x) > e]) if sum(len(x) > e for x in hs) >= 5 else np.nan
         for e in range(L)]
    ax[1, 0].plot(np.arange(1, L + 1), m, color=cols[a], label=LAB[a])
ax[1, 0].set_yscale("log")
ax[1, 0].set_title("E. h_exact of the validation graph\n(mean over seeds; never reaches 0)", fontsize=10)
ax[1, 0].set_xlabel("epoch"); ax[1, 0].set_ylabel("h_exact")
ax[1, 0].legend(title="lambda_causal", fontsize=8, ncol=2)

# F: AUC versus lambda_causal (+ fn_weight 5.0 arms)
rng = np.random.default_rng(0)
for i, a in enumerate(LAMS):
    v = np.array([rec(a, s)["auc"] for s in SEEDS])
    ax[1, 1].scatter(i + rng.uniform(-.12, .12, len(v)), v, s=14, color=cols[a], alpha=.7)
    ax[1, 1].errorbar(i, v.mean(), v.std(ddof=1), color="k", capsize=3, lw=1.2)
for a, i in (("fn5_lam0.05", 2.35), ("fn5_lam0.2", 4.35)):
    for s in SEEDS:
        r = rec(a, s)
        ax[1, 1].scatter(i + rng.uniform(-.1, .1), r["auc"], s=26 if r["diverged"] else 14,
                         marker="x" if r["diverged"] else "s",
                         color="crimson" if r["diverged"] else "grey", alpha=.9)
ax[1, 1].scatter([], [], marker="s", color="grey", s=14, label="fn_weight 5.0")
ax[1, 1].scatter([], [], marker="x", color="crimson", label="fn_weight 5.0, saturated (flagged)")
ax[1, 1].set_ylim(0.825, 0.905)
ax[1, 1].set_xticks(range(len(LAMS))); ax[1, 1].set_xticklabels([LAB[a] for a in LAMS])
ax[1, 1].set_xlabel("lambda_causal"); ax[1, 1].set_ylabel("validation AUC (best epoch)")
ax[1, 1].set_title("F. AUC vs lambda_causal\n(points: seeds; bar: mean +/- SD)", fontsize=10)
ax[1, 1].legend(fontsize=8, loc="lower right")

# G, H: density and acyclic fraction vs threshold
tx = [float(t) for t in TAUS]
for a in LAMS:
    ss = ok_seeds(a)
    dens, dag = [], []
    for t in TAUS:
        vals = [[h for h in hist(a, s) if h["epoch"] == rec(a, s)["best_epoch"]][0]["val_graph"]["per_tau"][t]
                for s in ss]
        dens.append(np.mean([v["density"] for v in vals]))
        dag.append(np.mean([v["is_dag"] for v in vals]))
    ax[1, 2].plot(tx, dens, "o-", color=cols[a], label=LAB[a], ms=4)
    ax[1, 3].plot(tx, dag, "o-", color=cols[a], label=LAB[a], ms=4)
for k in (2, 3):
    ax[1, k].set_xscale("log"); ax[1, k].set_xlabel("threshold tau  (edge kept if G[i,j] > tau)")
ax[1, 2].set_ylabel("edge density"); ax[1, 2].set_title("G. Density of the binarised graph", fontsize=10)
ax[1, 3].set_ylabel("fraction of seeds acyclic"); ax[1, 3].set_title("H. Binarised graph is a DAG only at large tau", fontsize=10)
ax[1, 3].set_ylim(-0.03, 1.03)
ax[1, 2].legend(title="lambda_causal", fontsize=8, ncol=2)

fig.suptitle("NOTEARS-layer analysis, Run 4 architecture, 10 seeds per arm (results tag notears-analysis-v1)", y=1.0)
fig.tight_layout()
fig.savefig(str(OUT) + ".png", dpi=200)
fig.savefig(str(OUT) + ".pdf")
print("saved", str(OUT) + ".png")
