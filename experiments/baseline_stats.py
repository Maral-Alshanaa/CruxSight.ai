# experiments/baseline_stats.py
"""
Statistics for the Section 5.2.7 baseline comparison. Reads results/baseline/*.json only.

Comparison direction everywhere: no_toc MINUS run4_retrain (positive = no-TOC baseline higher).
Pre-registered family (alpha = 0.05, two-sided): paired t-test, no_toc vs run4_retrain, on
AUC, PatAcc, RCS Top-1. Exact Wilcoxon signed-rank = sensitivity. CP-Recall is reported
descriptively only (saturated: random top-3 = 0.9458, PROTOCOL.md "CP-Recall saturation").
One-sample tests: each arm's RCS Top-1 vs 38.2 % (weighted random baseline).
SD = sample SD (ddof=1). CI = t-based 95 % CI of the mean paired difference.
Holm-adjusted p over the three paired t-tests is an additional (not pre-registered as primary) column.

Verification: recomputed values are compared with the paper numbers at the paper's rounding.
Mismatches are REPORTED; this script never edits a manuscript or a stored paper number.
Exit code 1 if any verification row is a MISMATCH or MISSING.

Usage: python experiments/baseline_stats.py [--results results/baseline]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from configs.runs import SEEDS                                          # noqa: E402

ARMS_NN = ("run4_retrain", "no_toc")
TEST_METRICS = ("auc", "pat_acc", "rcs_top1")
DESCRIPTIVE = ("auc", "cp_recall", "pat_acc", "rcs_top1")
RANDOM_RCS_TOP1 = 38.2
RANDOM_CP_RECALL = 0.9458
ALPHA = 0.05

# Paper numbers as supplied for verification (do NOT edit to match results).
# "wilcoxon_p"/"paired_p" are assumed to refer to RCS Top-1 (the only metric for which that pair is quoted).
PAPER = {
    "no_toc.auc.mean": (0.8737, 4), "no_toc.auc.sd": (0.0150, 4),
    "no_toc.pat_acc.mean": (0.8865, 4), "no_toc.pat_acc.sd": (0.0237, 4),
    "no_toc.rcs_top1.mean": (63.49, 2), "no_toc.rcs_top1.sd": (22.35, 2),
    "run4_retrain.auc.mean": (0.8705, 4), "run4_retrain.auc.sd": (0.0133, 4),
    "run4_retrain.pat_acc.mean": (0.9047, 4), "run4_retrain.pat_acc.sd": (0.0281, 4),
    "run4_retrain.rcs_top1.mean": (62.81, 2), "run4_retrain.rcs_top1.sd": (8.00, 2),
    "paired.rcs_top1.p_t": (0.902, 3), "paired.rcs_top1.p_wilcoxon": (0.625, 3),
}
# Values recorded in PROTOCOL.md (not in the paper list above); checked as a second block.
PROTOCOL_RECORDED = {
    "paired.auc.t": (0.600, 3), "paired.auc.p_t": (0.5636, 4),
    "paired.pat_acc.t": (-1.742, 3), "paired.pat_acc.p_t": (0.1156, 4),
    "paired.rcs_top1.t": (0.127, 3), "paired.rcs_top1.p_t": (0.9018, 4),
    "paired.rcs_top1.dz": (0.04, 2),
    "run4_retrain.cp_recall.mean": (0.9056, 4), "run4_retrain.cp_recall.sd": (0.0389, 4),
    "no_toc.cp_recall.mean": (0.9675, 4), "no_toc.cp_recall.sd": (0.0576, 4),
    "one_sample.run4_retrain.rcs_top1.t": (9.733, 3), "one_sample.run4_retrain.rcs_top1.mean_diff": (24.6, 1),
    "logreg.auc": (0.7120, 4), "logreg.cp_recall": (0.4527, 4), "logreg.pat_acc": (0.8816, 4),
    "logreg.rcs_top1": (41.72, 2),
}


# ------------------------------------------------------------------ loading
def load_arm(results: Path, arm: str) -> dict:
    recs = {}
    for s in SEEDS:
        f = results / f"{arm}_seed{s}.json"
        if not f.exists():
            raise FileNotFoundError(f"missing {f}")
        r = json.loads(f.read_text())
        assert r["seed"] == s and r["arm"] == arm, (f, r.get("seed"), r.get("arm"))
        for m in DESCRIPTIVE:
            assert r.get(m) is not None and np.isfinite(r[m]), (f, m, r.get(m))
        recs[s] = r
    return recs


def vec(recs: dict, metric: str) -> np.ndarray:
    return np.array([recs[s][metric] for s in SEEDS], dtype=float)


# ------------------------------------------------------------------ tests
def wilcoxon_exact(d: np.ndarray) -> dict:
    """Exact two-sided Wilcoxon signed-rank by enumerating all 2^n sign patterns (zeros dropped,
    mid-ranks for ties). Valid conditional on the observed |d| (permutation test)."""
    d = np.asarray(d, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return dict(W=float("nan"), p=float("nan"), n_eff=0, ties=False)
    assert n <= 22, "enumeration limited to n<=22"
    ranks = stats.rankdata(np.abs(nz))
    total = ranks.sum()
    w_plus = ranks[nz > 0].sum()
    w_obs = min(w_plus, total - w_plus)
    signs = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    wp = signs @ ranks
    stat = np.minimum(wp, total - wp)
    p = float(np.mean(stat <= w_obs + 1e-9))
    return dict(W=float(w_obs), p=p, n_eff=int(n), ties=bool(len(np.unique(np.abs(nz))) < n))


def t_ci95(x: np.ndarray) -> tuple:
    n = len(x)
    half = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    t, p = stats.ttest_rel(a, b)
    w = wilcoxon_exact(d)
    return dict(n=len(d), mean_a=float(a.mean()), sd_a=float(a.std(ddof=1)),
                mean_b=float(b.mean()), sd_b=float(b.std(ddof=1)),
                mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)),
                ci95=t_ci95(d), t=float(t), df=len(d) - 1, p_t=float(p),
                dz=float(d.mean() / d.std(ddof=1)),
                W=w["W"], p_wilcoxon=w["p"], wilcoxon_n_eff=w["n_eff"], wilcoxon_ties=w["ties"],
                higher_a=int((d > 0).sum()), higher_b=int((d < 0).sum()), tied=int((d == 0).sum()))


def one_sample(x: np.ndarray, mu: float) -> dict:
    t, p = stats.ttest_1samp(x, mu)
    w = wilcoxon_exact(x - mu)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)), mu=mu,
                mean_diff=float(x.mean() - mu), ci95_mean=t_ci95(x),
                t=float(t), df=len(x) - 1, p_t=float(p), dz=float((x.mean() - mu) / x.std(ddof=1)),
                W=w["W"], p_wilcoxon=w["p"], above=int((x > mu).sum()))


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = running
    return out


def weighted_random_top1(counts: dict) -> float:
    """Recompute the weighted random RCS Top-1 baseline from per-pattern positive-val counts."""
    sizes = {"A": 15, "B": 9, "C": 4, "D": 11, "E": 19, "F": 11}          # |flagged set|, src/core.py PATTERNS
    n = sum(c for p, c in counts.items() if p in sizes)
    return 100.0 * sum(c * sizes[p] / 30.0 for p, c in counts.items() if p in sizes) / n


# ------------------------------------------------------------------ report
def fmt(x, nd):
    return f"{x:.{nd}f}"


def check(table: dict, got: dict) -> list:
    rows = []
    for key, (val, nd) in table.items():
        g = got.get(key)
        if g is None:
            rows.append((key, val, None, nd, "MISSING"))
        else:
            ok = round(float(g), nd) == round(float(val), nd)
            rows.append((key, val, float(g), nd, "match" if ok else "MISMATCH"))
    return rows


def md_check(title: str, rows: list) -> str:
    out = [f"### {title}", "", "| quantity | paper / recorded | recomputed from results/baseline | status |", "|---|---|---|---|"]
    for key, val, g, nd, st in rows:
        out.append(f"| {key} | {fmt(val, nd)} | {'-' if g is None else fmt(g, nd + 2)} | {st} |")
    return "\n".join(out) + "\n"


def run(results: Path, out: Path) -> int:
    recs = {a: load_arm(results, a) for a in ARMS_NN}
    shas = {r["git_sha"] for a in ARMS_NN for r in recs[a].values()}
    prereg = {r.get("prereg_commit") for a in ARMS_NN for r in recs[a].values()}
    nvals = {r["rcs_top1_n"] for a in ARMS_NN for r in recs[a].values()}
    counts = recs["run4_retrain"][SEEDS[0]]["preflight"]["val_pos_pattern_counts"] if "preflight" in recs["run4_retrain"][SEEDS[0]] else None

    res = {"seeds": SEEDS, "git_shas": sorted(shas), "prereg_commits": sorted(map(str, prereg)),
           "rcs_top1_n_values": sorted(nvals), "arms": {}, "paired": {}, "one_sample": {}}
    got = {}
    for a in ARMS_NN:
        res["arms"][a] = {}
        for m in DESCRIPTIVE:
            v = vec(recs[a], m)
            res["arms"][a][m] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), values=v.tolist())
            got[f"{a}.{m}.mean"], got[f"{a}.{m}.sd"] = float(v.mean()), float(v.std(ddof=1))
    for m in TEST_METRICS:
        r = paired(vec(recs["no_toc"], m), vec(recs["run4_retrain"], m))
        res["paired"][m] = r
        for k in ("t", "p_t", "dz"):
            got[f"paired.{m}.{k}"] = r[k]
        got[f"paired.{m}.p_wilcoxon"] = r["p_wilcoxon"]
    hp = holm({m: res["paired"][m]["p_t"] for m in TEST_METRICS})
    for m in TEST_METRICS:
        res["paired"][m]["p_t_holm"] = hp[m]
    for a in ARMS_NN:
        r = one_sample(vec(recs[a], "rcs_top1"), RANDOM_RCS_TOP1)
        res["one_sample"][a] = r
        got[f"one_sample.{a}.rcs_top1.t"], got[f"one_sample.{a}.rcs_top1.mean_diff"] = r["t"], r["mean_diff"]
    if counts:
        res["recomputed_random_rcs_top1"] = weighted_random_top1(counts)

    lr = results / "logreg.json"
    if lr.exists():
        res["logreg"] = json.loads(lr.read_text())
        for m in ("auc", "cp_recall", "pat_acc", "rcs_top1"):
            got[f"logreg.{m}"] = res["logreg"][m]

    c_paper, c_prot = check(PAPER, got), check(PROTOCOL_RECORDED, got)
    res["verification"] = [dict(key=k, recorded=v, recomputed=g, status=s) for k, v, g, _, s in c_paper + c_prot]
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_stats.json").write_text(json.dumps(res, indent=2))

    # ---- markdown
    L = ["# Baseline comparison (Section 5.2.7), 10 seeds, recomputed from results/baseline",
         "", f"git_sha(s): {', '.join(sorted(shas))}; seeds: {SEEDS}; RCS Top-1 n per seed: {sorted(nvals)}", "",
         "## Table A: per arm (mean ± sample SD)", "",
         "| arm | AUC | PatAcc | RCS Top-1 (%) | CP-Recall (random top-3 = 0.9458; saturated) |", "|---|---|---|---|---|"]
    for a in ARMS_NN:
        s = res["arms"][a]
        L.append(f"| {a} | {fmt(s['auc']['mean'],4)} ± {fmt(s['auc']['sd'],4)} | {fmt(s['pat_acc']['mean'],4)} ± "
                 f"{fmt(s['pat_acc']['sd'],4)} | {fmt(s['rcs_top1']['mean'],2)} ± {fmt(s['rcs_top1']['sd'],2)} | "
                 f"{fmt(s['cp_recall']['mean'],4)} ± {fmt(s['cp_recall']['sd'],4)} |")
    if "logreg" in res:
        g = res["logreg"]
        L.append(f"| logreg F0-F6 (single deterministic fit) | {fmt(g['auc'],4)} | {fmt(g['pat_acc'],4)} | "
                 f"{fmt(g['rcs_top1'],2)} | {fmt(g['cp_recall'],4)} |")
    L += ["", "## Table B: paired tests, no_toc − run4_retrain (two-sided, df = 9)", "",
          "| metric | mean diff [95% CI] | t | p (t) | p Holm (3 tests) | exact Wilcoxon W / p | d_z | seeds no_toc higher / Run4 higher / tied |",
          "|---|---|---|---|---|---|---|---|"]
    for m in TEST_METRICS:
        r = res["paired"][m]
        nd = 2 if m == "rcs_top1" else 4
        flag = " (ties/zeros: exact p is conditional)" if (r["wilcoxon_ties"] or r["tied"]) else ""
        L.append(f"| {m} | {fmt(r['mean_diff'],nd)} [{fmt(r['ci95'][0],nd)}, {fmt(r['ci95'][1],nd)}] | {fmt(r['t'],3)} | "
                 f"{fmt(r['p_t'],4)} | {fmt(r['p_t_holm'],4)} | {fmt(r['W'],1)} / {fmt(r['p_wilcoxon'],4)}{flag} | "
                 f"{fmt(r['dz'],3)} | {r['higher_a']} / {r['higher_b']} / {r['tied']} |")
    L += ["", f"## Table C: one-sample tests, RCS Top-1 vs {RANDOM_RCS_TOP1}% (weighted random baseline), df = 9", "",
          "| arm | mean − 38.2 [95% CI of mean] | t | p (t) | exact Wilcoxon p | d_z | seeds above |", "|---|---|---|---|---|---|---|"]
    for a in ARMS_NN:
        r = res["one_sample"][a]
        L.append(f"| {a} | {fmt(r['mean_diff'],2)} [{fmt(r['ci95_mean'][0],2)}, {fmt(r['ci95_mean'][1],2)}] | "
                 f"{fmt(r['t'],3)} | {fmt(r['p_t'],5)} | {fmt(r['p_wilcoxon'],4)} | {fmt(r['dz'],3)} | {r['above']}/10 |")
    if "recomputed_random_rcs_top1" in res:
        L.append(f"\nWeighted random RCS Top-1 recomputed from val_pos_pattern_counts in the result files and "
                 f"src/core.py PATTERNS: {res['recomputed_random_rcs_top1']:.4f}% (pre-registered value used: {RANDOM_RCS_TOP1}%).")
    L += ["", md_check("Verification 1: paper numbers vs recomputed", c_paper),
          md_check("Verification 2: values recorded in PROTOCOL.md vs recomputed", c_prot)]
    bad = [r for r in c_paper + c_prot if r[4] != "match"]
    L.append(f"{'No mismatches.' if not bad else str(len(bad)) + ' row(s) not matching (reported, nothing adjusted).'}")
    (out / "results_table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results" / "baseline"))
    args = ap.parse_args()
    sys.exit(run(Path(args.results), Path(args.results)))
