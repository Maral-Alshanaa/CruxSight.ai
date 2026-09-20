# experiments/stage4_tables.py
import csv
import itertools
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import SEEDS

RES = Path("results/multiseed")
OUT = Path("results")
METRICS = [("auc", "AUC"), ("cp_recall", "CP-Recall"), ("pat_acc", "PatAcc")]
EXACT_BASELINE = 1 - comb(12, 3) / comb(30, 3)   # top-3 of 30 nodes, 18 on the critical path


def pm(m, s):
    return f"{m:.4f} ± {s:.4f}"


def tx(s):
    return (s.replace("±", r"$\pm$").replace("−", "$-$").replace("%", r"\%")
             .replace(">", "$>$").replace("d_z", "$d_z$"))


def signflip_exact(d):
    """Exact two-sided Wilcoxon signed-rank p by enumerating all sign patterns (valid with ties)."""
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    tot = r.sum()
    w_plus = r[d > 0].sum()
    dist = np.array([np.dot(s, r) for s in itertools.product([0, 1], repeat=len(d))])
    p = float(np.mean(np.abs(dist - tot / 2) >= abs(w_plus - tot / 2) - 1e-9))
    return float(min(w_plus, tot - w_plus)), p


def main():
    summ = json.loads((RES / "aggregated_summary.json").read_text(encoding="utf-8"))
    st = json.loads((RES / "statistical_analysis.json").read_text(encoding="utf-8"))
    base = st["baseline_comparison"]["run2"]["random_baseline"]
    runs = sorted(summ)

    # independent recomputation of the primary test; must match stage3
    x = np.array(summ["run2"]["cp_recall"]["raw_values"])
    y = np.array(summ["run4"]["cp_recall"]["raw_values"])
    d = y - x
    n = len(d)
    assert n == len(SEEDS)
    t, p = stats.ttest_rel(y, x)
    assert np.isclose(t, st["primary_test_paired_ttest"]["t_statistic"])
    assert np.isclose(p, st["primary_test_paired_ttest"]["p_value"])
    sd_d = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd_d / np.sqrt(n)
    ci = (d.mean() - half, d.mean() + half)
    dz = d.mean() / sd_d
    w_stat, p_exact = signflip_exact(d)
    p_scipy = st["sensitivity_test_wilcoxon"]["p_value"]

    # ---- Table A: mean ± SD per run ----
    rows = []
    for r in runs:
        cells = [pm(summ[r][m]["mean"], summ[r][m]["std"]) for m, _ in METRICS]
        above = sum(v > base for v in summ[r]["cp_recall"]["raw_values"])
        rows.append((r.replace("run", "Run "), *cells, f"{above}/{n}"))
    header = ("Run", *[lab for _, lab in METRICS], f"CP-Recall > {base:.4f}")

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    md.append(f"\nMean ± sample SD over {n} seeds. Baseline {base:.4f} (thesis, Monte Carlo); exact value {EXACT_BASELINE:.4f}.")
    (OUT / "table_multiseed.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    with open(OUT / "table_multiseed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run"] + [f"{m}_{k}" for m, _ in METRICS for k in ("mean", "std")])
        for r in runs:
            w.writerow([r] + [round(summ[r][m][k], 6) for m, _ in METRICS for k in ("mean", "std")])

    body = " \\\\\n".join(" & ".join(tx(c) for c in row) for row in rows) + " \\\\"
    tex_a = ("\\begin{table}[t]\n\\centering\n"
             f"\\caption{{Performance across {n} random seeds (mean $\\pm$ sample SD). "
             f"Random CP-Recall baseline: {base:.4f} (exact: {EXACT_BASELINE:.4f}).}}\n"
             "\\label{tab:multiseed}\n\\begin{tabular}{lcccc}\n\\hline\n"
             + " & ".join(tx(h) for h in header) + " \\\\ \\hline\n" + body + "\n\\hline\n"
             "\\end{tabular}\n\\end{table}\n")
    (OUT / "table_multiseed.tex").write_text(tex_a, encoding="utf-8")

    # ---- Table B: Run 2 vs Run 4 on CP-Recall ----
    b = [
        ("Run 2 CP-Recall (mean ± SD)", pm(x.mean(), x.std(ddof=1))),
        ("Run 4 CP-Recall (mean ± SD)", pm(y.mean(), y.std(ddof=1))),
        ("Mean difference, Run 4 − Run 2 (95% CI)", f"{d.mean():.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"),
        ("Paired t-test (two-sided)", f"t({n - 1}) = {t:.3f}, p = {p:.4f}"),
        ("Wilcoxon signed-rank (exact sign-flip)", f"W = {w_stat:.1f}, p = {p_exact:.4f}"),
        ("Cohen's d_z", f"{dz:.2f}"),
        ("Seeds with Run 4 > Run 2", f"{int((d > 0).sum())}/{n}"),
    ]
    mdb = ["| Statistic | Value |", "|---|---|"] + [f"| {a} | {v} |" for a, v in b]
    mdb.append(f"\nscipy default Wilcoxon p (stage3) = {p_scipy:.4f}. Primary test: paired t-test, alpha = 0.05.")
    (OUT / "table_stats.md").write_text("\n".join(mdb) + "\n", encoding="utf-8")

    tex_b = ("\\begin{table}[t]\n\\centering\n"
             f"\\caption{{Paired comparison of Run~2 and Run~4 on CP-Recall ($n={n}$ seeds). "
             f"Exact sign-flip Wilcoxon $p$; scipy default $p={p_scipy:.4f}$.}}\n"
             "\\label{tab:stats}\n\\begin{tabular}{ll}\n\\hline\n"
             + " \\\\\n".join(f"{tx(a)} & {tx(v)}" for a, v in b) + " \\\\\n\\hline\n"
             "\\end{tabular}\n\\end{table}\n")
    (OUT / "table_stats.tex").write_text(tex_b, encoding="utf-8")

    print("\n".join(md)); print(); print("\n".join(mdb))
    print("\nindependent recomputation matches stage3: OK")


if __name__ == "__main__":
    main()