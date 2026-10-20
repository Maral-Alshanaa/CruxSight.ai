# experiments/fig3_cp_recall.py
"""
Figure 3: CP-Recall across the four ablation runs versus the random baseline.

Presentation only. Reads the per-seed exports in results/<source>/run{1..4}_seed*.json
(default source: causal_fix, n = 10 seeds), recomputes mean and sample SD (ddof = 1), and
checks them against the stored stats file. It trains nothing and changes no number.

Random baseline: top-3 selection over 30 Compose nodes, 18 of which lie on CRITICAL_PATH,
so P(top-3 intersects CRITICAL_PATH) = 1 - C(12,3)/C(30,3) = 0.9458 (exact; asserted below).

Outputs (results/figures/): fig3_cp_recall.png (300 dpi) and fig3_cp_recall.pdf.

Usage:
    python experiments/fig3_cp_recall.py
    python experiments/fig3_cp_recall.py --width 3.5 --height 2.6      # inches
    python experiments/fig3_cp_recall.py --source multiseed_faithful   # NOT supported, see SOURCES
"""
import argparse
import glob
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from configs.runs import SEEDS  # noqa: E402

RUNS = ["run1", "run2", "run3", "run4"]
LABELS = ["Run 1", "Run 2", "Run 3", "Run 4"]
# source dir -> stats file used only as a cross-check of the recomputed values
SOURCES = {"causal_fix": "causal_fix_all4_stats.json"}

BASELINE = 1 - comb(12, 3) / comb(30, 3)
assert round(BASELINE, 4) == 0.9458, BASELINE
BASELINE_TXT = "0.9458"


def load(source):
    d = ROOT / "results" / source
    vals = {}
    for r in RUNS:
        files = sorted(glob.glob(str(d / f"{r}_seed*.json")))
        got = {json.load(open(f))["seed"]: json.load(open(f))["cp_recall"] for f in files}
        assert sorted(got) == sorted(SEEDS), (r, sorted(got))
        vals[r] = np.array([got[s] for s in SEEDS])
    mean = np.array([vals[r].mean() for r in RUNS])
    sd = np.array([vals[r].std(ddof=1) for r in RUNS])

    st = json.load(open(d / SOURCES[source]))
    assert abs(st["baseline_exact"] - BASELINE) < 1e-12, st["baseline_exact"]
    for i, r in enumerate(RUNS):
        assert abs(st["per_run"][r]["cp_recall"]["mean"] - mean[i]) < 1e-9, r
        assert abs(st["per_run"][r]["cp_recall"]["std"] - sd[i]) < 1e-9, r
    return mean, sd


def main(source="causal_fix", width=3.5, height=2.6, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 7,
        "legend.fontsize": 7, "hatch.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    mean, sd = load(source)
    out = Path(out) if out else ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(width, height))
    x = np.arange(len(RUNS))
    bars = ax.bar(x, mean, width=0.62, yerr=sd, capsize=3, color="#9ab0cf", edgecolor="black",
                  linewidth=0.8, error_kw=dict(elinewidth=0.8, capthick=0.8), zorder=2)
    bars[0].set_hatch("////")          # Run 1 hatched (kept from the previous version)
    ax.axhline(BASELINE, color="black", ls="--", lw=1.0, zorder=3,
               label=f"Random baseline ({BASELINE_TXT})")

    for xi, m, s in zip(x, mean, sd):
        ax.text(xi, 0.30, f"{m:.3f}", ha="center", va="center", fontsize=7,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("CP-Recall")
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=2.2,
              borderaxespad=0.2)
    ax.text(1.0, 1.01, "metric saturated", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, style="italic")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.yaxis.grid(True, lw=0.3, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)

    png, pdf = out / "fig3_cp_recall.png", out / "fig3_cp_recall.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"source={source}  n_seeds={len(SEEDS)}  baseline={BASELINE:.6f}")
    for r, m, s in zip(RUNS, mean, sd):
        print(f"  {r}: {m:.4f} +/- {s:.4f}")
    print(f"wrote {png}\nwrote {pdf}")
    return png, pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="causal_fix", choices=sorted(SOURCES))
    ap.add_argument("--width", type=float, default=3.5)
    ap.add_argument("--height", type=float, default=2.6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.source, a.width, a.height, a.out)
