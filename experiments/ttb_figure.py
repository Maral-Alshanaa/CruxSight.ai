# experiments/ttb_figure.py
"""
Final lead-time figure for the TTB-head evaluation. Presentation only: it reads the
same per-seed exports as experiments/ttb_stats.py and calls the same src.ttb_eval
functions, so it cannot change any number. It replaces the figure written by
ttb_stats.py (label overlap, and mean +/- SD whiskers that reach below 0 s and hide
how many seeds contribute per file). Reason documented in PROTOCOL.md.

Left : pooled lead times of detected (file, seed) pairs, split into alerts raised in a
       label-positive window and premature alerts (label-negative window).
Right: every (file, seed) outcome: dot = detected lead, x = missed, diamond = mean of
       detected leads; tick labels give detected/10 per file.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from configs.runs import SEEDS
from src import ttb_eval as E

DEFAULT_OUT = Path(os.environ.get("CRUX_OUT_TTB", "/content/drive/MyDrive/CruxSight/ttb_eval"))
BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#7f7f7f"


def collect(out):
    rows = []     # (seed, file, lead_s or None, premature or None)
    for s in SEEDS:
        z = np.load(Path(out) / f"ttb_val_seed{s}.npz")
        for r in E.lead_time_table(z["prob"], z["file_id"], z["label"], z["ttb_true"]):
            if r["status"] == "ok":
                rows.append((s, r["file"], None if r["missed"] else r["lead_s"], r["premature"]))
    return rows


def main(out_dir=None, results_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir) if out_dir else DEFAULT_OUT
    res = Path(results_dir) if results_dir else ROOT / "results" / "ttb_eval"
    res.mkdir(parents=True, exist_ok=True)
    rows = collect(out)
    files = sorted({r[1] for r in rows})
    det = [r for r in rows if r[2] is not None]
    missed = [r for r in rows if r[2] is None]
    ok_l = [r[2] for r in det if not r[3]]
    pre_l = [r[2] for r in det if r[3]]
    top = max([r[2] for r in det], default=60)
    ymax = int(np.ceil((top + 10) / 50.0) * 50)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    bins = np.arange(-5, ymax + 15, 10)
    ax[0].hist([ok_l, pre_l], bins=bins, stacked=True, color=[BLUE, ORANGE], edgecolor="white",
               label=["alert in label-positive window", "premature (label-negative window)"])
    ax[0].axvline(50, color="k", ls="--", lw=1, label="50 s: longest lead a label-positive window can show")
    ax[0].set_xlabel("Alert lead time (s)")
    ax[0].set_ylabel("Count of (file, seed) pairs")
    ax[0].set_title(f"Detected: {len(det)} of {len(rows)} (file, seed) pairs; missed: {len(missed)}", fontsize=9)
    ax[0].set_ylim(0, ax[0].get_ylim()[1] * 1.45)       # headroom so the legend never covers bars
    ax[0].legend(fontsize=7, loc="upper right", frameon=False)

    rng = np.random.default_rng(0)
    miss_y = -0.07 * ymax
    for j, f in enumerate(files):
        d = [r for r in det if r[1] == f]
        m = [r for r in missed if r[1] == f]
        if d:
            xs = j + rng.uniform(-0.2, 0.2, len(d))
            ax[1].scatter(xs, [r[2] for r in d], c=[ORANGE if r[3] else BLUE for r in d], s=28, zorder=3)
            ax[1].scatter([j], [np.mean([r[2] for r in d])], marker="D", s=60, facecolor="none",
                          edgecolor="k", zorder=4)
        if m:
            ax[1].scatter(j + rng.uniform(-0.2, 0.2, len(m)), [miss_y] * len(m), marker="x", c=GREY, s=28, zorder=3)
    ax[1].axhline(50, color="k", ls="--", lw=1)
    ax[1].axhline(0, color=GREY, lw=0.5)
    ticks = [miss_y] + list(range(0, ymax + 1, 50))
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(["missed"] + [str(t) for t in ticks[1:]])
    ax[1].set_ylim(miss_y * 1.8, ymax + 10)
    ax[1].set_xticks(range(len(files)))
    ax[1].set_xticklabels([f"F{f}\n{sum(1 for r in det if r[1] == f)}/{len(SEEDS)}" for f in files],
                          fontsize=8)
    ax[1].set_xlabel("Validation file (seeds with a stable alert / seeds)", fontsize=9)
    ax[1].set_xlim(-0.6, len(files) - 0.4)
    ax[1].set_ylabel("Alert lead time (s)")
    ax[1].set_title("Per validation file: each dot = one seed (diamond = mean of detected)", fontsize=9)
    fig.tight_layout()
    fig.savefig(res / "ttb_lead_time_distribution.png", dpi=200)
    fig.savefig(res / "ttb_lead_time_distribution.pdf")
    plt.close(fig)
    return dict(n_pairs=len(rows), n_detected=len(det), n_missed=len(missed), n_premature=len(pre_l))


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else None))
