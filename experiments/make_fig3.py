"""Figure 3: CP-Recall across ablation runs 1-4 vs the exact random baseline.
Grey (original protocol): mean +/- SD from results/table_multiseed.csv (per-seed files are not archived).
Blue (layer fixed): per-seed values from results/causal_fix/run{r}_seed*.json.
Open blue marker (Run 3): saturated predictions, positive-prediction rate 1.0 in 7/10 seeds
(PROTOCOL.md, section "Run 3 CP-Recall: confirmed probability saturation"; fixed-layer checkpoints only).
Upper whiskers are clipped at 1.0 (CP-Recall cannot exceed 1). Descriptive figure, no test.
Usage: python experiments/make_fig3.py <repo_root> <out_png> [--width 7.2 --height 4.68]
Writes <out_png> (300 dpi) and the same name with .pdf."""
import argparse, csv, glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ap = argparse.ArgumentParser()
ap.add_argument('root', nargs='?', default='.'); ap.add_argument('out', nargs='?', default='fig3.png')
ap.add_argument('--width', type=float, default=7.2); ap.add_argument('--height', type=float, default=4.68)
a = ap.parse_args()
R = a.root.rstrip('/') + '/results/'
BASE = 1 - 220 / 4060            # exact: 1 - C(12,3)/C(30,3) = 0.9458
assert round(BASE, 4) == 0.9458
assert abs(json.load(open(R + 'causal_fix/causal_fix_all4_stats.json'))['baseline_exact'] - BASE) < 1e-12
faith = {int(r['run'][3:]): (float(r['cp_recall_mean']), float(r['cp_recall_std']))
         for r in csv.DictReader(open(R + 'table_multiseed.csv')) if r['run'].startswith('run')}
fixed = {r: np.array([json.load(open(f))['cp_recall'] for f in sorted(glob.glob(R + f'causal_fix/run{r}_seed*.json'))]) for r in range(1, 5)}
assert all(len(v) == 10 for v in fixed.values()) and sorted(faith) == [1, 2, 3, 4]
plt.rcParams['pdf.fonttype'] = 42   # embedded TrueType (IEEE PDF eXpress rejects Type 3)
fig, ax = plt.subplots(figsize=(a.width, a.height))
ax.axhline(BASE, color='#d55e00', ls='--', lw=1.4)
ax.text(2.0, BASE + 0.012, 'Random baseline 0.9458 (metric saturated)', color='#d55e00', fontsize=7.5, ha='center', va='bottom')
dx = 0.17
for r in range(1, 5):
    open_ = (r == 3)                      # saturation documented for the fixed-layer arm only
    m, s = faith[r]
    ax.errorbar(r - dx, m, yerr=[[s], [min(s, 1.0 - m)]], fmt='s', ms=7, color='#555555', mfc='#9a9a9a', mec='#555555', capsize=4, lw=1.3, zorder=3)
    v = fixed[r]; mu, sd = v.mean(), v.std(ddof=1)
    jit = np.linspace(-0.07, 0.07, len(v))
    ax.scatter(r + dx + jit, v, s=11, color='#7f95c9', zorder=2)
    ax.errorbar(r + dx, mu, yerr=[[sd], [min(sd, 1.0 - mu)]], fmt='o', ms=7, color='#1f4e79', mfc='white' if open_ else '#1f4e79', mec='#1f4e79', capsize=4, lw=1.3, zorder=4)
ax.set_xticks(range(1, 5)); ax.set_xticklabels(['Run 1', 'Run 2', 'Run 3', 'Run 4 \u2605'])
ax.set_xlim(0.5, 4.65); ax.set_ylim(0.4, 1.05); ax.set_ylabel('CP-Recall (mean \u00b1 SD, 10 seeds)')
ax.grid(axis='y', alpha=0.3)
h = [Line2D([], [], marker='s', color='#555555', mfc='#9a9a9a', ls='', label='Original protocol (layer untrained): mean \u00b1 SD'),
     Line2D([], [], marker='o', color='#1f4e79', ls='', label='Layer fixed: mean \u00b1 SD'),
     Line2D([], [], marker='o', color='#1f4e79', mfc='white', ls='', label='Layer fixed, saturated predictions (Run 3)'),
     Line2D([], [], marker='o', color='#7f95c9', ls='', ms=4, label='Layer fixed: individual seeds')]
ax.legend(handles=h, fontsize=7.5, loc='lower right', ncol=1, frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig(a.out, dpi=300); plt.savefig(a.out.rsplit('.', 1)[0] + '.pdf'); print('saved', a.out)
