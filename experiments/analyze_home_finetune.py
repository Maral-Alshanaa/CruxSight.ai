# experiments/analyze_home_finetune.py
"""Standalone analysis for the Home fine-tuning re-run (10 seeds, PREBUILD_CAUSAL=1)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import SEEDS

RES = Path("results/home_finetune")
STAGES = ["zero_shot", "finetune_detection", "finetune_causal"]


def load():
    recs = []
    for seed in SEEDS:
        p = RES / f"home_seed{seed}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        recs.append(json.loads(p.read_text(encoding="utf-8")))
    return recs


def pm(vals):
    a = np.array(vals)
    return f"{a.mean():.4f} +- {a.std(ddof=1):.4f}  (min {a.min():.4f}, max {a.max():.4f})"


def check_finetune_set_balance():
    """Cheap diagnostic: does RCS Top-1 variance correlate with how many
    positive (bottleneck) samples ended up in each seed's fine-tune split?"""
    import torch
    from torch.utils.data import random_split
    from src.core import CachedWindowDataset

    cache_dir_str = os.environ.get("CRUX_CACHE_DIR", "")
    if not cache_dir_str:
        print("\nSkipping balance check: set CRUX_CACHE_DIR to the Drive dataset_cache path.")
        return
    cache_dir = Path(cache_dir_str)
    if not (cache_dir / "test.pt").exists():
        print(f"\nSkipping balance check: {cache_dir/'test.pt'} not found.")
        return

    home_samples = torch.load(cache_dir / "test.pt", weights_only=False)
    ds = CachedWindowDataset(home_samples)
    n_total = len(ds)
    n_ft = int(0.2 * n_total)

    print(f"\n{'seed':>6} {'n_ft':>6} {'n_pos_ft':>9} {'pos_rate':>9}")
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        ft_set, _ = random_split(ds, [n_ft, n_total - n_ft], generator=g)
        labels = []
        for i in ft_set.indices:
            lbl = ds.samples[i]["label"]
            labels.append(int(lbl.item()) if hasattr(lbl, "item") else int(lbl))
        n_pos = sum(labels)
        print(f"{seed:>6} {len(ft_set):>6} {n_pos:>9} {n_pos/len(ft_set):>9.2%}")


def main():
    recs = load()
    assert all(r["causal_w_absmax_after"] > 0 for r in recs), \
        "Fix did not take effect in one or more seeds"

    print(f"n = {len(recs)} seeds\n")
    for stage in STAGES:
        aucs = [r["stages"][stage]["auc"] for r in recs]
        tops = [r["stages"][stage]["rcs_top1"] for r in recs]
        print(f"[{stage}]")
        print(f"  AUC:      {pm(aucs)}")
        print(f"  RCS Top1: {pm(tops)}")
        print()

    w_before = [r["causal_w_absmax_before"] for r in recs]
    w_after = [r["causal_w_absmax_after"] for r in recs]
    print(f"causal_7.W_raw abs-max: before {pm(w_before)} -> after {pm(w_after)}")

    out = {
        "n": len(recs),
        "per_stage": {
            stage: {
                "auc": [r["stages"][stage]["auc"] for r in recs],
                "rcs_top1": [r["stages"][stage]["rcs_top1"] for r in recs],
            } for stage in STAGES
        },
        "seeds": list(SEEDS),
    }
    (RES / "home_finetune_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {RES/'home_finetune_stats.json'}")

    check_finetune_set_balance()


if __name__ == "__main__":
    main()