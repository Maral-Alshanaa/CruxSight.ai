# experiments/stage1_multiseed.py
"""
Multi-seed runner for CruxSight Runs 1-4.
Completion marker: OUT/{run}_seed{seed}.json, written atomically.
This file never fabricates metrics: if the adapter is missing, it stops.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede torch import

import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import RUNS, SEEDS, validate

# Checkpoints/history stay OUTSIDE the repo (Drive). Only the small JSONs get copied back.
OUT = Path(os.environ.get("CRUX_OUT", "/content/drive/MyDrive/CruxSight/multiseed"))
METRICS = ("auc", "cp_recall", "pat_acc")


def load_adapter():
    try:
        from src.pipeline import train_and_evaluate
    except ImportError as e:
        raise SystemExit(
            "Training adapter not found. Expose train_and_evaluate(cfg, seed, run_dir) "
            f"from your training module and fix the import above. ({e})"
        )
    return train_and_evaluate
# Adapter contract:
#   train_and_evaluate(cfg: dict, seed: int, run_dir: Path) -> dict
#   * fixed train/val/test split (NOT seed-dependent), same as Table 10
#   * best checkpoint chosen on VALIDATION; metrics computed on TEST
#   * same eval functions that produced Table 10
#   * writes checkpoint/history ONLY inside run_dir
#   * returns {"auc", "cp_recall", "pat_acc", "n_params", (optional) "best_epoch"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)  # PyG scatter ops


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def assert_prereg_intact() -> None:
    """Abort if the frozen config or the primary test changed since the tag."""
    r = subprocess.run(
        ["git", "diff", "--quiet", "multiseed-prereg-v2", "--",
         "configs/runs.py", "experiments/stage3_stats.py"]
    )
    if r.returncode == 1:
        raise SystemExit("configs/runs.py or stage3_stats.py differ from tag multiseed-prereg. Aborting.")
    if r.returncode not in (0, 1):
        print("WARNING: tag multiseed-prereg not found here (git fetch --tags?); integrity check skipped.")


def main() -> None:
    validate()
    assert len(SEEDS) == 10
    assert_prereg_intact()
    train_and_evaluate = load_adapter()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = [(r, s) for r in RUNS for s in SEEDS]
    only = os.environ.get("CRUX_ONLY")            # e.g. "run4:42" for the first check
    if only:
        r0, s0 = only.split(":")
        jobs = [(r0, int(s0))]

    failed = []
    for run, seed in jobs:
        marker = OUT / f"{run}_seed{seed}.json"
        if marker.exists():
            print(f"skip {run} seed={seed}")
            continue

        run_dir = OUT / run / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            set_seed(seed)
            t0 = time.time()
            cfg = RUNS[run]
            res = train_and_evaluate(cfg, seed, run_dir)
            for m in METRICS:
                assert m in res and math.isfinite(res[m]) and 0.0 <= res[m] <= 1.0, (m, res.get(m))
            record = dict(
                run=run, seed=seed, **{m: float(res[m]) for m in METRICS},
                n_params=res.get("n_params"), best_epoch=res.get("best_epoch"),
                causal_w_absmax=res.get("causal_w_absmax"),
                cfg_hash=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                git_sha=git_sha(), seconds=round(time.time() - t0, 1),
            )
            tmp = marker.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(marker)                    # atomic: marker exists only if complete
            print(f"done {run} seed={seed}: {record}")
        except Exception:
            (run_dir / "FAILED.txt").write_text(traceback.format_exc())
            failed.append((run, seed))
            print(f"FAILED {run} seed={seed} (see {run_dir / 'FAILED.txt'})")

    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}. Fix and re-run; completed jobs are skipped.")


if __name__ == "__main__":
    main()