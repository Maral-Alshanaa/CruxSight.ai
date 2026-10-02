# experiments/ttb_run.py
"""
TTB-head evaluation runner (pre-registration tag: ttb-eval-prereg).

For each seed: unchanged pipeline.train_and_evaluate (Run 4, PREBUILD_CAUSAL=1),
then src.ttb_export.export_val_predictions on best_model.pt. Saves
  OUT/ttb_val_seed{seed}.npz   raw per-window predictions + file ids
  OUT/ttb_seed{seed}.json      completion marker (written last, atomically)
  OUT/ttb_baselines.json       train-only constant baselines (written once)
CRUX_ONLY=42 runs a single seed (validation step d).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede torch import

import json
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
from src import pipeline as P
from src.pipeline import train_and_evaluate
from src.ttb_eval import train_constants
from src.ttb_export import export_val_predictions

RUN = "run4"
OUT = Path(os.environ.get("CRUX_OUT_TTB", "/content/drive/MyDrive/CruxSight/ttb_eval"))
LOCKED = ["configs/runs.py", "experiments/ttb_stats.py", "src/ttb_eval.py", "src/ttb_export.py"]


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def assert_prereg_intact() -> None:
    r = subprocess.run(["git", "diff", "--quiet", "ttb-eval-prereg", "--"] + LOCKED)
    if r.returncode == 1:
        raise SystemExit("Locked files differ from tag ttb-eval-prereg. Document the reason or revert. Aborting.")
    if r.returncode not in (0, 1):
        raise SystemExit("Tag ttb-eval-prereg not found (git fetch --tags?). Refusing to run without the pre-registration.")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def main() -> None:
    validate()
    assert len(SEEDS) == 10
    if not P.PREBUILD_CAUSAL:
        raise SystemExit("CRUX_PREBUILD_CAUSAL must be 1 (pre-registered).")
    assert_prereg_intact()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = RUNS[RUN]

    bpath = OUT / "ttb_baselines.json"
    if not bpath.exists():
        tr = torch.load(P.CACHE_DIR / "train.pt", weights_only=False)
        c = train_constants([float(s["ttb"]) for s in tr], [int(s["label"]) for s in tr])
        write_json_atomic(bpath, c)

    only = os.environ.get("CRUX_ONLY")
    seeds = [int(only)] if only else list(SEEDS)
    failed = []
    for seed in seeds:
        marker = OUT / f"ttb_seed{seed}.json"
        if marker.exists():
            print(f"skip seed={seed}"); continue
        run_dir = OUT / RUN / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            set_seed(seed)
            t0 = time.time()
            res = train_and_evaluate(cfg, seed, run_dir)
            ex = export_val_predictions(run_dir, cfg)
            assert abs(ex["val_auc"] - res["auc"]) < 1e-4, (ex["val_auc"], res["auc"])
            npz = OUT / f"ttb_val_seed{seed}.npz"
            tmp = npz.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                np.savez(f, **{k: ex[k] for k in ("prob", "ttb_pred", "label", "ttb_true", "pattern", "file_id")})
            tmp.replace(npz)
            rec = dict(seed=seed, run=RUN, auc=float(res["auc"]), best_epoch=int(res["best_epoch"]),
                       causal_w_absmax=res.get("causal_w_absmax"), n_params=res.get("n_params"),
                       n_val=int(len(ex["label"])), n_val_files=int(len(np.unique(ex["file_id"]))),
                       git_sha=git_sha(), seconds=round(time.time() - t0, 1))
            write_json_atomic(marker, rec)
            print("done", rec)
        except Exception:
            (run_dir / "FAILED.txt").write_text(traceback.format_exc())
            failed.append(seed)
            print(f"FAILED seed={seed} (see {run_dir / 'FAILED.txt'})")
    if failed:
        raise SystemExit(f"{len(failed)} seed(s) failed: {failed}")


if __name__ == "__main__":
    main()
