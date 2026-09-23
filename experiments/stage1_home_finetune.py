# experiments/stage1_home_finetune.py
"""
Home fine-tuning re-run, pre-registered under tag home-finetune-prereg.
For each seed, starts from that SAME seed's Run 4 compose checkpoint
(causal-fix-v1), so fine-tuning seed == compose-training seed throughout.
Completion marker: OUT/home_seed{S}.json, written atomically.
"""
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import SEEDS

OUT = Path(os.environ.get("CRUX_OUT", "/content/drive/MyDrive/CruxSight/home_finetune"))
COMPOSE_CKPT_ROOT = Path(os.environ.get(
    "CRUX_COMPOSE_CKPT_ROOT", "/content/drive/MyDrive/CruxSight/causal_fix"))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def assert_prereg_intact() -> None:
    r = subprocess.run(
        ["git", "diff", "--quiet", "home-finetune-prereg", "--",
         "configs/runs.py", "experiments/stage3_stats.py"]
    )
    if r.returncode == 1:
        raise SystemExit("configs/runs.py or stage3_stats.py differ from tag home-finetune-prereg. Aborting.")
    if r.returncode not in (0, 1):
        print("WARNING: tag home-finetune-prereg not found here; integrity check skipped.")


def main() -> None:
    assert os.environ.get("CRUX_PREBUILD_CAUSAL", "0") == "1", \
        "This study is pre-registered with CRUX_PREBUILD_CAUSAL=1 only."
    assert_prereg_intact()
    from src.pipeline import finetune_home  # import after env var is set

    OUT.mkdir(parents=True, exist_ok=True)
    failed = []

    only_seed = os.environ.get("CRUX_ONLY_SEED")
    seeds_to_run = [int(only_seed)] if only_seed else list(SEEDS)
    for seed in seeds_to_run:
        marker = OUT / f"home_seed{seed}.json"
        if marker.exists():
            print(f"skip seed={seed}")
            continue

        ckpt_path = COMPOSE_CKPT_ROOT / "run4" / f"seed{seed}" / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing compose checkpoint for seed {seed}: {ckpt_path}")

        run_dir = OUT / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            t0 = time.time()
            res = finetune_home(ckpt_path, seed=seed, run_dir=run_dir,
                                epochs_detect=8, epochs_causal=8)
            for stage_name, m in res["stages"].items():
                assert math.isfinite(m["auc"]) and 0.0 <= m["auc"] <= 1.0, (seed, stage_name, m)
                assert math.isfinite(m["rcs_top1"]) and 0.0 <= m["rcs_top1"] <= 100.0, (seed, stage_name, m)

            record = dict(
                seed=seed, compose_checkpoint=str(ckpt_path),
                stages=res["stages"],
                causal_w_absmax_before=res["causal_w_absmax_before"],
                causal_w_absmax_after=res["causal_w_absmax_after"],
                git_sha=git_sha(), seconds=round(time.time() - t0, 1),
            )
            tmp = marker.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(marker)
            print(f"done seed={seed}: {record}")
        except Exception:
            (run_dir / "FAILED.txt").write_text(traceback.format_exc())
            failed.append(seed)
            print(f"FAILED seed={seed} (see {run_dir / 'FAILED.txt'})")

    if failed:
        raise SystemExit(f"{len(failed)} seed(s) failed: {failed}. Fix and re-run; completed seeds are skipped.")


if __name__ == "__main__":
    main()