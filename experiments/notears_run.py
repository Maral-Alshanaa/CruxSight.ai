# experiments/notears_run.py
"""
NOTEARS / causal-layer analysis runner (pre-registration tag: notears-analysis-prereg).

For each (arm, seed) in configs/notears_runs.ARMS x configs.runs.SEEDS:
unchanged pipeline.train_and_evaluate(..., log_causal=True) with
CRUX_PREBUILD_CAUSAL=1. Writes under OUT:
  true_edges.json                 call-graph edge_index of graphs.pt[30] (once)
  <arm>/seed<S>/best_model.pt, causal_analysis.npz, causal_history.json,
                param_audit.json  (written by the pipeline)
  <arm>_seed<S>.json              completion marker (written last, atomically)
Resumable: finished (arm, seed) markers are skipped.
  CRUX_ONLY_ARM=lam0.05 CRUX_ONLY=42   -> single job (validation step d)
Rule 1 is enforced at run time: a non-diverged job FAILS unless the first-step
audit shows every parameter in the optimizer, with non-zero gradient and a
changed value after optimizer.step().
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
from configs.runs import SEEDS, validate as validate_runs
import configs.notears_runs as NR
from src import pipeline as P
from src.pipeline import train_and_evaluate

OUT = Path(os.environ.get("CRUX_OUT_NT", "/content/drive/MyDrive/CruxSight/notears"))
LOCKED = ["configs/runs.py", "configs/notears_runs.py", "experiments/notears_stats.py",
          "experiments/notears_run.py", "src/pipeline.py",
          "src/causal_analysis.py"]
TAG = "notears-analysis-prereg"


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
    r = subprocess.run(["git", "diff", "--quiet", TAG, "--"] + LOCKED)
    if r.returncode == 1:
        raise SystemExit(f"Locked files differ from tag {TAG}. Document the reason or revert. Aborting.")
    if r.returncode not in (0, 1):
        raise SystemExit(f"Tag {TAG} not found (git fetch --tags?). Refusing to run without the pre-registration.")


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def check_audit(run_dir: Path) -> None:
    audit = json.loads((run_dir / "param_audit.json").read_text())
    bad = {n: v for n, v in audit.items()
           if not (v["in_optimizer"] and v["grad_norm"] > 0 and v["max_abs_delta"] > 0)}
    if bad:
        raise AssertionError(f"parameter audit failed for {sorted(bad)}")
    n = sum(v["numel"] for v in audit.values())
    assert n == 209587, n


def main() -> None:
    validate_runs()
    n_jobs = NR.validate()
    if not P.PREBUILD_CAUSAL:
        raise SystemExit("CRUX_PREBUILD_CAUSAL must be 1 (pre-registered).")
    assert_prereg_intact()
    OUT.mkdir(parents=True, exist_ok=True)

    tpath = OUT / "true_edges.json"
    if not tpath.exists():
        g = torch.load(P.CACHE_DIR / "graphs.pt", weights_only=False)
        ei = g[30]["edge_index"].cpu().numpy()
        pairs = {(int(a), int(b)) for a, b in zip(ei[0], ei[1]) if a != b}
        write_json_atomic(tpath, dict(
            edge_index=ei.tolist(), n_edges_raw=int(ei.shape[1]),
            n_unique_directed=len(pairs),
            n_unique_undirected=len({(min(a, b), max(a, b)) for a, b in pairs})))

    only_arm, only_seed = os.environ.get("CRUX_ONLY_ARM"), os.environ.get("CRUX_ONLY")
    arms = [only_arm] if only_arm else list(NR.ARMS)
    seeds = [int(only_seed)] if only_seed else list(SEEDS)
    failed = []
    for arm in arms:
        for seed in seeds:
            marker = OUT / f"{arm}_seed{seed}.json"
            if marker.exists():
                print(f"skip {arm} seed={seed}"); continue
            run_dir = OUT / arm / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                set_seed(seed)
                t0 = time.time()
                res = train_and_evaluate(NR.ARMS[arm], seed, run_dir, log_causal=True)
                if not res.get("diverged") or res.get("auc") is not None:
                    check_audit(run_dir)
                rec = dict(arm=arm, seed=seed, git_sha=git_sha(),
                           seconds=round(time.time() - t0, 1),
                           **{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                              for k, v in res.items()})
                write_json_atomic(marker, rec)
                print("done", {k: rec[k] for k in ("arm", "seed", "auc", "diverged", "best_epoch")})
            except Exception:
                (run_dir / "FAILED.txt").write_text(traceback.format_exc())
                failed.append((arm, seed))
                print(f"FAILED {arm} seed={seed} (see {run_dir / 'FAILED.txt'})")
    print(f"jobs in design: {n_jobs}")
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}")


if __name__ == "__main__":
    main()
