# experiments/baseline_simple_run.py
"""
Runs the simple baselines (Section 5.2.7): rf_pernode, rf_flat (10 seeds each) and the static
threshold rule (one deterministic fit). Design: PROTOCOL.md, "Simple baselines (random forest and
static threshold rule)". Aborts unless tag `baseline-simple-prereg` exists and the locked files equal it.

Output: results/baseline_simple/{rf_pernode,rf_flat}_seed{seed}.json, rule.json (atomic; presence == done).
Env: CRUX_MODEL_DIR (dataset_cache/ with train.pt, val.pt), CRUX_SIMPLE_OUT, CRUX_ONLY="arm:seed" (gate).
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from configs import baseline_simple as C                                 # noqa: E402
from configs.runs import SEEDS                                           # noqa: E402
from src import simple_baselines as SB                                   # noqa: E402

OUT = Path(os.environ.get("CRUX_SIMPLE_OUT", REPO / "results" / "baseline_simple"))
PREREG_TAG = "baseline-simple-prereg"
LOCKED = ["configs/runs.py", "configs/baseline_simple.py", "src/simple_baselines.py", "src/core.py",
          "src/pipeline.py", "experiments/baseline_simple_stats.py", "experiments/baseline_stats.py",
          "results/baseline"]          # results/baseline = the Run 4 / no-TOC reference values


def git(*a):
    return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()


def assert_prereg_intact() -> str:
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{PREREG_TAG}^{{commit}}"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"Tag {PREREG_TAG} not found. Pre-registration must be tagged BEFORE any run.")
    if subprocess.run(["git", "diff", "--quiet", PREREG_TAG, "--", *LOCKED], cwd=REPO).returncode != 0:
        raise SystemExit(f"Locked files differ from tag {PREREG_TAG}: {LOCKED}")
    return r.stdout.strip()


def write_atomic(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(path)


def load():
    import torch
    from src.core import TOCPriorLoader as T
    from src.pipeline import CACHE_DIR
    tr = torch.load(CACHE_DIR / "train.pt", weights_only=False)
    va = torch.load(CACHE_DIR / "val.pt", weights_only=False)

    def arr(s):                                    # time-averaged F0-F6, as the logreg baseline
        return (np.stack([q["x"].float().mean(0).numpy() for q in s]),
                np.array([int(q["label"]) for q in s]), np.array([int(q["pattern_idx"]) for q in s]),
                np.stack([q["x"].float().numpy() for q in s[:5]]))
    (Xtr, ytr, ptr, raw), (Xva, yva, pva, _) = arr(tr), arr(va)
    return (Xtr, ytr, ptr), (Xva, yva, pva), T.PATTERNS, T.IDX_TO_PATTERN, raw, torch.__version__


def preflight(tr, va, patterns, idx_map, raw) -> dict:
    """Same windows/labels as the reference arms: compare with the recorded Run 4 / logreg records.
    (Amendment A1: the stored reference pre-flight has val_pos_pattern_counts, not n_train/n_val.)"""
    from collections import Counter
    ref = json.loads((REPO / "results/baseline/run4_retrain_seed42.json").read_text())["preflight"]
    lr = json.loads((REPO / "results/baseline/logreg.json").read_text())
    idx, _ = SB.eligible(va[1], va[2], patterns, idx_map)
    counts = dict(sorted(Counter(idx_map[int(va[2][i])] for i in idx).items()))
    pre = dict(n_train=len(tr[1]), n_val=len(va[1]), x_window_shape=list(raw.shape[1:]),
               rcs_top1_n_eligible_val=len(idx), val_pos_pattern_counts=counts,
               train_channel_mean=tr[0].mean((0, 1)).tolist(), train_channel_sd=tr[0].std((0, 1)).tolist())
    assert raw.shape[1:] == (12, 30, 7), raw.shape
    assert counts == ref["val_pos_pattern_counts"], (counts, ref["val_pos_pattern_counts"])
    assert pre["rcs_top1_n_eligible_val"] == lr["rcs_top1_n"], (pre["rcs_top1_n_eligible_val"], lr["rcs_top1_n"])
    return pre


def main() -> None:
    C.validate()
    prereg = assert_prereg_intact()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, patterns, idx_map, raw, tv = load()
    pre = preflight(tr, va, patterns, idx_map, raw)
    print("preflight OK:", json.dumps({k: pre[k] for k in ("n_train", "n_val", "rcs_top1_n_eligible_val")}))
    meta = dict(git_sha=git("rev-parse", "HEAD"), prereg_commit=prereg, preflight=pre,
                env=dict(python=platform.python_version(), torch=tv, numpy=np.__version__))
    import sklearn
    meta["env"]["sklearn"] = sklearn.__version__
    only = os.environ.get("CRUX_ONLY")
    jobs = [(a, s) for a in C.RF_ARMS for s in SEEDS]
    if only:
        a0, s0 = only.split(":")
        jobs = [(a0, int(s0))] if a0 != "rule" else []
    for arm, seed in jobs:
        path = OUT / f"{arm}_seed{seed}.json"
        if path.exists():
            continue
        t0 = time.time()
        fn = SB.rf_pernode if arm == "rf_pernode" else SB.rf_flat
        res = fn(*tr, *va, seed, patterns, idx_map)
        write_atomic(path, dict(arm=arm, seed=seed, **res, seconds=time.time() - t0, **meta))
        print("done", arm, seed, {k: res[k] for k in ("auc", "pat_acc", "rcs_top1", "rcs_top1_n")})
    if not only or only.startswith("rule"):
        path = OUT / "rule.json"
        if not path.exists():
            def once():
                m = SB.rule_fit(tr[0], tr[1])
                return m, SB.rule_eval(m, *va, patterns, idx_map)
            (m1, a), (m2, b) = once(), once()
            assert a == b, "rule not deterministic"
            write_atomic(path, dict(arm="rule", seed_independent=True, deterministic_verified_two_fits=True,
                                    n_normal_train_windows=m1["n_normal"], **a, **meta))
            print("done rule", {k: a[k] for k in ("auc", "rcs_top1", "rcs_top1_n", "tau")})


if __name__ == "__main__":
    main()
