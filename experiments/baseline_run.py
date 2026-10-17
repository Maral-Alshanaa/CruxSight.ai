# experiments/baseline_run.py
"""
Reproduces the Section 5.2.7 baseline comparison for the 10 fixed seeds.

  run4_retrain : CruxSight Run 4 (use_toc=True)            -> 10 jobs
  no_toc       : GAT+TFT, use_toc=False, node_weights == 1  -> 10 jobs
  logreg       : logistic regression on F0-F6               -> ONE deterministic fit
                 (convex, no random seed; a 10-seed sweep would repeat the same number)

Output: results/baseline/{arm}_seed{seed}.json (atomic; presence == job complete)
        results/baseline/logreg.json
Checkpoints/histories stay OUTSIDE the repo (CRUX_BASELINE_RUNS, default on Drive).

Requires a GPU runtime for the two network arms (~155 s/job on a T4 in earlier runs).
Aborts unless tag `baseline-corrected-prereg` exists and the locked files match it,
so no result can be produced before the pre-registration is tagged.

Env:
  CRUX_BASELINE_ARMS  comma list, default "run4_retrain,no_toc,logreg"
  CRUX_ONLY           "arm:seed" -> single job (verification-seed gate), e.g. "no_toc:42"
  CRUX_BASELINE_OUT   default <repo>/results/baseline
  CRUX_BASELINE_RUNS  default /content/drive/MyDrive/CruxSight/baseline_runs
  CRUX_MODEL_DIR      read by src.pipeline (model dir containing dataset_cache/)
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")          # must precede torch import
if os.environ.get("CRUX_PREBUILD_CAUSAL", "1") != "1":
    raise SystemExit("CRUX_PREBUILD_CAUSAL must be 1 (causal-fix convention used by every compared number).")
os.environ["CRUX_PREBUILD_CAUSAL"] = "1"                              # must precede src.pipeline import

import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from configs.runs import SEEDS                                          # noqa: E402
from configs.baseline_runs import ARMS, validate                        # noqa: E402

OUT = Path(os.environ.get("CRUX_BASELINE_OUT", REPO / "results" / "baseline"))
RUNS_DIR = Path(os.environ.get("CRUX_BASELINE_RUNS", "/content/drive/MyDrive/CruxSight/baseline_runs"))
PREREG_TAG = "baseline-corrected-prereg"
LOCKED = ["configs/runs.py", "configs/baseline_runs.py", "experiments/baseline_stats.py",
          "src/core.py", "src/pipeline.py"]
METRICS = ("auc", "cp_recall", "pat_acc")


# ------------------------------------------------------------------ housekeeping
def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def assert_prereg_intact() -> str:
    """Fail closed: the tag must exist and the locked files must equal it."""
    r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{PREREG_TAG}^{{commit}}"],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"Tag {PREREG_TAG} not found (git fetch --tags?). Pre-registration must be "
                         "tagged BEFORE any run. Aborting.")
    d = subprocess.run(["git", "diff", "--quiet", PREREG_TAG, "--", *LOCKED], cwd=REPO)
    if d.returncode != 0:
        raise SystemExit(f"Locked files differ from tag {PREREG_TAG}: {LOCKED}. Aborting.")
    return r.stdout.strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)             # PyG scatter ops


def write_atomic(path: Path, record: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2))
    tmp.replace(path)


def env_info() -> dict:
    return dict(torch=torch.__version__, cuda=torch.version.cuda, python=platform.python_version(),
                device=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))


# ------------------------------------------------------------------ pre-flight (no training)
def preflight() -> dict:
    """Asserts the corrected loss behaviour before any job runs, and inspects the cache."""
    from src.core import Config, TOCPriorLoader, TOCWeightedLoss, _TOCWeightedLossBase
    from src.pipeline import CACHE_DIR

    cfg = Config()
    toc = TOCPriorLoader(np.ones(30, dtype=np.float32), np.ones(7, dtype=np.float32))
    pcw = torch.ones(cfg.model.n_patterns)
    nt = _TOCWeightedLossBase(cfg, toc, pcw, use_toc=False)
    wt = TOCWeightedLoss(cfg, toc, pcw, use_toc=True)
    assert torch.all(nt.node_weights == 1.0), "no-TOC loss still carries CRITICAL_PATH weights"
    crit = torch.zeros(30, dtype=torch.bool)
    crit[list(toc.CRITICAL_PATH)] = True
    assert torch.all(wt.node_weights[crit] == cfg.training.constraint_mult)
    assert torch.all(wt.node_weights[~crit] == 1.0)

    train = torch.load(CACHE_DIR / "train.pt", weights_only=False)
    val = torch.load(CACHE_DIR / "val.pt", weights_only=False)
    pos_counts = Counter(toc.IDX_TO_PATTERN[int(s["pattern_idx"])] for s in val if int(s["label"]) == 1)
    f6 = torch.stack([s["x"][..., 6].float().mean(0) for s in val])      # [n_val, 30], time-averaged channel 6
    return dict(
        node_weights_no_toc_all_ones=True,
        node_weights_toc_critical_path=float(cfg.training.constraint_mult),
        n_train=len(train), n_val=len(val), val_pos_pattern_counts=dict(sorted(pos_counts.items())),
        x_feature_dim=int(val[0]["x"].shape[-1]),
        # Input-channel check (exploratory, see PROTOCOL.md): is channel 6 still present in x for BOTH arms?
        x_channel6_std_across_nodes_mean=float(f6.std(dim=1).mean()),
    )


# ------------------------------------------------------------------ network arms
def run_nn_job(arm: str, seed: int, pre: dict, prereg_commit: str) -> None:
    from src.pipeline import train_and_evaluate

    marker = OUT / f"{arm}_seed{seed}.json"
    if marker.exists():
        print(f"skip {arm} seed={seed}")
        return
    spec = ARMS[arm]
    run_dir = RUNS_DIR / arm / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        set_seed(seed)
        t0 = time.time()
        res = train_and_evaluate(spec["run_cfg"], seed, run_dir, use_toc=spec["use_toc"])
        for m in METRICS:
            assert res.get(m) is not None and math.isfinite(res[m]) and 0.0 <= res[m] <= 1.0, (m, res.get(m))
        assert res.get("rcs_top1") is not None and math.isfinite(res["rcs_top1"]), res.get("rcs_top1")
        assert res["use_toc"] == spec["use_toc"]
        assert res["causal_w_absmax"] > 0.0, "causal layer did not train (PREBUILD_CAUSAL not effective)"
        record = dict(
            arm=arm, seed=seed, **{m: float(res[m]) for m in METRICS},
            rcs_top1=float(res["rcs_top1"]), rcs_top1_n=int(res["rcs_top1_n"]),
            n_params=int(res["n_params"]), best_epoch=int(res["best_epoch"]),
            causal_w_absmax=float(res["causal_w_absmax"]), use_toc=bool(res["use_toc"]),
            lambda_sub=spec["run_cfg"]["lambda_sub"],
            cfg_hash=hashlib.sha256(json.dumps(spec["run_cfg"], sort_keys=True).encode()).hexdigest()[:12],
            git_sha=git("rev-parse", "HEAD"), prereg_commit=prereg_commit,
            seconds=round(time.time() - t0, 1), env=env_info(),
            preflight={k: pre[k] for k in ("node_weights_no_toc_all_ones", "val_pos_pattern_counts",
                                           "x_feature_dim", "x_channel6_std_across_nodes_mean")},
        )
        write_atomic(marker, record)
        print(f"done {arm} seed={seed}: auc={record['auc']:.4f} pat={record['pat_acc']:.4f} "
              f"rcs={record['rcs_top1']:.2f} (n={record['rcs_top1_n']})")
    except Exception:
        (run_dir / "FAILED.txt").write_text(traceback.format_exc())
        raise


# ------------------------------------------------------------------ logistic regression (F0-F6)
def run_logreg(prereg_commit: str) -> None:
    """
    Design as written in PROTOCOL.md ("Logistic regression baseline", Design):
      detection : StandardScaler -> LogisticRegression(class_weight="balanced"), 210-dim time-averaged F0-F6
      PatAcc    : StandardScaler -> multinomial LogisticRegression on pattern_idx (all windows)
      localize  : one shared classifier over (node, window) rows of eligible positive train windows;
                  per-node probability ranks nodes (CP-Recall top-3 vs CRITICAL_PATH; RCS Top-1 argmax vs flagged set)
    Reading choices NOT spelled out in PROTOCOL.md (flagged in the output JSON): no class_weight for the
    localization and pattern classifiers; PatAcc classifier trained on all train windows.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.core import TOCPriorLoader as T
    from src.pipeline import CACHE_DIR

    marker = OUT / "logreg.json"
    if marker.exists():
        print("skip logreg")
        return
    train = torch.load(CACHE_DIR / "train.pt", weights_only=False)
    val = torch.load(CACHE_DIR / "val.pt", weights_only=False)

    def arrays(samples):
        X = np.stack([s["x"].float().mean(0).numpy() for s in samples])  # [n, 30, 7]
        return X, np.array([int(s["label"]) for s in samples]), np.array([int(s["pattern_idx"]) for s in samples])

    def flagged_of(pat_idx):
        f = [n for n in T.PATTERNS.get(T.IDX_TO_PATTERN.get(int(pat_idx), "none"), frozenset()) if n < 30]
        return f if (f and len(f) < 30) else None                        # same exclusion rule as rcs_top1

    def pipe(**kw):
        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000, **kw))])

    def once():
        Xtr, ytr, ptr = arrays(train)
        Xva, yva, pva = arrays(val)
        det = pipe(class_weight="balanced").fit(Xtr.reshape(len(Xtr), -1), ytr)
        auc = roc_auc_score(yva, det.predict_proba(Xva.reshape(len(Xva), -1))[:, 1])
        pat = pipe().fit(Xtr.reshape(len(Xtr), -1), ptr)
        pat_acc = float((pat.predict(Xva.reshape(len(Xva), -1)) == pva).mean())

        rows, tgt = [], []
        for i in np.where(ytr == 1)[0]:
            fl = flagged_of(ptr[i])
            if fl is None:
                continue
            rows.append(Xtr[i])                                          # [30, 7]
            tgt.append(np.isin(np.arange(30), fl).astype(int))
        loc = pipe().fit(np.concatenate(rows), np.concatenate(tgt))
        cp_hit = cp_n = t1_hit = t1_n = 0
        for i in np.where(yva == 1)[0]:
            p = loc.predict_proba(Xva[i])[:, 1]
            cp_n += 1
            cp_hit += bool(set(np.argsort(p)[-3:].tolist()) & T.CRITICAL_PATH)
            fl = flagged_of(pva[i])
            if fl is not None:
                t1_n += 1
                t1_hit += int(np.argmax(p)) in fl
        return dict(auc=float(auc), cp_recall=cp_hit / max(cp_n, 1), cp_recall_n=cp_n,
                    pat_acc=pat_acc, rcs_top1=100.0 * t1_hit / max(t1_n, 1), rcs_top1_n=t1_n)

    a, b = once(), once()
    assert a == b, f"logistic regression not deterministic: {a} vs {b}"
    record = dict(arm="logreg", seed_independent=True, deterministic_verified_two_fits=True, **a,
                  unspecified_in_protocol=["no class_weight for localization/pattern classifiers",
                                           "pattern classifier trained on all train windows"],
                  git_sha=git("rev-parse", "HEAD"), prereg_commit=prereg_commit, env=env_info())
    write_atomic(marker, record)
    print("done logreg:", {k: record[k] for k in ("auc", "cp_recall", "pat_acc", "rcs_top1")})


# ------------------------------------------------------------------ main
def main() -> None:
    validate()
    prereg_commit = assert_prereg_intact()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in os.environ.get("CRUX_BASELINE_ARMS", "run4_retrain,no_toc,logreg").split(",")]
    jobs = [(a, s) for a in arms if a != "logreg" for s in SEEDS]
    only = os.environ.get("CRUX_ONLY")
    if only:
        a0, s0 = only.split(":")
        jobs, arms = [(a0, int(s0))], [a0]
    pre = preflight()
    print("preflight OK:", json.dumps({k: pre[k] for k in ("n_train", "n_val", "val_pos_pattern_counts",
                                                             "x_feature_dim")}))
    failed = []
    for arm, seed in jobs:
        try:
            run_nn_job(arm, seed, pre, prereg_commit)
        except Exception:
            failed.append((arm, seed))
            print(f"FAILED {arm} seed={seed}")
    if "logreg" in arms:
        run_logreg(prereg_commit)
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed: {failed}. Completed jobs are skipped on re-run.")


if __name__ == "__main__":
    main()
