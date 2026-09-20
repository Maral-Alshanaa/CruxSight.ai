# experiments/stage2_aggregate.py
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import RUNS, SEEDS, validate

RESULTS_DIR = Path("results/multiseed")
AGGREGATED_FILE = RESULTS_DIR / "aggregated_summary.json"
METRICS = ("auc", "cp_recall", "pat_acc")


def _load(run_id, seed):
    path = RESULTS_DIR / f"{run_id}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["run"] == run_id and rec["seed"] == seed, (path, rec.get("run"), rec.get("seed"))
    for m in METRICS:
        v = rec[m]
        assert isinstance(v, (int, float)) and np.isfinite(v) and 0.0 <= v <= 1.0, (path, m, v)
    return rec


def aggregate_results():
    validate()
    assert len(SEEDS) == 10
    summary, shas = {}, set()
    for run_id in RUNS:
        recs = [_load(run_id, s) for s in SEEDS]          # SEEDS order => pairing by seed
        assert len({r["cfg_hash"] for r in recs}) == 1, f"config drift inside {run_id}"
        shas |= {r["git_sha"] for r in recs}
        summary[run_id] = {"n_seeds": len(recs), "seeds": list(SEEDS), "cfg_hash": recs[0]["cfg_hash"]}
        for m in METRICS:
            vals = [float(r[m]) for r in recs]
            summary[run_id][m] = {"mean": float(np.mean(vals)),
                                  "std": float(np.std(vals, ddof=1)),   # sample SD
                                  "raw_values": vals}
    assert len(shas) == 1, f"results come from more than one code version: {shas}"
    for run_id in summary:
        summary[run_id]["git_sha"] = next(iter(shas))

    AGGREGATED_FILE.write_text(json.dumps(summary, indent=4), encoding="utf-8")
    print(f"Aggregation complete -> {AGGREGATED_FILE}")
    for run_id, v in summary.items():
        print(run_id, {m: f"{v[m]['mean']:.4f} +- {v[m]['std']:.4f}" for m in METRICS})


if __name__ == "__main__":
    aggregate_results()