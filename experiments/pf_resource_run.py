# experiments/pf_resource_run.py
"""
Runner for the Pattern F resource check (tag: pf-resource-prereg). Reads ONLY from the Kaggle zip:
  processed_dataset/compose/multi-modal-data-separate/<stem>_graph_1.csv   (trace starts + label_trace)
  raw_dataset/<stem>/prom_metrics/<service>_container_<metric>             (Prometheus series)
All statistics live in pf_resource_check.py. Deterministic; CPU only.
"""
import json
import re
import zipfile

import numpy as np
import pandas as pd

from experiments import pf_resource_check as P

CSV_DIR = "processed_dataset/compose/multi-modal-data-separate/"
KINDS = {"cpu": "container_cpu_usage_seconds_total",
         "memory": "container_memory_usage_bytes",
         "rx": "container_network_receive_bytes_total",
         "tx": "container_network_transmit_bytes_total"}


def stem_of(file_name):
    """'net_oct4_10min_800_0_graph_1.csv' -> 'net_oct4_10min_800_0'."""
    return re.sub(r"_graph_\d+\.csv$", "", file_name)


def load_traces(zf, stem):
    with zf.open(f"{CSV_DIR}{stem}_graph_1.csv") as fh:
        df = pd.read_csv(fh, usecols=lambda c: c.endswith("_start") or c == "label_trace")
    start_cols = [c for c in df.columns if c.endswith("_start")]
    t0 = df[start_cols].min(axis=1).values            # same reference as bin_by_time
    return P.to_seconds(t0), df["label_trace"].values


def list_services(zf, stem):
    pre = f"raw_dataset/{stem}/prom_metrics/"
    return sorted({n[len(pre):].split("_container_")[0] for n in zf.namelist()
                   if n.startswith(pre) and "_container_" in n})


def read_prom(zf, stem):
    pre = f"raw_dataset/{stem}/prom_metrics/"
    names = set(zf.namelist())
    out = {}
    for svc in list_services(zf, stem):
        for kind, metric in KINDS.items():
            n = f"{pre}{svc}_{metric}"
            if n not in names:
                continue
            try:
                raw = zf.read(n)
                out[(svc, kind)] = P.parse_prom_json(raw) if raw.strip() else np.zeros((0, 2))
            except Exception:
                out[(svc, kind)] = np.zeros((0, 2))
    return out


def usable_cpu_services(prom):
    """Services whose CPU series has >= 2 samples in this run."""
    return {svc for (svc, kind), a in prom.items() if kind == "cpu" and len(a) >= 2}


def common_cpu_services(prom_list):
    """Amendment A1: the cluster-CPU sum uses the services with a usable CPU series in EVERY
    F and A run, so the metric means the same thing in every file. Decided from series only."""
    sets = [usable_cpu_services(p) for p in prom_list]
    all_seen = set().union(*(set(s for (s, k) in p if k == "cpu") for p in prom_list))
    common = set.intersection(*sets) if sets else set()
    lacking = {svc: sum(svc not in st for st in sets) for svc in sorted(all_seen - common)}
    return sorted(common), lacking


def _jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k))): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run_analysis(zf, f_stems, a_stems, min_files_f=P.MIN_FILES_F, min_files_control=P.MIN_FILES_CONTROL):
    P.validate_f_manifest(f_stems)
    # 1) traces + prom for F and for the control set; coverage rules use series only (label-blind)
    f_in = {}
    for stem in f_stems:
        t, y = load_traces(zf, stem)
        f_in[stem] = (t, y, read_prom(zf, stem))
    a_in = {}
    for stem in a_stems:
        t, y = load_traces(zf, stem)
        a_in[stem] = (t, y, read_prom(zf, stem))
    counts = P.coverage_counts([v[2] for v in f_in.values()])
    sel = P.select_measurable(counts)
    extra = [(svc, kind) for kind, svcs in sel.items() for svc in svcs]
    cpu_set, cpu_lacking = common_cpu_services([v[2] for v in f_in.values()] + [v[2] for v in a_in.values()])

    f_recs = []
    for stem, (t, y, prom) in f_in.items():
        rec = P.build_file_record(stem, t, y, prom, cpu_set, extra_series=extra)
        rec["n_cpu_services"] = len(cpu_set)
        rec["label_seq"] = "".join(map(str, rec["label"]))
        f_recs.append(rec)

    # 2) positive control (CPU only), Pattern A, same service set
    a_recs = [P.build_file_record(stem, t, y, prom, cpu_set) for stem, (t, y, prom) in a_in.items()]

    res_a = P.analyze_group(a_recs, min_files_control, metrics=("cpu",))
    control_cpu = P.positive_control_passes(res_a, "cpu")

    # 3) primary analyses and verdicts
    keys = ["cpu"] + [f"{k}:{s}" for s, k in extra]
    res_f = P.analyze_group(f_recs, min_files_f, metrics=tuple(keys))
    verdicts = {"cpu": P.classify(res_f["metrics"]["cpu"], True, control_cpu)}
    for kind in ("memory", "rx", "tx"):
        if not sel[kind]:
            verdicts[kind] = "NOT_MEASURABLE"
        for svc in sel[kind]:
            verdicts[f"{kind}:{svc}"] = P.classify(res_f["metrics"][f"{kind}:{svc}"], True, None)

    # 4) sensitivity (descriptive only)
    sens = {"load_and_cpu_per_req": P.analyze_group(f_recs, min_files_f, metrics=("cpu_per_req", "load"))}
    post = [r2 for r in f_recs if (r2 := P.restrict_post_onset(r)) is not None]
    sens["post_onset_cpu"] = dict(n_records=len(post),
                                  result=P.analyze_group(post, min(len(post), 20), metrics=("cpu",))
                                  if len(post) >= 20 else None)

    out = dict(
        step_sec=P.STEP_SEC, margin=P.MARGIN, n_f=len(f_recs), n_a=len(a_recs),
        coverage={f"{svc}|{kind}": n for (svc, kind), n in sorted(counts.items()) if n >= 1 and n >= 10},
        selected_series=sel, control_cpu_passed=control_cpu,
        cpu_service_set=cpu_set, cpu_services_excluded=cpu_lacking,
        verdicts=verdicts, overall_claim=P.overall_claim(verdicts),
        primary=res_f, control=res_a, sensitivity=sens,
        gates=dict(
            n_cpu_services={r["id"]: r["n_cpu_services"] for r in f_recs},
            n_cpu_services_used=len(cpu_set),
            total_resets=int(sum(r["n_resets"] for r in f_recs)),
            bins_kept={r["id"]: r["n_bins_kept"] for r in f_recs},
            label_seq={r["id"]: r["label_seq"] for r in f_recs}),
    )
    return _jsonable(out)


def save(out, path):
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
