"""Sensitivity of capacity_compose to the validation files (Appendix C.2).
Env: CRUX_MODEL_DIR (dataset_manifest.json, capacity_compose.npy), CRUX_RESULTS_DIR (per-file *_results.json)."""
import json, os
import numpy as np
from scipy.stats import spearmanr

MD, RD = os.environ["CRUX_MODEL_DIR"], os.environ["CRUX_RESULTS_DIR"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "prior_sensitivity")
os.makedirs(OUT, exist_ok=True)

man = json.load(open(f"{MD}/dataset_manifest.json"))
nm = {i: m["file_name"] for i, m in enumerate(man) if m["workflow"].lower() != "home"}
VAL_CERTAIN, NO_WINDOWS = [0, 10, 15, 20], [7]
TIES = {"A": [4, 5], "D1": [23, 24, 25, 26, 27], "D2": [29, 30], "E": [35, 36], "F": [38, 39, 40]}  # one val file each
assert nm[0] == "cpu_aug12_25min_200_1_graph_1.csv" and nm[7] == "cpu_sept4_25min_800_5_graph_1.csv"
tied = [i for g in TIES.values() for i in g]
excl = {nm[i] for i in VAL_CERTAIN + tied}

J = {}
for f in os.listdir(RD):
    if f.endswith(".json"):
        r = json.load(open(os.path.join(RD, f))); J[os.path.basename(str(r.get("file_name", f)))] = r

def capacity(exclude=()):
    v, n = {i: [] for i in range(30)}, 0
    for b, r in J.items():
        if r.get("status", "OK") != "OK" or not r.get("node_ratios") or r.get("n_nodes", 30) != 30 or b in exclude:
            continue
        n += 1
        for k, s in r["node_ratios"].items():
            if int(k) < 30: v[int(k)].append(max(float(s.get("ratio", 1.0)) - 1.0, 0.0))
    c = np.array([np.mean(x) if x else 0.0 for x in v.values()], dtype=np.float32)
    return c / c.max(), n

stored = np.load(f"{MD}/capacity_compose.npy")
full, n_full = capacity()
assert np.abs(full - stored).max() < 1e-6, "regeneration does not match stored prior"
free, n_free = capacity(excl)
res = dict(n_full=n_full, n_excluded=len(excl), n_free=n_free,
           spearman=float(spearmanr(stored, free)[0]), max_abs_diff=float(np.abs(stored - free).max()),
           top5_overlap=len(set(np.argsort(-stored)[:5]) & set(np.argsort(-free)[:5])))
print(res)
assert (n_full, n_free, len(excl)) == (170, 152, 18) and res["top5_overlap"] == 5
assert abs(res["max_abs_diff"] - 0.0062) < 5e-4 and abs(res["spearman"] - 0.9964) < 5e-4
json.dump(res, open(f"{OUT}/prior_sensitivity.json", "w"), indent=1)
json.dump({"val_certain": [nm[i] for i in VAL_CERTAIN], "one_val_each_from": {k: [nm[i] for i in v] for k, v in TIES.items()},
           "no_windows": [nm[i] for i in NO_WINDOWS], "home_test": [m["file_name"] for m in man if m["workflow"].lower() == "home"]},
          open(f"{OUT}/split_files.json", "w"), indent=1)
