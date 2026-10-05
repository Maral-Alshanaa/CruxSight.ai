# experiments/pf_resource_posthoc.py
"""
POST HOC / EXPLORATORY. Written after the pre-registered results (tag pf-resource-v1 inputs) were seen,
to check one explanation of the CPU shift: that 'normal' bins are concentrated at the start of the
run (warm-up) so that bottleneck-vs-normal measures run phase. It changes no pre-registered verdict.

Per file, bins are split into three segments by label: normal BEFORE the first bottleneck bin (pre),
bottleneck bins (bn), normal AFTER the first bottleneck bin (post). Reported: mean(pre)/mean(post) and
mean(bn)/mean(post), averaged over files with >= MIN_BINS bins in each segment, with a file-level
bootstrap 90% interval (N_BOOT resamples, SEED). If warm-up explained the shift, pre/post would be
well above 1 and bn/post close to 1.
"""
import numpy as np

from experiments import pf_resource_check as P
from experiments import pf_resource_run as R

MIN_BINS = 3
MIN_FILES = 8
N_BOOT, SEED = 10000, 42


def segment_masks(label):
    y = np.asarray(label)
    if not (y == 1).any():
        return None
    first = int(np.argmax(y == 1))
    idx = np.arange(len(y))
    return {"pre": (idx < first) & (y == 0), "bn": y == 1, "post": (idx > first) & (y == 0)}


def seg_ratios(recs, key, min_bins=MIN_BINS):
    pre, bn, nb = [], [], []
    for r in recs:
        seg = segment_masks(r["label"])
        if seg is None:
            continue
        v = np.asarray(r["values"][key], dtype=float)
        m = {k: v[s][np.isfinite(v[s])] for k, s in seg.items()}
        if min(len(m["pre"]), len(m["bn"]), len(m["post"])) < min_bins or m["post"].mean() == 0:
            continue
        pre.append(m["pre"].mean() / m["post"].mean())
        bn.append(m["bn"].mean() / m["post"].mean())
        nb.append([len(m["pre"]), len(m["bn"]), len(m["post"])])
    return np.array(pre), np.array(bn), np.array(nb)


def boot_ci90(x, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return [float(v) for v in np.percentile(m, [5, 95])]


def summarize(recs, keys):
    out = {}
    for key in keys:
        pre, bn, nb = seg_ratios(recs, key)
        if len(pre) < MIN_FILES:
            out[key] = dict(n=int(len(pre)), insufficient=True)
            continue
        out[key] = dict(n=int(len(pre)),
                        pre_over_post=float(pre.mean()), pre_ci90=boot_ci90(pre),
                        bn_over_post=float(bn.mean()), bn_ci90=boot_ci90(bn),
                        bins_median_pre_bn_post=[int(v) for v in np.median(nb, axis=0)])
    return out


def run(zf, f_stems, a_stems, results):
    """results: parsed results_full.json (supplies the CPU service set and the selected series)."""
    cpu_set = results["cpu_service_set"]
    extra = [(s, k) for k, svcs in results["selected_series"].items() for s in svcs]

    def recs(stems, ex):
        res = []
        for s in stems:
            t, y = R.load_traces(zf, s)
            res.append(P.build_file_record(s, t, y, R.read_prom(zf, s), cpu_set, extra_series=ex))
        return res

    keys_f = ["cpu", "load"] + [f"{k}:{s}" for s, k in extra]
    return dict(posthoc=True, F=summarize(recs(f_stems, extra), keys_f),
                A=summarize(recs(a_stems, ()), ["cpu", "load"]))
