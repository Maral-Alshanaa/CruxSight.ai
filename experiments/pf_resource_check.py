# experiments/pf_resource_check.py
"""
Pattern F control-case check (tag: pf-resource-prereg). No torch, no training, deterministic.

Claim under test: Pattern F files (network stress only) show a bottleneck WITHOUT a resource
deviation, i.e. raw CPU / Memory / Network in bottleneck windows are equivalent to the normal
windows of the SAME files.

Unit of analysis = the FILE (bins are autocorrelated). The within-file unit is the 10 s BIN built
exactly as cell 7b `bin_by_time` does (bins with < 5 traces dropped; bin label = mode of
label_trace). Per file f and metric m:
  d_f = (mean_bn - mean_norm) / SD_norm      (SD_norm: ddof=1 over that file's label==0 bins)
Metrics come from Prometheus series (prom_metrics), NOT from the processed CSVs: CPU / rx / tx are
cumulative counters and are converted to per-bin RATES (reset-safe); memory is a gauge.

PRIMARY (frozen at the tag, not to be changed afterwards):
  TOST on mean(d) over usable Pattern F files, margin +-0.5 (SD_norm units), alpha=0.05 per side
  (== 90% CI of mean d inside [-0.5, 0.5]). The claim "no resource deviation" holds only if the
  three metrics (cpu, memory, network) are ALL equivalent (intersection-union, no correction).
POSITIVE CONTROL: same code on a group with known CPU/memory stress; it passes if the targeted
  metric is detected as shifted (mean d > MARGIN and 90% CI lower bound > 0). If it fails, the
  Pattern F result is void (the test cannot see a shift).
SENSITIVITY (descriptive only): margins 0.3 / 0.8, Wilcoxon signed-rank on d, file bootstrap
  (N_BOOT resamples, BOOT_SEED) percentile 90% CI.
VERDICTS per metric: EQUIVALENT | SHIFTED | INCONCLUSIVE | UNINFORMATIVE | NOT_MEASURABLE | BLOCKED
(see classify()). The claim 'no resource deviation' is SUPPORTED only if every metric is EQUIVALENT.
BLOCKING (artifact guards): a file/metric with non-finite, all-zero, constant or >=99% zero
  values is excluded and listed; if fewer than MIN_FILES usable files remain the verdict is
  BLOCKED, never "equivalent".
"""
import numpy as np
from scipy import stats

METRICS = ("cpu", "memory", "network")
WINDOW, HORIZON = 12, 6
MARGIN = 0.5
SENS_MARGINS = (0.3, 0.8)
ALPHA = 0.05
MIN_WIN = 8             # min bottleneck AND min normal BINS per file (decided from bin counts only)
MIN_FILES_F = 24        # of the 30 Pattern F files
MIN_FILES_CONTROL = 8
N_BOOT, BOOT_SEED = 10000, 42
ZERO_FRAC_MAX = 0.99
EXPECTED_F_FILES = 30


def n_windows(T: int) -> int:
    return T - (WINDOW + HORIZON - 1)


def window_means(series, window=WINDOW, horizon=HORIZON):
    """Mean of a raw 1-D series over the input steps of each window; length T-17."""
    s = np.asarray(series, dtype=float)
    assert s.ndim == 1, s.shape
    n = n_windows(len(s))
    assert n > 0, (len(s), "series shorter than WINDOW+HORIZON-1")
    c = np.concatenate([[0.0], np.cumsum(s)])
    return (c[window:window + n] - c[:n]) / window


def check_series(values):
    """Return None if the series is usable, else a short reason string."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return "empty"
    if not np.all(np.isfinite(v)):
        return "non_finite"
    if np.all(v == 0):
        return "all_zero"
    if float(np.mean(v == 0)) >= ZERO_FRAC_MAX:
        return "mostly_zero"
    if np.ptp(v) == 0:
        return "constant"
    return None


def file_effect(values, label):
    """Per-file standardized bottleneck-vs-normal difference. Never divides by SD<=0."""
    v, y = np.asarray(values, dtype=float), np.asarray(label)
    assert v.shape == y.shape, (v.shape, y.shape)
    bn, nm = v[y == 1], v[y == 0]
    out = dict(n_bn=int(len(bn)), n_norm=int(len(nm)), status="ok", d=np.nan,
               mu_bn=np.nan, sd_bn=np.nan, mu_norm=np.nan, sd_norm=np.nan)
    if len(bn) < MIN_WIN or len(nm) < MIN_WIN:
        out["status"] = "too_few_windows"
        return out
    out.update(mu_bn=float(bn.mean()), sd_bn=float(bn.std(ddof=1)),
               mu_norm=float(nm.mean()), sd_norm=float(nm.std(ddof=1)))
    if not np.isfinite(out["sd_norm"]) or out["sd_norm"] <= 1e-12:
        out["status"] = "zero_sd_norm"
        return out
    out["d"] = (out["mu_bn"] - out["mu_norm"]) / out["sd_norm"]
    return out


def tost(d, margin=MARGIN, alpha=ALPHA):
    """One-sample TOST on mean(d) against +-margin. Returns dict with p_tost and 1-2alpha CI."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    p_lower = float(1 - stats.t.cdf((m + margin) / se, df))   # H0: mean <= -margin
    p_upper = float(stats.t.cdf((m - margin) / se, df))       # H0: mean >= +margin
    tcrit = float(stats.t.ppf(1 - alpha, df))
    ci = (m - tcrit * se, m + tcrit * se)
    p = max(p_lower, p_upper)
    return dict(mean_d=m, sd_d=sd, se=float(se), n=n, ci90=ci, p_tost=p,
                equivalent=bool(p < alpha), margin=margin)


def boot_ci(d, n_boot=N_BOOT, seed=BOOT_SEED, level=0.90):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def analyze_group(files, min_files, metrics=METRICS):
    """
    files: list of dict(id, values={metric: (n_windows,) array}, label=(n_windows,) array).
    Returns per-metric results, exclusions, and a verdict in {EQUIVALENT, NOT_EQUIVALENT, BLOCKED}.
    """
    res, excluded = {}, []
    for m in metrics:
        rows, d_used = [], []
        for f in files:
            v = np.asarray(f["values"][m], dtype=float)
            y = np.asarray(f["label"])
            assert v.shape == y.shape, (f["id"], m, v.shape, y.shape)
            mask = np.isfinite(v)
            eff = None
            if mask.sum() == 0:
                reason = "empty"
            else:
                v, y = v[mask], y[mask]
                reason = check_series(v)
                if reason is None:
                    eff = file_effect(v, y)
                    if eff["status"] != "ok":
                        reason = eff["status"]
            if reason is not None:
                excluded.append(dict(file=f["id"], metric=m, reason=reason))
                continue
            rows.append(dict(file=f["id"], **eff))
            d_used.append(eff["d"])
        entry = dict(n_files=len(d_used), rows=rows)
        if len(d_used) >= min_files:
            d = np.array(d_used)
            entry["tost"] = tost(d)
            entry["sensitivity"] = {f"margin_{mg}": tost(d, margin=mg)["equivalent"]
                                    for mg in SENS_MARGINS}
            entry["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            entry["boot_ci90"] = boot_ci(d)
            entry["mu_bn_mean"] = float(np.mean([r["mu_bn"] for r in rows]))
            entry["mu_bn_sd"] = float(np.std([r["mu_bn"] for r in rows], ddof=1))
            entry["mu_norm_mean"] = float(np.mean([r["mu_norm"] for r in rows]))
            entry["mu_norm_sd"] = float(np.std([r["mu_norm"] for r in rows], ddof=1))
        else:
            entry["blocked"] = True
        res[m] = entry
    if any(res[m].get("blocked") for m in metrics):
        verdict = "BLOCKED"
    elif all(res[m]["tost"]["equivalent"] for m in metrics):
        verdict = "EQUIVALENT"
    else:
        verdict = "NOT_EQUIVALENT"
    return dict(metrics=res, excluded=excluded, verdict=verdict, n_input_files=len(files))


# ---------------------------------------------------------------------------
# Binning (literal re-implementation of cell 7b bin_by_time) and Prometheus series
# ---------------------------------------------------------------------------
STEP_SEC = 10
MIN_ROWS_BIN = 5
MIN_SAMPLES, MIN_DISTINCT, MIN_COVER_FILES = 10, 3, 24
MIN_EACH_POST_ONSET = 5


def to_seconds(t0):
    """Same unit rule as bin_by_time: us (>1e14), ms (>1e11), else seconds. Absolute unix seconds."""
    t0 = np.asarray(t0, dtype=float)
    mag = np.median(t0)
    if mag > 1e14:
        return t0 / 1e6
    if mag > 1e11:
        return t0 / 1e3
    return t0


def bin_traces(t_abs_sec, label_trace, step=STEP_SEC, min_rows=MIN_ROWS_BIN):
    """Bins = floor((t - t.min())/step); a bin is kept if it has >= min_rows traces;
    its label is the mode of label_trace (ties -> smallest value, as pandas mode()[0])."""
    t = np.asarray(t_abs_sec, dtype=float)
    y = np.asarray(label_trace)
    assert t.shape == y.shape and t.ndim == 1 and len(t) > 0
    tref = float(t.min())
    bid = np.floor((t - tref) / step).astype(int)
    bins, labels, n_req = [], [], []
    for b in np.unique(bid):
        m = bid == b
        if int(m.sum()) < min_rows:
            continue
        vals, cnt = np.unique(y[m], return_counts=True)
        bins.append(int(b)); labels.append(int(vals[np.argmax(cnt)])); n_req.append(int(m.sum()))
    return dict(tref=tref, bins=np.array(bins, int), label=np.array(labels, int),
                n_req=np.array(n_req, int), n_bins_total=int(bid.max()) + 1)


def counter_rates(ts, vs, edges):
    """Reset-safe per-bin rate of a cumulative counter, linearly interpolated at bin edges.
    A decrease between samples is a reset: the post-reset value is the increment (Prometheus rule).
    NaN where a bin is not fully inside [ts[0], ts[-1]]. Returns (rates, n_resets)."""
    ts = np.asarray(ts, dtype=float); vs = np.asarray(vs, dtype=float)
    o = np.argsort(ts, kind="stable"); ts, vs = ts[o], vs[o]
    assert len(ts) >= 2, "need >= 2 samples"
    inc = np.diff(vs)
    reset = inc < 0
    inc = np.where(reset, vs[1:], inc)
    cum = np.concatenate([[vs[0]], vs[0] + np.cumsum(inc)])
    edges = np.asarray(edges, dtype=float)
    c = np.interp(edges, ts, cum)
    inside = (edges >= ts[0]) & (edges <= ts[-1])
    r = np.diff(c) / np.diff(edges)
    return np.where(inside[:-1] & inside[1:], r, np.nan), int(reset.sum())


def gauge_at_mid(ts, vs, edges):
    ts = np.asarray(ts, dtype=float); vs = np.asarray(vs, dtype=float)
    o = np.argsort(ts, kind="stable"); ts, vs = ts[o], vs[o]
    mid = (np.asarray(edges, dtype=float)[:-1] + np.asarray(edges, dtype=float)[1:]) / 2
    g = np.interp(mid, ts, vs)
    return np.where((mid >= ts[0]) & (mid <= ts[-1]), g, np.nan)


def build_file_record(fid, t_abs_sec, label_trace, prom, cpu_services, extra_series=(),
                      step=STEP_SEC):
    """
    prom: {(service, kind): ndarray (n, 2) of [unix_ts, value]}, kind in {cpu, memory, rx, tx}.
    cpu_services: all services summed into the cluster CPU rate (every one must cover the bin).
    extra_series: iterable of (service, kind) for per-service memory/rx/tx tests.
    Returns a record for analyze_group(): values over KEPT bins only, aligned with `label`.
    Also provides n_req (load) and cpu_per_req (sensitivity-only metrics).
    """
    b = bin_traces(t_abs_sec, label_trace, step)
    nb = b["n_bins_total"]
    edges = b["tref"] + step * np.arange(nb + 1)
    keep = b["bins"]
    resets = 0

    tot = np.zeros(nb); missing = 0
    for svc in cpu_services:
        a = prom.get((svc, "cpu"))
        if a is None or len(a) < 2:
            missing += 1
            continue
        r, nr = counter_rates(a[:, 0], a[:, 1], edges)
        resets += nr
        tot = tot + r                      # NaN propagates: bin kept only if every service covers it
    cpu = np.full(len(keep), np.nan) if missing else tot[keep]

    values = {"cpu": cpu, "load": b["n_req"].astype(float),
              "cpu_per_req": cpu / b["n_req"]}
    for svc, kind in extra_series:
        a = prom.get((svc, kind))
        key = f"{kind}:{svc}"
        if a is None or len(a) < 2:
            values[key] = np.full(len(keep), np.nan)
        elif kind == "memory":
            values[key] = gauge_at_mid(a[:, 0], a[:, 1], edges)[keep]
        else:
            r, nr = counter_rates(a[:, 0], a[:, 1], edges); resets += nr
            values[key] = r[keep]
    return dict(id=fid, values=values, label=b["label"], n_bins_total=nb,
                n_bins_kept=len(keep), n_resets=resets, cpu_services_missing=missing)


def restrict_post_onset(rec, min_each=MIN_EACH_POST_ONSET):
    """Sensitivity: keep bins from the first label==1 bin onward (drops the pre-load baseline).
    Returns None if either class then has fewer than min_each bins."""
    y = np.asarray(rec["label"])
    if not (y == 1).any():
        return None
    k = int(np.argmax(y == 1))
    y2 = y[k:]
    if (y2 == 1).sum() < min_each or (y2 == 0).sum() < min_each:
        return None
    return dict(id=rec["id"], label=y2,
                values={m: np.asarray(v)[k:] for m, v in rec["values"].items()})


# ---------------------------------------------------------------------------
# Coverage rule (label-blind) and per-metric verdicts
# ---------------------------------------------------------------------------
def series_measurable(arr):
    return (arr is not None and len(arr) >= MIN_SAMPLES
            and len(np.unique(np.asarray(arr)[:, 1])) >= MIN_DISTINCT)


def coverage_counts(prom_list):
    """prom_list: one prom dict per file. Returns {(service, kind): n_files_measurable}."""
    cnt = {}
    for prom in prom_list:
        for key, arr in prom.items():
            if series_measurable(arr):
                cnt[key] = cnt.get(key, 0) + 1
    return cnt


def select_measurable(counts, kinds=("memory", "rx", "tx"), min_files=MIN_COVER_FILES):
    """Series tested individually; per kind, [] means the kind is NOT_MEASURABLE."""
    return {k: sorted(svc for (svc, kk), n in counts.items() if kk == k and n >= min_files)
            for k in kinds}


def ci95_excludes_zero(tost_entry):
    n, m, se = tost_entry["n"], tost_entry["mean_d"], tost_entry["se"]
    h = float(stats.t.ppf(0.975, n - 1)) * se
    return bool(m - h > 0 or m + h < 0)


def classify(entry, measurable=True, control_passed=None):
    """One verdict per metric. A shift is informative without a control; an EQUIVALENT claim
    additionally needs a passing positive control, otherwise it is UNINFORMATIVE."""
    if not measurable:
        return "NOT_MEASURABLE"
    if entry.get("blocked"):
        return "BLOCKED"
    t = entry["tost"]
    if t["equivalent"]:
        return "EQUIVALENT" if control_passed else "UNINFORMATIVE"
    return "SHIFTED" if ci95_excludes_zero(t) else "INCONCLUSIVE"


def overall_claim(verdicts):
    """'no resource deviation' for Pattern F: SUPPORTED only if every metric is EQUIVALENT."""
    v = list(verdicts.values())
    if any(x == "SHIFTED" for x in v):
        return "CONTRADICTED"
    return "SUPPORTED" if v and all(x == "EQUIVALENT" for x in v) else "NOT_ESTABLISHED"


def positive_control_passes(control_result, target_metric):
    """Control passes iff the targeted metric is detected as shifted upward."""
    e = control_result["metrics"][target_metric]
    if e.get("blocked"):
        return False
    t = e["tost"]
    return bool(t["mean_d"] > MARGIN and t["ci90"][0] > 0)


def validate_f_manifest(file_ids):
    ids = list(file_ids)
    assert len(ids) == len(set(ids)), "duplicate Pattern F file ids"
    assert len(ids) == EXPECTED_F_FILES, (len(ids), EXPECTED_F_FILES)
    return ids


def parse_prom_json(raw):
    """Prometheus range-query dump: [[unix_ts, "value"], ...] -> ndarray (n, 2). Empty -> (0, 2)."""
    import json
    v = json.loads(raw)
    return np.array([[float(a), float(b)] for a, b in v], dtype=float).reshape(-1, 2)
