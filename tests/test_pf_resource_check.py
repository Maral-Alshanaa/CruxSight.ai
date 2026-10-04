# tests/test_pf_resource_check.py
"""Synthetic verification of experiments/pf_resource_check.py (NumPy/SciPy only, CPU)."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from experiments import pf_resource_check as P


def make_file(fid, rng, shifts=None, n_bn=30, n_norm=100, scale=1.0):
    """Window-level values for 3 metrics; shifts[m] = true shift in SD_norm units."""
    shifts = shifts or {}
    label = np.concatenate([np.zeros(n_norm, int), np.ones(n_bn, int)])
    vals = {}
    for m in P.METRICS:
        base = 50.0 + 10 * rng.random()
        v = rng.normal(base, 2.0 * scale, len(label))
        v[label == 1] += shifts.get(m, 0.0) * 2.0 * scale
        vals[m] = v
    return dict(id=fid, values=vals, label=label)


def make_group(n, seed, shifts=None, **kw):
    rng = np.random.default_rng(seed)
    return [make_file(f"f{i}", rng, shifts, **kw) for i in range(n)]


class TestPFResourceCheck(unittest.TestCase):
    def test_window_means_length_and_values(self):
        T = 60
        w = P.window_means(np.arange(T, dtype=float))
        self.assertEqual(len(w), T - 17)                    # n = T - 17, not T - 12
        np.testing.assert_allclose(w, np.arange(T - 17) + 5.5)   # mean of i..i+11

    def test_no_shift_is_equivalent(self):
        r = P.analyze_group(make_group(30, 1), P.MIN_FILES_F)
        self.assertEqual(r["verdict"], "EQUIVALENT")
        self.assertEqual(r["excluded"], [])
        for m in P.METRICS:
            self.assertEqual(r["metrics"][m]["n_files"], 30)

    def test_planted_cpu_shift_is_detected(self):
        r = P.analyze_group(make_group(30, 2, shifts={"cpu": 1.5}), P.MIN_FILES_F)
        self.assertEqual(r["verdict"], "NOT_EQUIVALENT")
        self.assertFalse(r["metrics"]["cpu"]["tost"]["equivalent"])
        self.assertTrue(r["metrics"]["memory"]["tost"]["equivalent"])
        self.assertTrue(P.positive_control_passes(r, "cpu"))
        self.assertFalse(P.positive_control_passes(r, "memory"))

    def test_margin_boundaries(self):
        small = P.analyze_group(make_group(30, 3, shifts={"network": 0.3}), P.MIN_FILES_F)
        self.assertTrue(small["metrics"]["network"]["tost"]["equivalent"])      # 0.3 < margin
        for shift in (0.5, 0.7):
            r = P.analyze_group(make_group(30, 4, shifts={"network": shift}), P.MIN_FILES_F)
            self.assertFalse(r["metrics"]["network"]["tost"]["equivalent"], shift)

    def test_tost_matches_independent_scipy_path(self):
        d = np.random.default_rng(5).normal(0.1, 0.3, 30)
        t = P.tost(d)
        lo = stats.ttest_1samp(d, -P.MARGIN, alternative="greater").pvalue
        hi = stats.ttest_1samp(d, P.MARGIN, alternative="less").pvalue
        self.assertAlmostEqual(t["p_tost"], max(lo, hi), places=10)
        self.assertEqual(t["equivalent"], t["ci90"][0] > -P.MARGIN and t["ci90"][1] < P.MARGIN)

    def test_zero_filled_metric_is_blocked_not_equivalent(self):
        files = make_group(30, 6)
        for f in files:
            f["values"]["network"] = np.zeros_like(f["values"]["network"])   # missing column filled with 0
        r = P.analyze_group(files, P.MIN_FILES_F)
        self.assertEqual(r["verdict"], "BLOCKED")
        self.assertEqual({e["reason"] for e in r["excluded"] if e["metric"] == "network"}, {"all_zero"})

    def test_artifact_detectors(self):
        self.assertEqual(P.check_series(np.zeros(50)), "all_zero")
        self.assertEqual(P.check_series(np.full(50, 3.0)), "constant")
        self.assertEqual(P.check_series(np.r_[np.nan, np.ones(49)]), "non_finite")
        self.assertEqual(P.check_series(np.r_[np.ones(1), np.zeros(199)]), "mostly_zero")
        self.assertIsNone(P.check_series(np.random.default_rng(0).normal(5, 1, 50)))

    def test_zero_sd_and_short_files_are_excluded_not_divided(self):
        files = make_group(30, 7)
        files[0]["values"]["cpu"] = np.where(files[0]["label"] == 0, 5.0, files[0]["values"]["cpu"])  # SD_norm = 0
        files[1] = make_file("short", np.random.default_rng(1), n_bn=4)                                # < MIN_WIN
        r = P.analyze_group(files, P.MIN_FILES_F)
        reasons = {(e["file"], e["metric"]): e["reason"] for e in r["excluded"]}
        self.assertEqual(reasons[("f0", "cpu")], "zero_sd_norm")
        self.assertEqual(reasons[("short", "cpu")], "too_few_windows")
        self.assertEqual(r["metrics"]["cpu"]["n_files"], 28)
        self.assertTrue(np.all(np.isfinite([x["d"] for x in r["metrics"]["cpu"]["rows"]])))

    def test_too_many_exclusions_block(self):
        files = make_group(30, 8)
        for f in files[:10]:
            f["values"]["memory"] = np.zeros_like(f["values"]["memory"])
        r = P.analyze_group(files, P.MIN_FILES_F)        # 20 usable < 24
        self.assertEqual(r["verdict"], "BLOCKED")

    def test_deterministic_and_no_input_mutation(self):
        files = make_group(30, 9, shifts={"cpu": 0.2})
        snap = [{m: f["values"][m].copy() for m in P.METRICS} for f in files]
        a, b = P.analyze_group(files, P.MIN_FILES_F), P.analyze_group(files, P.MIN_FILES_F)
        self.assertEqual(a["metrics"]["cpu"]["boot_ci90"], b["metrics"]["cpu"]["boot_ci90"])
        self.assertEqual(a["metrics"]["cpu"]["tost"], b["metrics"]["cpu"]["tost"])
        for f, s in zip(files, snap):
            for m in P.METRICS:
                np.testing.assert_array_equal(f["values"][m], s[m])

    def test_manifest_gate(self):
        P.validate_f_manifest([f"f{i}" for i in range(30)])
        with self.assertRaises(AssertionError):
            P.validate_f_manifest([f"f{i}" for i in range(29)])
        with self.assertRaises(AssertionError):
            P.validate_f_manifest(["a"] * 30)

    def test_label_length_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            P.file_effect(np.arange(50.0), np.zeros(49, int))




# ---------------------------------------------------------------------------
# Bins, Prometheus rates, record builder, coverage rule, verdicts
# ---------------------------------------------------------------------------
def reference_bin_by_time(start_us, labels, step_sec=10):
    """Literal copy of cell 7b bin_by_time reduced to (t0 column, label_trace column)."""
    import pandas as pd
    df = pd.DataFrame({"t0": start_us, "label_trace": labels})
    t0 = df["t0"]
    magnitude = t0.median()
    if magnitude > 1e14:
        t_sec = t0 / 1e6
    elif magnitude > 1e11:
        t_sec = t0 / 1e3
    else:
        t_sec = t0
    t_sec = t_sec - t_sec.min()
    bin_id = (t_sec // step_sec).astype(int)
    out = []
    for b in sorted(bin_id.unique()):
        wdf = df[bin_id == b]
        if len(wdf) < 5:
            continue
        out.append(int(wdf["label_trace"].mode()[0]))
    return out


def make_traces(n_bins=60, per_bin=20, bn_range=(10, 30), tref=1_700_000_000.0, seed=0, sparse_bins=()):
    rng = np.random.default_rng(seed)
    t, y = [], []
    for b in range(n_bins):
        k = 2 if b in sparse_bins else per_bin
        tt = tref + 10 * b + np.sort(rng.random(k)) * 9.9
        t.extend(tt); y.extend([1 if bn_range[0] <= b < bn_range[1] else 0] * k)
    return np.array(t), np.array(y)


def make_counter(tref, rate_fn, n_bins=60, dt=15.0, pre=100.0, noise=0.2, seed=0, reset_at=None):
    """Cumulative counter sampled every dt s from tref-pre to tref+10*n_bins+100."""
    rng = np.random.default_rng(seed)
    ts = np.arange(tref - pre, tref + 10 * n_bins + 100, dt)
    inc = np.array([rate_fn(x) * dt * (1 + noise * rng.standard_normal()) for x in ts[:-1]])
    cum = np.concatenate([[5.0], 5.0 + np.cumsum(inc)])
    if reset_at is not None:
        cum = np.where(ts >= reset_at, cum - cum[ts >= reset_at][0], cum)
    return np.column_stack([ts, cum])


class TestBinsAndRates(unittest.TestCase):
    def test_to_seconds_units(self):
        np.testing.assert_allclose(P.to_seconds([1.7e15, 1.7e15 + 1e6]), [1.7e9, 1.7e9 + 1])   # us
        np.testing.assert_allclose(P.to_seconds([1.7e12]), [1.7e9])                              # ms
        np.testing.assert_allclose(P.to_seconds([1.7e9]), [1.7e9])                               # s

    def test_bin_traces_matches_cell_7b_reference(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not installed")
        t, y = make_traces(sparse_bins=(3, 4, 40))
        y = y.copy(); y[np.random.default_rng(1).choice(len(y), 40, replace=False)] ^= 1   # noisy labels incl. ties
        b = P.bin_traces(t, y)
        self.assertEqual(list(b["label"]), reference_bin_by_time(t * 1e6, y))
        self.assertEqual(len(b["label"]), 57)             # 3 sparse bins dropped (< 5 traces)

    def test_counter_rate_linear_and_nan_outside(self):
        ts = np.arange(0.0, 200.0, 15.0)
        r, nr = P.counter_rates(ts, 2.0 * ts, np.arange(0.0, 130.0, 10.0))
        np.testing.assert_allclose(r, 2.0)
        self.assertEqual(nr, 0)
        r2, _ = P.counter_rates(ts, 2.0 * ts, np.arange(-20.0, 60.0, 10.0))
        self.assertTrue(np.isnan(r2[:2]).all() and np.isfinite(r2[2:]).all())      # bins left of the first sample

    def test_counter_reset_never_negative(self):
        ts = np.arange(0.0, 150.0, 15.0)
        vs = np.where(ts < 40, 2.0 * ts, 2.0 * (ts - 40))
        r, nr = P.counter_rates(ts, vs, np.arange(0.0, 140.0, 10.0))
        self.assertEqual(nr, 1)
        self.assertGreaterEqual(np.nanmin(r), 0.0)
        np.testing.assert_allclose(r[:2], 2.0)             # before the reset
        np.testing.assert_allclose(r[-3:], 2.0)            # after it

    def test_gauge_midpoint(self):
        g = P.gauge_at_mid([0, 100], [0, 100], np.arange(0.0, 60.0, 10.0))
        np.testing.assert_allclose(g, [5, 15, 25, 35, 45])

    def test_cumulative_counter_artifact_is_removed_by_rates(self):
        """Constant true CPU rate + bottleneck late in the run: raw counter levels look 'shifted'
        (a time artifact), per-bin rates are equivalent."""
        raw_files, rate_files = [], []
        for i in range(30):
            tref = 1_700_000_000.0
            t, y = make_traces(tref=tref, seed=i, bn_range=(25, 55))
            prom = {(f"s{k}", "cpu"): make_counter(tref, lambda x: 2.0, seed=100 * i + k) for k in range(3)}
            rec = P.build_file_record(f"f{i}", t, y, prom, [f"s{k}" for k in range(3)])
            rate_files.append(rec)
            edges = tref + 10 * np.arange(61)
            lvl = sum(P.gauge_at_mid(prom[(f"s{k}", "cpu")][:, 0], prom[(f"s{k}", "cpu")][:, 1], edges)
                      for k in range(3))
            raw_files.append(dict(id=f"f{i}", values={"cpu": lvl[rec_bins(t, y)]}, label=rec["label"]))
        rate = P.analyze_group(rate_files, P.MIN_FILES_F, metrics=("cpu",))
        raw = P.analyze_group(raw_files, P.MIN_FILES_F, metrics=("cpu",))
        self.assertTrue(rate["metrics"]["cpu"]["tost"]["equivalent"])
        self.assertFalse(raw["metrics"]["cpu"]["tost"]["equivalent"])
        self.assertGreater(raw["metrics"]["cpu"]["tost"]["mean_d"], P.MARGIN)

    def test_planted_rate_shift_is_detected_through_the_pipeline(self):
        files = []
        for i in range(30):
            tref = 1_700_000_000.0
            t, y = make_traces(tref=tref, seed=i, bn_range=(25, 55))
            prom = {("s0", "cpu"): make_counter(tref, lambda x: 2.0 + (1.5 if tref + 250 <= x < tref + 550 else 0.0),
                                                seed=i)}
            files.append(P.build_file_record(f"f{i}", t, y, prom, ["s0"]))
        r = P.analyze_group(files, P.MIN_FILES_F, metrics=("cpu",))
        self.assertTrue(P.positive_control_passes(r, "cpu"))

    def test_record_alignment_and_missing_service(self):
        tref = 1_700_000_000.0
        t, y = make_traces(tref=tref, sparse_bins=(5,))
        prom = {("a", "cpu"): make_counter(tref, lambda x: 1.0), ("b", "cpu"): make_counter(tref, lambda x: 1.0, seed=1)}
        rec = P.build_file_record("f", t, y, prom, ["a", "b"])
        self.assertEqual(len(rec["label"]), len(rec["values"]["cpu"]))
        self.assertEqual(rec["n_bins_kept"], 59)
        self.assertTrue(np.isfinite(rec["values"]["cpu"]).all())
        self.assertEqual(rec["n_resets"], 0)
        rec2 = P.build_file_record("g", t, y, prom, ["a", "b", "c"])        # service c has no series
        self.assertEqual(rec2["cpu_services_missing"], 1)
        r = P.analyze_group([rec2] * 30, 24, metrics=("cpu",))
        self.assertEqual(r["verdict"], "BLOCKED")                           # never silently 'equivalent'

    def test_load_and_per_request_sensitivity_keys(self):
        tref = 1_700_000_000.0
        t, y = make_traces(tref=tref)
        prom = {("a", "cpu"): make_counter(tref, lambda x: 1.0)}
        rec = P.build_file_record("f", t, y, prom, ["a"])
        load = rec["values"]["load"]
        self.assertEqual(int(load.sum()), len(t))                         # every trace lands in a kept bin
        self.assertTrue(((load >= 15) & (load <= 25)).all())
        np.testing.assert_allclose(rec["values"]["cpu_per_req"], rec["values"]["cpu"] / load)

    def test_restrict_post_onset(self):
        y = np.r_[np.zeros(6, int), np.ones(10, int), np.zeros(6, int), np.ones(8, int)]
        rec = dict(id="f", label=y, values={"cpu": np.arange(len(y), dtype=float)})
        out = P.restrict_post_onset(rec)
        self.assertEqual(len(out["label"]), len(y) - 6)
        self.assertEqual(out["values"]["cpu"][0], 6.0)
        short = dict(id="g", label=np.r_[np.zeros(6, int), np.ones(10, int), np.zeros(3, int)],
                     values={"cpu": np.zeros(19)})
        self.assertIsNone(P.restrict_post_onset(short))                       # only 3 normal bins after onset

    def test_parse_prom_json(self):
        a = P.parse_prom_json('[[1696456142.805,"0.05"],[1696456152.917,"0.10"]]')
        self.assertEqual(a.shape, (2, 2)); self.assertAlmostEqual(a[1, 1], 0.10)
        self.assertEqual(P.parse_prom_json("[]").shape, (0, 2))


def rec_bins(t, y):
    return P.bin_traces(t, y)["bins"]


class TestCoverageAndVerdicts(unittest.TestCase):
    def prom(self, n_vals, n_samples=40):
        ts = np.arange(n_samples) * 15.0
        return np.column_stack([ts, np.arange(n_samples) % max(n_vals, 1) + 1.0])

    def test_series_measurable_rule(self):
        self.assertTrue(P.series_measurable(self.prom(5)))
        self.assertFalse(P.series_measurable(self.prom(2)))            # < 3 distinct values
        self.assertFalse(P.series_measurable(self.prom(5, n_samples=9)))   # < 10 samples
        self.assertFalse(P.series_measurable(None))

    def test_coverage_selection_uses_24_of_30(self):
        good, bad = self.prom(5), self.prom(1)
        lists = [{("db", "memory"): good if i < 25 else bad, ("mc", "memory"): good if i < 23 else bad,
                  ("db", "rx"): bad} for i in range(30)]
        sel = P.select_measurable(P.coverage_counts(lists))
        self.assertEqual(sel["memory"], ["db"])                         # 25 >= 24 passes, 23 fails
        self.assertEqual(sel["rx"], [])                                 # kind NOT_MEASURABLE
        self.assertEqual(sel["tx"], [])

    def test_verdict_matrix(self):
        g = lambda m, sd, n=30: P.analyze_group(
            [dict(id=f"f{i}", values={"x": np.r_[np.zeros(20), np.ones(20)] * 0 + np.random.default_rng(i).normal(0, 1, 40)
                                      + np.r_[np.zeros(20), np.full(20, m)]}, label=np.r_[np.zeros(20, int), np.ones(20, int)])
             for i in range(n)], 24, metrics=("x",))["metrics"]["x"]
        none, small, big = g(0.0, 1), g(0.0, 1), g(3.0, 1)
        eq = {"tost": dict(equivalent=True, n=30, mean_d=0.0, se=0.05)}
        self.assertEqual(P.classify(eq, True, True), "EQUIVALENT")
        self.assertEqual(P.classify(eq, True, False), "UNINFORMATIVE")      # no/failed control
        self.assertEqual(P.classify(eq, True, None), "UNINFORMATIVE")
        self.assertEqual(P.classify(eq, False, True), "NOT_MEASURABLE")
        self.assertEqual(P.classify({"blocked": True}, True, True), "BLOCKED")
        self.assertEqual(P.classify(big, True, False), "SHIFTED")           # a shift needs no control
        wide = {"tost": dict(equivalent=False, n=30, mean_d=0.2, se=0.3)}
        self.assertEqual(P.classify(wide, True, True), "INCONCLUSIVE")      # not equivalent, CI covers 0

    def test_overall_claim(self):
        ok = dict(cpu="EQUIVALENT", memory="EQUIVALENT")
        self.assertEqual(P.overall_claim(ok), "SUPPORTED")
        self.assertEqual(P.overall_claim({**ok, "rx": "NOT_MEASURABLE"}), "NOT_ESTABLISHED")
        self.assertEqual(P.overall_claim({**ok, "tx": "UNINFORMATIVE"}), "NOT_ESTABLISHED")
        self.assertEqual(P.overall_claim({**ok, "cpu": "INCONCLUSIVE"}), "NOT_ESTABLISHED")
        self.assertEqual(P.overall_claim({**ok, "cpu": "SHIFTED"}), "CONTRADICTED")


# ---------------------------------------------------------------------------
# End-to-end on a synthetic Kaggle-shaped zip (runner + statistics together)
# ---------------------------------------------------------------------------
import io
import json
import zipfile
import zlib
from experiments import pf_resource_run as R


def synth_zip(f_stems, a_stems, f_cpu_shift=0.0, a_cpu_shift=2.0, bn_range=(25, 55)):
    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, "w")
    tref = 1_700_000_000.0
    services = [f"svc{k}" for k in range(4)]
    for group, shift in ((f_stems, f_cpu_shift), (a_stems, a_cpu_shift)):
        for gi, stem in enumerate(group):
            seed = zlib.crc32(stem.encode()) % 10_000
            t, y = make_traces(tref=tref, seed=seed, bn_range=bn_range)
            csv = "0_start,1_start,label_trace\n" + "\n".join(
                f"{int(ti*1e6)},{int(ti*1e6)+500},{int(yi)}" for ti, yi in zip(t, y))
            zf.writestr(f"{R.CSV_DIR}{stem}_graph_1.csv", csv)
            pre = f"raw_dataset/{stem}/prom_metrics/"
            lo, hi = tref + 10 * bn_range[0], tref + 10 * bn_range[1]
            for k, svc in enumerate(services):
                arr = make_counter(tref, lambda x: 2.0 + (shift if lo <= x < hi else 0.0), seed=seed + k)
                zf.writestr(pre + f"{svc}_{R.KINDS['cpu']}", json.dumps([[a, str(b)] for a, b in arr]))
                # memory / rx / tx: only svc0 has a usable series; the others are constant (single value)
                ts = np.arange(tref - 50, tref + 650, 15.0)
                rng = np.random.default_rng(seed + 7 * k)
                for kind in ("memory", "rx", "tx"):
                    vals = (1e6 + np.cumsum(rng.random(len(ts)) * 1e3)) if k == 0 else np.full(len(ts), 5.0)
                    zf.writestr(pre + f"{svc}_{R.KINDS[kind]}", json.dumps([[a, str(b)] for a, b in zip(ts, vals)]))
    zf.close()
    return zipfile.ZipFile(io.BytesIO(buf.getvalue()))


class TestRunnerEndToEnd(unittest.TestCase):
    F = [f"net_x_{i}" for i in range(30)]
    A = [f"cpu_x_{i}" for i in range(12)]

    def test_stem_of(self):
        self.assertEqual(R.stem_of("net_oct4_10min_800_0_graph_1.csv"), "net_oct4_10min_800_0")

    def test_no_cpu_shift_in_F_with_working_control(self):
        zf = synth_zip(self.F, self.A, f_cpu_shift=0.0, a_cpu_shift=2.0)
        out = R.run_analysis(zf, self.F, self.A)
        self.assertTrue(out["control_cpu_passed"])
        self.assertEqual(out["verdicts"]["cpu"], "EQUIVALENT")
        self.assertEqual(out["selected_series"]["memory"], ["svc0"])       # constant services are not measurable
        self.assertIn(out["verdicts"]["memory:svc0"], ("UNINFORMATIVE", "SHIFTED", "INCONCLUSIVE"))
        self.assertNotEqual(out["verdicts"]["memory:svc0"], "EQUIVALENT")          # no control exists -> never EQUIVALENT
        self.assertIn(out["overall_claim"], ("NOT_ESTABLISHED", "CONTRADICTED"))   # no control for mem/net -> never SUPPORTED
        self.assertTrue(out["gates"]["all_have_30_cpu_services"] is False)         # synthetic has 4 services: gate must flag it
        self.assertEqual(out["gates"]["total_resets"], 0)
        json.dumps(out)                                                            # fully serialisable

    def test_cpu_shift_in_F_is_reported_as_shifted(self):
        zf = synth_zip(self.F, self.A, f_cpu_shift=1.5, a_cpu_shift=2.0)
        out = R.run_analysis(zf, self.F, self.A)
        self.assertEqual(out["verdicts"]["cpu"], "SHIFTED")
        self.assertEqual(out["overall_claim"], "CONTRADICTED")

    def test_control_failure_makes_equivalence_uninformative(self):
        zf = synth_zip(self.F, self.A, f_cpu_shift=0.0, a_cpu_shift=0.0)      # control files show no shift
        out = R.run_analysis(zf, self.F, self.A)
        self.assertFalse(out["control_cpu_passed"])
        self.assertEqual(out["verdicts"]["cpu"], "UNINFORMATIVE")

    def test_all_constant_memory_net_is_not_measurable(self):
        zf = synth_zip(self.F, self.A)
        # overwrite every memory/rx/tx series with a constant one
        buf = io.BytesIO(); new = zipfile.ZipFile(buf, "w")
        for n in zf.namelist():
            data = zf.read(n)
            if any(R.KINDS[k] in n for k in ("memory", "rx", "tx")):
                data = json.dumps([[1_700_000_000 + 15 * i, "7"] for i in range(40)]).encode()
            new.writestr(n, data)
        new.close()
        out = R.run_analysis(zipfile.ZipFile(io.BytesIO(buf.getvalue())), self.F, self.A)
        for kind in ("memory", "rx", "tx"):
            self.assertEqual(out["verdicts"][kind], "NOT_MEASURABLE")


if __name__ == "__main__":
    unittest.main()
