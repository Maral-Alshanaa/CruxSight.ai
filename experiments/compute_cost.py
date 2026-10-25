# experiments/compute_cost.py
"""
Compute-cost measurement for Appendix C.7 (Run 4 configuration).

Uses the repo's own code unchanged: src.pipeline.train_and_evaluate for training,
src.core.CSTGNN for inference. No file under src/ or configs/ is modified.
Per-epoch timing is obtained by wrapping pipeline.TOCEvaluator (it is constructed once
per train phase and once per validation phase, alternating, inside train_and_evaluate),
so the training loop itself is not re-implemented.

Measures, REPEATS times (default 3), on whatever GPU the session has (name recorded):
  A. training, N=30 (Run 4, seed CRUX_SEED): seconds/epoch, seconds to best epoch,
     epochs run, best_epoch, peak GPU memory (allocated / reserved) over the whole call
  B. inference, N=30 and N=7: latency per sample at batch size 1 and at batch size 64
     (amortised), peak GPU memory during inference
  C. N=7 vs N=30 training-step cost: one pass over the Home fine-tune split in the
     detection-only stage of pipeline.finetune_home (same loss, optimizer, batch size 32),
     seconds/epoch and peak memory
  D. parameter counts (N=30 model; with causal_7 added) and analytic cost terms
     (derived from src/core.py; analytic, not measured)
and compares the measured batch-size-1 latency at N=30 with the 5.78 ms figure.

Outputs results/compute_cost/{training,inference,n7_vs_n30,analytic,summary}.json
(+ meta). Nothing is written to Drive unless CRUX_OUT_CC points there.

Env: CRUX_MODEL_DIR (as pipeline.py), CRUX_PREBUILD_CAUSAL (default 1 here; see FLAG),
     CRUX_SEED (42), CRUX_REPEATS (3), CRUX_CLAIMED_MS (5.78),
     CRUX_DRY=1 -> tiny synthetic run (no files needed; numbers are NOT paper numbers).
"""
import os
os.environ.setdefault("CRUX_PREBUILD_CAUSAL", "1")   # FLAG: pipeline default is "0"; ttb-eval used "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import platform
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.runs import RUNS
from src import pipeline as P
from src.core import CSTGNN, CachedWindowDataset, TOCPriorLoader, TOCWeightedLoss

RUN = "run4"
SEED = int(os.environ.get("CRUX_SEED", "42"))
REPEATS = int(os.environ.get("CRUX_REPEATS", "3"))
CLAIMED_MS = float(os.environ.get("CRUX_CLAIMED_MS", "5.78"))
DRY = os.environ.get("CRUX_DRY", "0") == "1"
OUT = Path(os.environ.get("CRUX_OUT_CC", str(Path(__file__).resolve().parent.parent
                                              / "results" / "compute_cost")))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP, ITERS = (3, 10) if DRY else (50, 1000)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def reset_peak():
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mib():
    if DEV.type != "cuda":
        return dict(allocated=None, reserved=None)
    return dict(allocated=torch.cuda.max_memory_allocated() / 2**20,
                reserved=torch.cuda.max_memory_reserved() / 2**20)


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def meta():
    try:
        import torch_geometric
        pyg = torch_geometric.__version__
    except Exception:
        pyg = "unknown"
    return dict(git_sha=git_sha(), device=DEV.type,
                gpu=torch.cuda.get_device_name(0) if DEV.type == "cuda" else None,
                torch=torch.__version__, torch_geometric=pyg, cuda=torch.version.cuda,
                python=platform.python_version(), run=RUN, seed=SEED, repeats=REPEATS,
                prebuild_causal=P.PREBUILD_CAUSAL, dry=DRY,
                deterministic_flags_set=False)


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(OUT / name)


def summarize(xs):
    xs = [float(x) for x in xs]
    return dict(values=xs, mean=st.fmean(xs),
                sd=(st.stdev(xs) if len(xs) > 1 else None), min=min(xs), max=max(xs))


# ----------------------------------------------------------------------------- A. training
class _Timed:
    """Installed over P.TOCEvaluator. Records a synchronised timestamp when each train/val
    phase starts (evaluator constructed) and when it ends (compute() called)."""
    log = []

    @classmethod
    def install(cls, real):
        cls.log = []
        outer = cls

        class Wrapped(real):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                sync(); outer.log.append(("start", time.perf_counter()))

            def compute(self):
                r = super().compute()
                sync(); outer.log.append(("end", time.perf_counter()))
                return r
        P.TOCEvaluator = Wrapped


def train_once(rep):
    run_dir = OUT / f"_train_rep{rep}"
    run_dir.mkdir(parents=True, exist_ok=True)
    real = P.TOCEvaluator
    _Timed.install(real)
    reset_peak()
    try:
        sync(); t_call = time.perf_counter()
        res = P.train_and_evaluate(RUNS[RUN], SEED, run_dir)
        sync(); t_ret = time.perf_counter()
    finally:
        P.TOCEvaluator = real
    peaks = peak_mib()
    log = _Timed.log
    starts = [t for k, t in log if k == "start"]
    ends = [t for k, t in log if k == "end"]
    # phases alternate train, val, train, val ...
    assert len(starts) == len(ends) and len(starts) % 2 == 0, (len(starts), len(ends))
    n_ep = len(starts) // 2
    epoch_s = []
    for e in range(n_ep):
        nxt = starts[2 * e + 2] if e + 1 < n_ep else t_ret
        epoch_s.append(nxt - starts[2 * e])          # includes scheduler step and checkpoint save
    b = res["best_epoch"]
    assert 1 <= b <= n_ep, (b, n_ep)
    to_best = ends[2 * (b - 1) + 1] - starts[0]      # start of epoch 1 -> end of best epoch's val phase
    return dict(rep=rep, best_epoch=b, epochs_run=n_ep, val_auc=res["auc"],
                n_params=res["n_params"], sec_per_epoch_mean=st.fmean(epoch_s),
                sec_per_epoch_median=st.median(epoch_s), sec_per_epoch_first=epoch_s[0],
                sec_to_best_epoch=to_best, sec_total_loop=t_ret - starts[0],
                sec_call_total_incl_setup=t_ret - t_call, peak_mib=peaks,
                train_phase_sec_mean=st.fmean(ends[2 * e] - starts[2 * e] for e in range(n_ep)),
                val_phase_sec_mean=st.fmean(ends[2 * e + 1] - starts[2 * e + 1] for e in range(n_ep)))


# ----------------------------------------------------------------------------- B. inference
def load_inputs(n_nodes):
    """Returns (cfg, edge_index, toc_cap, samples x list[Tensor[T,N,F]])."""
    cfg = P._build_cfg(RUNS[RUN]) if not DRY else _dry_cfg()
    if DRY:
        E = 2 * n_nodes
        ei = torch.stack([torch.arange(E) % n_nodes, (torch.arange(E) * 7 + 1) % n_nodes]).to(DEV)
        cap = torch.rand(n_nodes, device=DEV)
        xs = [torch.randn(cfg.data.window_steps, n_nodes, cfg.data.n_features) for _ in range(128)]
        return cfg, ei, cap, xs
    graphs = torch.load(P.CACHE_DIR / "graphs.pt", weights_only=False)
    ei, cap = graphs[n_nodes]["edge_index"].to(DEV), graphs[n_nodes]["toc_cap"].to(DEV)
    split = "val" if n_nodes == 30 else "test"       # test.pt holds the Home (N=7) samples (pipeline.finetune_home)
    samples = torch.load(P.CACHE_DIR / f"{split}.pt", weights_only=False)
    xs = [s["x"] for s in samples]
    assert xs[0].shape[1] == n_nodes, (split, xs[0].shape)
    return cfg, ei, cap, xs


def _dry_cfg():
    from src.core import Config
    cfg = Config()
    r = RUNS[RUN]
    cfg.model.gat_hidden, cfg.model.tft_hidden = r["gat_hidden"], r["tft_hidden"]
    cfg.model.gat_dropout, cfg.model.tft_dropout = r["gat_dropout"], r["tft_dropout"]
    return cfg


def build_model(cfg, n_nodes, ei, cap):
    """Same construction as pipeline.train_and_evaluate (+ PREBUILD_CAUSAL branch)."""
    torch.manual_seed(SEED)
    m = CSTGNN(cfg).to(DEV)
    n_shared = sum(p.numel() for p in m.parameters())
    m.eval()
    with torch.no_grad():                            # forces causal_{N} to exist (as PREBUILD_CAUSAL=1)
        m(torch.zeros(1, cfg.data.window_steps, n_nodes, cfg.data.n_features, device=DEV), ei, cap)
    return m, n_shared


def infer_once(n_nodes, rep):
    cfg, ei, cap, xs = load_inputs(n_nodes)
    model, n_shared = build_model(cfg, n_nodes, ei, cap)
    out = dict(n_nodes=n_nodes, rep=rep, n_params_before_causal=n_shared,
               n_params_with_causal=sum(p.numel() for p in model.parameters()),
               n_edges=int(ei.shape[1]), n_samples_available=len(xs))
    for bs in (1, 64):
        reset_peak()
        batches = [torch.stack(xs[i:i + bs]).to(DEV) for i in range(0, len(xs) - bs + 1, bs)]
        assert batches, (bs, len(xs))
        with torch.no_grad():
            for i in range(WARMUP):
                model(batches[i % len(batches)], ei, cap)
            sync()
            ts = []
            for i in range(ITERS):
                x = batches[i % len(batches)]
                s = torch.cuda.Event(enable_timing=True) if DEV.type == "cuda" else None
                e = torch.cuda.Event(enable_timing=True) if DEV.type == "cuda" else None
                if s:
                    s.record(); model(x, ei, cap); e.record(); torch.cuda.synchronize()
                    ts.append(s.elapsed_time(e))     # ms, model forward only (inputs already on device)
                else:
                    t0 = time.perf_counter(); model(x, ei, cap)
                    ts.append((time.perf_counter() - t0) * 1e3)
        ts = np.array(ts)
        out[f"bs{bs}"] = dict(batch_size=bs, iters=ITERS, warmup=WARMUP,
                              ms_per_call_mean=float(ts.mean()), ms_per_call_median=float(np.median(ts)),
                              ms_per_call_p95=float(np.percentile(ts, 95)),
                              ms_per_sample_mean=float(ts.mean() / bs),
                              ms_per_sample_median=float(np.median(ts) / bs),
                              peak_mib=peak_mib())
    return out


# ----------------------------------------------------------------------------- C. N=7 vs N=30 training step
def finetune_epoch_once(n_nodes, rep, n_epochs=3):
    """One pass = one epoch of the detection-only stage of pipeline.finetune_home:
    BCE on sigmoid(bn_logit), AdamW(lr=1e-4, wd=1e-3), batch 32, same loop body.
    For N=30 the same step is applied to Compose val samples (the N=30 reference);
    for N=7 to the Home 20% fine-tune split (same split rule: seed, 0.2)."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split
    cfg, ei, cap, xs = load_inputs(n_nodes)
    labels = None
    if DRY:
        lab = torch.randint(0, 2, (len(xs),)).float()
        ds = list(zip(xs, lab))
    else:
        split = "val" if n_nodes == 30 else "test"
        samples = torch.load(P.CACHE_DIR / f"{split}.pt", weights_only=False)
        ds = [(s["x"], s["label"]) for s in samples]
    if n_nodes == 7:
        n_ft = int(0.2 * len(ds))
        g = torch.Generator().manual_seed(SEED)
        ds, _ = random_split(ds, [n_ft, len(ds) - n_ft], generator=g)
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    model, _ = build_model(cfg, n_nodes, ei, cap)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    model.train()
    reset_peak()
    times = []
    for _ in range(n_epochs):
        sync(); t0 = time.perf_counter()
        for x, label in loader:
            out = model(x.to(DEV), ei, cap)
            loss = F.binary_cross_entropy(torch.sigmoid(out["bn_logit"].squeeze(-1)),
                                          label.to(DEV).float())
            opt.zero_grad(); loss.backward(); opt.step()
        sync(); times.append(time.perf_counter() - t0)
    return dict(n_nodes=n_nodes, rep=rep, n_samples=len(ds), batch_size=32, epochs_timed=n_epochs,
                sec_per_epoch_mean=st.fmean(times), sec_per_epoch_first=times[0],
                sec_per_epoch_last=times[-1], ms_per_sample_train=1e3 * st.fmean(times) / len(ds),
                peak_mib=peak_mib())


# ----------------------------------------------------------------------------- D. analytic
def analytic(cfg, n_nodes, n_edges):
    """Multiply-add counts from the code in src/core.py. Analytic; not measured.
    B batch, T=window_steps, N nodes, E edges (before self-loops), H heads, C=hidden//H,
    F_in=7, d=tft_hidden."""
    m, T = cfg.model, cfg.data.window_steps
    H, hid, d, Fin = m.gat_heads, m.gat_hidden, m.tft_hidden, m.gat_in_feats
    C = hid // H
    Etot = n_edges + n_nodes                         # GATConv add_self_loops=True
    gat_l1 = n_nodes * Fin * H * C + Etot * H * C    # linear + attention/aggregation, per graph
    gat_l2 = n_nodes * hid * H * C + Etot * H * C
    proj = 2 * n_nodes * hid * hid                   # TOCGATLayer.proj, per layer, per graph
    dep = dict(sigmoid_mask_W=n_nodes ** 2,
               encoder=n_nodes * (d * (d // 2) + (d // 2) * n_nodes),   # per sample
               graph_build=n_nodes ** 2,                                   # per sample
               acyclicity_matmul=n_nodes ** 3 + n_nodes ** 2)              # once per forward (WW@WW)
    return dict(
        N=n_nodes, E=n_edges, E_with_self_loops=Etot, T=T, H=H, C=C, d=d,
        gat_per_sample_madds=T * (gat_l1 + gat_l2 + proj),
        gat_big_o="O(B*T*(N*F*H*C + (E+N)*H*C)); linear in N for a fixed-degree call graph, linear in E",
        dependency_per_sample_madds=dep["encoder"] + dep["graph_build"],
        dependency_per_forward_madds=dep["sigmoid_mask_W"] + dep["acyclicity_matmul"],
        dependency_terms=dep,
        dependency_big_o="per sample O(N*d^2/2 + N^2*d/2) (encoder) + O(N^2) (graph); "
                         "per forward O(N^3) (acyclicity WW@WW); graph memory O(B*N^2); W has N^2 parameters",
        temporal_per_sample_madds_lstm=N_LSTM(n_nodes, T, d, cfg.model.lstm_layers),
        temporal_big_o="O(B*N*T*d^2) (input_proj, feature_split, VSN, LSTM) + O(B*N*T^2*d) (attention)")


def N_LSTM(N, T, d, L):
    return N * T * L * 8 * d * d                     # 4 gates x (input + recurrent) d*d each, per layer


# ----------------------------------------------------------------------------- main
def main():
    if not DRY:
        P.MODEL_DIR.joinpath("config.yaml").stat()   # fail early with a clear error
    info = meta()
    print("meta:", json.dumps(info))
    write("meta.json", info)
    if DEV.type != "cuda":
        print("WARNING: no CUDA device; timings are CPU timings and are not usable for C.7")

    cfg30 = _dry_cfg() if DRY else P._build_cfg(RUNS[RUN])
    ei_e = {}
    for n in (30, 7):
        _, ei, _, _ = load_inputs(n)
        ei_e[n] = int(ei.shape[1])
    write("analytic.json", {str(n): analytic(cfg30, n, ei_e[n]) for n in (30, 7)})

    # B and C first: they are short; A last (long) so a session limit does not lose them.
    inf = [infer_once(n, r) for r in range(REPEATS) for n in (30, 7)]
    write("inference.json", inf)
    n7 = [finetune_epoch_once(n, r) for r in range(REPEATS) for n in (30, 7)]
    write("n7_vs_n30.json", n7)

    trn = []
    for r in range(0 if DRY else REPEATS):   # DRY has no cached training data
        trn.append(train_once(r))
        write("training.json", trn)                  # rewritten after each repeat

    lat = [x["bs1"]["ms_per_sample_mean"] for x in inf if x["n_nodes"] == 30]
    summ = dict(
        meta=info,
        training=None if not trn else dict(sec_per_epoch=summarize([t["sec_per_epoch_mean"] for t in trn]),
                      sec_to_best_epoch=summarize([t["sec_to_best_epoch"] for t in trn]),
                      best_epoch=[t["best_epoch"] for t in trn],
                      epochs_run=[t["epochs_run"] for t in trn],
                      peak_mib_allocated=summarize([t["peak_mib"]["allocated"] for t in trn])
                      if (DEV.type == "cuda" and trn) else None) if trn else None,
        inference={f"N{n}": dict(
            bs1_ms_per_sample=summarize([x["bs1"]["ms_per_sample_mean"] for x in inf if x["n_nodes"] == n]),
            bs64_ms_per_sample=summarize([x["bs64"]["ms_per_sample_mean"] for x in inf if x["n_nodes"] == n]),
            peak_mib_allocated_bs1=summarize([x["bs1"]["peak_mib"]["allocated"] for x in inf if x["n_nodes"] == n])
            if DEV.type == "cuda" else None) for n in (30, 7)},
        n7_vs_n30_finetune_epoch={f"N{n}": summarize([x["sec_per_epoch_mean"] for x in n7 if x["n_nodes"] == n])
                                  for n in (30, 7)},
        latency_check=dict(claimed_ms=CLAIMED_MS, measured_bs1_N30_ms_mean_of_repeats=st.fmean(lat),
                           measured_values=lat, difference_ms=st.fmean(lat) - CLAIMED_MS,
                           note="claimed figure not found in repo; protocol (bs, warmup, sync) of the original "
                                "measurement is unknown"))
    write("summary.json", summ)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
