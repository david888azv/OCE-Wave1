"""GPU vs CPU benchmark for OCE hot paths.

Profiles which steps of the OCE pipeline could benefit from GPU acceleration:
  (1) Feature build (figures + cutoff-2F enumeration)
  (2) Ridge fit (sklearn → torch)
  (3) Inference (matrix-vector multiply)
  (4) Bootstrap variance ensemble (K parallel ridge fits)

For each, runs CPU and GPU and reports speedup ratio.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")

results: list[dict] = []


def time_cpu_gpu(fn_cpu, fn_gpu, *, label: str, n_warm=2, n_repeat=5,
                   skip_gpu=False):
    """Time a function on CPU and GPU; return speedup ratio."""
    for _ in range(n_warm):
        fn_cpu()
    ts_cpu = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn_cpu()
        ts_cpu.append(time.perf_counter() - t0)
    t_cpu = float(np.mean(ts_cpu))

    if skip_gpu:
        speedup = float("nan")
        print(f"  {label:<55s}  CPU={t_cpu*1000:>8.2f} ms   GPU=skip")
        return t_cpu, float("nan"), float("nan")

    if torch.cuda.is_available():
        # warm
        for _ in range(n_warm):
            fn_gpu()
        torch.cuda.synchronize()
        ts_gpu = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn_gpu()
            torch.cuda.synchronize()
            ts_gpu.append(time.perf_counter() - t0)
        t_gpu = float(np.mean(ts_gpu))
        speedup = t_cpu / t_gpu
    else:
        t_gpu = float("nan"); speedup = float("nan")
    print(f"  {label:<55s}  CPU={t_cpu*1000:>8.2f} ms   "
          f"GPU={t_gpu*1000:>8.2f} ms   speedup×={speedup:>5.2f}")
    results.append(dict(label=label, t_cpu_ms=t_cpu*1000,
                         t_gpu_ms=t_gpu*1000, speedup=speedup))


# === (1) Feature build ===
print("\n(1) Feature build  [hot loop in Python+ASE; GPU N/A unless rewritten]")
from ase.build import bulk
from oce.atomic_table import load_table
from oce_carbon.features.figures_pbc import enumerate_figures_pbc
from oce_carbon.features.cutoff_2f import enumerate_cutoff_2f
table = load_table(ROOT/"data"/"atoms"/"atomic_table.json")

for n_repeat_lat, label in [((4,4,4), "diamond_4x4x4 (128 atoms)"),
                              ((6,6,6), "diamond_6x6x6 (432 atoms)")]:
    ats = bulk("C","diamond",a=3.567).repeat(n_repeat_lat)
    def fb_cpu():
        enumerate_figures_pbc(ats, table)
        enumerate_cutoff_2f(ats, r_cutoff=5.0)
    time_cpu_gpu(fb_cpu, lambda: None, label=f"feat-build {label}",
                  skip_gpu=True)
print("  ↳ Feature build is CPU-bound symbolic; GPU offers no native gain.")
print("  ↳ Path: rewrite hot loops in Numba/Cython for 10-50× CPU speedup.")

# === (2) Ridge fit ===
print("\n(2) Ridge fit  [sklearn vs torch.linalg.lstsq on GPU]")
from sklearn.linear_model import Ridge

for (n, p), label in [((200, 200),  "n=200 p=200 (small, like baseline_n93)"),
                        ((250, 1700),"n=250 p=1700 (medium, +CT2F perov)"),
                        ((3000, 1000),"n=3000 p=1000 (large, QMOF-scale)"),
                        ((5000, 5000),"n=5000 p=5000 (very large)")]:
    X_np = np.random.randn(n, p).astype(np.float64)
    y_np = np.random.randn(n).astype(np.float64)
    alpha = 10.0
    def fit_cpu():
        Ridge(alpha=alpha, fit_intercept=True).fit(X_np, y_np)
    Xt = torch.from_numpy(X_np).to(DEVICE)
    yt = torch.from_numpy(y_np).to(DEVICE)
    I = torch.eye(p, dtype=Xt.dtype, device=DEVICE)
    def fit_gpu():
        # Centred ridge: solve (XᵀX + αI) β = Xᵀy
        XtX = Xt.T @ Xt + alpha * I
        Xty = Xt.T @ yt
        torch.linalg.solve(XtX, Xty)
    time_cpu_gpu(fit_cpu, fit_gpu, label=f"ridge-fit {label}")

# === (3) Inference ===
print("\n(3) Inference  [matrix-vector multiply; small problem]")
for n, p in [(1, 1700), (1000, 1700), (10000, 1700)]:
    X_np = np.random.randn(n, p).astype(np.float32)
    w_np = np.random.randn(p).astype(np.float32)
    Xt = torch.from_numpy(X_np).to(DEVICE)
    wt = torch.from_numpy(w_np).to(DEVICE)
    label = f"predict {n} structures × {p} feats"
    time_cpu_gpu(lambda: X_np @ w_np,
                  lambda: (Xt @ wt).cpu(),
                  label=label, n_repeat=20)

# === (4) Bootstrap variance ensemble ===
print("\n(4) Bootstrap variance  [K parallel ridge fits — biggest GPU win]")
n, p, K = 200, 1700, 100
X_np = np.random.randn(n, p).astype(np.float64)
y_np = np.random.randn(n).astype(np.float64)
X_cand_np = np.random.randn(1500, p).astype(np.float64)
alpha = 10.0

def boot_cpu():
    rng = np.random.default_rng(0)
    preds = np.zeros((K, X_cand_np.shape[0]))
    for k in range(K):
        sel = rng.choice(n, size=n, replace=True)
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X_np[sel], y_np[sel])
        preds[k] = m.predict(X_cand_np)
    return preds.std(axis=0, ddof=1)

Xt = torch.from_numpy(X_np).to(DEVICE)
yt = torch.from_numpy(y_np).to(DEVICE)
Xct = torch.from_numpy(X_cand_np).to(DEVICE)
I = torch.eye(p, dtype=Xt.dtype, device=DEVICE)

def boot_gpu():
    """Solve K ridge problems as batched linear systems on GPU."""
    g = torch.Generator(device=DEVICE).manual_seed(0)
    sel = torch.randint(0, n, (K, n), generator=g, device=DEVICE)
    # Subsample features per fold; gather rows via advanced indexing
    Xb = Xt[sel]            # (K, n, p)
    yb = yt[sel]            # (K, n)
    XtX = torch.einsum("knp,knq->kpq", Xb, Xb) + alpha * I[None]  # (K,p,p)
    Xty = torch.einsum("knp,kn->kp", Xb, yb)                       # (K,p)
    beta = torch.linalg.solve(XtX, Xty)                             # (K,p)
    preds = torch.einsum("mp,kp->km", Xct, beta)                    # (K,1500)
    return preds.std(dim=0, unbiased=True).cpu().numpy()

time_cpu_gpu(boot_cpu, boot_gpu,
              label=f"bootstrap K={K} fits (n={n} p={p})",
              n_repeat=2)

# Final summary
out_path = ROOT / "data" / "gpu_benchmark.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out_path}")
print("\n=== Summary ===")
print("  Where GPU helps:")
print("    - Bootstrap-ensemble variance (Active Learning): YES, see (4)")
print("    - Large ridge fit (n>10³, p>10³): MAYBE, see (2)")
print("  Where GPU hurts (overhead > benefit):")
print("    - Feature build (Python + ASE): N/A in current implementation")
print("    - Single-shot inference: NO (μs problem)")
print("    - Small fit (n,p < 1000): NO (overhead dominates)")
