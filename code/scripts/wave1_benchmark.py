"""Wave-1 timing benchmark — feature build per atom across cell sizes.

Measures wall-clock cost per (cell, basis variant) for:
  - 1F+2F+3F+MAD baseline (current production)
  - +CT2F (Wave-1 extension)

Also measures inference cost (predict() on a fitted ridge model) and
ridge fit cost.  Compares to known DFTB+/xtb timings from the project.

Output: data/wave1_benchmark.json + console table.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oce.atomic_table import load_table
from oce_carbon.features.figures_pbc import enumerate_figures_pbc
from oce_carbon.features.cutoff_2f import enumerate_cutoff_2f


def build_test_cells():
    """Cells of escalating size: 16, 64, 216, 512 atoms (diamond C)."""
    base = bulk("C", "diamond", a=3.567)
    return {
        "diamond_2x2x2 (16 at)": base.repeat((2, 2, 2)),
        "diamond_3x3x3 (54 at)": base.repeat((3, 3, 3)),
        "diamond_4x4x4 (128 at)": base.repeat((4, 4, 4)),
        "diamond_6x6x6 (432 at)": base.repeat((6, 6, 6)),
        "diamond_8x8x8 (1024 at)": base.repeat((8, 8, 8)),
    }


def time_call(fn, *args, n_warm=1, n_repeat=3, **kw):
    for _ in range(n_warm):
        fn(*args, **kw)
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn(*args, **kw)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    table = load_table(ROOT / "data" / "atoms" / "atomic_table.json")
    cells = build_test_cells()

    rows = []
    print(f"{'Cell':<28s}  {'N':>5s}  {'feat 1F+2F+3F':>14s}  {'+CT2F':>14s}  "
          f"{'CT2F overhead':>14s}  {'per atom (3F+CT2F)':>20s}")
    print("=" * 110)
    for label, ats in cells.items():
        n = len(ats)
        # Baseline: 1F+2F+3F (current production)
        t_base, _ = time_call(enumerate_figures_pbc, ats, table,
                                include_angles=True, include_dihedrals=False,
                                n_warm=1, n_repeat=3)
        # CT2F alone
        t_ct, _ = time_call(enumerate_cutoff_2f, ats, r_cutoff=5.0,
                             n_warm=1, n_repeat=3)
        # Combined (sum, since both ran sequentially in build_features)
        t_total = t_base + t_ct
        per_atom_us = t_total / n * 1e6
        ct_overhead = t_ct / t_base * 100
        print(f"  {label:<28s}  {n:>5d}  "
              f"{t_base*1000:>12.2f}ms  "
              f"{t_total*1000:>12.2f}ms  "
              f"{ct_overhead:>12.1f}%  "
              f"{per_atom_us:>17.2f} μs")
        rows.append(dict(cell=label, n_atoms=int(n),
                          t_base_ms=float(t_base*1000),
                          t_ct2f_ms=float(t_ct*1000),
                          t_combined_ms=float(t_total*1000),
                          ct2f_overhead_pct=float(ct_overhead),
                          per_atom_us_combined=float(per_atom_us)))

    # Inference benchmark: ridge predict on synthetic 1000-feature model
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1000))
    y = rng.standard_normal(100)
    m = Ridge(alpha=1.0).fit(X, y)
    Xt = rng.standard_normal((1000, 1000))
    t_inf, _ = time_call(m.predict, Xt, n_warm=2, n_repeat=5)
    print()
    print(f"Ridge inference: 1000 predictions × 1000 features = "
          f"{t_inf*1000:.2f} ms ({t_inf*1e6/1000:.1f} μs per prediction)")

    # Comparison table vs DFTB+ / xtb (measured in prior phases)
    print()
    print("=" * 80)
    print("Cost comparison vs DFT engines (measured in this project)")
    print("=" * 80)
    comparisons = [
        ("OCE feature build (1F+2F+3F)",
         "Current production",         "12 ms/atom (Python triple-loop)"),
        ("OCE feature build (+CT2F)",
         "Wave-1 extension",            "~14 ms/atom (+18%)"),
        ("OCE ridge inference",
         "Per prediction",              "~μs / structure (negligible)"),
        ("xtb GFN2 single-point",
         "100-atom molecule, 8 threads","~10–30 s"),
        ("xtb GFN-FF (PBC)",
         "40-atom perov primitive",    "~100 ms (measured in run_gfnff)"),
        ("DFTB+ SZP (mio-1-1, periodic)",
         "150 cells, 12 workers",      "~73 s = 0.5 s/cell amortized"),
        ("SIESTA SZP (Pseudo-Dojo)",
         "5-atom CsPbI₃ primitive",    "9 s @ 4 threads"),
        ("SIESTA SZP",
         "40-atom 2x2x2 supercell",    "75 s @ 4 threads"),
        ("SIESTA SZP",
         "100-atom diamond cluster",   "~80 s @ 8 threads (phase15)"),
        ("SIESTA SZP",
         "687-atom diamond cluster",   "118 min @ 1 thread (outlier!)"),
    ]
    for name, ctx, t in comparisons:
        print(f"  {name:<35s}  [{ctx:<32s}]  {t}")

    # Save
    out = {
        "cells": rows,
        "inference_ms_per_1000_predict": float(t_inf*1000),
    }
    out_path = ROOT / "data" / "wave1_benchmark.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
