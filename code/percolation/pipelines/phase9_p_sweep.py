"""Phase 9 — off-criticality p sweep.

Vary site-occupation probability p around p_c on the honeycomb lattice
(C-AIREBO) to test whether:

  (a)  D_f → 91/48 ≈ 1.896  is unique to p = p_c    (we expect yes)
  (b)  γ ≈ 0  holds at all p where the largest cluster grows with L
       (we expect yes — it is a "compact density" property of any
        sufficiently extensive cluster, not strictly tied to fractality)
  (c)  D_f → 2 for p > p_c                               (compact 2D)
  (d)  D_f saturates / becomes ill-defined for p < p_c   (cluster size
       saturates with L)

Single lattice (honeycomb), single material (C-AIREBO), L ∈ {16, 24, 32, 48, 64}.
p ∈ {0.7·p_c, 0.85·p_c, p_c, 1.15·p_c, 1.30·p_c}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms

from lattices import build_honeycomb, P_C
from percolation import sample_many
from runners_lammps import lammps_airebo
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw, R_VDW_C)
from analysis import fit_loglog


L_VALUES = [16, 24, 32, 48, 64]
N_PER_L = {16: 18, 24: 14, 32: 12, 48: 10, 64: 8}
P_C_HONEYCOMB = P_C["honeycomb"]
P_FRAC = [0.70, 0.85, 1.00, 1.15, 1.30]
RANDOM_SEED = 20260514
MC_SAMPLES = 300_000
MPI_PROCS = 8
PARALLEL_REPLICAS = 2

DATA_DIR = ROOT / "data" / "phase9_p_sweep"
RESULTS_DIR = ROOT / "results" / "phase9_p_sweep"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _process_one(args) -> dict | None:
    p_label, p_value, L, ic, cl_pos, cl_seed, cl_n, cl_bonds, mpi = args
    atoms = Atoms(symbols=["C"] * cl_n, positions=cl_pos)
    sp = lammps_airebo(atoms, optimize=False, timeout=1800, mpi_procs=mpi)
    opt = lammps_airebo(atoms, optimize=True, timeout=3600, mpi_procs=mpi)
    if not (sp["converged"] and opt["converged"]):
        return None
    pos_rel = opt["opt_atoms"].get_positions()
    return dict(
        p_label=p_label, p_value=float(p_value),
        L=int(L), realisation=int(ic),
        seed=int(cl_seed), n_atoms=int(cl_n), n_bonds=int(cl_bonds),
        E_relaxed_eV=float(opt["E_eV"]),
        V_mc_A3=float(volume_mc_vdw(pos_rel, r=R_VDW_C,
                                      n_samples=MC_SAMPLES, seed=cl_seed)),
        V_atomic_A3=float(volume_atomic(cl_n, r=R_VDW_C)),
        wall_s=float(sp["wall_time_s"] + opt["wall_time_s"]),
    )


def gather():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = DATA_DIR / "p_sweep.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rng = np.random.default_rng(RANDOM_SEED)
    tasks = []
    for frac in P_FRAC:
        p = P_C_HONEYCOMB * frac
        p_label = f"p={frac:.2f}p_c"
        for L in L_VALUES:
            lat = build_honeycomb(L)
            n = N_PER_L[L]
            clusters = sample_many(lat, p, n,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=4)
            for ic, cl in enumerate(clusters):
                tasks.append((p_label, p, L, ic, cl.positions, cl.seed,
                              cl.n_atoms, cl.n_bonds, MPI_PROCS))
    print(f"  queued {len(tasks)} clusters")

    records: list[dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL_REPLICAS) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            if r is None:
                continue
            print(f"  {r['p_label']:11s}  L={r['L']:3d} rep={r['realisation']:2d} "
                  f"N={r['n_atoms']:5d}  E={r['E_relaxed_eV']:+10.2f}  "
                  f"E/V={r['E_relaxed_eV']/r['V_mc_A3']:+.4f}  "
                  f"({r['wall_s']:.1f}s)", flush=True)
            records.append(r)
            cache.write_text(json.dumps(records, indent=2))
    print(f"  total {len(records)} clusters in {time.perf_counter() - t0:.1f}s")
    cache.write_text(json.dumps(records, indent=2))
    return records


def per_L_means(records, key) -> tuple[np.ndarray, np.ndarray]:
    Ls = sorted({r["L"] for r in records})
    mean = np.array([np.mean([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    return np.array(Ls, dtype=float), mean


def analyse(records: list[dict]) -> dict:
    summaries = {}
    for frac in P_FRAC:
        p_label = f"p={frac:.2f}p_c"
        sub = [r for r in records if r["p_label"] == p_label]
        if not sub:
            continue
        arrL, n_mean = per_L_means(sub, "n_atoms")
        _, e_mean = per_L_means(sub, "E_relaxed_eV")
        _, v_mc = per_L_means(sub, "V_mc_A3")
        _, v_at = per_L_means(sub, "V_atomic_A3")
        eov_mc = e_mean / v_mc
        eov_at = e_mean / v_at
        epn = e_mean / n_mean

        fit_N = fit_loglog(arrL, n_mean)
        fit_E = fit_loglog(arrL, -e_mean)
        fit_Vmc = fit_loglog(arrL, v_mc)
        fit_eov_mc = fit_loglog(arrL, -eov_mc)
        fit_eov_at = fit_loglog(arrL, -eov_at)
        fit_epn = fit_loglog(arrL, -epn)

        summaries[p_label] = dict(
            p_frac=float(frac), p_value=float(P_C_HONEYCOMB * frac),
            L=[int(L) for L in arrL],
            N=[float(x) for x in n_mean],
            E=[float(x) for x in e_mean],
            V_mc=[float(x) for x in v_mc],
            eov_mc=[float(x) for x in eov_mc],
            eov_atomic=[float(x) for x in eov_at],
            epn=[float(x) for x in epn],
            D_f=float(fit_N.exponent),
            D_E=float(fit_E.exponent),
            D_V_mc=float(fit_Vmc.exponent),
            gamma_eov_mc=float(fit_eov_mc.exponent),
            gamma_eov_atomic=float(fit_eov_at.exponent),
            alpha_epn=float(fit_epn.exponent),
            eps_V_star_mc=float(-eov_mc[-1]),
            eps_per_atom=float(-epn[-1]),
        )
    return summaries


def main():
    print(f"=== Phase 9 — honeycomb site percolation, p sweep around p_c ===")
    print(f"  p_c = {P_C_HONEYCOMB:.6f}\n")
    records = gather()
    summaries = analyse(records)

    print(f"\n{'='*86}")
    print(f"            P-SWEEP MASTER TABLE — honeycomb C-AIREBO")
    print(f"{'='*86}")
    print(f"  {'p / p_c':10s}  {'p_value':10s}  {'D_f':6s}  {'D_E':6s}  {'D_V_mc':7s}  "
          f"{'γ_mc':6s}  {'ε_V*':7s}  {'−E/N':6s}")
    for label in sorted(summaries, key=lambda k: summaries[k]["p_frac"]):
        s = summaries[label]
        marker = " ← p_c" if abs(s["p_frac"] - 1.0) < 1e-3 else ""
        print(f"  {s['p_frac']:.2f}        {s['p_value']:.4f}      "
              f"{s['D_f']:+.3f}  {s['D_E']:+.3f}  {s['D_V_mc']:+.3f}    "
              f"{s['gamma_eov_mc']:+.3f}  {s['eps_V_star_mc']:.3f}   "
              f"{s['eps_per_atom']:.3f}{marker}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fracs = sorted(set(s["p_frac"] for s in summaries.values()))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(fracs)))
    for color, frac in zip(cmap, fracs):
        s = next(s for s in summaries.values() if s["p_frac"] == frac)
        arrL = np.array(s["L"], dtype=float)
        marker = "o" if abs(frac - 1.0) < 1e-3 else "."
        ms = 8 if abs(frac - 1.0) < 1e-3 else 6
        axes[0].plot(arrL, s["N"], "-", marker=marker, color=color,
                     markersize=ms, label=f"p={frac:.2f}p_c  D_f={s['D_f']:+.3f}")
        axes[1].plot(arrL, [-x for x in s["E"]], "-", marker=marker,
                     color=color, markersize=ms,
                     label=f"p={frac:.2f}p_c  D_E={s['D_E']:+.3f}")
        axes[2].plot(arrL, [-x for x in s["eov_mc"]], "-", marker=marker,
                     color=color, markersize=ms,
                     label=f"p={frac:.2f}p_c  γ={s['gamma_eov_mc']:+.3f}")
    for ax in axes:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("L")
        ax.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("⟨N_atoms⟩")
    axes[0].set_title("Mass scaling vs p")
    axes[1].set_ylabel("−⟨E⟩ (eV)")
    axes[1].set_title("Energy scaling vs p")
    axes[2].set_ylabel(r"$-\langle E\rangle/\langle V\rangle$ (eV/Å³)")
    axes[2].set_title(r"Energy density vs L")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_p_sweep.png", dpi=180)
    plt.close(fig)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_p_sweep.png'}")


if __name__ == "__main__":
    main()
