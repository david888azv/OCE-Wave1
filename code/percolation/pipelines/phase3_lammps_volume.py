"""Phase 3 — LAMMPS+AIREBO relaxed clusters: scaling of E, V, E/V vs L.

Workflow per L ∈ L_VALUES:
  1. Sample N_realisations site-percolation clusters at p_c on honeycomb.
  2. AIREBO single-point + minimisation (LAMMPS).
  3. Estimate volume of relaxed cluster (3 estimators for cross-check).
  4. Average ⟨E_relaxed⟩, ⟨V⟩, ⟨E/V⟩ over realisations.
  5. Log-log fit of E(L), V(L), E/V(L) → exponents.

Why AIREBO: cheap, well-parameterised reactive force-field for carbon,
handles dangling bonds without explicit spin assignment.  Scales O(N)
with small prefactor — comfortable up to L = 32 (~700 atoms) on this box.
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
from volume import (volume_atomic, volume_inflated_hull,
                     volume_obb_pad, planar_area_pad)
from analysis import fit_loglog


LATTICE = "honeycomb"
P = P_C[LATTICE]
L_VALUES = [4, 6, 8, 12, 16, 24, 32]
N_PER_L = {4: 24, 6: 24, 8: 18, 12: 16, 16: 12, 24: 10, 32: 8}
RANDOM_SEED = 20260508

DATA_DIR = ROOT / "data" / "phase3_lammps_volume"
RESULTS_DIR = ROOT / "results" / "phase3_lammps_volume"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "clusters_lammps_relax.json"


def gather():
    if CACHE.exists():
        records = json.loads(CACHE.read_text())
        print(f"Loaded {len(records)} cached records from {CACHE}")
        return records
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []
    t0 = time.perf_counter()
    for L in L_VALUES:
        lat = build_honeycomb(L)
        n = N_PER_L[L]
        clusters = sample_many(lat, P, n,
                                base_seed=int(rng.integers(2**31)),
                                min_size=4)
        for ic, cl in enumerate(clusters):
            atoms = Atoms(symbols=["C"] * cl.n_atoms, positions=cl.positions)
            sp = lammps_airebo(atoms, optimize=False)
            opt = lammps_airebo(atoms, optimize=True)
            if not (sp["converged"] and opt["converged"]):
                print(f"  [skip] L={L} rep={ic}  sp={sp['converged']} "
                      f"opt={opt['converged']}")
                continue

            pos_rel = opt["opt_atoms"].get_positions()
            v_atomic = volume_atomic(cl.n_atoms)
            v_obb = volume_obb_pad(pos_rel)
            v_hull = volume_inflated_hull(pos_rel)
            _, v_planar = planar_area_pad(pos_rel)

            rms_disp = float(np.sqrt(np.mean(np.sum(
                (pos_rel - cl.positions) ** 2, axis=1))))
            print(f"  L={L:2d}  rep={ic:2d}  N={cl.n_atoms:4d}  "
                  f"E_sp={sp['E_eV']:+10.3f}  E_opt={opt['E_eV']:+10.3f}  "
                  f"⟨|Δr|⟩={rms_disp:.2f}  "
                  f"V_obb={v_obb:6.0f}  V_hull={v_hull:6.0f}  "
                  f"({sp['wall_time_s']+opt['wall_time_s']:.1f}s)")
            records.append(dict(
                L=int(L), realisation=int(ic), seed=int(cl.seed),
                n_atoms=int(cl.n_atoms), n_bonds=int(cl.n_bonds),
                positions_unrelaxed=cl.positions.tolist(),
                positions_relaxed=pos_rel.tolist(),
                E_airebo_sp_eV=float(sp["E_eV"]),
                E_airebo_relaxed_eV=float(opt["E_eV"]),
                rms_displacement_A=rms_disp,
                V_atomic_A3=float(v_atomic),
                V_obb_A3=float(v_obb),
                V_inflated_hull_A3=float(v_hull),
                V_planar_A3=float(v_planar),
            ))
    print(f"\nTotal LAMMPS wall: {time.perf_counter() - t0:.1f}s "
          f"for {len(records)} clusters")
    CACHE.write_text(json.dumps(records, indent=2))
    return records


def per_L_means(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (L_array, mean(key), std(key)) grouped by L."""
    Ls = sorted({r["L"] for r in records})
    arrL = np.array(Ls, dtype=float)
    mean = []
    std = []
    for L in Ls:
        vals = np.array([r[key] for r in records if r["L"] == L
                          and np.isfinite(r[key])])
        mean.append(float(vals.mean()) if len(vals) else float("nan"))
        std.append(float(vals.std()) if len(vals) else float("nan"))
    return arrL, np.array(mean), np.array(std)


def main():
    print(f"=== Phase 3 LAMMPS+AIREBO: relaxed clusters, L = {L_VALUES} ===")
    records = gather()

    # Per-L aggregations
    arrL, n_mean, n_std = per_L_means(records, "n_atoms")
    arrL, e_mean, e_std = per_L_means(records, "E_airebo_relaxed_eV")
    _, v_obb_mean, v_obb_std = per_L_means(records, "V_obb_A3")
    _, v_hull_mean, v_hull_std = per_L_means(records, "V_inflated_hull_A3")
    _, v_planar_mean, v_planar_std = per_L_means(records, "V_planar_A3")
    _, v_atomic_mean, _ = per_L_means(records, "V_atomic_A3")

    # E/V means
    eov_obb = e_mean / v_obb_mean
    eov_hull = e_mean / v_hull_mean
    eov_planar = e_mean / v_planar_mean
    eov_atomic = e_mean / v_atomic_mean

    print(f"\n--- Per-L means (N realisations: {[N_PER_L[int(L)] for L in arrL]}) ---")
    print(f"  L    ⟨N⟩      ⟨E⟩(eV)     ⟨V_obb⟩(Å³)  ⟨V_hull⟩  ⟨V_planar⟩  ⟨V_atomic⟩")
    for i, L in enumerate(arrL):
        print(f"  {int(L):2d}   {n_mean[i]:6.1f}   {e_mean[i]:+10.2f}    "
              f"{v_obb_mean[i]:8.1f}    {v_hull_mean[i]:7.1f}   "
              f"{v_planar_mean[i]:7.1f}    {v_atomic_mean[i]:7.1f}")

    # ---------- Power-law fits ----------
    fit_N = fit_loglog(arrL, n_mean, label="⟨N⟩(L)")
    fit_E = fit_loglog(arrL, -e_mean, label="−⟨E⟩(L)")
    fit_V_obb = fit_loglog(arrL, v_obb_mean, label="⟨V_obb⟩(L)")
    fit_V_hull = fit_loglog(arrL, v_hull_mean, label="⟨V_hull⟩(L)")
    fit_V_planar = fit_loglog(arrL, v_planar_mean, label="⟨V_planar⟩(L)")
    fit_V_atomic = fit_loglog(arrL, v_atomic_mean, label="⟨V_atomic⟩(L)")
    fit_eov_obb = fit_loglog(arrL, -eov_obb, label="−⟨E/V_obb⟩(L)")
    fit_eov_hull = fit_loglog(arrL, -eov_hull, label="−⟨E/V_hull⟩(L)")
    fit_eov_planar = fit_loglog(arrL, -eov_planar, label="−⟨E/V_planar⟩(L)")

    print(f"\n--- Power-law exponents from log–log linear fit ---")
    print(f"  N(L)              ~ L^{fit_N.exponent:+.3f}    "
          f"(theory D_f = 91/48 ≈ 1.896)")
    print(f"  -E(L)             ~ L^{fit_E.exponent:+.3f}    "
          f"(extensive ≈ D_f)")
    print(f"  V_obb(L)          ~ L^{fit_V_obb.exponent:+.3f}    "
          f"oriented bbox (×vdW pad)")
    print(f"  V_hull(L)         ~ L^{fit_V_hull.exponent:+.3f}    "
          f"vdW-inflated convex hull")
    print(f"  V_planar(L)       ~ L^{fit_V_planar.exponent:+.3f}    "
          f"best-plane area × 2r_vdW")
    print(f"  V_atomic(L)       ~ L^{fit_V_atomic.exponent:+.3f}    "
          f"N·(4π/3)r³ — should equal D_f")
    print(f"\n--- E/V ratios (these are the user's question) ---")
    print(f"  −E/V_obb(L)       ~ L^{fit_eov_obb.exponent:+.3f}")
    print(f"  −E/V_hull(L)      ~ L^{fit_eov_hull.exponent:+.3f}")
    print(f"  −E/V_planar(L)    ~ L^{fit_eov_planar.exponent:+.3f}")
    print(f"  (−E/V_atomic(L)   ~ L^{(fit_E.exponent - fit_V_atomic.exponent):+.3f})  "
          f"-- exactly L^(D_E−D_f), trivial cross-check")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    ax.errorbar(arrL, n_mean, yerr=n_std, fmt="o-", label=rf"$\langle N\rangle$  $\sim L^{{{fit_N.exponent:.3f}}}$")
    ax.plot(arrL, arrL ** (91 / 48), ":", color="red", label=r"theory $L^{91/48}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L"); ax.set_ylabel(r"$\langle N\rangle$")
    ax.set_title("Cluster mass scaling")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.errorbar(arrL, -e_mean, yerr=e_std, fmt="o-", label=rf"$-\langle E\rangle$  $\sim L^{{{fit_E.exponent:.3f}}}$")
    ax.errorbar(arrL, v_obb_mean, yerr=v_obb_std, fmt="s-", color="C2",
                label=rf"$\langle V_{{obb}}\rangle$  $\sim L^{{{fit_V_obb.exponent:.3f}}}$")
    ax.errorbar(arrL, v_hull_mean, yerr=v_hull_std, fmt="x-", color="C3",
                label=rf"$\langle V_{{hull}}\rangle$  $\sim L^{{{fit_V_hull.exponent:.3f}}}$")
    ax.errorbar(arrL, v_planar_mean, yerr=v_planar_std, fmt="d-", color="C4",
                label=rf"$\langle V_{{planar}}\rangle$  $\sim L^{{{fit_V_planar.exponent:.3f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L"); ax.set_ylabel("eV  /  Å³")
    ax.set_title("Energy & volume vs L")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(arrL, -eov_obb, "s-", color="C2",
            label=rf"$-E/V_{{obb}}$  $\sim L^{{{fit_eov_obb.exponent:.3f}}}$")
    ax.plot(arrL, -eov_hull, "x-", color="C3",
            label=rf"$-E/V_{{hull}}$  $\sim L^{{{fit_eov_hull.exponent:.3f}}}$")
    ax.plot(arrL, -eov_planar, "d-", color="C4",
            label=rf"$-E/V_{{planar}}$  $\sim L^{{{fit_eov_planar.exponent:.3f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L"); ax.set_ylabel(r"$-\langle E\rangle / \langle V\rangle$  (eV/Å³)")
    ax.set_title("Energy density vs L")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_scaling.png", dpi=180)
    plt.close(fig)

    # Plain (linear) E/V vs L for the eyeball check
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(arrL, -eov_hull, "x-", color="C3", label=r"$-E/V_{hull}$")
    ax.plot(arrL, -eov_planar, "d-", color="C4", label=r"$-E/V_{planar}$")
    ax.plot(arrL, -eov_obb, "s-", color="C2", label=r"$-E/V_{obb}$")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$-\langle E\rangle / \langle V\rangle$  (eV/Å³)")
    ax.set_title("Energy density (linear axes)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_E_over_V_linear.png", dpi=180)
    plt.close(fig)

    summary = dict(
        lattice=LATTICE, p=float(P),
        L_values=[int(L) for L in arrL],
        n_clusters=len(records),
        D_f_empirical=fit_N.exponent,
        D_E_relaxed=fit_E.exponent,
        D_V_obb=fit_V_obb.exponent,
        D_V_hull=fit_V_hull.exponent,
        D_V_planar=fit_V_planar.exponent,
        D_V_atomic=fit_V_atomic.exponent,
        gamma_E_over_V_obb=fit_eov_obb.exponent,
        gamma_E_over_V_hull=fit_eov_hull.exponent,
        gamma_E_over_V_planar=fit_eov_planar.exponent,
        per_L=dict(
            L=[int(L) for L in arrL],
            N_atoms=[float(x) for x in n_mean],
            E_relaxed_eV=[float(x) for x in e_mean],
            V_obb_A3=[float(x) for x in v_obb_mean],
            V_hull_A3=[float(x) for x in v_hull_mean],
            V_planar_A3=[float(x) for x in v_planar_mean],
            V_atomic_A3=[float(x) for x in v_atomic_mean],
        ),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plots    → {RESULTS_DIR}/fig_*.png")


if __name__ == "__main__":
    main()
