"""Phase 4 — cross-lattice universality: site-percolation at p_c.

Tests whether the L-independent E/V observed in Phase 3 (honeycomb only)
generalises to a different lattice (square) and whether the asymptotic
constant ε_V* = lim_{L→∞} ⟨E⟩/⟨V⟩ is universal across lattices for the
same chemistry (carbon, AIREBO).

Also upgrades the volume estimator: precise Monte-Carlo integration of
the vdW-sphere union, parameter-free up to the choice of r_vdW.

Settings:
  L_VALUES = {16, 24, 32, 48, 64}
  lattices = {honeycomb (p_c=0.6970), square (p_c=0.5927)}
  N_per_L  ≈ 12 ... 6 (more samples at small L)
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

from lattices import build_honeycomb, build_square, P_C
from percolation import sample_many
from runners_lammps import lammps_airebo
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw)
from analysis import fit_loglog


L_VALUES = [16, 24, 32, 48, 64]
N_PER_L = {16: 12, 24: 10, 32: 8, 48: 6, 64: 5}
LATTICES = {
    "honeycomb": (build_honeycomb, P_C["honeycomb"]),
    "square":    (build_square,    P_C["square"]),
}
RANDOM_SEED = 20260509
MC_SAMPLES = 300_000

DATA_DIR = ROOT / "data" / "phase4_cross_lattice"
RESULTS_DIR = ROOT / "results" / "phase4_cross_lattice"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def gather_lattice(name: str, builder, p: float) -> list[dict]:
    cache = DATA_DIR / f"{name}_clusters.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
        print(f"  [{name}] loaded {len(recs)} cached")
        return recs
    rng = np.random.default_rng(RANDOM_SEED + hash(name) % 10_000)
    records: list[dict] = []
    t0 = time.perf_counter()
    for L in L_VALUES:
        lat = builder(L)
        n = N_PER_L[L]
        clusters = sample_many(lat, p, n,
                                base_seed=int(rng.integers(2**31)),
                                min_size=4)
        for ic, cl in enumerate(clusters):
            atoms = Atoms(symbols=["C"] * cl.n_atoms, positions=cl.positions)
            sp = lammps_airebo(atoms, optimize=False)
            opt = lammps_airebo(atoms, optimize=True)
            if not (sp["converged"] and opt["converged"]):
                print(f"  [skip] {name} L={L} rep={ic}")
                continue
            pos_rel = opt["opt_atoms"].get_positions()
            v_atomic = volume_atomic(cl.n_atoms)
            v_obb = volume_obb_pad(pos_rel)
            v_hull = volume_inflated_hull(pos_rel)
            _, v_planar = planar_area_pad(pos_rel)
            v_mc = volume_mc_vdw(pos_rel, n_samples=MC_SAMPLES,
                                  seed=cl.seed)
            rms = float(np.sqrt(np.mean(np.sum(
                (pos_rel - cl.positions) ** 2, axis=1))))
            records.append(dict(
                lattice=name, L=int(L), realisation=int(ic),
                seed=int(cl.seed), n_atoms=int(cl.n_atoms),
                n_bonds=int(cl.n_bonds),
                positions_relaxed=pos_rel.tolist(),
                E_airebo_relaxed_eV=float(opt["E_eV"]),
                E_airebo_sp_eV=float(sp["E_eV"]),
                rms_displacement_A=rms,
                V_atomic_A3=v_atomic,
                V_obb_A3=v_obb,
                V_hull_A3=v_hull,
                V_planar_A3=v_planar,
                V_mc_A3=v_mc,
            ))
            print(f"  [{name}] L={L:2d} rep={ic:2d} N={cl.n_atoms:4d}  "
                  f"E={opt['E_eV']:+9.2f}  V_mc={v_mc:7.0f}  "
                  f"V_hull={v_hull:7.0f}  E/V_mc={opt['E_eV']/v_mc:+.4f}  "
                  f"({sp['wall_time_s']+opt['wall_time_s']:.1f}s)")
    print(f"  [{name}] {len(records)} clusters in "
          f"{time.perf_counter() - t0:.1f}s")
    cache.write_text(json.dumps(records, indent=2))
    return records


def per_L_means(records, key) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ls = sorted({r["L"] for r in records})
    arrL = np.array(Ls, dtype=float)
    mean = np.array([np.mean([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    std = np.array([np.std([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    return arrL, mean, std


def analyse(name: str, records: list[dict]) -> dict:
    arrL, n_mean, n_std = per_L_means(records, "n_atoms")
    arrL, e_mean, e_std = per_L_means(records, "E_airebo_relaxed_eV")
    _, v_mc_mean, v_mc_std = per_L_means(records, "V_mc_A3")
    _, v_hull_mean, _ = per_L_means(records, "V_hull_A3")
    _, v_atomic_mean, _ = per_L_means(records, "V_atomic_A3")

    eov_mc = e_mean / v_mc_mean
    eov_hull = e_mean / v_hull_mean
    eov_atomic = e_mean / v_atomic_mean
    epn = e_mean / n_mean

    fit_N = fit_loglog(arrL, n_mean)
    fit_E = fit_loglog(arrL, -e_mean)
    fit_V_mc = fit_loglog(arrL, v_mc_mean)
    fit_V_hull = fit_loglog(arrL, v_hull_mean)
    fit_V_atomic = fit_loglog(arrL, v_atomic_mean)
    fit_eov_mc = fit_loglog(arrL, -eov_mc)
    fit_eov_hull = fit_loglog(arrL, -eov_hull)
    fit_eov_atomic = fit_loglog(arrL, -eov_atomic)

    print(f"\n=== Lattice: {name} ===")
    print(f"  N(L)            ~ L^{fit_N.exponent:+.3f}    "
          f"(theory D_f = 91/48 ≈ 1.896)")
    print(f"  -E(L)           ~ L^{fit_E.exponent:+.3f}")
    print(f"  V_mc(L)         ~ L^{fit_V_mc.exponent:+.3f}    "
          f"(MC vdW union, parameter-free)")
    print(f"  V_hull(L)       ~ L^{fit_V_hull.exponent:+.3f}    "
          f"(vdW-inflated convex hull)")
    print(f"  V_atomic(L)     ~ L^{fit_V_atomic.exponent:+.3f}    "
          f"(N · 4πr³/3)")
    print(f"  -E/V_mc(L)      ~ L^{fit_eov_mc.exponent:+.3f}    "
          f"⟨−E/V⟩(L_max) = {-eov_mc[-1]:.4f} eV/Å³")
    print(f"  -E/V_hull(L)    ~ L^{fit_eov_hull.exponent:+.3f}    "
          f"⟨−E/V⟩(L_max) = {-eov_hull[-1]:.4f} eV/Å³")
    print(f"  -E/V_atomic(L)  ~ L^{fit_eov_atomic.exponent:+.3f}    "
          f"⟨−E/V⟩(L_max) = {-eov_atomic[-1]:.4f} eV/Å³")
    print(f"  -E/N(L) (per atom) {-epn[-1]:+.3f} eV/atom at L_max={int(arrL[-1])}")

    return dict(
        L=[int(L) for L in arrL],
        N_atoms=[float(x) for x in n_mean],
        E_relaxed=[float(x) for x in e_mean],
        E_relaxed_std=[float(x) for x in e_std],
        V_mc=[float(x) for x in v_mc_mean],
        V_mc_std=[float(x) for x in v_mc_std],
        V_hull=[float(x) for x in v_hull_mean],
        V_atomic=[float(x) for x in v_atomic_mean],
        eov_mc=[float(x) for x in eov_mc],
        eov_hull=[float(x) for x in eov_hull],
        eov_atomic=[float(x) for x in eov_atomic],
        epn=[float(x) for x in epn],
        D_f=float(fit_N.exponent),
        D_E=float(fit_E.exponent),
        D_V_mc=float(fit_V_mc.exponent),
        D_V_hull=float(fit_V_hull.exponent),
        D_V_atomic=float(fit_V_atomic.exponent),
        gamma_eov_mc=float(fit_eov_mc.exponent),
        gamma_eov_hull=float(fit_eov_hull.exponent),
        gamma_eov_atomic=float(fit_eov_atomic.exponent),
        eps_V_star_mc=float(-eov_mc[-1]),
        eps_V_star_hull=float(-eov_hull[-1]),
        eps_V_star_atomic=float(-eov_atomic[-1]),
        eps_per_atom_at_Lmax=float(-epn[-1]),
    )


def main():
    print(f"=== Phase 4 cross-lattice (honeycomb + square), L = {L_VALUES} ===")
    summaries = {}
    for name, (builder, p) in LATTICES.items():
        recs = gather_lattice(name, builder, p)
        summaries[name] = analyse(name, recs)

    # Universality check
    print(f"\n=== Universality: same ε_V* across lattices? ===")
    for est in ("mc", "hull", "atomic"):
        a = summaries["honeycomb"][f"eps_V_star_{est}"]
        b = summaries["square"][f"eps_V_star_{est}"]
        print(f"  ε_V* via V_{est:6s}  honeycomb={a:.4f} eV/Å³   "
              f"square={b:.4f} eV/Å³   ratio={b/a:.3f}")
    print(f"\n  ε_per_atom (cohesive proxy)  "
          f"honeycomb={summaries['honeycomb']['eps_per_atom_at_Lmax']:.3f} eV/atom   "
          f"square={summaries['square']['eps_per_atom_at_Lmax']:.3f} eV/atom")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"honeycomb": "C0", "square": "C1"}
    for est in ("mc", "hull"):
        ax = axes[0 if est == "mc" else 1]
        for name in summaries:
            s = summaries[name]
            arrL = np.array(s["L"], dtype=float)
            ax.plot(arrL, [-x for x in s[f"eov_{est}"]],
                    "o-", color=colors[name],
                    label=f"{name}   exponent γ = {s[f'gamma_eov_{est}']:+.3f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("L")
        ax.set_ylabel(rf"$-\langle E\rangle / \langle V_{{{est}}}\rangle$  (eV/Å³)")
        ax.set_title(f"E/V using V_{est}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_universal.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in summaries:
        s = summaries[name]
        arrL = np.array(s["L"], dtype=float)
        ax.plot(arrL, [-x for x in s["eov_mc"]],
                "o-", color=colors[name], label=f"{name}")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$-\langle E\rangle / \langle V_{mc}\rangle$  (eV/Å³)")
    ax.set_title(f"Energy density (precise MC vdW volume)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_energy_density_linear.png", dpi=180)
    plt.close(fig)

    summaries["meta"] = dict(
        L_values=L_VALUES, MC_SAMPLES=MC_SAMPLES,
        N_PER_L=N_PER_L, p_honeycomb=P_C["honeycomb"],
        p_square=P_C["square"],
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plots    → {RESULTS_DIR}/fig_*.png")


if __name__ == "__main__":
    main()
