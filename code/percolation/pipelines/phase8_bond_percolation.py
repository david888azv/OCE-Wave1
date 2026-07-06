"""Phase 8 — bond percolation universality test (C and Si).

Repeats phase 5 (C-AIREBO) and phase 6 (Si-Tersoff) with **bond**
percolation instead of site percolation, on the same 3 lattices:

    honeycomb  bond p_c = 1 − 2 sin(π/18) ≈ 0.6527036
    square     bond p_c = 0.5                exact (Kesten 1980)
    triangular bond p_c = 2 sin(π/18) ≈ 0.3472964

The 2D percolation universality class predicts identical critical
exponents (D_f = 91/48) and γ → 0 across BOTH percolation modes
(site and bond).  If ε_V* coincides between the two modes per material,
that is strong evidence for a "law" tied only to the universality class.
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

from lattices import build_honeycomb, build_square, build_triangular, P_C_BOND
from percolation import sample_many_bond
from runners_lammps import lammps_airebo, lammps_si_tersoff
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw, R_VDW_C, R_VDW_SI)
from analysis import fit_loglog


L_VALUES = [16, 24, 32, 48, 64, 96, 128]
N_PER_L = {16: 18, 24: 14, 32: 12, 48: 10, 64: 8, 96: 6, 128: 5}
A_C, A_SI = 1.42, 2.35
RANDOM_SEED = 20260513
MC_SAMPLES = 300_000
MPI_PROCS = 8
PARALLEL_REPLICAS = 2

LATTICES_C = {
    "honeycomb":  ((lambda L: build_honeycomb(L, a=A_C)),  P_C_BOND["honeycomb"]),
    "square":     ((lambda L: build_square(L, a=A_C)),      P_C_BOND["square"]),
    "triangular": ((lambda L: build_triangular(L, a=A_C)), P_C_BOND["triangular"]),
}
LATTICES_SI = {
    "honeycomb":  ((lambda L: build_honeycomb(L, a=A_SI)),  P_C_BOND["honeycomb"]),
    "square":     ((lambda L: build_square(L, a=A_SI)),      P_C_BOND["square"]),
    "triangular": ((lambda L: build_triangular(L, a=A_SI)), P_C_BOND["triangular"]),
}

DATA_DIR = ROOT / "data" / "phase8_bond"
RESULTS_DIR = ROOT / "results" / "phase8_bond"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _process_one(args) -> dict | None:
    material, lattice_name, L, ic, cl_pos, cl_seed, cl_n, cl_bonds, mpi_procs = args
    runner = lammps_airebo if material == "C" else lammps_si_tersoff
    atoms = Atoms(symbols=[material] * cl_n, positions=cl_pos)
    sp = runner(atoms, optimize=False, timeout=1800, mpi_procs=mpi_procs)
    opt = runner(atoms, optimize=True, timeout=3600, mpi_procs=mpi_procs)
    if not (sp["converged"] and opt["converged"]):
        return dict(_skip=True, name=lattice_name, L=L, ic=ic, n=cl_n,
                    err=sp.get("error", "") + " | " + opt.get("error", ""))
    pos_rel = opt["opt_atoms"].get_positions()
    rvdw = R_VDW_C if material == "C" else R_VDW_SI
    v_atomic = volume_atomic(cl_n, r=rvdw)
    v_obb = volume_obb_pad(pos_rel, r=rvdw)
    v_hull = volume_inflated_hull(pos_rel, r=rvdw)
    _, v_planar = planar_area_pad(pos_rel, r=rvdw)
    v_mc = volume_mc_vdw(pos_rel, r=rvdw, n_samples=MC_SAMPLES, seed=cl_seed)
    rms = float(np.sqrt(np.mean(np.sum((pos_rel - cl_pos) ** 2, axis=1))))
    e_key = "E_airebo_relaxed_eV" if material == "C" else "E_tersoff_relaxed_eV"
    e_sp_key = "E_airebo_sp_eV" if material == "C" else "E_tersoff_sp_eV"
    return {
        "lattice": lattice_name, "material": material,
        "L": int(L), "realisation": int(ic),
        "seed": int(cl_seed), "n_atoms": int(cl_n), "n_bonds": int(cl_bonds),
        e_key: float(opt["E_eV"]),
        e_sp_key: float(sp["E_eV"]),
        "rms_displacement_A": rms,
        "V_atomic_A3": float(v_atomic),
        "V_obb_A3": float(v_obb), "V_hull_A3": float(v_hull),
        "V_planar_A3": float(v_planar), "V_mc_A3": float(v_mc),
        "wall_sp_s": float(sp["wall_time_s"]),
        "wall_opt_s": float(opt["wall_time_s"]),
    }


def gather(material: str, lattices: dict) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = DATA_DIR / f"{material}_clusters.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
        print(f"  [{material}] loaded {len(recs)} cached")
        return recs
    rng = np.random.default_rng(RANDOM_SEED + (1 if material == "Si" else 0))
    records: list[dict] = []
    t0 = time.perf_counter()
    tasks = []
    for lattice_name, (builder, p) in lattices.items():
        for L in L_VALUES:
            lat = builder(L)
            n = N_PER_L[L]
            clusters = sample_many_bond(lat, p, n,
                                         base_seed=int(rng.integers(2**31)),
                                         min_size=4)
            for ic, cl in enumerate(clusters):
                tasks.append((material, lattice_name, L, ic, cl.positions,
                              cl.seed, cl.n_atoms, cl.n_bonds, MPI_PROCS))

    with ThreadPoolExecutor(max_workers=PARALLEL_REPLICAS) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for f in as_completed(futures):
            t = futures[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f"  [exc] {material} L={t[2]} ic={t[3]}: {e}")
                continue
            if rec.get("_skip"):
                print(f"  [skip] {material}/{rec['name']} L={rec['L']} "
                      f"ic={rec['ic']}  N={rec['n']}")
                continue
            tt = (rec["wall_sp_s"] + rec["wall_opt_s"])
            e_key = "E_airebo_relaxed_eV" if material == "C" else "E_tersoff_relaxed_eV"
            print(f"  [{material}/{rec['lattice']}] L={rec['L']:3d} "
                  f"rep={rec['realisation']:2d} N={rec['n_atoms']:5d}  "
                  f"E={rec[e_key]:+10.2f}  V_mc={rec['V_mc_A3']:8.0f}  "
                  f"E/V={rec[e_key]/rec['V_mc_A3']:+.4f}  "
                  f"({tt:.1f}s)", flush=True)
            records.append(rec)
            cache.write_text(json.dumps(sorted(
                records, key=lambda r: (r["lattice"], r["L"], r["realisation"])
            ), indent=2))
    print(f"  [{material}] {len(records)} clusters in "
          f"{time.perf_counter() - t0:.1f}s\n")
    cache.write_text(json.dumps(sorted(
        records, key=lambda r: (r["lattice"], r["L"], r["realisation"])
    ), indent=2))
    return records


def per_L_means(records, key) -> tuple[np.ndarray, np.ndarray]:
    Ls = sorted({r["L"] for r in records})
    mean = np.array([np.mean([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    return np.array(Ls, dtype=float), mean


def analyse(material: str, records: list[dict]) -> dict:
    e_key = "E_airebo_relaxed_eV" if material == "C" else "E_tersoff_relaxed_eV"
    out = {}
    for lattice in ("honeycomb", "square", "triangular"):
        sub = [r for r in records if r["lattice"] == lattice]
        if not sub:
            continue
        arrL, n_mean = per_L_means(sub, "n_atoms")
        _, e_mean = per_L_means(sub, e_key)
        _, v_mc_mean = per_L_means(sub, "V_mc_A3")
        _, v_atomic_mean = per_L_means(sub, "V_atomic_A3")

        eov_mc = e_mean / v_mc_mean
        eov_atomic = e_mean / v_atomic_mean
        epn = e_mean / n_mean

        fit_N = fit_loglog(arrL, n_mean)
        fit_E = fit_loglog(arrL, -e_mean)
        fit_Vmc = fit_loglog(arrL, v_mc_mean)
        fit_eov_mc = fit_loglog(arrL, -eov_mc)
        fit_epn = fit_loglog(arrL, -epn)

        out[lattice] = dict(
            material=material, L=[int(L) for L in arrL],
            N=[float(x) for x in n_mean],
            E=[float(x) for x in e_mean],
            V_mc=[float(x) for x in v_mc_mean],
            eov_mc=[float(x) for x in eov_mc],
            eov_atomic=[float(x) for x in eov_atomic],
            epn=[float(x) for x in epn],
            D_f=float(fit_N.exponent),
            D_E=float(fit_E.exponent),
            D_V_mc=float(fit_Vmc.exponent),
            gamma=float(fit_eov_mc.exponent),
            alpha=float(fit_epn.exponent),
            eps_V_star_mc=float(-eov_mc[-1]),
            eps_V_star_atomic=float(-eov_atomic[-1]),
            eps_per_atom=float(-epn[-1]),
        )
    return out


def main():
    print("=== Phase 8 — bond percolation, both materials ===")
    print(f"  honeycomb  bond p_c = {P_C_BOND['honeycomb']:.6f}")
    print(f"  square     bond p_c = {P_C_BOND['square']:.6f}")
    print(f"  triangular bond p_c = {P_C_BOND['triangular']:.6f}\n")

    summaries = {}
    for material, lattices in (("C", LATTICES_C), ("Si", LATTICES_SI)):
        recs = gather(material, lattices)
        summaries[material] = analyse(material, recs)

    print(f"\n{'='*92}")
    print(f"            BOND-PERCOLATION SUMMARY (master table)")
    print(f"{'='*92}")
    print(f"  {'mat':3s} {'lattice':10s}  {'D_f':6s} {'D_E':6s} {'D_Vmc':6s} "
          f"{'γ':7s}  {'ε_V*(mc)':9s} {'ε_V*(at)':9s} {'−E/N':7s}")
    for mat, s in summaries.items():
        for lat, r in s.items():
            print(f"   {mat:2s}  {lat:10s}  "
                  f"{r['D_f']:+.3f} {r['D_E']:+.3f} {r['D_V_mc']:+.3f}  "
                  f"{r['gamma']:+.3f}  {r['eps_V_star_mc']:.4f}    "
                  f"{r['eps_V_star_atomic']:.4f}    "
                  f"{r['eps_per_atom']:.3f}")

    # Compare to phase 5/6 (site percolation)
    site5 = ROOT / "results" / "phase5_three_lattices" / "summary.json"
    site6 = ROOT / "results" / "phase6_silicon" / "summary.json"
    if site5.exists() and site6.exists():
        site_C = json.loads(site5.read_text())
        site_Si = json.loads(site6.read_text())
        print(f"\n{'='*78}")
        print(f"            SITE vs BOND comparison — ε_V*(MC) (eV/Å³)")
        print(f"{'='*78}")
        print(f"  {'':12s}  {'site (P5/P6)':24s} {'bond (P8)':24s}")
        for mat in ("C", "Si"):
            site_summ = site_C if mat == "C" else site_Si
            print(f"  ----- {mat} -----")
            for lat in ("honeycomb", "square", "triangular"):
                site_v = site_summ[lat]["eps_V_star_mc"]
                bond_v = summaries[mat].get(lat, {}).get("eps_V_star_mc", float("nan"))
                ratio = bond_v / site_v if site_v else float("nan")
                print(f"   {lat:10s}  site={site_v:.4f}             "
                      f"bond={bond_v:.4f}  ratio={ratio:.3f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"honeycomb": "C0", "square": "C1", "triangular": "C2"}
    for ax, (mat, s) in zip(axes, summaries.items()):
        for lat, r in s.items():
            arrL = np.array(r["L"], dtype=float)
            ax.plot(arrL, [-x for x in r["eov_mc"]], "o-",
                     color=colors[lat],
                     label=f"{lat}  γ={r['gamma']:+.3f}, ε_V*={-r['eov_mc'][-1]:.3f}")
        ax.set_xlabel("L"); ax.set_ylabel(r"$-\langle E\rangle/\langle V_{mc}\rangle$ (eV/Å³)")
        ax.set_title(f"{mat} — bond percolation E/V vs L")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_bond_universal.png", dpi=180)
    plt.close(fig)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_bond_universal.png'}")


if __name__ == "__main__":
    main()
