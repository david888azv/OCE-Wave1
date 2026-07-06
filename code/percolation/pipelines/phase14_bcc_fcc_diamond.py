"""Phase 14 — extend 3D universality test to BCC, FCC, and diamond cubic.

Adds 3 more 3D Bravais lattices to phase 13 (which had only simple cubic):

  bcc_3d      z=8   p_c = 0.24596   (Lorenz & Ziff 1998)
  fcc_3d      z=12  p_c = 0.19923   (Lorenz & Ziff 1998)
  diamond_3d  z=4   p_c = 0.43003   (van der Marck 1998)

Diamond is the natural ground-state lattice of Si and Ge, so the spread
across {SC, BCC, FCC, diamond} for those materials should *shrink*
(removing the "frustrated coordination" artefact of phase 6).

Same 3 materials (C, Si, Ge), same volume / energy protocol.
L = {8, 12, 16, 24, 32}; we keep L_max=32 here because diamond has 8 atoms
per conventional cell so the lattice itself becomes large quickly.
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

from lattices import (build_bcc_3d, build_fcc_3d, build_diamond_3d, P_C)
from percolation import sample_many
from runners_lammps import (lammps_airebo, lammps_si_tersoff,
                              lammps_ge_tersoff)
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw,
                     R_VDW_C, R_VDW_SI, R_VDW_GE)
from analysis import fit_loglog


L_VALUES = [8, 12, 16, 24, 32]
N_PER_L = {8: 18, 12: 14, 16: 12, 24: 10, 32: 8}

A_C, A_SI, A_GE = 1.42, 2.35, 2.45
A_BY_MATERIAL = {"C": A_C, "Si": A_SI, "Ge": A_GE}
RVDW_BY_MATERIAL = {"C": R_VDW_C, "Si": R_VDW_SI, "Ge": R_VDW_GE}
RUNNER_BY_MATERIAL = {
    "C":  lammps_airebo,
    "Si": lammps_si_tersoff,
    "Ge": lammps_ge_tersoff,
}
LATTICE_BUILDERS = {
    "bcc_3d":     build_bcc_3d,
    "fcc_3d":     build_fcc_3d,
    "diamond_3d": build_diamond_3d,
}

CONFIGS = [(mat, lat, LATTICE_BUILDERS[lat], P_C[lat])
           for mat in ("C", "Si", "Ge")
           for lat in ("bcc_3d", "fcc_3d", "diamond_3d")]

RANDOM_SEED = 20260517
MC_SAMPLES = 300_000
MPI_PROCS = 8
PARALLEL_REPLICAS = 2

DATA_DIR = ROOT / "data" / "phase14_bcc_fcc_diamond"
RESULTS_DIR = ROOT / "results" / "phase14_bcc_fcc_diamond"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "clusters.json"


def _process_one(args) -> dict | None:
    material, lat_name, L, ic, cl_pos, cl_seed, cl_n, cl_bonds, mpi = args
    runner = RUNNER_BY_MATERIAL[material]
    rvdw = RVDW_BY_MATERIAL[material]
    atoms = Atoms(symbols=[material] * cl_n, positions=cl_pos)
    sp = runner(atoms, optimize=False, timeout=1800, mpi_procs=mpi)
    opt = runner(atoms, optimize=True, timeout=3600, mpi_procs=mpi)
    if not (sp["converged"] and opt["converged"]):
        return dict(_skip=True, mat=material, lat=lat_name, ic=ic, n=cl_n)
    pos_rel = opt["opt_atoms"].get_positions()
    v_atomic = volume_atomic(cl_n, r=rvdw)
    v_obb = volume_obb_pad(pos_rel, r=rvdw)
    v_hull = volume_inflated_hull(pos_rel, r=rvdw)
    _, v_planar = planar_area_pad(pos_rel, r=rvdw)
    v_mc = volume_mc_vdw(pos_rel, r=rvdw, n_samples=MC_SAMPLES, seed=cl_seed)
    rms = float(np.sqrt(np.mean(np.sum((pos_rel - cl_pos) ** 2, axis=1))))
    return dict(
        material=material, lattice=lat_name, dim="3D",
        L=int(L), realisation=int(ic),
        seed=int(cl_seed), n_atoms=int(cl_n), n_bonds=int(cl_bonds),
        E_relaxed_eV=float(opt["E_eV"]),
        E_sp_eV=float(sp["E_eV"]),
        rms_displacement_A=rms,
        V_atomic_A3=float(v_atomic),
        V_obb_A3=float(v_obb), V_hull_A3=float(v_hull),
        V_planar_A3=float(v_planar), V_mc_A3=float(v_mc),
        wall_sp_s=float(sp["wall_time_s"]),
        wall_opt_s=float(opt["wall_time_s"]),
    )


def gather():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    rng = np.random.default_rng(RANDOM_SEED)
    tasks = []
    for material, lat_name, builder, p in CONFIGS:
        a_bond = A_BY_MATERIAL[material]
        for L in L_VALUES:
            lat = builder(L, a=a_bond)
            n = N_PER_L[L]
            clusters = sample_many(lat, p, n,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=4)
            for ic, cl in enumerate(clusters):
                tasks.append((material, lat_name, L, ic, cl.positions,
                              cl.seed, cl.n_atoms, cl.n_bonds, MPI_PROCS))
    print(f"  queued {len(tasks)} tasks")

    records: list[dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL_REPLICAS) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for f in as_completed(futures):
            t = futures[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f"  [exc] {t[0]}/{t[1]} L={t[2]}: {e}")
                continue
            if rec.get("_skip"):
                print(f"  [skip] {rec['mat']}/{rec['lat']} L={t[2]} ic={rec['ic']}")
                continue
            tt = (rec["wall_sp_s"] + rec["wall_opt_s"])
            print(f"  [{rec['material']}/{rec['lattice']}] L={rec['L']:3d} "
                  f"rep={rec['realisation']:2d} N={rec['n_atoms']:5d}  "
                  f"E/N={rec['E_relaxed_eV']/rec['n_atoms']:+5.3f}  "
                  f"V_mc={rec['V_mc_A3']:8.0f}  "
                  f"E/V={rec['E_relaxed_eV']/rec['V_mc_A3']:+.4f}  "
                  f"({tt:.1f}s)", flush=True)
            records.append(rec)
            cache_data = sorted(records,
                key=lambda r: (r["material"], r["lattice"], r["L"], r["realisation"]))
            CACHE.write_text(json.dumps(cache_data, indent=2))
    print(f"  total {len(records)} clusters in {time.perf_counter() - t0:.1f}s")
    cache_data = sorted(records,
        key=lambda r: (r["material"], r["lattice"], r["L"], r["realisation"]))
    CACHE.write_text(json.dumps(cache_data, indent=2))
    return records


def per_L_means(records, key) -> tuple[np.ndarray, np.ndarray]:
    Ls = sorted({r["L"] for r in records})
    mean = np.array([np.mean([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    return np.array(Ls, dtype=float), mean


def main():
    print("=== Phase 14 — BCC + FCC + diamond 3D ===\n")
    records = gather()
    if not records:
        return

    summary = []
    for mat in ("C", "Si", "Ge"):
        for lat in ("bcc_3d", "fcc_3d", "diamond_3d"):
            sub = [r for r in records
                    if r["material"] == mat and r["lattice"] == lat]
            if not sub:
                continue
            arrL, n_mean = per_L_means(sub, "n_atoms")
            _, e_mean = per_L_means(sub, "E_relaxed_eV")
            _, v_mc = per_L_means(sub, "V_mc_A3")
            eov = e_mean / v_mc
            epn = e_mean / n_mean
            fit_N = fit_loglog(arrL, n_mean)
            fit_E = fit_loglog(arrL, -e_mean)
            fit_V = fit_loglog(arrL, v_mc)
            fit_eov = fit_loglog(arrL, -eov)
            summary.append(dict(
                material=mat, lattice=lat, dim="3D",
                L=[int(L) for L in arrL],
                N=[float(x) for x in n_mean],
                E=[float(x) for x in e_mean],
                V_mc=[float(x) for x in v_mc],
                eov_mc=[float(x) for x in eov],
                epn=[float(x) for x in epn],
                D_f=float(fit_N.exponent),
                D_E=float(fit_E.exponent),
                D_V=float(fit_V.exponent),
                gamma=float(fit_eov.exponent),
                eps_V_star_at_Lmax=float(-eov[-1]),
                eps_per_atom_at_Lmax=float(-epn[-1]),
                n_clusters=len(sub),
            ))

    print(f"\n{'='*100}")
    print(f"            PHASE 14 MASTER TABLE — BCC/FCC/diamond, 3 materials")
    print(f"{'='*100}")
    print(f"  {'mat':3s} {'lattice':12s} {'D_f':6s}  {'D_E':6s}  {'D_V':6s}  "
          f"{'γ':6s}  {'ε_V*':8s}  {'−E/N':8s}  n")
    for s in summary:
        print(f"   {s['material']:2s}  {s['lattice']:12s} "
              f"{s['D_f']:+.3f} {s['D_E']:+.3f} {s['D_V']:+.3f}  "
              f"{s['gamma']:+.3f}  {s['eps_V_star_at_Lmax']:.4f}  "
              f"{s['eps_per_atom_at_Lmax']:5.3f}    {s['n_clusters']}")

    # Overall cross-lattice spread (4 lattices: SC + BCC + FCC + diamond)
    p13_summ = ROOT / "results" / "phase13_ge_and_3d" / "summary.json"
    if p13_summ.exists():
        p13 = json.loads(p13_summ.read_text())
        print(f"\n{'='*88}")
        print(f"            CROSS-LATTICE SPREAD ε_V* per material (4 3D lattices)")
        print(f"{'='*88}")
        print(f"  {'mat':3s}  {'SC':10s} {'BCC':10s} {'FCC':10s} {'diamond':10s}    mean    spread")
        for mat in ("C", "Si", "Ge"):
            sc_v = next((s["eps_V_star_at_Lmax"] for s in p13
                         if s["material"] == mat and s["lattice"] == "cubic_3d"), None)
            vals = [sc_v] if sc_v is not None else [None]
            for lat in ("bcc_3d", "fcc_3d", "diamond_3d"):
                v = next((s["eps_V_star_at_Lmax"] for s in summary
                          if s["material"] == mat and s["lattice"] == lat), None)
                vals.append(v)
            if all(v is not None for v in vals):
                arr = np.array(vals)
                m = arr.mean()
                spread = (arr.max() - arr.min()) / m * 100
                print(f"   {mat:2s}   "
                      f"{vals[0]:.4f}    {vals[1]:.4f}    "
                      f"{vals[2]:.4f}    {vals[3]:.4f}      "
                      f"{m:.4f}   {spread:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"C": "C0", "Si": "C1", "Ge": "C2"}
    markers = {"bcc_3d": "v", "fcc_3d": "s", "diamond_3d": "D"}
    for s in summary:
        c = colors[s["material"]]
        m = markers[s["lattice"]]
        arrL = np.array(s["L"], dtype=float)
        ax.plot(arrL, [-x for x in s["eov_mc"]], "-",
                marker=m, color=c,
                label=f"{s['material']}/{s['lattice']}  γ={s['gamma']:+.3f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("L"); ax.set_ylabel(r"$-\langle E\rangle/\langle V_{mc}\rangle$ (eV/Å³)")
    ax.set_title("3D BCC/FCC/diamond — energy density")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_3d_bcc_fcc_diamond.png", dpi=180)
    plt.close(fig)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
