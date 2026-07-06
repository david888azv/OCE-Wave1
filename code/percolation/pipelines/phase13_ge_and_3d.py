"""Phase 13 — extend universality test to (a) Ge Tersoff and (b) 3D cubic.

  (a) Ge with Tersoff PRB 1989 (Ge.tersoff): 3rd material, between C
      (light, 4-valence, sp²/sp³) and Si (heavier, 4-valence, sp³ only).
      Covalent radius / vdW radius slightly larger than Si.
      Bond length a₀ = 2.45 Å (diamond Ge).

  (b) Simple-cubic 3D lattice, p_c (site) = 0.31161 (Lorenz & Ziff 1998).
      Universal D_f for 3D percolation = 2.523 (vs. 91/48 ≈ 1.896 in 2D).
      Tests dimensional robustness of γ ≈ 0 and ε_V* universality.

For (a) we run Ge on the same 3 2D lattices as before.
For (b) we run all 3 materials (C, Si, Ge) on cubic 3D.
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

from lattices import (build_honeycomb, build_square, build_triangular,
                       build_cubic_3d, P_C)
from percolation import sample_many
from runners_lammps import (lammps_airebo, lammps_si_tersoff,
                              lammps_ge_tersoff)
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw,
                     R_VDW_C, R_VDW_SI, R_VDW_GE)
from analysis import fit_loglog


L_VALUES_2D = [16, 24, 32, 48, 64, 96, 128]
N_PER_L_2D = {16: 18, 24: 14, 32: 12, 48: 10, 64: 8, 96: 6, 128: 5}
L_VALUES_3D = [8, 12, 16, 24, 32, 48, 64]
N_PER_L_3D = {8: 18, 12: 14, 16: 12, 24: 10, 32: 8, 48: 6, 64: 5}

A_C, A_SI, A_GE = 1.42, 2.35, 2.45
A_BY_MATERIAL = {"C": A_C, "Si": A_SI, "Ge": A_GE}
RVDW_BY_MATERIAL = {"C": R_VDW_C, "Si": R_VDW_SI, "Ge": R_VDW_GE}
RUNNER_BY_MATERIAL = {
    "C":  lammps_airebo,
    "Si": lammps_si_tersoff,
    "Ge": lammps_ge_tersoff,
}

# Phase 13a: Ge in 3 2D lattices
PART_A = [
    ("Ge", "honeycomb",  build_honeycomb,  P_C["honeycomb"], "2D"),
    ("Ge", "square",     build_square,      P_C["square"], "2D"),
    ("Ge", "triangular", build_triangular, P_C["triangular"], "2D"),
]
# Phase 13b: 3 materials on 3D cubic
PART_B = [
    ("C", "cubic_3d",  build_cubic_3d, P_C["cubic_3d"], "3D"),
    ("Si", "cubic_3d", build_cubic_3d, P_C["cubic_3d"], "3D"),
    ("Ge", "cubic_3d", build_cubic_3d, P_C["cubic_3d"], "3D"),
]

RANDOM_SEED = 20260516
MC_SAMPLES = 300_000
MPI_PROCS = 8
PARALLEL_REPLICAS = 2

DATA_DIR = ROOT / "data" / "phase13_ge_and_3d"
RESULTS_DIR = ROOT / "results" / "phase13_ge_and_3d"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _process_one(args) -> dict | None:
    material, lat_name, dim, L, ic, cl_pos, cl_seed, cl_n, cl_bonds, mpi = args
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
        material=material, lattice=lat_name, dim=dim,
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


def gather_one_set(label: str, configs: list, L_values, N_per_L) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = DATA_DIR / f"{label}.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
        print(f"  [{label}] loaded {len(recs)} cached")
        return recs
    rng = np.random.default_rng(RANDOM_SEED + abs(hash(label)) % 10_000)
    tasks = []
    for material, lat_name, builder, p, dim in configs:
        a_bond = A_BY_MATERIAL[material]
        for L in L_values:
            lat = builder(L, a=a_bond) if "cubic" in lat_name or lat_name in (
                "honeycomb", "square", "triangular") else builder(L)
            n = N_per_L[L]
            clusters = sample_many(lat, p, n,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=4)
            for ic, cl in enumerate(clusters):
                tasks.append((material, lat_name, dim, L, ic, cl.positions,
                              cl.seed, cl.n_atoms, cl.n_bonds, MPI_PROCS))
    print(f"  [{label}] queued {len(tasks)} tasks")
    records: list[dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL_REPLICAS) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for f in as_completed(futures):
            t = futures[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f"  [exc] {label} {t[0]}/{t[1]} L={t[3]} ic={t[4]}: {e}")
                continue
            if rec.get("_skip"):
                print(f"  [skip] {label} {rec['mat']}/{rec['lat']} L={t[3]} "
                      f"ic={rec['ic']} N={rec['n']}")
                continue
            tt = (rec["wall_sp_s"] + rec["wall_opt_s"])
            print(f"  [{label}/{rec['material']}/{rec['lattice']}] "
                  f"L={rec['L']:3d} rep={rec['realisation']:2d} "
                  f"N={rec['n_atoms']:5d}  "
                  f"E/N={rec['E_relaxed_eV']/rec['n_atoms']:+5.3f}  "
                  f"V_mc={rec['V_mc_A3']:8.0f}  "
                  f"E/V={rec['E_relaxed_eV']/rec['V_mc_A3']:+.4f}  "
                  f"({tt:.1f}s)", flush=True)
            records.append(rec)
            cache.write_text(json.dumps(sorted(
                records, key=lambda r: (r["material"], r["lattice"],
                                          r["L"], r["realisation"])
            ), indent=2))
    print(f"  [{label}] {len(records)} clusters in "
          f"{time.perf_counter() - t0:.1f}s\n")
    cache.write_text(json.dumps(sorted(
        records, key=lambda r: (r["material"], r["lattice"],
                                  r["L"], r["realisation"])
    ), indent=2))
    return records


def per_L_means(records, key) -> tuple[np.ndarray, np.ndarray]:
    Ls = sorted({r["L"] for r in records})
    mean = np.array([np.mean([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    return np.array(Ls, dtype=float), mean


def analyse(records: list[dict]) -> list[dict]:
    out = []
    keys = sorted(set((r["material"], r["lattice"]) for r in records))
    for mat, lat in keys:
        sub = [r for r in records if r["material"] == mat
                and r["lattice"] == lat]
        arrL, n_mean = per_L_means(sub, "n_atoms")
        _, e_mean = per_L_means(sub, "E_relaxed_eV")
        _, v_mc_mean = per_L_means(sub, "V_mc_A3")
        eov_mc = e_mean / v_mc_mean
        epn = e_mean / n_mean
        fit_N = fit_loglog(arrL, n_mean)
        fit_E = fit_loglog(arrL, -e_mean)
        fit_V = fit_loglog(arrL, v_mc_mean)
        fit_eov = fit_loglog(arrL, -eov_mc)
        out.append(dict(
            material=mat, lattice=lat, dim=sub[0]["dim"],
            L=[int(L) for L in arrL],
            N=[float(x) for x in n_mean],
            E=[float(x) for x in e_mean],
            V_mc=[float(x) for x in v_mc_mean],
            eov_mc=[float(x) for x in eov_mc],
            epn=[float(x) for x in epn],
            D_f=float(fit_N.exponent),
            D_E=float(fit_E.exponent),
            D_V=float(fit_V.exponent),
            gamma=float(fit_eov.exponent),
            eps_V_star_at_Lmax=float(-eov_mc[-1]),
            eps_per_atom_at_Lmax=float(-epn[-1]),
            n_clusters=len(sub),
        ))
    return out


def main():
    print("=== Phase 13 — Ge Tersoff + 3D cubic percolation ===\n")

    print("Part A: Ge in 3 2D lattices")
    recs_A = gather_one_set("part_A_Ge_2D", PART_A, L_VALUES_2D, N_PER_L_2D)

    print("\nPart B: C, Si, Ge in 3D cubic")
    recs_B = gather_one_set("part_B_3D_cubic", PART_B, L_VALUES_3D, N_PER_L_3D)

    all_recs = recs_A + recs_B
    summary = analyse(all_recs)

    print(f"\n{'='*100}")
    print(f"            PHASE 13 MASTER TABLE — Ge 2D + 3D cubic universality")
    print(f"{'='*100}")
    print(f"  {'mat':3s} {'lattice':10s} {'dim':3s}  {'D_f':6s}  {'D_E':6s}  "
          f"{'D_V':6s}  {'γ':6s}  {'ε_V*':8s}  {'−E/N':8s}  n")
    for s in summary:
        print(f"   {s['material']:2s}  {s['lattice']:10s} {s['dim']:3s}  "
              f"{s['D_f']:+.3f} {s['D_E']:+.3f} {s['D_V']:+.3f}  "
              f"{s['gamma']:+.3f}  {s['eps_V_star_at_Lmax']:.4f}  "
              f"{s['eps_per_atom_at_Lmax']:5.3f}    {s['n_clusters']}")

    # Compare with phase 5/6 (existing C/Si 2D)
    site_C = ROOT / "results" / "phase5_three_lattices" / "summary.json"
    site_Si = ROOT / "results" / "phase6_silicon" / "summary.json"
    if site_C.exists() and site_Si.exists():
        c_summ = json.loads(site_C.read_text())
        si_summ = json.loads(site_Si.read_text())
        print(f"\n{'='*100}")
        print(f"            CROSS-MATERIAL CROSS-DIMENSION ε_V* COMPARISON")
        print(f"{'='*100}")
        print(f"  {'lattice':12s} {'dim':3s}    {'C':10s} {'Si':10s} {'Ge':10s}    "
              f"{'ratio C/Si':10s} {'ratio C/Ge':10s} {'ratio Si/Ge':10s}")
        # 2D lattices
        for lat in ("honeycomb", "square", "triangular"):
            c_v = c_summ.get(lat, {}).get("eps_V_star_mc")
            si_v = si_summ.get(lat, {}).get("eps_V_star_mc")
            ge_v = next((s["eps_V_star_at_Lmax"] for s in summary
                          if s["material"] == "Ge" and s["lattice"] == lat),
                         None)
            if c_v is None or si_v is None or ge_v is None:
                continue
            print(f"  {lat:12s} 2D     "
                  f"{c_v:.4f}     {si_v:.4f}     {ge_v:.4f}      "
                  f"{c_v/si_v:.2f}        {c_v/ge_v:.2f}        {si_v/ge_v:.2f}")
        # 3D cubic
        c_3d = next((s["eps_V_star_at_Lmax"] for s in summary
                     if s["material"] == "C" and s["lattice"] == "cubic_3d"), None)
        si_3d = next((s["eps_V_star_at_Lmax"] for s in summary
                      if s["material"] == "Si" and s["lattice"] == "cubic_3d"), None)
        ge_3d = next((s["eps_V_star_at_Lmax"] for s in summary
                      if s["material"] == "Ge" and s["lattice"] == "cubic_3d"), None)
        if all(x is not None for x in (c_3d, si_3d, ge_3d)):
            print(f"  cubic_3d      3D     "
                  f"{c_3d:.4f}     {si_3d:.4f}     {ge_3d:.4f}      "
                  f"{c_3d/si_3d:.2f}        {c_3d/ge_3d:.2f}        "
                  f"{si_3d/ge_3d:.2f}")

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"C": "C0", "Si": "C1", "Ge": "C2"}
    markers = {"honeycomb": "v", "square": "s", "triangular": "^",
               "cubic_3d": "o"}
    for s in summary:
        c = colors[s["material"]]
        m = markers[s["lattice"]]
        arrL = np.array(s["L"], dtype=float)
        ax = axes[0] if s["dim"] == "2D" else axes[1]
        ax.plot(arrL, [-x for x in s["eov_mc"]], "-",
                marker=m, color=c,
                label=f"{s['material']}/{s['lattice']}  γ={s['gamma']:+.3f}")
    for ax, title in zip(axes, ("2D lattices", "3D cubic")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("L")
        ax.set_ylabel(r"$-\langle E\rangle/\langle V_{mc}\rangle$ (eV/Å³)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_ge_and_3d.png", dpi=180)
    plt.close(fig)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_ge_and_3d.png'}")


if __name__ == "__main__":
    main()
