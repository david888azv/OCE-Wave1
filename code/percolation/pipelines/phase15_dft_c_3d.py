"""Phase 15 — DFT-PBE validation for C in 4 3D lattices.

Mirrors phase 12 (2D) but for 3D: SC, BCC, FCC, diamond.  C only —
this phase is the carbon-only DFT closure that supports the PRL claim
that ε_V*(C) is universal across topology and dimension.

Setup: AIREBO relaxation + SIESTA single-point on the relaxed positions.
L=8, 2 reps per lattice → 8 clusters total.  Clusters at L=8 in 3D have
~30-100 atoms each, ≤ 1 min/cluster with SZP basis.
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
from ase import Atoms

from lattices import (build_cubic_3d, build_bcc_3d, build_fcc_3d,
                       build_diamond_3d, P_C)
from percolation import sample_many
from runners_lammps import lammps_airebo
from runners_siesta import siesta_energy, siesta_isolated_atom_energy
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw, R_VDW_C)


L_TARGET = 8
N_PER_LATTICE = 2
A_C = 1.42
RANDOM_SEED = 20260518
MC_SAMPLES = 300_000
MPI_PROCS_LMP = 8
MPI_PROCS_SIESTA = 4

LATTICES_3D = {
    "cubic_3d":   (build_cubic_3d,   P_C["cubic_3d"]),
    "bcc_3d":     (build_bcc_3d,     P_C["bcc_3d"]),
    "fcc_3d":     (build_fcc_3d,     P_C["fcc_3d"]),
    "diamond_3d": (build_diamond_3d, P_C["diamond_3d"]),
}

DATA_DIR = ROOT / "data" / "phase15_dft_c_3d"
RESULTS_DIR = ROOT / "results" / "phase15_dft_c_3d"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "dft_3d.json"


def main():
    if CACHE.exists():
        records = json.loads(CACHE.read_text())
    else:
        records = []
    if not records:
        print("=== Phase 15 — DFT C-3D validation, L=8 ===")
        e_atom_C = siesta_isolated_atom_energy("C", mpi_procs=4)
        print(f"  E_atom(C) = {e_atom_C:.4f} eV\n")
        rng = np.random.default_rng(RANDOM_SEED)
        for lat_name, (builder, p) in LATTICES_3D.items():
            lat = builder(L_TARGET, a=A_C)
            clusters = sample_many(lat, p, N_PER_LATTICE,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=8)
            for ic, cl in enumerate(clusters):
                atoms = Atoms(symbols=["C"] * cl.n_atoms,
                              positions=cl.positions)
                t0 = time.perf_counter()
                opt = lammps_airebo(atoms, optimize=True, timeout=1800,
                                     mpi_procs=MPI_PROCS_LMP)
                t_lmp = time.perf_counter() - t0
                if not opt["converged"]:
                    print(f"  [skip] {lat_name}/{ic} (LAMMPS failed)")
                    continue
                pos_rel = opt["opt_atoms"].get_positions()
                relaxed = Atoms(symbols=["C"] * cl.n_atoms, positions=pos_rel)
                tag = f"3d_{lat_name}_{ic}"
                t1 = time.perf_counter()
                sie = siesta_energy(relaxed, tag=tag, timeout=3600,
                                     threads=2, mpi_procs=MPI_PROCS_SIESTA)
                t_sie = time.perf_counter() - t1
                if not sie["converged"]:
                    print(f"  [siesta-fail] {lat_name}/{ic}: "
                          f"{sie.get('error', '?')[:200]}")
                    continue
                v_mc = volume_mc_vdw(pos_rel, r=R_VDW_C,
                                      n_samples=MC_SAMPLES, seed=cl.seed)
                E_coh_dft = sie["E_eV"] - cl.n_atoms * e_atom_C
                rec = dict(
                    material="C", lattice=lat_name, dim="3D",
                    L=int(L_TARGET), realisation=int(ic),
                    seed=int(cl.seed), n_atoms=int(cl.n_atoms),
                    E_classical_eV=float(opt["E_eV"]),
                    E_siesta_total_eV=float(sie["E_eV"]),
                    E_atom_ref_eV=float(e_atom_C),
                    E_siesta_cohesive_eV=float(E_coh_dft),
                    E_per_atom_classical=float(opt["E_eV"] / cl.n_atoms),
                    E_per_atom_siesta=float(E_coh_dft / cl.n_atoms),
                    V_mc_A3=float(v_mc),
                    eps_V_classical=float(-opt["E_eV"] / v_mc),
                    eps_V_siesta=float(-E_coh_dft / v_mc),
                    wall_lammps_s=t_lmp, wall_siesta_s=t_sie,
                )
                print(f"  [{lat_name}/{ic}] N={cl.n_atoms:3d}  "
                      f"E/N_clas={rec['E_per_atom_classical']:+6.3f}  "
                      f"E/N_dft={rec['E_per_atom_siesta']:+6.3f}  "
                      f"ε_V*_clas={rec['eps_V_classical']:.4f}  "
                      f"ε_V*_dft={rec['eps_V_siesta']:.4f}  "
                      f"({t_lmp:.1f}s LMP, {t_sie:.1f}s SIE)",
                      flush=True)
                records.append(rec)
                CACHE.write_text(json.dumps(records, indent=2))

    # Combined analysis with phase 12 (2D) data
    print(f"\n{'='*100}")
    print(f"            PHASE 15 — C 3D DFT vs CLASSICAL (per cluster)")
    print(f"{'='*100}")
    print(f"  {'lat':12s} {'N':3s}  E/N_clas   E/N_dft   ratio_DFT/clas   ε_V*_clas  ε_V*_dft")
    for r in records:
        ratio = r["eps_V_siesta"] / r["eps_V_classical"]
        print(f"   {r['lattice']:12s}  {r['n_atoms']:3d}    "
              f"{r['E_per_atom_classical']:+6.3f}    "
              f"{r['E_per_atom_siesta']:+6.3f}     {ratio:.3f}        "
              f"{r['eps_V_classical']:.4f}    {r['eps_V_siesta']:.4f}")

    # Aggregate per lattice
    print(f"\n{'='*92}")
    print(f"            AGGREGATE — DFT/classical ratios per 3D lattice")
    print(f"{'='*92}")
    by_lat: dict = {}
    for r in records:
        by_lat.setdefault(r["lattice"], []).append(r)
    print(f"  {'lattice':12s} n  ⟨ε_V_class⟩  ⟨ε_V_dft⟩  ⟨ratio_DFT/clas⟩")
    for lat, recs in by_lat.items():
        ec = float(np.mean([r["eps_V_classical"] for r in recs]))
        ed = float(np.mean([r["eps_V_siesta"] for r in recs]))
        rat = float(np.mean([r["eps_V_siesta"] / r["eps_V_classical"]
                              for r in recs]))
        print(f"  {lat:12s} {len(recs)}    {ec:.4f}      {ed:.4f}    {rat:.3f}")

    # All C records (2D phase 12 + 3D phase 15)
    p12_path = ROOT / "data" / "phase12_siesta_validation" / "siesta_validation.json"
    if p12_path.exists():
        p12_recs = json.loads(p12_path.read_text())
        c_2d = [r for r in p12_recs if r["material"] == "C"]
        all_c = c_2d + records
        ratios = np.array([r["eps_V_siesta"] / r["eps_V_classical"]
                           for r in all_c])
        print(f"\n{'='*70}")
        print(f"            COMBINED C 2D + 3D — DFT/AIREBO ratio")
        print(f"{'='*70}")
        print(f"  N_clusters: {len(all_c)} (2D: {len(c_2d)}, 3D: {len(records)})")
        print(f"  ⟨ratio DFT/AIREBO⟩ = {ratios.mean():.4f} ± {ratios.std():.4f}")
        print(f"  spread (max−min)/mean = "
              f"{(ratios.max() - ratios.min()) / ratios.mean() * 100:.2f}%")

    summary = dict(
        L_TARGET=L_TARGET, records=records,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
