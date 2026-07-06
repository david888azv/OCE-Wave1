"""Phase 12 — DFT (SIESTA / PBE / DZP) spot-check at L=12.

We validate the classical-potential ε_V* values (AIREBO for C, Tersoff for Si)
against DFT-PBE energies on the AIREBO/Tersoff-RELAXED geometry.  Comparing
ε_V*(DFT) vs ε_V*(classical) per cluster:

  * if the classical and DFT values agree on ε_V*, the law is independent
    of the level of theory — strong support for it being a real physical
    statement, not an artefact of the empirical force field;

  * if they disagree, we learn the systematic shift of AIREBO/Tersoff
    relative to PBE for percolation clusters of carbon and silicon.

Configuration:
  L = 12, 6 (material, lattice) combos, ~2 realisations each ≈ 12 SIESTA runs.
  Single point only (no DFT geometry optimisation).
  AIREBO/Tersoff geometry → DFT energy.
  V_mc remains the same (geometry-only).

Why L=12?  honeycomb ⟨N⟩≈109, square ⟨N⟩≈53, triangular ⟨N⟩≈68 — all
manageable in 5-10 min per SIESTA single-point with mpirun=4.
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

from lattices import build_honeycomb, build_square, build_triangular, P_C
from percolation import sample_many
from runners_lammps import lammps_airebo, lammps_si_tersoff
from runners_siesta import siesta_energy, siesta_isolated_atom_energy
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw, R_VDW_C, R_VDW_SI)


L_TARGET = 12
N_PER_LM = 2          # realisations per (material, lattice)
A_C, A_SI = 1.42, 2.35
RANDOM_SEED = 20260515
MC_SAMPLES = 300_000
MPI_PROCS_LMP = 8
MPI_PROCS_SIESTA = 4

LATTICES_C = {
    "honeycomb":  ((lambda L: build_honeycomb(L, a=A_C)),  P_C["honeycomb"]),
    "square":     ((lambda L: build_square(L, a=A_C)),      P_C["square"]),
    "triangular": ((lambda L: build_triangular(L, a=A_C)), P_C["triangular"]),
}
LATTICES_SI = {
    "honeycomb":  ((lambda L: build_honeycomb(L, a=A_SI)),  P_C["honeycomb"]),
    "square":     ((lambda L: build_square(L, a=A_SI)),      P_C["square"]),
    "triangular": ((lambda L: build_triangular(L, a=A_SI)), P_C["triangular"]),
}

DATA_DIR = ROOT / "data" / "phase12_siesta_validation"
RESULTS_DIR = ROOT / "results" / "phase12_siesta_validation"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "siesta_validation.json"


def gather():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    print("Computing SIESTA atom references...", flush=True)
    e_atom_C = siesta_isolated_atom_energy("C", mpi_procs=4)
    e_atom_Si = siesta_isolated_atom_energy("Si", mpi_procs=4)
    print(f"  E_atom(C)  = {e_atom_C:.4f} eV")
    print(f"  E_atom(Si) = {e_atom_Si:.4f} eV\n")
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []
    for material, lattices in (("C", LATTICES_C), ("Si", LATTICES_SI)):
        runner = lammps_airebo if material == "C" else lammps_si_tersoff
        rvdw = R_VDW_C if material == "C" else R_VDW_SI
        e_atom_ref = e_atom_C if material == "C" else e_atom_Si
        for lattice_name, (builder, p) in lattices.items():
            lat = builder(L_TARGET)
            clusters = sample_many(lat, p, N_PER_LM,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=8)
            for ic, cl in enumerate(clusters):
                atoms = Atoms(symbols=[material] * cl.n_atoms,
                              positions=cl.positions)
                # AIREBO/Tersoff opt
                t0 = time.perf_counter()
                opt = runner(atoms, optimize=True, timeout=1800,
                              mpi_procs=MPI_PROCS_LMP)
                t_lmp = time.perf_counter() - t0
                if not opt["converged"]:
                    print(f"  [skip] {material}/{lattice_name} ic={ic} (LAMMPS failed)")
                    continue
                pos_rel = opt["opt_atoms"].get_positions()
                relaxed_atoms = Atoms(symbols=[material] * cl.n_atoms,
                                       positions=pos_rel)

                # SIESTA single-point on the relaxed positions
                tag = f"{material}_{lattice_name}_{ic}"
                t1 = time.perf_counter()
                siesta_res = siesta_energy(relaxed_atoms, tag=tag,
                                            timeout=3600,
                                            threads=2,
                                            mpi_procs=MPI_PROCS_SIESTA)
                t_siesta = time.perf_counter() - t1
                if not siesta_res["converged"]:
                    print(f"  [siesta-fail] {material}/{lattice_name}/{ic}: "
                          f"{siesta_res.get('error', '?')[:200]}")
                    continue

                v_mc = volume_mc_vdw(pos_rel, r=rvdw,
                                      n_samples=MC_SAMPLES, seed=cl.seed)
                v_at = volume_atomic(cl.n_atoms, r=rvdw)
                # Cohesive energy: E_cluster - N · E_atom
                E_coh_dft = siesta_res["E_eV"] - cl.n_atoms * e_atom_ref
                # AIREBO/Tersoff already report cohesive (vacuum atom = 0)
                E_coh_classical = opt["E_eV"]
                rec = dict(
                    material=material, lattice=lattice_name,
                    realisation=int(ic), seed=int(cl.seed),
                    n_atoms=int(cl.n_atoms), n_bonds=int(cl.n_bonds),
                    L=int(L_TARGET),
                    E_classical_eV=float(E_coh_classical),
                    E_siesta_total_eV=float(siesta_res["E_eV"]),
                    E_atom_ref_eV=float(e_atom_ref),
                    E_siesta_cohesive_eV=float(E_coh_dft),
                    E_per_atom_classical=float(E_coh_classical / cl.n_atoms),
                    E_per_atom_siesta=float(E_coh_dft / cl.n_atoms),
                    V_mc_A3=float(v_mc),
                    V_atomic_A3=float(v_at),
                    eps_V_classical=float(-E_coh_classical / v_mc),
                    eps_V_siesta=float(-E_coh_dft / v_mc),
                    wall_lammps_s=float(t_lmp),
                    wall_siesta_s=float(t_siesta),
                    positions_relaxed=pos_rel.tolist(),
                )
                print(f"  [{material}/{lattice_name}/{ic}] N={cl.n_atoms:3d}  "
                      f"E_coh_clas={E_coh_classical:+9.2f}  "
                      f"E_coh_dft={E_coh_dft:+9.2f}  "
                      f"E/N_clas={rec['E_per_atom_classical']:+6.3f}  "
                      f"E/N_dft={rec['E_per_atom_siesta']:+6.3f}  "
                      f"ε_V*_clas={rec['eps_V_classical']:.4f}  "
                      f"ε_V*_dft={rec['eps_V_siesta']:.4f}  "
                      f"(LMP={t_lmp:.1f}s, SIE={t_siesta:.1f}s)",
                      flush=True)
                records.append(rec)
                CACHE.write_text(json.dumps(records, indent=2))
    return records


def main():
    print(f"=== Phase 12 — SIESTA DFT-PBE validation at L={L_TARGET} ===\n")
    records = gather()
    if not records:
        print("No converged records.")
        return

    print(f"\n{'='*100}")
    print(f"            DFT vs CLASSICAL — per cluster")
    print(f"{'='*100}")
    print(f"  {'mat':3s} {'lat':10s} {'N':4s}   "
          f"{'E_class (eV)':14s} {'E_dft (eV)':12s}   "
          f"{'E/N_class':10s} {'E/N_dft':10s} ΔE/N")
    for r in records:
        delta = r["E_per_atom_siesta"] - r["E_per_atom_classical"]
        print(f"   {r['material']:2s}  {r['lattice']:10s} {r['n_atoms']:3d}    "
              f"{r['E_classical_eV']:+10.2f}    "
              f"{r['E_siesta_cohesive_eV']:+10.2f}      "
              f"{r['E_per_atom_classical']:+8.3f}   "
              f"{r['E_per_atom_siesta']:+8.3f}   {delta:+.3f}")

    print(f"\n{'='*92}")
    print(f"            ε_V*(DFT) vs ε_V*(classical) — per cluster")
    print(f"{'='*92}")
    print(f"  {'mat':3s} {'lat':10s} {'N':4s}   "
          f"{'V_mc (Å³)':10s} {'ε_V_class':10s} {'ε_V_dft':10s}  ratio_dft/class")
    for r in records:
        ratio = r["eps_V_siesta"] / r["eps_V_classical"]
        print(f"   {r['material']:2s}  {r['lattice']:10s} {r['n_atoms']:3d}    "
              f"{r['V_mc_A3']:8.0f}     "
              f"{r['eps_V_classical']:.4f}     {r['eps_V_siesta']:.4f}     "
              f"{ratio:.3f}")

    # Aggregate per (material, lattice)
    print(f"\n{'='*92}")
    print(f"            AGGREGATE — DFT/classical ratios per (material, lattice)")
    print(f"{'='*92}")
    by_ml: dict = {}
    for r in records:
        key = (r["material"], r["lattice"])
        by_ml.setdefault(key, []).append(r)
    print(f"  {'mat':3s} {'lat':10s} n  "
          f"⟨ε_V_class⟩  ⟨ε_V_dft⟩  ⟨ratio⟩    ⟨E/N_class⟩  ⟨E/N_dft⟩")
    for (mat, lat), recs in by_ml.items():
        ec = float(np.mean([r["eps_V_classical"] for r in recs]))
        ed = float(np.mean([r["eps_V_siesta"] for r in recs]))
        rat = float(np.mean([r["eps_V_siesta"] / r["eps_V_classical"]
                              for r in recs]))
        epc = float(np.mean([r["E_per_atom_classical"] for r in recs]))
        epd = float(np.mean([r["E_per_atom_siesta"] for r in recs]))
        print(f"   {mat:2s}  {lat:10s} {len(recs)}    "
              f"{ec:.4f}      {ed:.4f}    {rat:.3f}      "
              f"{epc:+.3f}      {epd:+.3f}")

    # Overall
    arr = np.array([r["eps_V_siesta"] / r["eps_V_classical"] for r in records])
    print(f"\n  --- Overall ---")
    print(f"  ⟨ε_V_dft / ε_V_classical⟩  =  {arr.mean():.3f} ± {arr.std():.3f}")
    print(f"  N_clusters validated: {len(records)}")

    summary = dict(
        L_TARGET=L_TARGET, N_PER_LM=N_PER_LM,
        records=records,
        overall_dft_classical_ratio_mean=float(arr.mean()),
        overall_dft_classical_ratio_std=float(arr.std()),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
