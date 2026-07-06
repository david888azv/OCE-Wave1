"""Phase 16 — DFT strengthening for PRL submission.

(A) Add 3 more SZP reps per lattice for C (2D × 3 + 3D × 4 = 7 lattices)
     → final n=5 reps per lattice, 35 clusters total
(B) DZP spot-check on 4 representative clusters (1 per dim × 2 lattices)
     → confirms SZP→DZP basis convergence

Both run with the same AIREBO-relaxed protocol as phases 12 + 15.
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

from lattices import (build_honeycomb, build_square, build_triangular,
                       build_cubic_3d, build_bcc_3d, build_fcc_3d,
                       build_diamond_3d, P_C)
from percolation import sample_many
from runners_lammps import lammps_airebo
from runners_siesta import siesta_energy, siesta_isolated_atom_energy
from volume import volume_mc_vdw, R_VDW_C


# Part A: more reps per lattice
LATTICES_2D_A = {
    "honeycomb":  (build_honeycomb,  P_C["honeycomb"], 12),
    "square":     (build_square,      P_C["square"], 12),
    "triangular": (build_triangular, P_C["triangular"], 12),
}
LATTICES_3D_A = {
    "cubic_3d":   (build_cubic_3d,   P_C["cubic_3d"], 8),
    "bcc_3d":     (build_bcc_3d,     P_C["bcc_3d"], 8),
    "fcc_3d":     (build_fcc_3d,     P_C["fcc_3d"], 8),
    "diamond_3d": (build_diamond_3d, P_C["diamond_3d"], 8),
}
N_NEW_REPS_A = 3   # 2 already done in phase 12/15 → 5 total per lattice
A_C = 1.42
RANDOM_SEED = 20260520
MC_SAMPLES = 300_000
MPI_PROCS_LMP = 8
MPI_PROCS_SIESTA = 4

# Part B: DZP spot-check on representative clusters
DZP_LATTICES = {
    "honeycomb":  (build_honeycomb,  P_C["honeycomb"], 12, "2D"),
    "triangular": (build_triangular, P_C["triangular"], 12, "2D"),
    "cubic_3d":   (build_cubic_3d,   P_C["cubic_3d"], 8, "3D"),
    "diamond_3d": (build_diamond_3d, P_C["diamond_3d"], 8, "3D"),
}

DATA_DIR = ROOT / "data" / "phase16_dft_strengthen"
RESULTS_DIR = ROOT / "results" / "phase16_dft_strengthen"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_A = DATA_DIR / "more_reps_szp.json"
CACHE_B = DATA_DIR / "dzp_spot_check.json"


def run_one_dft(lat_name: str, builder, p: float, L: int,
                 ic: int, e_atom_C: float,
                 basis: str = "SZP") -> dict | None:
    """Sample one cluster, AIREBO opt, SIESTA single-point."""
    rng = np.random.default_rng(RANDOM_SEED + ic * 7 + abs(hash(lat_name)) % 100)
    lat = builder(L, a=A_C)
    clusters = sample_many(lat, p, 1,
                            base_seed=int(rng.integers(2**31)),
                            min_size=8)
    if not clusters:
        return None
    cl = clusters[0]
    atoms = Atoms(symbols=["C"] * cl.n_atoms, positions=cl.positions)
    opt = lammps_airebo(atoms, optimize=True, timeout=1800,
                         mpi_procs=MPI_PROCS_LMP)
    if not opt["converged"]:
        return None
    pos_rel = opt["opt_atoms"].get_positions()
    relaxed = Atoms(symbols=["C"] * cl.n_atoms, positions=pos_rel)
    tag = f"{basis}_{lat_name}_{ic}"

    # tweak SIESTA basis
    import runners_siesta as rs
    orig_basis_size = "SZP" if "PAO.BasisSize       SZP" in rs.INPUT_TEMPLATE else "DZP"
    if basis != orig_basis_size:
        rs.INPUT_TEMPLATE = rs.INPUT_TEMPLATE.replace(
            f"PAO.BasisSize       {orig_basis_size}",
            f"PAO.BasisSize       {basis}")
    sie = rs.siesta_energy(relaxed, tag=tag, timeout=7200,
                            threads=2, mpi_procs=MPI_PROCS_SIESTA)
    # restore
    if basis != orig_basis_size:
        rs.INPUT_TEMPLATE = rs.INPUT_TEMPLATE.replace(
            f"PAO.BasisSize       {basis}",
            f"PAO.BasisSize       {orig_basis_size}")
    if not sie["converged"]:
        return None

    v_mc = volume_mc_vdw(pos_rel, r=R_VDW_C, n_samples=MC_SAMPLES, seed=cl.seed)
    E_coh_dft = sie["E_eV"] - cl.n_atoms * e_atom_C
    return dict(
        lattice=lat_name, basis=basis,
        L=int(L), realisation=int(ic),
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
        wall_lammps_s=float(opt["wall_time_s"]),
        wall_siesta_s=float(sie["wall_time_s"]),
    )


def part_A_more_reps(e_atom_C: float):
    if CACHE_A.exists():
        return json.loads(CACHE_A.read_text())
    records = []
    print(f"\n=== Phase 16A: SZP additional reps (n+={N_NEW_REPS_A}) ===")
    for lat_dict in (LATTICES_2D_A, LATTICES_3D_A):
        for lat_name, (builder, p, L) in lat_dict.items():
            # Use ic offset 100..103 so we don't collide with phase 12/15 seeds
            for ic in range(100, 100 + N_NEW_REPS_A):
                t0 = time.perf_counter()
                rec = run_one_dft(lat_name, builder, p, L, ic, e_atom_C, basis="SZP")
                if rec is None:
                    print(f"  [skip] {lat_name}/{ic}")
                    continue
                records.append(rec)
                CACHE_A.write_text(json.dumps(records, indent=2))
                ratio = rec["eps_V_siesta"] / rec["eps_V_classical"]
                print(f"  [{lat_name}/{ic}] N={rec['n_atoms']:3d}  "
                      f"E/N_clas={rec['E_per_atom_classical']:+5.3f}  "
                      f"E/N_dft={rec['E_per_atom_siesta']:+5.3f}  "
                      f"ratio={ratio:.3f}  ({time.perf_counter()-t0:.0f}s)",
                      flush=True)
    CACHE_A.write_text(json.dumps(records, indent=2))
    return records


def part_B_dzp(e_atom_C: float):
    if CACHE_B.exists():
        return json.loads(CACHE_B.read_text())
    records = []
    print(f"\n=== Phase 16B: DZP spot-check (4 clusters) ===")
    for lat_name, (builder, p, L, dim) in DZP_LATTICES.items():
        ic = 200  # unique seed offset
        t0 = time.perf_counter()
        rec = run_one_dft(lat_name, builder, p, L, ic, e_atom_C, basis="DZP")
        if rec is None:
            print(f"  [skip] {lat_name}")
            continue
        rec["dim"] = dim
        records.append(rec)
        CACHE_B.write_text(json.dumps(records, indent=2))
        ratio = rec["eps_V_siesta"] / rec["eps_V_classical"]
        print(f"  [{lat_name}/{dim}] N={rec['n_atoms']:3d}  "
              f"E/N_dft(DZP)={rec['E_per_atom_siesta']:+5.3f}  "
              f"ratio={ratio:.3f}  ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    CACHE_B.write_text(json.dumps(records, indent=2))
    return records


def main():
    print("=== Phase 16: DFT strengthening ===")
    e_atom_C = siesta_isolated_atom_energy("C", mpi_procs=4)
    print(f"  E_atom(C) = {e_atom_C:.4f} eV")

    recs_A = part_A_more_reps(e_atom_C)
    recs_B = part_B_dzp(e_atom_C)

    # Aggregate with phases 12 and 15
    p12 = ROOT / "data" / "phase12_siesta_validation" / "siesta_validation.json"
    p15 = ROOT / "data" / "phase15_dft_c_3d" / "dft_3d.json"
    all_szp_C = []
    if p12.exists():
        for r in json.loads(p12.read_text()):
            if r.get("material") == "C":
                all_szp_C.append(r)
    if p15.exists():
        all_szp_C.extend(json.loads(p15.read_text()))
    all_szp_C.extend(recs_A)
    print(f"\n=== COMBINED (phases 12+15+16A) — SZP, all C clusters ===")
    print(f"  N_clusters: {len(all_szp_C)}")
    by_lat: dict = {}
    for r in all_szp_C:
        by_lat.setdefault(r["lattice"], []).append(r)
    print(f"  {'lattice':12s} n   ⟨ratio⟩    std")
    overall = []
    for lat, rs in by_lat.items():
        ratios = np.array([r["eps_V_siesta"] / r["eps_V_classical"] for r in rs])
        overall.extend(ratios.tolist())
        print(f"  {lat:12s} {len(rs):2d}   {ratios.mean():.4f}    {ratios.std():.4f}")
    overall = np.array(overall)
    print(f"  {'OVERALL':12s} {len(overall):2d}   {overall.mean():.4f}    "
          f"{overall.std():.4f}   spread={(overall.max()-overall.min())/overall.mean()*100:.1f}%")

    # DZP comparison
    if recs_B:
        print(f"\n=== Phase 16B: DZP vs SZP comparison ===")
        dzp_ratios = np.array([r["eps_V_siesta"] / r["eps_V_classical"]
                                for r in recs_B])
        print(f"  DZP   n={len(recs_B)}  ⟨ratio⟩={dzp_ratios.mean():.4f} ± {dzp_ratios.std():.4f}")
        print(f"  SZP overall n={len(overall)}  ⟨ratio⟩={overall.mean():.4f}")
        print(f"  DZP/SZP shift = {dzp_ratios.mean()/overall.mean():.3f}")

    summary = dict(
        szp_clusters_total=len(all_szp_C),
        szp_ratio_mean=float(overall.mean()),
        szp_ratio_std=float(overall.std()),
        szp_spread_pct=float((overall.max()-overall.min())/overall.mean()*100),
        dzp_clusters=len(recs_B),
        dzp_ratio_mean=float(dzp_ratios.mean()) if recs_B else None,
        dzp_ratio_std=float(dzp_ratios.std()) if recs_B else None,
        per_lattice={lat: dict(
            n=len(rs),
            ratio_mean=float(np.mean([r["eps_V_siesta"]/r["eps_V_classical"] for r in rs])),
            ratio_std=float(np.std([r["eps_V_siesta"]/r["eps_V_classical"] for r in rs])),
        ) for lat, rs in by_lat.items()},
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
