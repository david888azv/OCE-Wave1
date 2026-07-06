"""Phase 17 — drive DFT/AIREBO ratio spread below 1% via more realizations.

Variance decomposition (phase 12+15+16) showed σ_within = 0.0716 dominates
σ_between = 0.0313; the 2.4% cross-lattice spread is statistically consistent
with a single common ratio.  Need n=30 per lattice → ratio SEM ~1%.

Adds 24 more clusters per lattice (offset ic=300..323) to bring each lattice
to n=29-31.  Runs LAMMPS+SIESTA in parallel (6 workers × 1-2 OMP threads).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from runners_siesta import siesta_isolated_atom_energy
from volume import volume_mc_vdw, R_VDW_C


LATTICES_2D = {
    "honeycomb":  (build_honeycomb,  P_C["honeycomb"], 12),
    "square":     (build_square,      P_C["square"], 12),
    "triangular": (build_triangular, P_C["triangular"], 12),
}
LATTICES_3D = {
    "cubic_3d":   (build_cubic_3d,   P_C["cubic_3d"], 8),
    "bcc_3d":     (build_bcc_3d,     P_C["bcc_3d"], 8),
    "fcc_3d":     (build_fcc_3d,     P_C["fcc_3d"], 8),
    "diamond_3d": (build_diamond_3d, P_C["diamond_3d"], 8),
}
ALL_LATTICES = {**LATTICES_2D, **LATTICES_3D}

N_NEW = 24                       # 24 per lattice × 7 lattices = 168 new clusters
A_C = 1.42
RANDOM_SEED = 20260530
MC_SAMPLES = 300_000
MPI_PROCS_LMP = 1
SIESTA_THREADS = 1               # 6 parallel × 1 thread = 6 cores total
N_WORKERS = 6

DATA_DIR = ROOT / "data" / "phase17_dft_more_reps"
RESULTS_DIR = ROOT / "results" / "phase17_dft_more_reps"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "more_reps_szp.json"


def _process_one(args):
    """Worker: build cluster, AIREBO opt, SIESTA SZP, MC volume."""
    lat_name, ic, e_atom_C = args
    builder, p, L = ALL_LATTICES[lat_name]
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
    tag = f"szp_{lat_name}_{ic}"

    from runners_siesta import siesta_energy
    sie = siesta_energy(relaxed, tag=tag, timeout=7200,
                         threads=SIESTA_THREADS, mpi_procs=1)
    if not sie["converged"]:
        return None

    v_mc = volume_mc_vdw(pos_rel, r=R_VDW_C, n_samples=MC_SAMPLES, seed=cl.seed)
    E_coh_dft = sie["E_eV"] - cl.n_atoms * e_atom_C
    return dict(
        lattice=lat_name, basis="SZP",
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


def main():
    print("=== Phase 17: drive DFT/AIREBO spread to <1% ===")
    e_atom_C = siesta_isolated_atom_energy("C", mpi_procs=1)
    print(f"  E_atom(C) = {e_atom_C:.4f} eV")

    # Resume support
    records = []
    if CACHE.exists():
        records = json.loads(CACHE.read_text())
        done_keys = {(r["lattice"], r["realisation"]) for r in records}
    else:
        done_keys = set()

    queue = []
    for lat in ALL_LATTICES:
        for ic in range(300, 300 + N_NEW):
            if (lat, ic) not in done_keys:
                queue.append((lat, ic, e_atom_C))
    print(f"Queue: {len(queue)} (already done: {len(done_keys)})")
    print(f"Workers: {N_WORKERS} × {SIESTA_THREADS} OMP, "
          f"LAMMPS MPI {MPI_PROCS_LMP} per worker")

    t0 = time.time()
    n_ok = 0; n_fail = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_process_one, q): q for q in queue}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = None
                print(f"  EXCEPTION on {q[0]}/{q[1]}: {type(e).__name__}: {e}")
            if rec is None:
                n_fail += 1
                continue
            n_ok += 1
            records.append(rec)
            CACHE.write_text(json.dumps(records, indent=2))
            ratio = rec["eps_V_siesta"] / rec["eps_V_classical"]
            elapsed = time.time() - t0
            avg = elapsed / max(1, n_ok + n_fail)
            remaining = (len(queue) - n_ok - n_fail) * avg
            print(f"  [{n_ok+n_fail}/{len(queue)}] "
                  f"{rec['lattice']:>10s}/{rec['realisation']:<4d}  "
                  f"N={rec['n_atoms']:>3d}  ratio={ratio:.4f}  "
                  f"wall={rec['wall_lammps_s']+rec['wall_siesta_s']:.0f}s  "
                  f"elapsed={elapsed/60:.1f}min  ETA={remaining/60:.1f}min",
                  flush=True)
    print(f"\nPhase 17 done: {n_ok} OK / {n_fail} FAIL")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")

    # Final summary
    p12 = ROOT / "data" / "phase12_siesta_validation" / "siesta_validation.json"
    p15 = ROOT / "data" / "phase15_dft_c_3d" / "dft_3d.json"
    p16 = ROOT / "data" / "phase16_dft_strengthen" / "more_reps_szp.json"

    all_szp = []
    for path in (p12, p15, p16):
        if path.exists():
            for r in json.loads(path.read_text()):
                if r.get("material", "C") == "C" and r.get("basis", "SZP") == "SZP":
                    all_szp.append(r)
    all_szp.extend(records)

    by_lat = {}
    for r in all_szp:
        by_lat.setdefault(r["lattice"], []).append(r)
    means = []
    print(f"\n=== COMBINED summary (phases 12+15+16+17) — SZP, all C ===")
    print(f"  N_clusters: {len(all_szp)}")
    print(f"  {'lattice':>12s}  {'n':>3s}  {'⟨ratio⟩':>8s}  {'std':>7s}  {'sem':>7s}")
    for lat in sorted(by_lat):
        arr = np.array([r["eps_V_siesta"]/r["eps_V_classical"] for r in by_lat[lat]])
        sem = arr.std(ddof=1)/np.sqrt(len(arr)) if len(arr)>1 else float('nan')
        print(f"  {lat:>12s}  {len(arr):>3d}  {arr.mean():>8.4f}  "
              f"{arr.std(ddof=1) if len(arr)>1 else 0:>7.4f}  {sem:>7.4f}")
        means.append(arr.mean())
    print(f"\n  spread (between-lattice means) = "
          f"{np.std(means, ddof=1)/np.mean(means)*100:.2f}%")

    summary = dict(
        n_total_szp_C=len(all_szp),
        per_lattice={lat: dict(
            n=len(by_lat[lat]),
            ratio_mean=float(np.mean([r["eps_V_siesta"]/r["eps_V_classical"]
                                       for r in by_lat[lat]])),
            ratio_std=float(np.std([r["eps_V_siesta"]/r["eps_V_classical"]
                                     for r in by_lat[lat]], ddof=1)
                            if len(by_lat[lat]) > 1 else 0.0),
            ratio_sem=float(np.std([r["eps_V_siesta"]/r["eps_V_classical"]
                                     for r in by_lat[lat]], ddof=1)
                            / np.sqrt(len(by_lat[lat])))
                            if len(by_lat[lat]) > 1 else float("nan"),
        ) for lat in sorted(by_lat)},
        between_lattice_spread_pct=float(np.std(means, ddof=1)/np.mean(means)*100),
        between_lattice_mean_ratio=float(np.mean(means)),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary  → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
