"""Phase 17 (parametric) — DFT spread campaign for C / Si / Ge.

Same protocol as phase17_dft_more_reps.py but accepts --material argument.
Resumable from disk cache (per-material).

Usage:
  python pipelines/phase17_xx_dft.py --material Si
  python pipelines/phase17_xx_dft.py --material Ge --workers 6
  python pipelines/phase17_xx_dft.py --material C    # rerun (already done)

For Si we use Tersoff via lammps_si_tersoff for the relaxation engine.
For Ge we use Tersoff via lammps_ge_tersoff.
For C we use AIREBO (lammps_airebo) — same as before.
"""
from __future__ import annotations

import argparse
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
from runners_lammps import lammps_airebo, lammps_si_tersoff, lammps_ge_tersoff
from runners_siesta import siesta_isolated_atom_energy
from volume import volume_mc_vdw, R_VDW_C


# --- Per-material configuration ---
ENGINE = {
    "C":  lammps_airebo,
    "Si": lammps_si_tersoff,
    "Ge": lammps_ge_tersoff,
}
A_LAT = {"C": 1.42, "Si": 2.35, "Ge": 2.45}  # nominal nearest-neighbor distance
R_VDW = {"C": R_VDW_C, "Si": 2.10, "Ge": 2.11}  # Bondi vdW radii (Å)

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

N_NEW = 24
RANDOM_SEED = 20260530
MC_SAMPLES = 300_000
SIESTA_THREADS = 1
N_WORKERS_DEFAULT = 6


def _process_one(args):
    mat, lat_name, ic, e_atom, a_lat, r_vdw = args
    builder, p, L = ALL_LATTICES[lat_name]
    rng = np.random.default_rng(RANDOM_SEED + ic * 7 + abs(hash(lat_name)) % 100
                                  + abs(hash(mat)) % 100)
    lat = builder(L, a=a_lat)
    clusters = sample_many(lat, p, 1,
                            base_seed=int(rng.integers(2**31)),
                            min_size=8)
    if not clusters:
        return dict(error="empty cluster", mat=mat, lat=lat_name, ic=ic)
    cl = clusters[0]
    atoms = Atoms(symbols=[mat] * cl.n_atoms, positions=cl.positions)

    engine = ENGINE[mat]
    opt = engine(atoms, optimize=True, timeout=1800, mpi_procs=1)
    if not opt["converged"]:
        return dict(error=f"engine_{mat} failed", mat=mat, lat=lat_name, ic=ic)
    pos_rel = opt["opt_atoms"].get_positions()
    relaxed = Atoms(symbols=[mat] * cl.n_atoms, positions=pos_rel)
    tag = f"szp_{mat}_{lat_name}_{ic}"

    from runners_siesta import siesta_energy
    sie = siesta_energy(relaxed, tag=tag, timeout=7200,
                         threads=SIESTA_THREADS, mpi_procs=1)
    if not sie["converged"]:
        return dict(error="siesta failed", mat=mat, lat=lat_name, ic=ic,
                     err_msg=sie.get("error", "")[:300])

    v_mc = volume_mc_vdw(pos_rel, r=r_vdw, n_samples=MC_SAMPLES, seed=cl.seed)
    E_coh_dft = sie["E_eV"] - cl.n_atoms * e_atom
    return dict(
        material=mat, lattice=lat_name, basis="SZP",
        L=int(L), realisation=int(ic),
        seed=int(cl.seed), n_atoms=int(cl.n_atoms),
        E_classical_eV=float(opt["E_eV"]),
        E_siesta_total_eV=float(sie["E_eV"]),
        E_atom_ref_eV=float(e_atom),
        E_siesta_cohesive_eV=float(E_coh_dft),
        E_per_atom_classical=float(opt["E_eV"] / cl.n_atoms),
        E_per_atom_siesta=float(E_coh_dft / cl.n_atoms),
        V_mc_A3=float(v_mc),
        eps_V_classical=float(-opt["E_eV"] / v_mc),
        eps_V_siesta=float(-E_coh_dft / v_mc),
        wall_lammps_s=float(opt["wall_time_s"]),
        wall_siesta_s=float(sie["wall_time_s"]),
    )


def main(material: str, n_workers: int):
    print(f"=== Phase 17 — material={material} ===", flush=True)
    if material not in ENGINE:
        raise ValueError(f"Unknown material: {material}; supported: {list(ENGINE)}")

    # Atom reference (cached)
    e_atom = siesta_isolated_atom_energy(material, threads=2, mpi_procs=1)
    print(f"  E_atom({material}) = {e_atom:.4f} eV", flush=True)

    a_lat = A_LAT[material]; r_vdw = R_VDW[material]
    DATA_DIR = ROOT / "data" / f"phase17_{material}_dft_more_reps"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE = DATA_DIR / "more_reps_szp.json"

    records = json.loads(CACHE.read_text()) if CACHE.exists() else []
    done_keys = {(r.get("lattice"), r.get("realisation")) for r in records
                 if "lattice" in r}
    queue = []
    for lat in ALL_LATTICES:
        for ic in range(300, 300 + N_NEW):
            if (lat, ic) not in done_keys:
                queue.append((material, lat, ic, e_atom, a_lat, r_vdw))
    print(f"  Already done: {len(done_keys)}  /  To run: {len(queue)}",
          flush=True)
    print(f"  Workers: {n_workers} × {SIESTA_THREADS} OMP", flush=True)

    if not queue:
        print("  Nothing to do — all 168 already cached", flush=True)
        return

    # Progress reporting cadence: every 10 minutes wall, dump rolling stats
    REPORT_EVERY_S = 600
    last_report = time.time()
    t0 = time.time()
    n_ok = 0; n_fail = 0
    fail_reasons = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_one, q): q for q in queue}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"error": f"exception: {type(e).__name__}: {e}",
                       "mat": q[0], "lat": q[1], "ic": q[2]}
            if rec.get("error"):
                n_fail += 1
                fail_reasons.append((rec.get("lat"), rec.get("ic"),
                                      rec["error"][:100]))
                continue
            n_ok += 1
            records.append(rec)
            CACHE.write_text(json.dumps(records, indent=2))

            elapsed = time.time() - t0
            avg = elapsed / max(1, n_ok + n_fail)
            remaining = (len(queue) - n_ok - n_fail) * avg
            ratio = rec["eps_V_siesta"] / rec["eps_V_classical"]
            print(f"  [{n_ok+n_fail}/{len(queue)}]  "
                  f"{rec['lattice']:>10s}/{rec['realisation']:<4d}  "
                  f"N={rec['n_atoms']:>3d}  ratio={ratio:.4f}  "
                  f"wall={rec['wall_lammps_s']+rec['wall_siesta_s']:.0f}s  "
                  f"OK/FAIL={n_ok}/{n_fail}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={remaining/60:.1f}min",
                  flush=True)

            now = time.time()
            if now - last_report > REPORT_EVERY_S:
                last_report = now
                # Per-lattice progress snapshot
                from collections import Counter
                counts = Counter(r["lattice"] for r in records)
                print(f"  *** 10-min checkpoint  ***  "
                      f"OK={n_ok} FAIL={n_fail}  "
                      f"per-lattice: {dict(counts)}",
                      flush=True)

    # Final
    print(f"\n=== {material} phase 17 done ===", flush=True)
    print(f"  OK: {n_ok}  FAIL: {n_fail}  Wall: {(time.time()-t0)/60:.1f} min",
          flush=True)
    if fail_reasons:
        print("  Failure summary (first 10):")
        for lat, ic, msg in fail_reasons[:10]:
            print(f"    {lat}/{ic}: {msg}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True, choices=["C", "Si", "Ge"])
    ap.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    args = ap.parse_args()
    main(args.material, args.workers)
