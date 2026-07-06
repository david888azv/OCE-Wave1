"""Phase 7 — focused coordination study.

Re-samples small subsets of percolation clusters (L = 32, 64) for both
materials × 3 lattices, runs LAMMPS optimisation, and stores BOTH the
relaxed positions AND the post-relaxation coordination distribution.

Goal: directly correlate post-relaxation ⟨z⟩ and the fraction of
4-coordinated atoms with E/N, to test the user's valence-vs-z_lattice
hypothesis explaining why Si shows larger cross-lattice spread than C.
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
from scipy.spatial import cKDTree

from lattices import build_honeycomb, build_square, build_triangular, P_C
from percolation import sample_many
from runners_lammps import lammps_airebo, lammps_si_tersoff
from volume import R_COV_C, R_COV_SI


L_VALUES = [32, 64]
N_PER_L = {32: 12, 64: 8}
A_C, A_SI = 1.42, 2.35
BOND_SCALE = 1.30
R_COV = {"C": R_COV_C, "Si": R_COV_SI}
MPI_PROCS = 8
PARALLEL_REPLICAS = 2

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

DATA_DIR = ROOT / "data" / "phase7_coordination"
RESULTS_DIR = ROOT / "results" / "phase7_coordination"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def coordination_distribution(positions: np.ndarray, element: str) -> dict:
    cutoff = BOND_SCALE * 2.0 * R_COV[element]
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    z = np.zeros(len(positions), dtype=int)
    if len(pairs):
        np.add.at(z, pairs[:, 0], 1)
        np.add.at(z, pairs[:, 1], 1)
    return dict(
        z_array=z.tolist(),
        z_mean=float(z.mean()),
        z_std=float(z.std()),
        frac_z0=float(np.mean(z == 0)),
        frac_z1=float(np.mean(z == 1)),
        frac_z2=float(np.mean(z == 2)),
        frac_z3=float(np.mean(z == 3)),
        frac_z4=float(np.mean(z == 4)),
        frac_z5=float(np.mean(z == 5)),
        frac_z6plus=float(np.mean(z >= 6)),
    )


def _process_one(args) -> dict | None:
    material, lattice_name, L, ic, cl_pos, cl_seed, cl_n = args
    runner = lammps_airebo if material == "C" else lammps_si_tersoff
    atoms = Atoms(symbols=[material] * cl_n, positions=cl_pos)
    opt = runner(atoms, optimize=True, timeout=3600, mpi_procs=MPI_PROCS)
    if not opt["converged"]:
        return None
    pos_rel = opt["opt_atoms"].get_positions()
    cstats = coordination_distribution(pos_rel, material)
    return dict(
        material=material, lattice=lattice_name, L=int(L),
        realisation=int(ic), seed=int(cl_seed),
        n_atoms=int(cl_n),
        E_relaxed_eV=float(opt["E_eV"]),
        E_per_atom_eV=float(opt["E_eV"] / cl_n),
        positions_relaxed=pos_rel.tolist(),
        wall_s=float(opt["wall_time_s"]),
        **{k: v for k, v in cstats.items() if k != "z_array"},
        z_array=cstats["z_array"],
    )


def gather_material(material: str, lattices: dict) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = DATA_DIR / f"{material}_clusters.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
        print(f"  [{material}] loaded {len(recs)} cached")
        return recs
    rng = np.random.default_rng(20260512 + (1 if material == "Si" else 0))
    tasks = []
    for lattice_name, (builder, p) in lattices.items():
        for L in L_VALUES:
            lat = builder(L)
            n = N_PER_L[L]
            clusters = sample_many(lat, p, n,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=4)
            for ic, cl in enumerate(clusters):
                tasks.append((material, lattice_name, L, ic,
                              cl.positions, cl.seed, cl.n_atoms))
    print(f"  [{material}] queued {len(tasks)} tasks")
    records: list[dict] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PARALLEL_REPLICAS) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            if r is None:
                continue
            print(f"  [{material}/{r['lattice']}] L={r['L']:3d} "
                  f"rep={r['realisation']:2d} N={r['n_atoms']:5d}  "
                  f"⟨z⟩={r['z_mean']:.2f}  −E/N={-r['E_per_atom_eV']:5.3f}  "
                  f"f4={r['frac_z4']:.3f}  ({r['wall_s']:.1f}s)",
                  flush=True)
            records.append(r)
    print(f"  [{material}] {len(records)} clusters in "
          f"{time.perf_counter() - t0:.1f}s")
    cache.write_text(json.dumps(records, indent=2))
    return records


def aggregate(records: list[dict]) -> list[dict]:
    out = []
    for material in ("C", "Si"):
        for lattice in ("honeycomb", "square", "triangular"):
            sub = [r for r in records
                    if r["material"] == material and r["lattice"] == lattice]
            if not sub:
                continue
            out.append(dict(
                material=material, lattice=lattice,
                z_lattice={"honeycomb": 3, "square": 4, "triangular": 6}[lattice],
                n=len(sub),
                z_post_mean=float(np.mean([r["z_mean"] for r in sub])),
                z_post_std=float(np.std([r["z_mean"] for r in sub])),
                E_per_atom_mean=float(np.mean([r["E_per_atom_eV"] for r in sub])),
                E_per_atom_std=float(np.std([r["E_per_atom_eV"] for r in sub])),
                frac_z3_mean=float(np.mean([r["frac_z3"] for r in sub])),
                frac_z4_mean=float(np.mean([r["frac_z4"] for r in sub])),
                frac_z5plus_mean=float(np.mean([r["frac_z5"] + r["frac_z6plus"] for r in sub])),
            ))
    return out


def main():
    print("=== Phase 7: coordination vs energy across lattices/materials ===\n"
          f"  L_VALUES = {L_VALUES},  N_PER_L = {N_PER_L}\n")
    all_recs = []
    all_recs += gather_material("C", LATTICES_C)
    all_recs += gather_material("Si", LATTICES_SI)
    rows = aggregate(all_recs)

    print(f"\n{'='*92}")
    print(f"            POST-RELAXATION COORDINATION × COHESION (mean over L={L_VALUES})")
    print(f"{'='*92}")
    print(f"  {'mat':3s} {'lattice':10s} {'z_lat':5s}  "
          f"{'⟨z_post⟩':10s} {'⟨−E/N⟩(eV)':12s}  "
          f"{'f3':6s}    {'f4':6s}    {'f5+':6s}")
    for r in rows:
        print(f"   {r['material']:2s}  {r['lattice']:10s}  {r['z_lattice']}     "
              f"{r['z_post_mean']:.2f}±{r['z_post_std']:.2f}    "
              f"{-r['E_per_atom_mean']:5.3f}±{r['E_per_atom_std']:.3f}  "
              f"{r['frac_z3_mean']:.3f}    {r['frac_z4_mean']:.3f}    "
              f"{r['frac_z5plus_mean']:.3f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    markers = {"honeycomb": "v", "square": "s", "triangular": "^"}
    colors = {"C": "C0", "Si": "C1"}
    for r in all_recs:
        m = markers[r["lattice"]]
        c = colors[r["material"]]
        axes[0].scatter(r["z_mean"], -r["E_per_atom_eV"], marker=m, c=c,
                         s=18, alpha=0.55, edgecolor="k", linewidth=0.2)
        axes[1].scatter(r["frac_z4"], -r["E_per_atom_eV"], marker=m, c=c,
                         s=18, alpha=0.55, edgecolor="k", linewidth=0.2)
    # legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="grey", label="honeycomb (z=3)", markersize=8),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="grey", label="square (z=4)", markersize=8),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="grey", label="triangular (z=6)", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", label="C", markersize=8),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C1", label="Si", markersize=8),
    ]
    axes[0].legend(handles=handles, fontsize=7, loc="upper left")
    axes[0].set_xlabel(r"$\langle z\rangle$ post-relaxation")
    axes[0].set_ylabel(r"$-E/N$  (eV/atom)")
    axes[0].set_title("Cohesion vs avg coordination")
    axes[1].set_xlabel(r"fraction of 4-coordinated atoms (post-relaxation)")
    axes[1].set_ylabel(r"$-E/N$  (eV/atom)")
    axes[1].set_title("Cohesion vs sp³ fraction")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_coordination_vs_energy.png", dpi=180)
    plt.close(fig)

    summary = dict(BOND_SCALE=BOND_SCALE, R_COV=R_COV,
                   N_PER_L=N_PER_L, L_VALUES=L_VALUES, rows=rows)
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_coordination_vs_energy.png'}")


if __name__ == "__main__":
    main()
