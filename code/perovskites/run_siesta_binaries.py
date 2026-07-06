"""Compute SIESTA-PBE single-point energies for 15 binary phases needed
to build the SIESTA decomposition energy

    E_decomp(ABX3) = E_coh(ABX3) - E_coh(AX) - E_coh(BX2)

so that it becomes directly comparable to Mannodi's decomposition energy
(currently reported as VASP-PBE and VASP-HSE06). Without this conversion,
SIESTA E_coh (vs isolated atoms) and Mannodi E_decomp (vs binary phases)
measure different quantities and do not correlate (rho approx 0.08).

Structures used:
  AX (A = Cs)  -> CsCl-type B2 (Pm-3m), 2 atoms / f.u. (1 f.u. per primitive)
  AX (A = K)   -> rocksalt B1 (Fm-3m), 2 atoms / f.u. (1 f.u. per primitive)
  BX2          -> CdI2-type P-3m1 layered, 3 atoms / f.u. (1 f.u. per primitive)

Lattice constants are taken from experiment / literature for the
ground-state polymorph where available, otherwise interpolated from
ionic radii. The exact polymorph choice affects the absolute E_decomp
but should leave the ranking of the 18 ABX3 endmembers nearly invariant,
because the same binary reference is used for every (A,B) combination
that contains that binary.

Output: data/perovskites/siesta_binaries.json
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.perovskites.siesta_perov import siesta_szp_energy


# --------------------------------------------------------------- structures

CSX_LATTICE = {"CsCl": 4.12, "CsBr": 4.29, "CsI": 4.57}     # CsCl-type B2 (A)
KX_LATTICE  = {"KCl": 6.29,  "KBr": 6.60,  "KI": 7.07}      # rocksalt B1 (conventional cubic edge)

# Iodides are CdI2-type P-3m1 (layered hexagonal, 3 atoms / f.u.) — ground state
# Chlorides and bromides are cotunnite-type PNMA (orthorhombic, 4 f.u. = 12 atoms) — ground state
# Reference lattices: PbCl2 a=7.62 b=4.53 c=9.05 Å; we scale Br/Sn/Ge variants.
BX2_CDI2 = {  # iodides only
    "PbI2":  (4.557, 6.99),
    "SnI2":  (4.53,  7.00),
    "GeI2":  (4.13,  6.83),
}

# Cotunnite (PNMA) structure: 4 B + 8 X per orthorhombic conventional cell.
# Wyckoff positions (general PNMA setting):
#   B  at 4c (x_B, 1/4, z_B), default x_B≈0.25, z_B≈0.10
#   X1 at 4c (x1, 1/4, z1),    default x1≈0.36, z1≈0.43
#   X2 at 4c (x2, 1/4, z2),    default x2≈0.03, z2≈0.83
# (and the four equivalent positions via PNMA symmetry)
# Reference: PbCl2 a=7.62 b=4.53 c=9.05 Å. PbBr2 a=8.05 b=4.76 c=9.55. SnCl2 a=7.79 b=4.32 c=9.18.
# Bromides ~ 5% larger than chlorides; Sn/Ge variants ~ 2-3% smaller than Pb.
BX2_COTUNNITE = {
    "PbCl2": (7.62, 4.53, 9.05),
    "PbBr2": (8.05, 4.76, 9.55),
    "SnCl2": (7.79, 4.32, 9.18),
    "SnBr2": (8.20, 4.55, 9.65),
    "GeCl2": (7.50, 4.20, 8.95),
    "GeBr2": (7.95, 4.45, 9.40),
}


def build_cdi2(B: str, X: str, a: float, c: float, z: float = 0.25) -> Atoms:
    """CdI2-type structure: hexagonal P-3m1, 1 B + 2 X per primitive cell."""
    cell = np.array([
        [a,          0.0,            0.0],
        [-a/2.0,     a*np.sqrt(3)/2, 0.0],
        [0.0,        0.0,            c],
    ])
    syms = [B, X, X]
    frac = np.array([
        [0.0,       0.0,      0.0],
        [1.0/3.0,   2.0/3.0,  z],
        [2.0/3.0,   1.0/3.0, -z],
    ])
    return Atoms(symbols=syms, scaled_positions=frac, cell=cell, pbc=True)


def build_cotunnite(B: str, X: str, a: float, b: float, c: float) -> Atoms:
    """Cotunnite-type (PNMA) structure: 4 B + 8 X = 12 atoms per conv. cell.

    Wyckoff 4c positions in PNMA setting: (x, 1/4, z), (-x, 3/4, -z),
    (1/2 - x, 3/4, 1/2 + z), (1/2 + x, 1/4, 1/2 - z).
    Default (x, z) for B, X1, X2 from PbCl2 experimental structure.
    """
    xB, zB = 0.25, 0.10
    x1, z1 = 0.36, 0.43
    x2, z2 = 0.03, 0.83

    def four_c(x: float, z: float) -> list[tuple[float, float, float]]:
        return [
            (x,       0.25,  z      ),
            (-x % 1,  0.75,  (-z) % 1),
            ((0.5 - x) % 1, 0.75, (0.5 + z) % 1),
            ((0.5 + x) % 1, 0.25, (0.5 - z) % 1),
        ]

    pos_B  = four_c(xB, zB)
    pos_X1 = four_c(x1, z1)
    pos_X2 = four_c(x2, z2)

    syms = [B]*4 + [X]*8
    frac = pos_B + pos_X1 + pos_X2
    cell = np.diag([a, b, c])
    return Atoms(symbols=syms, scaled_positions=frac, cell=cell, pbc=True)


def build_b2(A: str, X: str, a: float) -> Atoms:
    """CsCl-type B2 structure (primitive, 1 A + 1 X)."""
    # ASE bulk supports 'cesiumchloride'
    return bulk(f"{A}{X}", "cesiumchloride", a=a)


def build_rocksalt(A: str, X: str, a_conv: float) -> Atoms:
    """B1 rocksalt structure. ASE 'rocksalt' uses primitive (2 atoms)."""
    return bulk(f"{A}{X}", "rocksalt", a=a_conv)


def all_binaries() -> dict[str, Atoms]:
    out: dict[str, Atoms] = {}
    for f, a in CSX_LATTICE.items():
        A, X = f[:2], f[2:]
        out[f] = build_b2(A, X, a)
    for f, a in KX_LATTICE.items():
        A, X = f[:1], f[1:]
        out[f] = build_rocksalt(A, X, a)
    for f, (a, c) in BX2_CDI2.items():
        B = f[:2]
        X = f[2:].rstrip("2")
        out[f] = build_cdi2(B, X, a, c)
    for f, (a, b, c) in BX2_COTUNNITE.items():
        B = f[:2]
        X = f[2:].rstrip("2")
        out[f] = build_cotunnite(B, X, a, b, c)
    return out


# ----------------------------------------------------------------- main

def run_one(name_atoms: tuple[str, Atoms], threads: int = 2,
             keep_dir: Path | None = None) -> dict:
    name, atoms = name_atoms
    t0 = time.time()
    r = siesta_szp_energy(atoms, tag=f"bin_{name}", threads=threads,
                            timeout=3600,
                            keep_dir=keep_dir/name if keep_dir else None)
    r["binary"] = name
    r["symbols"] = atoms.get_chemical_symbols()
    r["cell"] = atoms.get_cell().array.tolist()
    r["positions"] = atoms.get_positions().tolist()
    r["wall_total_s"] = time.time() - t0
    print(f"  {name:>8s}  n={len(atoms)} atoms  E={r.get('E_eV','nan'):>14}  "
          f"ok={r.get('converged')}  wall={r['wall_total_s']:.1f}s")
    return r


def main(threads_per_job: int = 2, max_parallel: int = 4):
    binaries = all_binaries()
    print(f"Building SIESTA-PBE energies for {len(binaries)} binaries:")
    for name, atoms in binaries.items():
        print(f"  {name:>8s}  {atoms.get_chemical_symbols()}  cell={atoms.cell.array.diagonal()}")
    print()
    t0 = time.time()
    results: dict[str, dict] = {}
    keep_dir = Path(__file__).resolve().parent / "siesta_binaries_outputs"
    keep_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = {ex.submit(run_one, (name, atoms),
                          threads=threads_per_job, keep_dir=keep_dir): name
                for name, atoms in binaries.items()}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["binary"]] = r

    out = Path(__file__).resolve().parent / "siesta_binaries.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print(f"Total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=2,
                    help="OMP threads per SIESTA job")
    p.add_argument("--parallel", type=int, default=4,
                    help="number of parallel SIESTA jobs")
    args = p.parse_args()
    main(threads_per_job=args.threads, max_parallel=args.parallel)
