"""Diagnostic: where does the relaxation energy go in the OCE basis?

Decompose Π¹F, Π²F, Π³F values and totals before/after xtb relaxation,
using the SAME free-atom ε_μ in both cases (the table is composition-only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from ase import Atoms

from oce_predict import (featurise, atomic_table_mod, figures_mod,
                          correlations_mod, ATOMIC_TABLE_PATH)


CACHE = ROOT / "data" / "phase2_smoke" / "clusters_relax.json"


def decompose(atoms: Atoms, table) -> tuple[float, float, float, dict]:
    one, two, three, _ = figures_mod.enumerate_figures(
        atoms, table, include_angles=True, include_dihedrals=False)
    cv = correlations_mod.correlations_for_molecule(
        one, two, table, three_figs=three)
    pi1 = sum(v for k, v in cv.pi.items() if k[0] == "1F")
    pi2 = sum(v for k, v in cv.pi.items() if k[0] == "2F")
    pi3 = sum(v for k, v in cv.pi.items() if k[0] == "3F")
    return pi1, pi2, pi3, dict(cv.pi)


def main():
    if not CACHE.exists():
        print(f"Need cache {CACHE}.  Run pipelines/phase2_relax_smoke.py first.")
        return
    records = json.loads(CACHE.read_text())
    table = atomic_table_mod.load_table(ATOMIC_TABLE_PATH)

    print(f"=== OCE Π_F values: UN-relaxed vs RELAXED geometry ===")
    print(f"(Same free-atom ε_μ for both; only positions change.)\n")

    print(f"  N    ΔE_xtb (eV)   Δ(Σ Π¹F)   Δ(Σ Π²F)   Δ(Σ Π³F)   |Δ|Π²F| / Σ Π²F")
    for r in records:
        atoms_un = Atoms(symbols=["C"] * r["n_atoms"],
                         positions=np.array(r["positions_unrelaxed"]))
        atoms_rel = Atoms(symbols=["C"] * r["n_atoms"],
                          positions=np.array(r["positions_relaxed"]))

        pi1_un, pi2_un, pi3_un, _ = decompose(atoms_un, table)
        pi1_rel, pi2_rel, pi3_rel, _ = decompose(atoms_rel, table)

        d1 = pi1_rel - pi1_un
        d2 = pi2_rel - pi2_un
        d3 = pi3_rel - pi3_un
        dE = r["E_xtb_sp_eV"] - r["E_xtb_relaxed_eV"]
        rel2 = abs(d2) / abs(pi2_un) if pi2_un != 0 else float("nan")
        print(f"  {r['n_atoms']:3d}    {dE:+8.3f}     "
              f"{d1:+8.3f}    {d2:+9.3f}    {d3:+8.3f}    {rel2*100:5.2f}%")

    # Also show a representative single bond and angle change
    print(f"\n  --- Bond / angle distance distributions ---")
    for r in records[:3]:
        from scipy.spatial.distance import pdist
        pos_un = np.array(r["positions_unrelaxed"])
        pos_rel = np.array(r["positions_relaxed"])
        d_un = pdist(pos_un)
        d_rel = pdist(pos_rel)
        # Only nearest-neighbour-ish distances (<2 Å)
        nn_un = d_un[(d_un > 0.5) & (d_un < 2.0)]
        nn_rel = d_rel[(d_rel > 0.5) & (d_rel < 2.0)]
        print(f"  N={r['n_atoms']:3d}   unrelaxed bonds: ⟨r⟩={nn_un.mean():.3f} ± "
              f"{nn_un.std():.3f} Å    relaxed: ⟨r⟩={nn_rel.mean():.3f} ± "
              f"{nn_rel.std():.3f} Å")


if __name__ == "__main__":
    main()
