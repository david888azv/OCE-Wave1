"""Phase 7 — coordination analysis: how does valence-vs-z_lattice
constrain the cross-lattice universality of ε_V*?

User hypothesis (May 2026):
  Si has 4 valence electrons, prefers sp³ → z_pref = 4.
  C has 4 valence electrons, sp² (z=3) and sp³ (z=4) both stable.
  Lattice coordinations: honeycomb 3, square 4, triangular 6.
  Therefore Si on honeycomb (z=3) is *under*-coordinated, on square is
  matched, on triangular is *over*-coordinated → relaxation freedom
  varies between lattices and breaks the universality of ε_V* for Si.
  Carbon, which tolerates both 3- and 4-coordination, should suffer less.

Test: from the cached relaxed clusters of phase 5 (C-AIREBO) and phase 6
(Si-Tersoff), compute the per-cluster *post-relaxation* coordination
distribution.  Then correlate ⟨z⟩_post with ⟨E/N⟩ across lattices.

Bond perception: r_ij < scale × (r_cov_i + r_cov_j) with scale = 1.30
(same as oce/figures.py and standard for organics / Si).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from volume import R_COV_C, R_COV_SI

R_COV = {"C": R_COV_C, "Si": R_COV_SI}
BOND_SCALE = 1.30

DATA_C = ROOT / "data" / "phase5_three_lattices"
DATA_SI = ROOT / "data" / "phase6_silicon"
RESULTS_DIR = ROOT / "results" / "phase7_coordination"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def coordination_stats(positions: np.ndarray, element: str) -> dict:
    """Per-cluster coordination distribution after relaxation."""
    from scipy.spatial import cKDTree
    rcov = R_COV[element]
    cutoff = BOND_SCALE * 2.0 * rcov
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    z = np.zeros(len(positions), dtype=int)
    if len(pairs):
        np.add.at(z, pairs[:, 0], 1)
        np.add.at(z, pairs[:, 1], 1)
    return dict(
        z_mean=float(z.mean()) if len(z) else 0.0,
        z_std=float(z.std()) if len(z) else 0.0,
        frac_z0=float(np.mean(z == 0)),
        frac_z1=float(np.mean(z == 1)),
        frac_z2=float(np.mean(z == 2)),
        frac_z3=float(np.mean(z == 3)),
        frac_z4=float(np.mean(z == 4)),
        frac_z5=float(np.mean(z == 5)),
        frac_z6plus=float(np.mean(z >= 6)),
    )


def gather(material: str, datadir: Path, energy_key: str) -> list[dict]:
    out = []
    for lattice in ("honeycomb", "square", "triangular"):
        cache = datadir / f"{lattice}_clusters.json"
        if not cache.exists():
            continue
        for r in json.loads(cache.read_text()):
            pos = np.array(r["positions_relaxed"]) if "positions_relaxed" in r \
                  else np.array(r.get("positions", []))
            if pos.size == 0:
                # Phase 6 records don't keep positions — recompute via energy/V?
                # We need positions to compute coordination.  Skip if missing.
                # (We'll add a side-cache for relaxed positions.)
                continue
            stats = coordination_stats(pos, material)
            out.append(dict(
                material=material,
                lattice=lattice,
                L=r["L"],
                n_atoms=r["n_atoms"],
                E_eV=r[energy_key],
                e_per_atom=r[energy_key] / r["n_atoms"],
                **stats,
            ))
    return out


def main():
    print("=== Phase 7 — coordination vs energy across lattices/materials ===\n")
    c_records = gather("C", DATA_C, "E_airebo_relaxed_eV")
    si_records = gather("Si", DATA_SI, "E_tersoff_relaxed_eV")
    print(f"  C records loaded: {len(c_records)}    "
          f"Si records loaded: {len(si_records)}")
    if not si_records:
        print("\n  [warn] Si phase-6 cache has no positions_relaxed field.")
        print("  Re-running phase-6 with position retention is needed for Si.")

    rows = []
    for mat, recs in (("C", c_records), ("Si", si_records)):
        if not recs:
            continue
        for lattice in ("honeycomb", "square", "triangular"):
            sub = [r for r in recs if r["lattice"] == lattice]
            if not sub:
                continue
            zs = [r["z_mean"] for r in sub]
            epn = [r["e_per_atom"] for r in sub]
            f3 = [r["frac_z3"] for r in sub]
            f4 = [r["frac_z4"] for r in sub]
            f5p = [r["frac_z5"] + r["frac_z6plus"] for r in sub]
            rows.append(dict(
                material=mat, lattice=lattice,
                z_lattice={"honeycomb": 3, "square": 4, "triangular": 6}[lattice],
                n=len(sub),
                z_post_mean=float(np.mean(zs)),
                z_post_std=float(np.std(zs)),
                E_per_atom_mean=float(np.mean(epn)),
                E_per_atom_std=float(np.std(epn)),
                frac_z3_mean=float(np.mean(f3)),
                frac_z4_mean=float(np.mean(f4)),
                frac_z5plus_mean=float(np.mean(f5p)),
            ))

    print(f"\n{'='*88}")
    print(f"            POST-RELAXATION COORDINATION SUMMARY (averaged over all L)")
    print(f"{'='*88}")
    print(f"  {'mat':3s} {'lattice':10s} z_lat  ⟨z_post⟩      ⟨−E/N⟩    "
          f"frac(z=3)  frac(z=4)  frac(z≥5)")
    for r in rows:
        print(f"   {r['material']:2s}  {r['lattice']:10s}  {r['z_lattice']}    "
              f"{r['z_post_mean']:.2f} ± {r['z_post_std']:.2f}    "
              f"{-r['E_per_atom_mean']:5.3f}     "
              f"{r['frac_z3_mean']:.3f}      {r['frac_z4_mean']:.3f}      "
              f"{r['frac_z5plus_mean']:.3f}")

    # Plots
    if c_records and si_records:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for mat, recs, marker, color in (
            ("C", c_records, "o", "C0"),
            ("Si", si_records, "s", "C1"),
        ):
            for lattice, lattice_marker in (
                ("honeycomb", "v"), ("square", "s"), ("triangular", "^"),
            ):
                sub = [r for r in recs if r["lattice"] == lattice]
                if not sub:
                    continue
                z = [r["z_mean"] for r in sub]
                epn = [-r["e_per_atom"] for r in sub]
                axes[0].scatter(z, epn, marker=lattice_marker,
                                 c=color, s=20, alpha=0.6,
                                 label=f"{mat} {lattice}" if lattice == "honeycomb"
                                 else None)
                axes[1].scatter([r["frac_z4"] for r in sub],
                                 epn, marker=lattice_marker, c=color,
                                 s=20, alpha=0.6,
                                 label=f"{mat} {lattice}" if lattice == "honeycomb"
                                 else None)
        axes[0].set_xlabel(r"⟨z⟩ post-relaxation"); axes[0].set_ylabel(r"$-E/N$ (eV/atom)")
        axes[0].set_title("Cohesion vs avg coordination")
        axes[1].set_xlabel(r"fraction of 4-coordinated atoms")
        axes[1].set_ylabel(r"$-E/N$ (eV/atom)")
        axes[1].set_title("Cohesion vs sp³ fraction")
        for ax in axes:
            ax.legend(fontsize=8)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "fig_coordination_vs_energy.png", dpi=180)
        plt.close(fig)

    # Save summary
    summary = dict(
        BOND_SCALE=BOND_SCALE,
        R_COV=R_COV,
        rows=rows,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_coordination_vs_energy.png'}")


if __name__ == "__main__":
    main()
