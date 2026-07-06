"""End-to-end characterisation of OCE forces driving an ASE optimiser.

This script answers the operational question that closes Phase 1 of
the JCTC roadmap:

    With a *real* (production) ridge-fit OCEModel and the analytic
    gradients now in :mod:`oce.forces`, can the v1.0.0 basis act as
    a useful force field for geometry optimisation?

It is **not** a unit test — it is a quantitative experiment whose
output goes to ``results/optimization_demo.log``.  Three diagnostics
per molecule:

    1.  ``|F|_max`` and ``|F|_rms`` at the xtb-optimised geometry.
        Small values mean the OCE energy gradient is well-aligned
        with the xtb gradient at that point.

    2.  Force response to a 0.05 Å random perturbation (same diagnostic
        evaluated at the displaced geometry).  The displaced |F|
        should be larger than the equilibrium |F| if the basis
        encodes any restoring physics; we report the ratio.

    3.  A short BFGS run (≤5 steps, ``fmax=0.01`` eV/Å, conservative
        ``maxstep=0.05`` Å) starting from the perturbed geometry.
        We report energy trajectory, bond-length deltas vs the xtb
        equilibrium, and whether BFGS converged or had to be cut off.

Known caveat (motivates Phase 2)
--------------------------------
The 1F+2F Wolfsberg-Helmholz term is monotonically *more bonding* as
``r → 0`` (no Pauli/short-range repulsion).  With positive J_F
coefficients on the bond features, the OCE energy is unbounded below
under bond compression.  Unconstrained BFGS will therefore not
converge to a physical equilibrium — it will collapse bonds.  This
script *measures* that pathology rather than papering over it.  The
fix is the Slater-Koster directional bonding planned for v1.2.0.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from oce.atomic_table import load_table
from oce.calculator import OCECalculator
from oce.dataset import entry_to_atoms, load_dataset
from oce.fit import OCEModel


# ------------------------------------------------------------------- #
# Helpers                                                              #
# ------------------------------------------------------------------- #

def load_ridge_model(path: Path) -> OCEModel:
    """Reconstruct an OCEModel from the JSON dump in results/."""
    raw = json.loads(path.read_text())
    keys = [tuple(_freeze(k) for k in entry) for entry in raw["feature_keys"]]
    return OCEModel(
        feature_keys=keys,
        J=raw["J"],
        intercept=raw["intercept"],
        method=raw["method"],
    )


def _freeze(x):
    """JSON arrays come back as lists; OCE feature keys use tuples
    everywhere."""
    if isinstance(x, list):
        return tuple(_freeze(e) for e in x)
    return x


def bond_lengths(atoms: Atoms, scale: float = 1.30) -> dict[tuple, float]:
    """Return {(i,j,sym): distance} for all bonded pairs (covalent-radii
    cutoff identical to oce.figures.perceive_bonds)."""
    from ase.data import covalent_radii
    Z = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    out: dict[tuple, float] = {}
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            r_cov = covalent_radii[Z[i]] + covalent_radii[Z[j]]
            if d / r_cov <= scale:
                key = (i, j, f"{syms[i]}-{syms[j]}")
                out[key] = d
    return out


def fmax_frms(F: np.ndarray) -> tuple[float, float]:
    fnorms = np.linalg.norm(F, axis=1)
    return float(fnorms.max()), float(np.sqrt(np.mean(fnorms ** 2)))


def perturb(atoms: Atoms, magnitude: float, seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    pos = atoms.get_positions()
    pos = pos + rng.uniform(-magnitude, magnitude, size=pos.shape)
    new = atoms.copy()
    new.set_positions(pos)
    return new


def diagnose(atoms: Atoms, model: OCEModel, table) -> tuple[float, float, float]:
    """Energy and force diagnostics at a single geometry."""
    calc = OCECalculator(model=model, atomic_table=table,
                         freeze_topology=False)
    a = atoms.copy()
    a.calc = calc
    E = float(a.get_potential_energy())
    F = np.asarray(a.get_forces())
    fmax, frms = fmax_frms(F)
    return E, fmax, frms


def run_bfgs(atoms: Atoms, model: OCEModel, table,
             max_steps: int = 5, fmax_target: float = 1e-2,
             max_step_size: float = 0.05) -> tuple[Atoms, list[float], int, bool]:
    """Run a short BFGS, returning final atoms + energy trajectory + step
    count + converged flag.  The conservative ``max_step_size`` keeps
    BFGS away from regions where the v1.0.0 basis goes pathological."""
    a = atoms.copy()
    a.calc = OCECalculator(model=model, atomic_table=table,
                           freeze_topology=True)
    energies: list[float] = []

    def _logE():
        energies.append(float(a.get_potential_energy()))

    opt = BFGS(a, logfile=None, maxstep=max_step_size)
    _logE()
    converged = False
    for step in range(max_steps):
        opt.step()
        _logE()
        F = np.asarray(a.get_forces())
        fmax, _ = fmax_frms(F)
        if fmax < fmax_target:
            converged = True
            break
    n_steps = len(energies) - 1
    return a, energies, n_steps, converged


# ------------------------------------------------------------------- #
# Main experiment                                                      #
# ------------------------------------------------------------------- #

def main():
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    model = load_ridge_model(base / "results" / "model_ridge.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")

    print("=" * 78)
    print(f"OCE optimization demo  (Phase 1 milestone)")
    print(f"  model:   {model.method}")
    print(f"  features: {len(model.J)}  "
          f"({sum(1 for k in model.feature_keys if k[0]=='1F')} 1F + "
          f"{sum(1 for k in model.feature_keys if k[0]=='2F')} 2F)")
    print(f"  intercept: {model.intercept:+.4f} eV")
    print("=" * 78)

    # Targets: small, well-trained, tractable.
    targets = ["water", "methane", "methanol", "ethylene", "formaldehyde"]
    by_name = {e.name: e for e in train}
    picked = [by_name[n] for n in targets if n in by_name]

    summary_rows = []
    for entry in picked:
        atoms_eq = entry_to_atoms(entry)        # xtb-optimised geometry
        n = len(atoms_eq)

        E_eq, fmax_eq, frms_eq = diagnose(atoms_eq, model, table)
        bonds_eq = bond_lengths(atoms_eq)

        atoms_pert = perturb(atoms_eq, magnitude=0.05, seed=42)
        E_pert, fmax_pert, frms_pert = diagnose(atoms_pert, model, table)

        # Short BFGS from the perturbed geometry.
        atoms_opt, traj_E, n_steps, converged = run_bfgs(
            atoms_pert, model, table,
            max_steps=5, fmax_target=1e-2, max_step_size=0.05)
        bonds_opt = bond_lengths(atoms_opt)

        # Bond-length comparison: matched by index pair.
        bond_deltas = []
        for key in bonds_eq:
            if key in bonds_opt:
                bond_deltas.append((key[2], bonds_eq[key], bonds_opt[key],
                                    bonds_opt[key] - bonds_eq[key]))

        # RMSD vs xtb equilibrium (atom-aligned, no centring/rotation).
        rmsd = float(np.sqrt(np.mean(np.sum(
            (atoms_opt.get_positions() - atoms_eq.get_positions()) ** 2,
            axis=1))))

        # ----- print per-molecule report -----
        print()
        print(f"--- {entry.name}  ({entry.formula}, n_atoms={n}) ---")
        print(f"  At xtb equilibrium:")
        print(f"     E_OCE = {E_eq:+11.4f} eV     E_xtb (train) = "
              f"{entry.energy_eV:+11.4f} eV     ΔE = "
              f"{E_eq - entry.energy_eV:+8.4f} eV")
        print(f"     |F|_max = {fmax_eq:8.4f} eV/Å    "
              f"|F|_rms = {frms_eq:8.4f} eV/Å")
        print(f"  After ±0.05 Å perturbation:")
        print(f"     E_OCE = {E_pert:+11.4f} eV     "
              f"|F|_max = {fmax_pert:8.4f} eV/Å    "
              f"|F|_rms = {frms_pert:8.4f} eV/Å")
        print(f"     restoring force ratio (pert/eq): "
              f"{fmax_pert / max(fmax_eq, 1e-9):6.2f}×")
        print(f"  BFGS from perturbed (≤5 steps, maxstep=0.05 Å):")
        print(f"     n_steps = {n_steps}    converged = {converged}    "
              f"E trajectory = "
              + "  →  ".join(f"{e:+8.3f}" for e in traj_E))
        print(f"     RMSD vs xtb_eq = {rmsd:.4f} Å")
        if bond_deltas:
            print(f"     bond-length deltas (Å):")
            for sym, b_eq, b_opt, db in bond_deltas:
                marker = "  ←contracted" if db < -0.005 else (
                    "  ←stretched" if db > 0.005 else "")
                print(f"        {sym:6s}  xtb={b_eq:.4f}  "
                      f"oce={b_opt:.4f}  Δ={db:+.4f}{marker}")

        summary_rows.append({
            "name": entry.name,
            "fmax_eq": fmax_eq, "frms_eq": frms_eq,
            "fmax_pert": fmax_pert,
            "rmsd": rmsd,
            "n_steps": n_steps, "converged": converged,
            "dE_traj": traj_E[-1] - traj_E[0],
        })

    # -------- final compact summary --------
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'molecule':14s}  {'|F|_max@eq':>11s}  {'|F|_max@pert':>13s}  "
          f"{'RMSD':>7s}  {'BFGS':>10s}  {'ΔE BFGS':>9s}")
    for row in summary_rows:
        bfgs_state = (f"{row['n_steps']}/{'OK' if row['converged'] else 'cut'}")
        print(f"{row['name']:14s}  "
              f"{row['fmax_eq']:>11.4f}  "
              f"{row['fmax_pert']:>13.4f}  "
              f"{row['rmsd']:>7.4f}  "
              f"{bfgs_state:>10s}  "
              f"{row['dE_traj']:>+9.4f}")
    print("=" * 78)
    print()
    print("Reading the diagnostics:")
    print("  * |F|_max@eq close to zero → OCE energy gradient agrees with")
    print("    xtb gradient at the xtb minimum (well-aligned forces).")
    print("  * |F|_max@pert > |F|_max@eq → forces respond to displacement.")
    print("  * RMSD small (<0.05 Å) and ΔE_BFGS < 0 → optimisation moves")
    print("    geometry back toward xtb equilibrium.")
    print("  * RMSD ≫ 0.05 Å with bond-length contractions → the v1.0.0")
    print("    basis lacks short-range repulsion (expected; fix in v1.2.0).")


if __name__ == "__main__":
    main()
