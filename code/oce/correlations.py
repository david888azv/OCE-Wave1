"""Correlation functions Π_F for OCE.

For each figure class F (defined by symmetry/composition key) we build a
single scalar feature value Π_F(σ) per molecule σ. The molecular energy is
expanded as

    E_OCE(σ) = Σ_F  J_F · Π_F(σ)

with J_F obtained by linear regression on a training set.

Π functions implemented (v0.2):
    1-figure:   Π = occupation · ε_shell
    2-figure:   LCAO-bonding form,
                Π = bo · [ (ε_μ+ε_ν)/2  −  √( ((ε_μ−ε_ν)/2)² + h(r)² ) ]
                with h(r) = h0 · exp(-α (r - r0))  (Wolfsberg-Helmholz-like)
    3-figure:   angle-modulated mean-orbital,
                Π = cos(θ) · (ε_i + ε_j + ε_k) / 3
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from oce.figures import FourFigure, OneFigure, ThreeFigure, TwoFigure


# Default hopping parameters (eV, Å). Shared across orbital types in v0.1.
H0_DEFAULT = -8.0   # eV, magnitude of resonance integral at reference distance
ALPHA_DEFAULT = 1.5  # Å^-1, exponential decay
R0_DEFAULT = 1.5     # Å, reference distance


def lcao_pair_energy(eps_a: float, eps_b: float, r: float,
                     h0: float = H0_DEFAULT,
                     alpha: float = ALPHA_DEFAULT,
                     r0: float = R0_DEFAULT) -> float:
    """Bonding eigenvalue of a 2×2 secular problem with diagonal ε_a, ε_b
    and off-diagonal hopping h(r) = h0 · exp(-α(r-r0))."""
    h = h0 * math.exp(-alpha * (r - r0))
    half_sum = 0.5 * (eps_a + eps_b)
    half_diff = 0.5 * (eps_a - eps_b)
    return half_sum - math.sqrt(half_diff * half_diff + h * h)


# Coulomb prefactor e²/(4πε₀) in eV·Å. Source: 2018 CODATA, k_C = 14.3996454.
COULOMB_eV_A = 14.39964547842567


def madelung_feature(positions_A, charges,
                     r_min: float = 1.5) -> float:
    """Sum of pairwise Coulomb energy over atom pairs separated by more
    than `r_min` Å, expressed in eV.

    The short-range cutoff `r_min` excludes covalently bonded pairs
    (those are already represented by the 2F LCAO bonding term in the
    OCE basis); only longer-range, non-bonded electrostatics enter
    here.  This is the standard "Coulomb-without-double-counting"
    decomposition used in DFTB+ and tight-binding QM/MM hybrids.

    Parameters
    ----------
    positions_A : array-like, shape (N, 3) — atomic positions in Å
    charges     : array-like, shape (N,)   — atomic charges (electron
                  units; can be formal or partial)
    r_min       : float — short-range cutoff in Å.  Pairs with
                  r < r_min are skipped (covalent regime).

    Returns
    -------
    Total Madelung energy in eV (extensive).  Caller normalises
    by N_atoms when used as a per-atom feature.
    """
    import numpy as np
    pos = np.asarray(positions_A, dtype=float)
    q = np.asarray(charges, dtype=float)
    n = len(q)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((diff * diff).sum(-1))
    qq = q[:, None] * q[None, :]
    mask = (r > r_min)
    iu = np.triu_indices(n, k=1)
    valid = mask[iu]
    rr = r[iu][valid]
    qprod = qq[iu][valid]
    if rr.size == 0:
        return 0.0
    return float(COULOMB_eV_A * (qprod / rr).sum())


@dataclass
class CorrelationVector:
    """Per-molecule feature vector indexed by figure-class key."""
    pi: dict[tuple, float]                # key -> Π_F
    feature_keys: tuple[tuple, ...]       # ordered list of keys

    def as_array(self, master_keys: list[tuple]):
        import numpy as np
        return np.array([self.pi.get(k, 0.0) for k in master_keys], dtype=float)


def correlations_for_molecule(
    one_figs: list[OneFigure],
    two_figs: list[TwoFigure],
    atomic_table,
    three_figs: list[ThreeFigure] | None = None,
    four_figs: list[FourFigure] | None = None,
) -> CorrelationVector:
    """Compute Π_F for each figure class found in the molecule."""
    pi: dict[tuple, float] = defaultdict(float)

    # 1-figures: sum over atoms in this class of (occ * ε)
    for fig in one_figs:
        atom = atomic_table[fig.element]
        shell = next(s for s in atom.shells if s.label == fig.shell_label)
        pi[fig.key()] += shell.occupation * shell.epsilon_eV

    # 2-figures: sum over bonded pairs of LCAO bonding eigenvalue * bond_order
    for fig in two_figs:
        atom_i = atomic_table[fig.element_i]
        atom_j = atomic_table[fig.element_j]
        sh_i = next(s for s in atom_i.shells if s.label == fig.shell_i)
        sh_j = next(s for s in atom_j.shells if s.label == fig.shell_j)
        bond_eig = lcao_pair_energy(sh_i.epsilon_eV, sh_j.epsilon_eV, fig.distance)
        pi[fig.key()] += fig.bond_order * bond_eig

    # 3-figures: Π = cos(θ) · (ε_i + ε_j + ε_k)/3
    if three_figs is not None:
        for fig in three_figs:
            sh_i = next(s for s in atomic_table[fig.element_i].shells
                        if s.label == fig.shell_i)
            sh_j = next(s for s in atomic_table[fig.element_j].shells
                        if s.label == fig.shell_j)
            sh_k = next(s for s in atomic_table[fig.element_k].shells
                        if s.label == fig.shell_k)
            mean_eps = (sh_i.epsilon_eV + sh_j.epsilon_eV + sh_k.epsilon_eV) / 3.0
            pi[fig.key()] += math.cos(fig.angle_rad) * mean_eps

    # 4-figures: Π = cos(2φ) · mean(ε_i, ε_j, ε_k, ε_l)
    # cos(2φ) is the standard V₂ torsion form — distinguishes eclipsed
    # (φ=0,π) from staggered (φ=±π/2) without breaking i↔l symmetry.
    if four_figs is not None:
        for fig in four_figs:
            sh_i = next(s for s in atomic_table[fig.element_i].shells
                        if s.label == fig.shell_i)
            sh_j = next(s for s in atomic_table[fig.element_j].shells
                        if s.label == fig.shell_j)
            sh_k = next(s for s in atomic_table[fig.element_k].shells
                        if s.label == fig.shell_k)
            sh_l = next(s for s in atomic_table[fig.element_l].shells
                        if s.label == fig.shell_l)
            mean_eps = 0.25 * (sh_i.epsilon_eV + sh_j.epsilon_eV
                               + sh_k.epsilon_eV + sh_l.epsilon_eV)
            pi[fig.key()] += math.cos(2.0 * fig.dihedral_rad) * mean_eps

    keys = tuple(sorted(pi.keys()))
    return CorrelationVector(pi=dict(pi), feature_keys=keys)


def collect_feature_keys(vectors: list[CorrelationVector]) -> list[tuple]:
    """Union of all feature keys seen across a dataset, sorted for stability."""
    all_keys: set[tuple] = set()
    for v in vectors:
        all_keys.update(v.pi.keys())
    return sorted(all_keys)


if __name__ == "__main__":
    from pathlib import Path
    from ase.build import molecule
    from oce.atomic_table import load_table
    from oce.figures import enumerate_figures

    table = load_table(Path(__file__).resolve().parents[1]
                       / "data" / "atoms" / "atomic_table.json")

    for name in ["H2O", "CH4", "C2H4", "C2H2", "CH3OH", "HCOOH"]:
        atoms = molecule(name)
        one, two, three, four = enumerate_figures(atoms, table)
        cv = correlations_for_molecule(one, two, table, three_figs=three)
        n1 = sum(1 for k in cv.feature_keys if k[0] == "1F")
        n2 = sum(1 for k in cv.feature_keys if k[0] == "2F")
        n3 = sum(1 for k in cv.feature_keys if k[0] == "3F")
        print(f"\n{name}: {len(cv.feature_keys)} feature classes "
              f"({n1} 1F, {n2} 2F, {n3} 3F)")
        for k in cv.feature_keys:
            if k[0] == "3F":
                print(f"   {str(k):75s}  Π = {cv.pi[k]:+10.3f}")
