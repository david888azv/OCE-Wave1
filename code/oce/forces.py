"""Analytic forces for OCE.

For each figure class F, we compute the partial derivatives ∂Π_F/∂r_a
of the correlation feature with respect to atomic positions r_a. The
total force on atom `a` is

    F_a = -Σ_F  J_F · ∂Π_F/∂r_a

with J_F obtained from a fitted ``oce.fit.OCEModel``.

Topology assumption
-------------------
Bond perception (covalent-radii cutoff) is held *fixed* at the
reference geometry that was used to build ``FigureSet`` via
:func:`freeze_topology`. Forces are defined within that fixed
topology. Geometric quantities (distances, angles, dihedrals) are
recomputed from current ``atoms.get_positions()`` inside each
gradient routine — the values stored in the
:class:`oce.figures.TwoFigure` / :class:`ThreeFigure` /
:class:`FourFigure` dataclasses are only valid at the reference
geometry.

Implementation status
---------------------
v1.1.0:
    1F (atom)        : zero by construction (no position dependence).
    2F (bond)        : analytic Wolfsberg-Helmholz chain rule.        ✓
    3F (angle)       : analytic ∂cos θ_ijk/∂r_a (Bekker formulae).    ✓
    4F (dihedral)    : MOCK — returns zero.  Sign convention of φ
                       must be cross-validated against
                       :func:`oce.figures.enumerate_figures` before
                       turning on; deferred to v1.1.1.  Tests with
                       ``include_dihedrals=False`` are unaffected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from oce.correlations import ALPHA_DEFAULT, H0_DEFAULT, R0_DEFAULT
from oce.figures import (FourFigure, OneFigure, ThreeFigure, TwoFigure,
                          enumerate_figures)


@dataclass
class FigureSet:
    """Bundles the four figure lists returned by ``enumerate_figures``.

    A ``FigureSet`` represents a *frozen topology*: the connectivity
    (which atoms participate in which figures) is fixed, while the
    geometric values inside each figure are only meaningful at the
    reference geometry from which the set was built.
    """
    one: list[OneFigure]
    two: list[TwoFigure]
    three: list[ThreeFigure]
    four: list[FourFigure]

    @property
    def n_figures(self) -> int:
        return len(self.one) + len(self.two) + len(self.three) + len(self.four)


def freeze_topology(atoms: Atoms, atomic_table,
                    include_angles: bool = True,
                    include_dihedrals: bool = False) -> FigureSet:
    """Enumerate figures at the current geometry and return them as a
    frozen :class:`FigureSet` to be reused across force evaluations."""
    one, two, three, four = enumerate_figures(
        atoms, atomic_table,
        include_angles=include_angles,
        include_dihedrals=include_dihedrals,
    )
    return FigureSet(one=one, two=two, three=three, four=four)


# --------------------------------------------------------------------- #
# Per-feature gradients   (MOCK — all return zeros in v0.1.0)            #
# --------------------------------------------------------------------- #

def grad_one_figure(fig: OneFigure, n_atoms: int, atomic_table) -> np.ndarray:
    """∂Π/∂r for a 1-figure.  Identically zero (no position dependence).

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
    """
    return np.zeros((n_atoms, 3), dtype=float)


def grad_two_figure(fig: TwoFigure, atoms: Atoms, atomic_table,
                    h0: float = H0_DEFAULT,
                    alpha: float = ALPHA_DEFAULT,
                    r0: float = R0_DEFAULT) -> np.ndarray:
    """∂Π/∂r for a 2-figure (Wolfsberg-Helmholz bonding term).

    With ε_a, ε_b atomic shell eigenenergies, h(r) = h0·exp(-α(r-r0))
    and λ_- = (ε_a+ε_b)/2 - sqrt(Δ² + h²) where Δ = (ε_a-ε_b)/2,
    the chain rule gives

        ∂λ_-/∂h   = -h / sqrt(Δ² + h²)
        ∂h/∂r    = -α · h
        ∂λ_-/∂r  = (∂λ_-/∂h)(∂h/∂r) = α · h² / sqrt(Δ² + h²)
        ∂r/∂r_i  = (r_i - r_j) / r       (= -∂r/∂r_j)

    so ``∂Π/∂r_i = bond_order · (α h²/sqrt(Δ² + h²)) · (r_i-r_j)/r``
    and ``∂Π/∂r_j = -∂Π/∂r_i``.  All other atoms get zero.

    The distance r is recomputed from ``atoms.get_positions()`` so the
    routine is valid at any geometry within the frozen topology — the
    ``fig.distance`` field is not used.
    """
    n = len(atoms)
    grad = np.zeros((n, 3), dtype=float)
    pos = atoms.get_positions()
    v = pos[fig.i] - pos[fig.j]
    r = float(np.linalg.norm(v))
    if r < 1e-12:
        return grad

    h = h0 * math.exp(-alpha * (r - r0))
    eps_i = next(s for s in atomic_table[fig.element_i].shells
                 if s.label == fig.shell_i).epsilon_eV
    eps_j = next(s for s in atomic_table[fig.element_j].shells
                 if s.label == fig.shell_j).epsilon_eV
    half_diff = 0.5 * (eps_i - eps_j)
    sqrt_term = math.sqrt(half_diff * half_diff + h * h)

    dlambda_dr = alpha * h * h / sqrt_term       # = (α h²) / sqrt(Δ²+h²)
    coeff = fig.bond_order * dlambda_dr / r      # bo · dλ/dr · (1/r)
    grad[fig.i] =  coeff * v
    grad[fig.j] = -coeff * v
    return grad


def grad_three_figure(fig: ThreeFigure, atoms: Atoms,
                      atomic_table) -> np.ndarray:
    """∂Π/∂r for a 3-figure (angle figure).

    Π = cos(θ_ijk) · (ε_i + ε_j + ε_k) / 3 with j the central atom.
    ε̄ is r-independent, so ``∂Π/∂r_a = ε̄ · ∂cos θ/∂r_a`` for
    ``a ∈ {i, j, k}`` and zero elsewhere.

    Standard angle gradient (j is central; v1 = r_i-r_j, v2 = r_k-r_j;
    L1, L2 their lengths; c = cos θ = v1·v2 / (L1 L2)):

        ∂c/∂r_i = (v2/(L1·L2)) − c · v1/L1²
        ∂c/∂r_k = (v1/(L1·L2)) − c · v2/L2²
        ∂c/∂r_j = − ∂c/∂r_i − ∂c/∂r_k          (translation invariance)

    Geometric quantities are recomputed from ``atoms.get_positions()``;
    ``fig.angle_rad`` is not used.
    """
    n = len(atoms)
    grad = np.zeros((n, 3), dtype=float)
    pos = atoms.get_positions()
    v1 = pos[fig.i] - pos[fig.j]
    v2 = pos[fig.k] - pos[fig.j]
    L1 = float(np.linalg.norm(v1))
    L2 = float(np.linalg.norm(v2))
    if L1 < 1e-12 or L2 < 1e-12:
        return grad

    cos_t = float(np.dot(v1, v2) / (L1 * L2))
    inv_L1L2 = 1.0 / (L1 * L2)
    inv_L1sq = 1.0 / (L1 * L1)
    inv_L2sq = 1.0 / (L2 * L2)
    dc_di = v2 * inv_L1L2 - cos_t * v1 * inv_L1sq
    dc_dk = v1 * inv_L1L2 - cos_t * v2 * inv_L2sq
    dc_dj = -dc_di - dc_dk

    eps_i = next(s for s in atomic_table[fig.element_i].shells
                 if s.label == fig.shell_i).epsilon_eV
    eps_j = next(s for s in atomic_table[fig.element_j].shells
                 if s.label == fig.shell_j).epsilon_eV
    eps_k = next(s for s in atomic_table[fig.element_k].shells
                 if s.label == fig.shell_k).epsilon_eV
    eps_bar = (eps_i + eps_j + eps_k) / 3.0

    grad[fig.i] = eps_bar * dc_di
    grad[fig.j] = eps_bar * dc_dj
    grad[fig.k] = eps_bar * dc_dk
    return grad


def grad_four_figure(fig: FourFigure, atoms: Atoms,
                     atomic_table) -> np.ndarray:
    """∂Π/∂r for a 4-figure.

    Π = cos(2φ_ijkl) · (ε_i + ε_j + ε_k + ε_l) / 4.

    TODO(v1.1.0): implement ∂cos 2φ/∂r_a via the standard dihedral
    gradient.  Note the factor of 2 from the chain rule on cos(2φ).
    """
    return np.zeros((len(atoms), 3), dtype=float)


# --------------------------------------------------------------------- #
# Aggregation                                                            #
# --------------------------------------------------------------------- #

def compute_forces(atoms: Atoms, atomic_table, model,
                   topology: FigureSet | None = None) -> np.ndarray:
    """Total forces ``F_a = -Σ_F J_F · ∂Π_F/∂r_a`` in eV/Å.

    Parameters
    ----------
    atoms : ase.Atoms
        Current geometry.
    atomic_table : dict
        Orbital ε table (see :mod:`oce.atomic_table`).
    model : oce.fit.OCEModel
        Fitted model carrying ``feature_keys`` and ``J``.
    topology : FigureSet, optional
        Frozen reference topology. If ``None``, perceive at the
        current geometry.

    Returns
    -------
    np.ndarray, shape (n_atoms, 3)
        Forces in eV/Å.
    """
    if topology is None:
        topology = freeze_topology(atoms, atomic_table)

    n = len(atoms)
    F = np.zeros((n, 3), dtype=float)

    J_by_key = dict(zip(model.feature_keys, model.J))

    for fig in topology.one:
        J = J_by_key.get(fig.key())
        if not J:
            continue
        F -= J * grad_one_figure(fig, n, atomic_table)

    for fig in topology.two:
        J = J_by_key.get(fig.key())
        if not J:
            continue
        F -= J * grad_two_figure(fig, atoms, atomic_table)

    for fig in topology.three:
        J = J_by_key.get(fig.key())
        if not J:
            continue
        F -= J * grad_three_figure(fig, atoms, atomic_table)

    for fig in topology.four:
        J = J_by_key.get(fig.key())
        if not J:
            continue
        F -= J * grad_four_figure(fig, atoms, atomic_table)

    return F
