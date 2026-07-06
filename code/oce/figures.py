"""Bond perception and figure enumeration for OCE.

A "figure" here is a subset of (atom, shell) tuples that share a sub-graph
on the molecular bond graph. Supports:
    - 1-figures: each (atom, shell) — the on-site/atomic-reference term
    - 2-figures: bonded (atom_i, shell_μ)–(atom_j, shell_ν) pairs
    - 3-figures: bonded angle i–j–k (j is the central atom; i–j and j–k
      are both bonds), shell-decorated, with the geometric angle θ_ijk
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers


# Bond-order classification by ratio (r / (r_cov_A + r_cov_B))
# These are heuristic and tuned for first/second-row organics.
BO_CUTOFFS = [
    (0.00, 0.83, 3),   # triple   (e.g. C≡C ≈ 1.20 Å, ratio ≈ 0.80)
    (0.83, 0.93, 2),   # double   (e.g. C=C ≈ 1.34 Å, ratio ≈ 0.88)
    (0.93, 1.30, 1),   # single   (e.g. C-C ≈ 1.54 Å, ratio ≈ 1.01)
    # > 1.30 → not bonded
]


@dataclass(frozen=True)
class OneFigure:
    atom_idx: int
    element: str
    shell_label: str  # 's' / 'p' / 'd'

    def key(self) -> tuple:
        # symmetry-equivalent on-site figures share this key (drops atom_idx)
        return ("1F", self.element, self.shell_label)


@dataclass(frozen=True)
class TwoFigure:
    i: int
    j: int
    element_i: str
    element_j: str
    shell_i: str
    shell_j: str
    distance: float
    bond_order: int

    def key(self) -> tuple:
        # symmetry-equivalent pair: sort by (element, shell) so (C,s)–(O,p)
        # and (O,p)–(C,s) hash to the same class
        a = (self.element_i, self.shell_i)
        b = (self.element_j, self.shell_j)
        a, b = sorted([a, b])
        return ("2F", a, b, self.bond_order)


@dataclass(frozen=True)
class FourFigure:
    """Dihedral i–j–k–l. j–k is the central bond, i is bonded to j, l to k.

    Dihedral angle φ is the rotation of the (j,k,l) plane about j–k relative
    to (i,j,k) — measured in (-π, π].  We use cos(2φ) as the geometric
    factor (consistent with V₂ torsion potentials) so eclipsed (φ=0,π) and
    staggered (φ=±π/2) configurations get distinct features.
    """
    i: int
    j: int
    k: int
    l: int
    element_i: str
    element_j: str
    element_k: str
    element_l: str
    shell_i: str
    shell_j: str
    shell_k: str
    shell_l: str
    dihedral_rad: float
    bond_order_jk: int

    def key(self) -> tuple:
        # symmetry: invert (i↔l, j↔k) — dihedral reverses sign but cos(2φ)
        # is invariant.  Sort the two end-atom labels.
        end_i = (self.element_i, self.shell_i)
        end_l = (self.element_l, self.shell_l)
        mid_j = (self.element_j, self.shell_j)
        mid_k = (self.element_k, self.shell_k)
        # mirror: pair up arms so (end_i, mid_j) ↔ (end_l, mid_k)
        a = (end_i, mid_j)
        b = (end_l, mid_k)
        a, b = sorted([a, b])
        return ("4F", a, b, self.bond_order_jk)


@dataclass(frozen=True)
class ThreeFigure:
    """Angle figure i–j–k. j is the central atom; i–j and j–k are bonds.

    `ring_size` is the smallest ring (≤6) that contains the angle, or 0 for
    chain angles. Including it in the symmetry key separates strained
    ring angles (e.g. 60° in 3-membered) from chain angles (~110°), which
    is decisive for distinguishing oxirane-like vs aldehyde-like isomers.
    """
    i: int
    j: int
    k: int
    element_i: str
    element_j: str
    element_k: str
    shell_i: str
    shell_j: str
    shell_k: str
    angle_rad: float
    bond_order_ij: int
    bond_order_jk: int
    ring_size: int = 0

    def key(self) -> tuple:
        # symmetry-equivalent angle: keep j (center) fixed, sort the two
        # arms (i, k) so the figure (A-B-C) and (C-B-A) hash identically.
        arm_i = (self.element_i, self.shell_i, self.bond_order_ij)
        arm_k = (self.element_k, self.shell_k, self.bond_order_jk)
        arm_a, arm_b = sorted([arm_i, arm_k])
        center = (self.element_j, self.shell_j)
        return ("3F", center, arm_a, arm_b, self.ring_size)


def perceive_bonds(atoms: Atoms, scale: float = 1.30) -> list[TwoFigure]:
    """Infer bonds from interatomic distances using covalent radii.

    For each pair (i,j), compute r/(r_cov_i + r_cov_j); if within a cutoff,
    classify bond order via BO_CUTOFFS.
    """
    n = len(atoms)
    pos = atoms.get_positions()
    Z = atoms.get_atomic_numbers()
    syms = atoms.get_chemical_symbols()
    bonds: list[tuple[int, int, float, int]] = []  # (i, j, dist, BO)
    for i, j in combinations(range(n), 2):
        d = float(np.linalg.norm(pos[i] - pos[j]))
        rcov = covalent_radii[Z[i]] + covalent_radii[Z[j]]
        ratio = d / rcov
        if ratio > scale:
            continue
        bo = None
        for lo, hi, order in BO_CUTOFFS:
            if lo <= ratio < hi:
                bo = order
                break
        if bo is None:
            continue
        bonds.append((i, j, d, bo))

    # Heuristic correction: for organic molecules, an atom's total bond order
    # should not exceed its typical valence. If it does, downgrade longest
    # high-order bonds to single.
    typical_valence = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1}
    bo_sum = [0] * n
    # Sort by descending BO so we accept high-order bonds first
    bonds.sort(key=lambda x: (-x[3], x[2]))
    accepted: list[tuple[int, int, float, int]] = []
    for i, j, d, bo in bonds:
        vi = typical_valence.get(syms[i], 99)
        vj = typical_valence.get(syms[j], 99)
        if bo_sum[i] + bo > vi or bo_sum[j] + bo > vj:
            # try downgrading to single
            if bo_sum[i] + 1 <= vi and bo_sum[j] + 1 <= vj:
                bo = 1
            else:
                continue
        bo_sum[i] += bo
        bo_sum[j] += bo
        accepted.append((i, j, d, bo))
    return accepted


def angle_ring_size(i: int, j: int, k: int,
                    neighbors: dict[int, list[int]],
                    max_size: int = 6) -> int:
    """Smallest ring (≤max_size) that contains the angle i-j-k. 0 if none.

    A ring "contains" the angle iff there is a cycle through both edges
    i–j and j–k.  Ring size = 2 + shortest path from i to k that does NOT
    pass through j.  Direct i–k bond → triangle (size 3).
    """
    from collections import deque
    if k in neighbors.get(i, []):
        return 3
    visited = {i, j}
    queue = deque([(i, 0)])
    while queue:
        node, dist = queue.popleft()
        if dist + 1 > max_size - 2:
            continue
        for nb in neighbors.get(node, []):
            if nb == j:
                continue
            if nb == k:
                return 2 + (dist + 1)
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return 0


def shells_for_element(element: str, atomic_table) -> list[str]:
    """Return shell labels available for an element from the atomic table."""
    return [sh.label for sh in atomic_table[element].shells]


def enumerate_figures(
    atoms: Atoms,
    atomic_table: dict,
    include_angles: bool = True,
    include_dihedrals: bool = False,
) -> tuple[list[OneFigure], list[TwoFigure], list[ThreeFigure],
            list[FourFigure]]:
    """Enumerate all 1-, 2-, 3-, and (optionally) 4-figures present in a molecule.

    Returns a 4-tuple even when dihedrals are off (last element is empty).
    """
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    one_figs: list[OneFigure] = []
    for idx, el in enumerate(syms):
        for sh in shells_for_element(el, atomic_table):
            one_figs.append(OneFigure(atom_idx=idx, element=el, shell_label=sh))

    bonds = perceive_bonds(atoms)
    two_figs: list[TwoFigure] = []
    for i, j, d, bo in bonds:
        ei, ej = syms[i], syms[j]
        for shi in shells_for_element(ei, atomic_table):
            for shj in shells_for_element(ej, atomic_table):
                two_figs.append(TwoFigure(
                    i=i, j=j,
                    element_i=ei, element_j=ej,
                    shell_i=shi, shell_j=shj,
                    distance=d, bond_order=bo,
                ))

    three_figs: list[ThreeFigure] = []
    if include_angles:
        # Adjacency with bond order
        adj: dict[int, list[tuple[int, int]]] = {}
        # Plain neighbor list (no BO) for ring detection
        nbr: dict[int, list[int]] = {}
        for i, j, _d, bo in bonds:
            adj.setdefault(i, []).append((j, bo))
            adj.setdefault(j, []).append((i, bo))
            nbr.setdefault(i, []).append(j)
            nbr.setdefault(j, []).append(i)

        for j_center, neighbors in adj.items():
            if len(neighbors) < 2:
                continue
            for (i_idx, bo_ij), (k_idx, bo_jk) in combinations(neighbors, 2):
                v1 = pos[i_idx] - pos[j_center]
                v2 = pos[k_idx] - pos[j_center]
                cos_t = float(np.dot(v1, v2)
                              / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
                cos_t = max(-1.0, min(1.0, cos_t))
                theta = float(np.arccos(cos_t))
                rsize = angle_ring_size(i_idx, j_center, k_idx, nbr,
                                         max_size=6)
                ei, ej, ek = syms[i_idx], syms[j_center], syms[k_idx]
                for shi in shells_for_element(ei, atomic_table):
                    for shj in shells_for_element(ej, atomic_table):
                        for shk in shells_for_element(ek, atomic_table):
                            three_figs.append(ThreeFigure(
                                i=i_idx, j=j_center, k=k_idx,
                                element_i=ei, element_j=ej, element_k=ek,
                                shell_i=shi, shell_j=shj, shell_k=shk,
                                angle_rad=theta,
                                bond_order_ij=bo_ij,
                                bond_order_jk=bo_jk,
                                ring_size=rsize,
                            ))

    four_figs: list[FourFigure] = []
    if include_dihedrals and include_angles:
        # dihedrals i-j-k-l: enumerate over central bonds (j,k) and pick
        # one neighbor of j (≠k) and one neighbor of k (≠j)
        for j, k, _d, bo_jk in bonds:
            for i_idx, _bo in adj.get(j, []):
                if i_idx == k:
                    continue
                for l_idx, _bo in adj.get(k, []):
                    if l_idx == j or l_idx == i_idx:
                        continue
                    # compute dihedral
                    p_i = pos[i_idx]; p_j = pos[j]
                    p_k = pos[k]; p_l = pos[l_idx]
                    b1 = p_j - p_i
                    b2 = p_k - p_j
                    b3 = p_l - p_k
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
                    x = float(np.dot(n1, n2))
                    y = float(np.dot(m1, n2))
                    phi = float(np.arctan2(y, x))
                    ei, ej, ek, el = syms[i_idx], syms[j], syms[k], syms[l_idx]
                    for shi in shells_for_element(ei, atomic_table):
                        for shj in shells_for_element(ej, atomic_table):
                            for shk in shells_for_element(ek, atomic_table):
                                for shl in shells_for_element(el, atomic_table):
                                    four_figs.append(FourFigure(
                                        i=i_idx, j=j, k=k, l=l_idx,
                                        element_i=ei, element_j=ej,
                                        element_k=ek, element_l=el,
                                        shell_i=shi, shell_j=shj,
                                        shell_k=shk, shell_l=shl,
                                        dihedral_rad=phi,
                                        bond_order_jk=bo_jk,
                                    ))

    return one_figs, two_figs, three_figs, four_figs


if __name__ == "__main__":
    from ase.build import molecule
    from oce.atomic_table import load_table
    from pathlib import Path

    table = load_table(Path(__file__).resolve().parents[1]
                       / "data" / "atoms" / "atomic_table.json")

    import math
    for name in ["H2O", "CH4", "C2H4", "C2H2", "CH3OH", "HCOOH"]:
        atoms = molecule(name)
        one_figs, two_figs, three_figs, _four = enumerate_figures(atoms, table)
        bonds = perceive_bonds(atoms)
        print(f"\n{name}: {len(atoms)} atoms, "
              f"{len(bonds)} bonds, "
              f"{len(one_figs)} 1-figs, {len(two_figs)} 2-figs, "
              f"{len(three_figs)} 3-figs")
        # Print unique angles only (one shell-decorated rep per geometric angle)
        seen = set()
        for f in three_figs:
            geom = (f.i, f.j, f.k)
            if geom in seen:
                continue
            seen.add(geom)
            syms = atoms.get_chemical_symbols()
            print(f"   {syms[f.i]}{f.i}-{syms[f.j]}{f.j}-{syms[f.k]}{f.k}  "
                  f"θ = {math.degrees(f.angle_rad):6.2f}°  "
                  f"BO = {f.bond_order_ij}/{f.bond_order_jk}")
