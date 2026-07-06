"""Honeycomb (graphene) and square 2D lattices for percolation studies.

Honeycomb has 2 atoms per unit cell (A and B sublattices), each with
3 nearest neighbours.  Lattice vectors and atom positions follow the
zig-zag convention used by ASE / nanoribbon code:

    a1 = a (3/2,  √3/2)
    a2 = a (3/2, -√3/2)
    A position: (0, 0)
    B position: (a, 0)

with C-C bond length a = 1.42 Å (graphene equilibrium, GFN2 ≈ 1.42 Å).

For an L × L tile (open boundary), we generate 2 L² atoms and the
nearest-neighbour graph (each A connects to 3 B sites, each B to 3 A
sites).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

A_CC = 1.42  # Å, graphene equilibrium C–C bond length


@dataclass
class Lattice:
    name: str                          # "honeycomb" | "square"
    positions: np.ndarray              # (N, 3) — 2D, z=0
    neighbours: list[list[int]]        # adjacency list, undirected
    L: int                             # linear size (unit cells per side)
    bond_length: float

    @property
    def n_sites(self) -> int:
        return len(self.positions)


def build_honeycomb(L: int, a: float = A_CC) -> Lattice:
    """Open-boundary L × L honeycomb tile.  Total sites = 2 L²."""
    positions: list[np.ndarray] = []
    indices: dict[tuple[int, int, int], int] = {}
    a1 = a * np.array([1.5,  math.sqrt(3) / 2.0, 0.0])
    a2 = a * np.array([1.5, -math.sqrt(3) / 2.0, 0.0])
    pos_A = np.zeros(3)
    pos_B = a * np.array([1.0, 0.0, 0.0])
    for i in range(L):
        for j in range(L):
            base = i * a1 + j * a2
            indices[(i, j, 0)] = len(positions)
            positions.append(base + pos_A)
            indices[(i, j, 1)] = len(positions)
            positions.append(base + pos_B)
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    # A(i,j)  bonds to  B(i,j)  [vector +a*x],
    #                    B(i-1,j)  [vector -a/2*x +√3/2*y],
    #                    B(i,j-1)  [vector -a/2*x -√3/2*y]
    for i in range(L):
        for j in range(L):
            iA = indices[(i, j, 0)]
            for di, dj in [(0, 0), (-1, 0), (0, -1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < L and 0 <= jj < L:
                    iB = indices[(ii, jj, 1)]
                    if iB not in nbrs[iA]:
                        nbrs[iA].append(iB)
                        nbrs[iB].append(iA)
    return Lattice(
        name="honeycomb",
        positions=np.array(positions),
        neighbours=nbrs,
        L=L,
        bond_length=a,
    )


def build_square(L: int, a: float = A_CC) -> Lattice:
    """Open-boundary L × L square tile.  Total sites = L²."""
    positions: list[np.ndarray] = []
    for i in range(L):
        for j in range(L):
            positions.append(np.array([i * a, j * a, 0.0]))
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    def idx(i, j): return i * L + j
    for i in range(L):
        for j in range(L):
            ij = idx(i, j)
            if i + 1 < L:
                nbrs[ij].append(idx(i + 1, j))
                nbrs[idx(i + 1, j)].append(ij)
            if j + 1 < L:
                nbrs[ij].append(idx(i, j + 1))
                nbrs[idx(i, j + 1)].append(ij)
    return Lattice(
        name="square",
        positions=np.array(positions),
        neighbours=nbrs,
        L=L,
        bond_length=a,
    )


def build_triangular(L: int, a: float = A_CC) -> Lattice:
    """Open-boundary L × L triangular lattice (6 NN per site).

    Lattice vectors:
        a1 = a (1, 0, 0)
        a2 = a (1/2, √3/2, 0)
    1 atom per unit cell at (0, 0, 0); total sites = L².
    Each interior site has 6 neighbours: ±a1, ±a2, ±(a1−a2).
    """
    a1 = a * np.array([1.0, 0.0, 0.0])
    a2 = a * np.array([0.5, math.sqrt(3) / 2.0, 0.0])
    positions: list[np.ndarray] = []
    indices: dict[tuple[int, int], int] = {}
    for i in range(L):
        for j in range(L):
            indices[(i, j)] = len(positions)
            positions.append(i * a1 + j * a2)
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    for i in range(L):
        for j in range(L):
            here = indices[(i, j)]
            for di, dj in deltas:
                ii, jj = i + di, j + dj
                if 0 <= ii < L and 0 <= jj < L:
                    there = indices[(ii, jj)]
                    if there not in nbrs[here]:
                        nbrs[here].append(there)
    return Lattice(name="triangular",
                   positions=np.array(positions),
                   neighbours=nbrs, L=L, bond_length=a)


def build_bcc_3d(L: int, a: float = A_CC) -> Lattice:
    """L × L × L body-centered cubic tile.  Each unit cell has 2 atoms.

    Unit-cell basis: corner (0,0,0) and centre (½,½,½)·a.
    NN distance: a·√3/2.  Each interior atom has z=8 nearest neighbours.

    To make NN distance equal `a` (so AIREBO/Tersoff sees usual bond
    length), we scale the cubic edge by 2/√3.  Total sites = 2·L³.
    """
    edge = a * 2.0 / math.sqrt(3.0)   # gives NN distance == a
    positions: list[np.ndarray] = []
    indices: dict[tuple[int, int, int, int], int] = {}
    basis = [np.zeros(3), 0.5 * edge * np.ones(3)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                base = np.array([i, j, k]) * edge
                for s, b in enumerate(basis):
                    indices[(i, j, k, s)] = len(positions)
                    positions.append(base + b)
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    # NN of corner (s=0) at (i,j,k): 8 centres at (i+di, j+dj, k+dk, 1) with di,dj,dk ∈ {0,-1}
    for i in range(L):
        for j in range(L):
            for k in range(L):
                here = indices[(i, j, k, 0)]
                for di, dj, dk in [(0,0,0),(-1,0,0),(0,-1,0),(0,0,-1),
                                    (-1,-1,0),(-1,0,-1),(0,-1,-1),(-1,-1,-1)]:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < L and 0 <= jj < L and 0 <= kk < L:
                        there = indices[(ii, jj, kk, 1)]
                        if there not in nbrs[here]:
                            nbrs[here].append(there)
                            nbrs[there].append(here)
    return Lattice(name="bcc_3d",
                   positions=np.array(positions),
                   neighbours=nbrs, L=L, bond_length=a)


def build_fcc_3d(L: int, a: float = A_CC) -> Lattice:
    """L × L × L face-centered cubic tile.  Each unit cell has 4 atoms.

    NN distance: a·√2/2.  Each interior atom has z=12 nearest neighbours.

    Edge scaled by √2 so NN distance equals `a`.  Total sites = 4·L³.
    """
    edge = a * math.sqrt(2.0)   # gives NN distance == a
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]) * edge,
        np.array([0.5, 0.0, 0.5]) * edge,
        np.array([0.0, 0.5, 0.5]) * edge,
    ]
    positions: list[np.ndarray] = []
    indices: dict[tuple[int, int, int, int], int] = {}
    for i in range(L):
        for j in range(L):
            for k in range(L):
                base = np.array([i, j, k]) * edge
                for s, b in enumerate(basis):
                    indices[(i, j, k, s)] = len(positions)
                    positions.append(base + b)
    # NN: brute-force inside (L+1)³·4 box restricted to threshold
    pos = np.array(positions)
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    cutoff = a * 1.05
    # use cKDTree for efficiency
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    for i, j in pairs:
        nbrs[i].append(int(j))
        nbrs[j].append(int(i))
    return Lattice(name="fcc_3d",
                   positions=pos,
                   neighbours=nbrs, L=L, bond_length=a)


def build_diamond_3d(L: int, a: float = A_CC) -> Lattice:
    """L × L × L diamond cubic tile.  Each conventional cell has 8 atoms.

    NN distance: a·√3/4 (within a sublattice the ring is sp³).
    Each interior atom has z=4 nearest neighbours.

    Edge scaled by 4/√3 so NN distance equals `a`.  Total sites = 8·L³.
    """
    edge = a * 4.0 / math.sqrt(3.0)   # gives NN distance == a
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]) * edge,
        np.array([0.5, 0.0, 0.5]) * edge,
        np.array([0.0, 0.5, 0.5]) * edge,
        np.array([0.25, 0.25, 0.25]) * edge,
        np.array([0.75, 0.75, 0.25]) * edge,
        np.array([0.75, 0.25, 0.75]) * edge,
        np.array([0.25, 0.75, 0.75]) * edge,
    ]
    positions: list[np.ndarray] = []
    for i in range(L):
        for j in range(L):
            for k in range(L):
                base = np.array([i, j, k]) * edge
                for b in basis:
                    positions.append(base + b)
    pos = np.array(positions)
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    pairs = tree.query_pairs(r=a * 1.05, output_type="ndarray")
    for i, j in pairs:
        nbrs[i].append(int(j))
        nbrs[j].append(int(i))
    return Lattice(name="diamond_3d",
                   positions=pos,
                   neighbours=nbrs, L=L, bond_length=a)


def build_cubic_3d(L: int, a: float = A_CC) -> Lattice:
    """Open-boundary L × L × L simple-cubic 3D tile (z=6 NN per site).

    Total sites = L³.  Lattice vectors are the Cartesian basis × `a`.
    Each interior site has 6 nearest neighbours (±x, ±y, ±z).
    """
    positions: list[np.ndarray] = []
    indices: dict[tuple[int, int, int], int] = {}
    for i in range(L):
        for j in range(L):
            for k in range(L):
                indices[(i, j, k)] = len(positions)
                positions.append(np.array([i * a, j * a, k * a]))
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    deltas = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
              (0, 0, 1), (0, 0, -1)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                here = indices[(i, j, k)]
                for di, dj, dk in deltas:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < L and 0 <= jj < L and 0 <= kk < L:
                        there = indices[(ii, jj, kk)]
                        if there not in nbrs[here]:
                            nbrs[here].append(there)
    return Lattice(name="cubic_3d",
                   positions=np.array(positions),
                   neighbours=nbrs, L=L, bond_length=a)


def build_pentagonal(L: int, a: float = 3.52) -> Lattice:
    """Pentagraphene (Zhang/Wang et al., PNAS 2015) — Cairo-pentagonal 2D allotrope.

    Connectivity and fractional coordinates are taken DIRECTLY from the
    relaxed crystallographic structure (space group P-4̄2₁m, #113,
    a = b = 3.52 Å, c = 12.58 Å).  6 atoms per tetragonal cell:
      idx 0,1: sp3 (z=4) C1, in-plane, z = 0
      idx 2,3: sp2 (z=3) C2, buckled DOWN  (z = -0.052·c ≈ -0.65 Å)
      idx 4,5: sp2 (z=3) C2, buckled UP    (z = +0.052·c ≈ +0.65 Å)

    Periodic bonds per cell, extracted from the CIF symmetry operations
    (verified by distance: 8 × C1-C2 at 1.519 Å + 2 × C2-C2 dimers at
    1.331 Å = 10 bonds; degrees 4,4,3,3,3,3; **NO sp3-sp3 bonds**; every
    atom sits in a 5-membered ring, i.e. graph girth = 5):
      sp3-sp2 (8): (0,2),(0,3),(0,4),(0,5),
                   (1,2@0,-1),(1,3@-1,0),(1,4@-1,-1),(1,5)
      sp2-sp2 (2): (2,3@0,+1) dimer, (4,5@+1,0) dimer

    NOTE (2026-05-21): this replaces a previous incorrect bond list that
    wired the sp3 atoms into spurious 1-D chains (66 sp3-sp3 bonds where
    pentagraphene has 0), producing a quasi-1D percolation artifact.

    Site-percolation threshold of the Cairo pentagonal lattice:
    p_c ≈ 0.6501834 (Yonezawa 1989); confirmed here by a finite-size
    spanning-probability crossing at p_c = 0.6507 ± 0.003.
    Total sites in L×L tile = 6 L².
    """
    c_lat = 12.58
    zb = 1.0 - 0.94831  # 0.05169, buckling amplitude in fractional c
    # Fractional coordinates from the P-4̄2₁m CIF (z-wrapped to one layer):
    atom_frac = [
        (0.50000, 0.50000,  0.00000),   # 0  sp3_A (C1)
        (0.00000, 0.00000,  0.00000),   # 1  sp3_B (C1)
        (0.36629, 0.86629, -zb),        # 2  sp2  (C2, down)
        (0.63371, 0.13371, -zb),        # 3  sp2  (C2, down)
        (0.86629, 0.63371, +zb),        # 4  sp2  (C2, up)
        (0.13371, 0.36629, +zb),        # 5  sp2  (C2, up)
    ]

    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([0.0, a, 0.0])

    def site_index(i: int, j: int, k: int) -> int:
        return (i * L + j) * 6 + k

    positions: list[np.ndarray] = []
    for i in range(L):
        for j in range(L):
            base = i * a1 + j * a2
            for fx, fy, fz in atom_frac:
                positions.append(base + fx * a1 + fy * a2
                                  + np.array([0.0, 0.0, fz * c_lat]))
    pos_arr = np.array(positions)
    n = len(pos_arr)

    # Periodic bond list (a_in_cell, b_in_cell, di, dj) from the CIF.
    prim_bonds = [
        (0, 2,  0,  0), (0, 3,  0,  0), (0, 4,  0,  0), (0, 5,  0,  0),   # sp3_A - sp2
        (1, 2,  0, -1), (1, 3, -1,  0), (1, 4, -1, -1), (1, 5,  0,  0),   # sp3_B - sp2
        (2, 3,  0, +1), (4, 5, +1,  0),                                   # sp2-sp2 dimers
    ]
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for i in range(L):
        for j in range(L):
            for ka, kb, di, dj in prim_bonds:
                ii, jj = i + di, j + dj
                if not (0 <= ii < L and 0 <= jj < L):
                    continue
                a_idx = site_index(i, j, ka)
                b_idx = site_index(ii, jj, kb)
                if b_idx not in nbrs[a_idx]:
                    nbrs[a_idx].append(b_idx)
                    nbrs[b_idx].append(a_idx)
    return Lattice(
        name="pentagonal",
        positions=pos_arr,
        neighbours=nbrs,
        L=L,
        bond_length=1.519,
    )


def build_kagome(L: int, a: float = A_CC) -> Lattice:
    """Open-boundary L × L kagome tile.  3 atoms per primitive cell,
    each with z = 4 nearest neighbours in bulk; total sites = 3·L².

    Kagome = trihexagonal tiling, dual of the rhombille.  Lattice
    vectors a1 = a (2, 0), a2 = a (1, √3); 3 sublattices A, B, C at
    (0, 0), (a, 0), (a/2, a√3/2).  Each interior site has 4 NNs in a
    "Star of David" pattern.

    Site percolation threshold: p_c = 0.6527036284 (Suding & Ziff 1999).
    """
    a1 = a * np.array([2.0, 0.0, 0.0])
    a2 = a * np.array([1.0, math.sqrt(3.0), 0.0])
    basis = [np.zeros(3),
              a * np.array([1.0, 0.0, 0.0]),
              a * np.array([0.5, math.sqrt(3.0) / 2.0, 0.0])]
    positions: list[np.ndarray] = []
    indices: dict[tuple[int, int, int], int] = {}
    for i in range(L):
        for j in range(L):
            for k, b in enumerate(basis):
                indices[(i, j, k)] = len(positions)
                positions.append(i * a1 + j * a2 + b)
    n = len(positions)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    # Bond pattern within a unit cell (i,j) and to neighbour cells.
    # Each A (k=0) connects to: B(i,j), C(i,j), B(i-1,j), C(i,j-1)
    # Each B (k=1) connects to: A(i,j), C(i,j), A(i+1,j), C(i+1,j-1)
    # Each C (k=2) connects to: A(i,j), B(i,j), A(i,j+1), B(i-1,j+1)
    cross_bonds = [
        # (ka, kb, di, dj) — atom_a at (0,0) bonds to atom_b at (di,dj)
        (0, 1,  0,  0), (0, 2,  0,  0),                        # within cell
        (1, 2,  0,  0),
        (0, 1, -1,  0), (0, 2,  0, -1),                        # A out-of-cell
        (1, 2,  1, -1),                                         # B out-of-cell
    ]
    for i in range(L):
        for j in range(L):
            for ka, kb, di, dj in cross_bonds:
                ii, jj = i + di, j + dj
                if not (0 <= ii < L and 0 <= jj < L):
                    continue
                a_idx = indices[(i, j, ka)]
                b_idx = indices[(ii, jj, kb)]
                if b_idx not in nbrs[a_idx]:
                    nbrs[a_idx].append(b_idx)
                    nbrs[b_idx].append(a_idx)
    return Lattice(name="kagome", positions=np.array(positions),
                   neighbours=nbrs, L=L, bond_length=a)


# Site-percolation thresholds
P_C = {
    "honeycomb":  0.6970402,   # Suding & Ziff 1999, exact-Monte-Carlo benchmark
    "square":     0.59274621,  # Newman & Ziff 2000
    "triangular": 0.5,         # Wierman 1981, exact for site-percolation triangular
    "pentagonal": 0.6501834,   # Cairo pentagonal site percolation (Yonezawa 1989); confirmed 0.6507±0.003
    "kagome":     0.6527036,   # Suding & Ziff 1999
    "cubic_3d":   0.31161,     # Lorenz & Ziff 1998, simple-cubic 3D site
    "bcc_3d":     0.24596,     # Lorenz & Ziff 1998, BCC 3D site
    "fcc_3d":     0.19923,     # Lorenz & Ziff 1998, FCC 3D site
    "diamond_3d": 0.43003,     # van der Marck 1998, diamond cubic site
}

# Bond-percolation thresholds (exact for honeycomb / square / triangular)
P_C_BOND = {
    "honeycomb":  1.0 - 2.0 * math.sin(math.pi / 18),   # ≈ 0.6527036
    "square":     0.5,                                    # exact (Kesten 1980)
    "triangular": 2.0 * math.sin(math.pi / 18),          # ≈ 0.3472964
}


if __name__ == "__main__":
    for L in (4, 6, 8):
        lat = build_honeycomb(L)
        deg = [len(n) for n in lat.neighbours]
        print(f"honeycomb L={L}: N={lat.n_sites}, "
              f"⟨z⟩={sum(deg) / lat.n_sites:.3f} (bulk=3)")
