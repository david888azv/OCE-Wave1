"""Site-percolation cluster sampler at p = p_c.

Given a Lattice (lattices.py), generate Bernoulli site-occupations with
probability p, identify the largest connected cluster of occupied sites
and return it as a sub-lattice (positions + adjacency restricted to the
cluster).  We retain the original (lattice) neighbour list so the
"radical count" of each cluster atom can be inferred from the number of
*occupied* neighbours (out of 3 for honeycomb).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lattices import Lattice


@dataclass
class PercoCluster:
    """Largest connected component of occupied sites at p = p_c."""
    lattice_name: str
    L: int
    p: float
    seed: int
    site_idx: list[int]                    # original lattice indices in cluster
    positions: np.ndarray                  # (N, 3)
    cluster_neighbours: list[list[int]]    # local indexing within the cluster
    n_dangling: list[int]                  # per-atom = 3 − (intra-cluster degree)
    bonds: list[tuple[int, int]]           # local-index bonds (single sigma)

    @property
    def n_atoms(self) -> int:
        return len(self.site_idx)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)


def sample_percolation(lattice: Lattice, p: float, seed: int) -> PercoCluster:
    """Site-percolate `lattice` at occupation prob p; keep the largest cluster."""
    rng = np.random.default_rng(seed)
    occupied = rng.random(lattice.n_sites) < p

    # Largest connected component
    parent_cluster = _largest_component(occupied, lattice.neighbours)
    site_idx = sorted(parent_cluster)
    pos = lattice.positions[site_idx]
    local_of = {orig: loc for loc, orig in enumerate(site_idx)}

    cluster_nbrs: list[list[int]] = [[] for _ in site_idx]
    bonds_set: set[tuple[int, int]] = set()
    for orig, loc in local_of.items():
        for nb in lattice.neighbours[orig]:
            if nb in local_of:
                cluster_nbrs[loc].append(local_of[nb])
                a, b = sorted((loc, local_of[nb]))
                bonds_set.add((a, b))

    # Per-atom dangling = (original lattice degree) − (intra-cluster degree).
    # Works for mixed-coordination lattices such as pentagraphene.
    n_dangling = []
    for orig, loc in local_of.items():
        full_z_here = len(lattice.neighbours[orig])
        intra_deg = len(cluster_nbrs[loc])
        n_dangling.append(full_z_here - intra_deg)

    return PercoCluster(
        lattice_name=lattice.name,
        L=lattice.L,
        p=p,
        seed=seed,
        site_idx=site_idx,
        positions=pos,
        cluster_neighbours=cluster_nbrs,
        n_dangling=n_dangling,
        bonds=sorted(bonds_set),
    )


def sample_bond_percolation(lattice: Lattice, p: float, seed: int) -> PercoCluster:
    """Bond percolation: each lattice bond is kept with probability p.

    The cluster is the set of sites connected via occupied bonds.  Sites
    with no occupied bonds are isolated and excluded from the largest
    cluster (which contains ≥ 2 sites by construction unless the
    realisation is trivial).
    """
    rng = np.random.default_rng(seed)
    # build bond list
    bond_list: set[tuple[int, int]] = set()
    for i, nbs in enumerate(lattice.neighbours):
        for j in nbs:
            bond_list.add((min(i, j), max(i, j)))
    bonds = list(bond_list)
    keep = rng.random(len(bonds)) < p
    occupied_bonds = [bonds[k] for k, ok in enumerate(keep) if ok]

    # build adjacency of occupied bonds only
    occ_nbrs: dict[int, list[int]] = {}
    for i, j in occupied_bonds:
        occ_nbrs.setdefault(i, []).append(j)
        occ_nbrs.setdefault(j, []).append(i)

    # connected components
    visited: set[int] = set()
    best: set[int] = set()
    for s in occ_nbrs:
        if s in visited:
            continue
        comp: set[int] = set()
        queue = deque([s])
        visited.add(s)
        while queue:
            u = queue.popleft()
            comp.add(u)
            for v in occ_nbrs.get(u, []):
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        if len(comp) > len(best):
            best = comp
    site_idx = sorted(best)
    pos = lattice.positions[site_idx]
    local_of = {orig: loc for loc, orig in enumerate(site_idx)}
    occ_set = set(occupied_bonds)

    cluster_nbrs: list[list[int]] = [[] for _ in site_idx]
    bonds_local: set[tuple[int, int]] = set()
    for i, j in occupied_bonds:
        if i in local_of and j in local_of:
            li, lj = local_of[i], local_of[j]
            cluster_nbrs[li].append(lj)
            cluster_nbrs[lj].append(li)
            bonds_local.add((min(li, lj), max(li, lj)))

    # Per-atom dangling derived from original lattice degree.
    n_dangling = [len(lattice.neighbours[site_idx[loc]]) - len(cluster_nbrs[loc])
                  for loc in range(len(site_idx))]

    return PercoCluster(
        lattice_name=lattice.name, L=lattice.L, p=p, seed=seed,
        site_idx=site_idx, positions=pos,
        cluster_neighbours=cluster_nbrs,
        n_dangling=n_dangling,
        bonds=sorted(bonds_local),
    )


def sample_many_bond(lattice: Lattice, p: float, n_samples: int,
                     base_seed: int = 0,
                     min_size: int = 4,
                     max_attempts: int = 50) -> list[PercoCluster]:
    out: list[PercoCluster] = []
    seed = base_seed
    attempts = 0
    while len(out) < n_samples and attempts < n_samples * max_attempts:
        c = sample_bond_percolation(lattice, p, seed)
        if c.n_atoms >= min_size:
            out.append(c)
        seed += 1
        attempts += 1
    return out


def _largest_component(occupied: np.ndarray,
                       neighbours: list[list[int]]) -> set[int]:
    """Return the set of indices forming the largest occupied component."""
    n = len(occupied)
    visited = np.zeros(n, dtype=bool)
    best: set[int] = set()
    for s in range(n):
        if not occupied[s] or visited[s]:
            continue
        comp: set[int] = set()
        queue = deque([s])
        visited[s] = True
        while queue:
            u = queue.popleft()
            comp.add(u)
            for v in neighbours[u]:
                if occupied[v] and not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if len(comp) > len(best):
            best = comp
    return best


def sample_many(lattice: Lattice, p: float, n_samples: int,
                base_seed: int = 0,
                min_size: int = 2,
                max_attempts: int = 50) -> list[PercoCluster]:
    """Sample n_samples clusters; reject realisations smaller than min_size."""
    out: list[PercoCluster] = []
    seed = base_seed
    attempts = 0
    while len(out) < n_samples and attempts < n_samples * max_attempts:
        c = sample_percolation(lattice, p, seed)
        if c.n_atoms >= min_size:
            out.append(c)
        seed += 1
        attempts += 1
    return out


if __name__ == "__main__":
    from lattices import build_honeycomb, P_C
    p_c = P_C["honeycomb"]
    print(f"p_c = {p_c}")
    for L in (4, 6, 8, 12, 16):
        lat = build_honeycomb(L)
        sizes = []
        for seed in range(40):
            c = sample_percolation(lat, p_c, seed)
            sizes.append(c.n_atoms)
        sizes = np.array(sizes)
        print(f"  L={L:2d}  N_total={lat.n_sites:4d}  "
              f"⟨N_largest⟩ = {sizes.mean():6.1f} ± {sizes.std():.1f}   "
              f"(L^1.896 ≈ {L**1.896:6.1f})")
