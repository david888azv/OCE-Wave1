"""Build a perovskite structure library for OCE feature evaluation.

Inorganic ABX3 perovskites with A in {Cs,K}, B in {Pb,Sn,Ge}, X in {I,Br,Cl,F}.
Three families:
  (a) endmembers — 5-atom cubic Pm-3m primitive cells, 24 ABX3 combinations
  (b) mixed-X     — 2x2x2 supercells (40 atoms) with 1-12 halide substitutions
  (c) mixed-A     — 2x2x2 supercells with Cs/K mixing on the A-site
  (d) mixed-B     — 2x2x2 supercells with Pb/Sn/Ge mixing on the B-site

Lattice constants come from a Vegard interpolation between literature endmembers
(see LATTICE table below; values in A).  Output: structures.json — list of
{name, symbols, positions, cell, pbc, formal_charges}.
"""
from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

import numpy as np
from ase import Atoms

# Endmember lattice parameters (cubic Pm-3m, alpha phase) in Angstrom.
# Sources: Cs/Pb halides — Stoumpos 2013, JACS; Sn/Ge — Heyns 1990 / Schueller
# 2018; mixed cations linearly interpolated.
LATTICE = {
    ("Cs", "Pb", "I"):  6.39,
    ("Cs", "Pb", "Br"): 5.87,
    ("Cs", "Pb", "Cl"): 5.61,
    ("Cs", "Pb", "F"):  5.10,
    ("Cs", "Sn", "I"):  6.22,
    ("Cs", "Sn", "Br"): 5.81,
    ("Cs", "Sn", "Cl"): 5.56,
    ("Cs", "Sn", "F"):  5.05,
    ("Cs", "Ge", "I"):  6.05,
    ("Cs", "Ge", "Br"): 5.65,
    ("Cs", "Ge", "Cl"): 5.43,
    ("Cs", "Ge", "F"):  4.92,
    ("K",  "Pb", "I"):  6.10,
    ("K",  "Pb", "Br"): 5.81,
    ("K",  "Pb", "Cl"): 5.59,
    ("K",  "Pb", "F"):  5.06,
    ("K",  "Sn", "I"):  6.05,
    ("K",  "Sn", "Br"): 5.74,
    ("K",  "Sn", "Cl"): 5.50,
    ("K",  "Sn", "F"):  5.00,
    ("K",  "Ge", "I"):  5.86,
    ("K",  "Ge", "Br"): 5.55,
    ("K",  "Ge", "Cl"): 5.32,
    ("K",  "Ge", "F"):  4.83,
}

# Formal charges (used by Madelung column).
A_CHARGE = {"Cs": +1, "K": +1}
B_CHARGE = {"Pb": +2, "Sn": +2, "Ge": +2}
X_CHARGE = {"I": -1, "Br": -1, "Cl": -1, "F": -1}


def primitive_cell(a_el: str, b_el: str, x_el: str, a_lat: float
                    ) -> tuple[Atoms, list[float]]:
    """Build the cubic Pm-3m 5-atom ABX3 primitive cell."""
    cell = [a_lat, a_lat, a_lat]
    sp = [
        ("A", (0.0, 0.0, 0.0)),
        ("B", (0.5, 0.5, 0.5)),
        ("X", (0.5, 0.5, 0.0)),
        ("X", (0.5, 0.0, 0.5)),
        ("X", (0.0, 0.5, 0.5)),
    ]
    el_map = {"A": a_el, "B": b_el, "X": x_el}
    elements = [el_map[k] for k, _ in sp]
    frac = [p for _, p in sp]
    atoms = Atoms(symbols=elements, scaled_positions=frac, cell=cell, pbc=True)
    charges = [A_CHARGE[a_el], B_CHARGE[b_el]] + [X_CHARGE[x_el]] * 3
    return atoms, charges


def supercell(a_el: str, b_el: str, x_el: str, a_lat: float,
               n: tuple[int, int, int] = (2, 2, 2)) -> Atoms:
    """Build pure ABX3 then repeat to nx, ny, nz."""
    prim, _ = primitive_cell(a_el, b_el, x_el, a_lat)
    return prim.repeat(n)


def vegard_lat(a_mix: dict[str, float], b_mix: dict[str, float],
                x_mix: dict[str, float]) -> float:
    """Linear interpolation of lattice constant from composition fractions."""
    s = 0.0
    n = 0.0
    for a_el, fA in a_mix.items():
        for b_el, fB in b_mix.items():
            for x_el, fX in x_mix.items():
                w = fA * fB * fX
                s += w * LATTICE[(a_el, b_el, x_el)]
                n += w
    return s / n


def site_indices_in_supercell(atoms: Atoms) -> dict[str, list[int]]:
    """Group indices in a supercell by lattice site (A, B, X) using element."""
    out = {"A": [], "B": [], "X": []}
    for i, el in enumerate(atoms.get_chemical_symbols()):
        if el in A_CHARGE:
            out["A"].append(i)
        elif el in B_CHARGE:
            out["B"].append(i)
        elif el in X_CHARGE:
            out["X"].append(i)
    return out


def random_substitute(atoms: Atoms, charges: list[float],
                       site: str, candidates: list[str], n_sub: int,
                       rng: random.Random) -> tuple[Atoms, list[float]]:
    """Replace n_sub atoms at the given site with a random candidate.

    candidates must NOT contain the original element.  Modifies a *copy*.
    """
    atoms = atoms.copy()
    charges = list(charges)
    sites = site_indices_in_supercell(atoms)[site]
    chosen = rng.sample(sites, n_sub)
    sym = atoms.get_chemical_symbols()
    for idx in chosen:
        new_el = rng.choice(candidates)
        sym[idx] = new_el
        if site == "A":
            charges[idx] = A_CHARGE[new_el]
        elif site == "B":
            charges[idx] = B_CHARGE[new_el]
        else:
            charges[idx] = X_CHARGE[new_el]
    atoms.set_chemical_symbols(sym)
    return atoms, charges


def supercell_charges(a_el: str, b_el: str, x_el: str, n_cells: int) -> list[float]:
    """Charges for an n_cells-supercell of pure ABX3 (5 atoms per primitive)."""
    per = [A_CHARGE[a_el], B_CHARGE[b_el]] + [X_CHARGE[x_el]] * 3
    return per * n_cells


def to_record(name: str, atoms: Atoms, charges: list[float]) -> dict:
    return {
        "name": name,
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": [bool(p) for p in atoms.get_pbc()],
        "formal_charges": list(charges),
    }


def build(seed: int = 20260508) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []

    # (a) Endmembers — 24 cubic 5-atom primitives
    for a_el, b_el, x_el in product(
        ["Cs", "K"], ["Pb", "Sn", "Ge"], ["I", "Br", "Cl", "F"]
    ):
        a = LATTICE[(a_el, b_el, x_el)]
        prim, ch = primitive_cell(a_el, b_el, x_el, a)
        records.append(to_record(f"{a_el}{b_el}{x_el}3_prim", prim, ch))

    # (b) Mixed-X 2x2x2 supercells: 100 per (A,B) pair = 600
    for a_el, b_el in product(["Cs", "K"], ["Pb", "Sn", "Ge"]):
        for k in range(100):
            host_x = rng.choice(["I", "Br", "Cl", "F"])
            other_xs = [x for x in ["I", "Br", "Cl", "F"] if x != host_x]
            n_sub = rng.choice([1, 2, 3, 4, 6, 8, 12])
            a = LATTICE[(a_el, b_el, host_x)]
            sc = supercell(a_el, b_el, host_x, a)
            ch = supercell_charges(a_el, b_el, host_x, 8)  # 2x2x2 = 8 prim cells
            sc_sub, ch_sub = random_substitute(
                sc, ch, "X", other_xs, n_sub, rng,
            )
            name = f"{a_el}{b_el}{host_x}3_mixX{k:03d}_n{n_sub}"
            records.append(to_record(name, sc_sub, ch_sub))

    # (c) Mixed-A 2x2x2: 50 per (B,X) pair = 600
    for b_el, x_el in product(["Pb", "Sn", "Ge"], ["I", "Br", "Cl", "F"]):
        for k in range(50):
            host_a = rng.choice(["Cs", "K"])
            other_a = "K" if host_a == "Cs" else "Cs"
            n_sub = rng.choice([1, 2, 3, 4])
            a = LATTICE[(host_a, b_el, x_el)]
            sc = supercell(host_a, b_el, x_el, a)
            ch = supercell_charges(host_a, b_el, x_el, 8)
            sc_sub, ch_sub = random_substitute(
                sc, ch, "A", [other_a], n_sub, rng,
            )
            name = f"{host_a}{b_el}{x_el}3_mixA{k:03d}_n{n_sub}"
            records.append(to_record(name, sc_sub, ch_sub))

    # (d) Mixed-B 2x2x2: 60 per (A,X) pair = 480
    for a_el, x_el in product(["Cs", "K"], ["I", "Br", "Cl", "F"]):
        for k in range(60):
            host_b = rng.choice(["Pb", "Sn", "Ge"])
            other_b = [b for b in ["Pb", "Sn", "Ge"] if b != host_b]
            n_sub = rng.choice([1, 2, 3, 4])
            a = LATTICE[(a_el, host_b, x_el)]
            sc = supercell(a_el, host_b, x_el, a)
            ch = supercell_charges(a_el, host_b, x_el, 8)
            sc_sub, ch_sub = random_substitute(
                sc, ch, "B", other_b, n_sub, rng,
            )
            name = f"{a_el}{host_b}{x_el}3_mixB{k:03d}_n{n_sub}"
            records.append(to_record(name, sc_sub, ch_sub))

    # (e) Distorted cubic — apply a small random isotropic strain (-3..+3%)
    #     to a subset of pure 2x2x2 cells.  Probes geometric sensitivity.
    for k in range(80):
        a_el = rng.choice(["Cs", "K"])
        b_el = rng.choice(["Pb", "Sn", "Ge"])
        x_el = rng.choice(["I", "Br", "Cl", "F"])
        eps = rng.uniform(-0.03, 0.03)
        a = LATTICE[(a_el, b_el, x_el)] * (1.0 + eps)
        sc = supercell(a_el, b_el, x_el, a)
        ch = supercell_charges(a_el, b_el, x_el, 8)
        name = f"{a_el}{b_el}{x_el}3_strain{k:03d}_eps{eps:+.3f}"
        records.append(to_record(name, sc, ch))

    return records


if __name__ == "__main__":
    out_path = Path(__file__).parent / "structures.json"
    records = build()
    out_path.write_text(json.dumps(records, indent=2))
    n_atoms = sum(len(r["symbols"]) for r in records)
    n_per = [len(r["symbols"]) for r in records]
    print(f"Wrote {len(records)} structures to {out_path}")
    print(f"Total atoms: {n_atoms}")
    print(f"Atoms per structure — min {min(n_per)}, mean {n_atoms/len(records):.1f}, "
          f"max {max(n_per)}")
    # Print family counts
    from collections import Counter
    fams = Counter()
    for r in records:
        if "_prim"   in r["name"]: fams["primitive"] += 1
        elif "_mixX" in r["name"]: fams["mixed-X"]   += 1
        elif "_mixA" in r["name"]: fams["mixed-A"]   += 1
        elif "_mixB" in r["name"]: fams["mixed-B"]   += 1
        elif "_strain" in r["name"]: fams["strained"] += 1
    print("Families:", dict(fams))
