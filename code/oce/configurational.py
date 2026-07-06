"""Configurational CE example: H/F decoration of benzene.

Six aromatic carbons form a fixed hexagonal scaffold; each of the 6 ortho
positions is either H or F.  This gives 2⁶ = 64 binary configurations σ.
By C6v symmetry there are 13 unique necklaces (orbits under D6h):

   C6H6  (1)
   C6H5F (1)
   C6H4F2 — ortho, meta, para (3)
   C6H3F3 — 1,2,3-, 1,2,4-, 1,3,5- (3)
   C6H2F4 — 1,2,3,4-, 1,2,3,5-, 1,2,4,5- (3)
   C6HF5  (1)
   C6F6   (1)

For OCE we generate ALL 64 configurations (positions matter for the
x,y coordinates in the design matrix even if the energy is the same),
optimise each with xtb, and ask whether the trained model preserves
the chemical-symmetry-equivalent ordering and ranks the 3 disubstituted
isomers correctly (the standard ortho/meta/para test).
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms

from oce.xtb_runner import xtb_energy

# Geometry constants (Å) — slightly idealised, xtb will relax
R_CC = 1.395
R_CH = 1.087
R_CF = 1.355
RING_RADIUS = R_CC  # carbons sit on a circle of this radius


def benzene_decoration(mask: tuple[int, ...]) -> Atoms:
    """Build a planar benzene-H/F geometry. mask[i]=1 → F at position i."""
    elements = []
    positions = []
    for i in range(6):
        ang = 2.0 * math.pi * i / 6.0
        cx = RING_RADIUS * math.cos(ang)
        cy = RING_RADIUS * math.sin(ang)
        elements.append("C")
        positions.append([cx, cy, 0.0])
    for i in range(6):
        ang = 2.0 * math.pi * i / 6.0
        if mask[i]:
            r = RING_RADIUS + R_CF
            el = "F"
        else:
            r = RING_RADIUS + R_CH
            el = "H"
        ex = r * math.cos(ang)
        ey = r * math.sin(ang)
        elements.append(el)
        positions.append([ex, ey, 0.0])
    return Atoms(symbols=elements, positions=positions)


@dataclass
class ConfigEntry:
    name: str          # human label
    mask: list[int]    # 6-bit pattern, position 0..5
    formula: str       # canonical formula
    n_F: int
    elements: list[str]
    positions: list[list[float]]
    energy_eV: float


def enumerate_all_64() -> list[tuple[int, ...]]:
    return [tuple(b) for b in itertools.product([0, 1], repeat=6)]


def canonical_orbit_label(mask: tuple[int, ...]) -> str:
    """Return a label that is identical for D6h-equivalent masks.

    D6h on a 6-ring = 12 symmetry ops (6 rotations × 2 reflections via
    flipping the order).  We compute the canonical (lex-smallest) string
    over all rotations of the original AND its reverse.
    """
    s = "".join(str(b) for b in mask)
    candidates = []
    for shift in range(6):
        rot = s[shift:] + s[:shift]
        candidates.append(rot)
        candidates.append(rot[::-1])
    return min(candidates)


def isomer_name(mask: tuple[int, ...]) -> str:
    """Human-readable name like '1,2-F2' for a fluorine pattern."""
    nF = sum(mask)
    if nF == 0:
        return "benzene"
    if nF == 6:
        return "C6F6"
    # rotate so first 1 is at position 0
    s = "".join(str(b) for b in mask)
    best = canonical_orbit_label(mask)
    posF = [i for i, c in enumerate(best) if c == "1"]
    indices = ",".join(str(p + 1) for p in posF)
    return f"C6H{6-nF}F{nF}-({indices})"


def build_dataset(cache: Path,
                  optimize: bool = True) -> list[ConfigEntry]:
    """Generate all 64 H/F decorations of benzene, run xtb, cache to JSON."""
    entries: list[ConfigEntry] = []
    for mask in enumerate_all_64():
        atoms = benzene_decoration(mask)
        E_eV, opt = xtb_energy(atoms, optimize=optimize)
        nF = sum(mask)
        formula = opt.get_chemical_formula()
        entries.append(ConfigEntry(
            name=isomer_name(mask),
            mask=list(mask),
            formula=formula,
            n_F=nF,
            elements=opt.get_chemical_symbols(),
            positions=opt.get_positions().tolist(),
            energy_eV=E_eV,
        ))
        print(f"  mask={''.join(str(b) for b in mask)}  "
              f"nF={nF}  E={E_eV:+12.4f} eV   {entries[-1].name}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([e.__dict__ for e in entries], indent=2))
    return entries


def load_dataset(cache: Path) -> list[ConfigEntry]:
    raw = json.loads(cache.read_text())
    return [ConfigEntry(**e) for e in raw]


def entry_to_atoms(entry: ConfigEntry) -> Atoms:
    return Atoms(symbols=entry.elements, positions=entry.positions)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1] / "data" / "configurational"
    print("=== Building benzene H/F configurational dataset (64 configs) ===")
    build_dataset(base / "benzene_hf.json")
