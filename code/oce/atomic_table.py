"""Atomic orbital eigenenergy table built from xtb single-atom calculations.

Each element gets a list of (shell_label, eigenenergy_eV, occupation) tuples,
grouped by shell (s, p, d) — orbitals with identical eigenenergies are
collapsed into a single shell entry.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

XTB_BIN = "/home/xtb/bin/xtb"

# Ground-state spin multiplicities (unpaired electrons).  xtb GFN2 covers
# elements 1–86; entries beyond Rn (Z>86) require GFN-FF or external refs and
# are intentionally absent here — caller code should filter them out.
GROUND_STATE_UHF = {
    "H": 1, "He": 0,
    "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1, "Ne": 0,
    "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1, "Ar": 0,
    "K": 1, "Ca": 0,
    # 3d transition metals
    "Ti": 2, "V": 3, "Cr": 6, "Mn": 5, "Fe": 4, "Co": 3, "Ni": 2,
    "Cu": 1, "Zn": 0,
    # 4p
    "Ga": 1, "Ge": 2, "As": 3, "Se": 2, "Br": 1, "Kr": 0,
    # 5s/4d
    "Rb": 1, "Sr": 0, "Y": 1, "Zr": 2, "Nb": 5, "Mo": 6, "Tc": 5,
    "Ru": 4, "Rh": 3, "Pd": 0, "Ag": 1, "Cd": 0,
    # 5p
    "In": 1, "Sn": 2, "Sb": 3, "Te": 2, "I": 1, "Xe": 0,
    # 6s + lanthanides (4f^n + 5d^0/1)
    "Cs": 1, "Ba": 0,
    "La": 1, "Ce": 2, "Pr": 3, "Nd": 4, "Pm": 5, "Sm": 6, "Eu": 7,
    "Gd": 8, "Tb": 5, "Dy": 4, "Ho": 3, "Er": 2, "Tm": 1, "Yb": 0,
    "Lu": 1,
    # 5d
    "Hf": 2, "Ta": 3, "W": 4, "Re": 5, "Os": 4, "Ir": 3, "Pt": 2,
    "Au": 1, "Hg": 0,
    # 6p
    "Tl": 1, "Pb": 2, "Bi": 3, "Po": 2, "At": 1, "Rn": 0,
}


@dataclass
class Shell:
    label: str            # 's', 'p', 'd' (assigned heuristically by degeneracy)
    epsilon_eV: float     # eigenenergy
    occupation: float     # total electrons in that shell


@dataclass
class AtomEntry:
    element: str
    total_energy_Eh: float
    shells: list[Shell]


def _run_xtb_atom(element: str, workdir: Path) -> str:
    """Run xtb GFN2 single-atom and return stdout."""
    xyz = workdir / f"{element}.xyz"
    xyz.write_text(f"1\n{element} atom\n{element}  0.0  0.0  0.0\n")
    uhf = GROUND_STATE_UHF[element]
    cmd = [XTB_BIN, str(xyz), "--gfn", "2", "--uhf", str(uhf), "--sp"]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"xtb failed on {element}:\n{res.stderr}")
    return res.stdout


def _parse_orbitals(stdout: str) -> tuple[float, list[tuple[float, float]]]:
    """Return (total_energy_Eh, [(occ, eV), ...]) from xtb stdout."""
    total_re = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", stdout)
    if not total_re:
        raise RuntimeError("could not find TOTAL ENERGY in xtb output")
    total = float(total_re.group(1))

    # The block under "* Orbital Energies and Occupations"
    block_match = re.search(
        r"\* Orbital Energies and Occupations(.*?)HL-Gap",
        stdout, flags=re.DOTALL,
    )
    if not block_match:
        raise RuntimeError("could not find orbital block in xtb output")
    block = block_match.group(1)

    orbitals: list[tuple[float, float]] = []
    for line in block.splitlines():
        m = re.match(
            r"\s*\d+\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)",
            line,
        )
        if m:
            occ, _eh, ev = float(m.group(1)), float(m.group(2)), float(m.group(3))
            orbitals.append((occ, ev))
    if not orbitals:
        raise RuntimeError("no orbitals parsed")
    return total, orbitals


def _group_into_shells(orbitals: list[tuple[float, float]], tol: float = 1e-3) -> list[Shell]:
    """Collapse orbitals with (nearly) identical eigenenergies into shells.

    Shell label heuristic: degeneracy 1 -> 's', 3 -> 'p', 5 -> 'd', else 'x'.
    """
    sorted_orb = sorted(orbitals, key=lambda x: x[1])
    shells: list[Shell] = []
    current: list[tuple[float, float]] = []
    current_eps: float | None = None
    for occ, ev in sorted_orb:
        if current_eps is None or abs(ev - current_eps) < tol:
            current.append((occ, ev))
            current_eps = ev if current_eps is None else current_eps
        else:
            shells.append(_make_shell(current))
            current = [(occ, ev)]
            current_eps = ev
    if current:
        shells.append(_make_shell(current))
    return shells


def _make_shell(group: list[tuple[float, float]]) -> Shell:
    deg = len(group)
    label = {1: "s", 3: "p", 5: "d"}.get(deg, "x")
    occ_total = sum(o for o, _ in group)
    eps = sum(e for _, e in group) / deg
    return Shell(label=label, epsilon_eV=eps, occupation=occ_total)


def build_table(elements: list[str], cache: Path | None = None,
                merge: bool = False) -> dict[str, AtomEntry]:
    """Run xtb on each atom and return dict element -> AtomEntry.

    If merge=True and cache exists, load existing entries and only run xtb
    for the elements not already present (or all listed if explicitly forced).
    """
    table: dict[str, AtomEntry] = {}
    if merge and cache is not None and cache.exists():
        table = load_table(cache)
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for el in elements:
            if merge and el in table:
                continue
            stdout = _run_xtb_atom(el, wd)
            tot, orbs = _parse_orbitals(stdout)
            shells = _group_into_shells(orbs)
            table[el] = AtomEntry(element=el, total_energy_Eh=tot, shells=shells)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {k: asdict(v) for k, v in table.items()}, indent=2,
        ))
    return table


def load_table(cache: Path) -> dict[str, AtomEntry]:
    raw = json.loads(cache.read_text())
    return {
        k: AtomEntry(
            element=v["element"],
            total_energy_Eh=v["total_energy_Eh"],
            shells=[Shell(**s) for s in v["shells"]],
        )
        for k, v in raw.items()
    }


if __name__ == "__main__":
    import sys
    cache = Path(__file__).resolve().parents[1] / "data" / "atoms" / "atomic_table.json"
    args = sys.argv[1:]
    merge = True
    if args and args[0] == "--rebuild":
        merge = False
        args = args[1:]
    elements = args or ["H", "C", "N", "O", "F"]
    table = build_table(elements, cache=cache, merge=merge)
    for el, entry in table.items():
        print(f"\n{el}: total = {entry.total_energy_Eh:+.6f} Eh")
        for sh in entry.shells:
            print(f"   shell {sh.label}: ε = {sh.epsilon_eV:+8.3f} eV, "
                  f"occ = {sh.occupation:.3f}")
    print(f"\nCached at {cache}")
