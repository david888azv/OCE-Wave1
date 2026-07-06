"""Run xtb GFN-FF single-point on every perovskite in structures.json.

Saves energies back to the same structures.json (adds energy_total_eV and
engine fields) so fit_subset.py can read them directly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oce.parallel import energy_batch
from oce.xtb_periodic import gfnff_energy


def main(n_workers: int = 12):
    sd = ROOT / "data" / "perovskites"
    records = json.loads((sd / "structures.json").read_text())
    n = len(records)
    print(f"Loaded {n} perovskite structures")

    atoms_list = [
        Atoms(symbols=r["symbols"], positions=r["positions"],
              cell=r["cell"], pbc=r["pbc"])
        for r in records
    ]

    t0 = time.time()
    results = energy_batch(
        atoms_list, engine=gfnff_energy,
        kwargs_list={"threads": 1},
        n_workers=n_workers, executor="thread", progress=True,
    )
    dt = time.time() - t0
    print(f"\nWall time: {dt:.1f} s   ({dt/n*1000:.1f} ms/struct, "
          f"{n_workers} workers)")

    n_ok = 0
    energies = []
    for r, (E, _opt) in zip(records, results):
        if E is None:
            r["energy_total_eV"] = None
            r["engine"] = "gfnff_failed"
        else:
            r["energy_total_eV"] = float(E)
            r["engine"] = "xtb-gfnff-sp"
            energies.append(E / len(r["symbols"]))
            n_ok += 1
    print(f"Success: {n_ok}/{n}")
    if energies:
        print(f"E/atom: min {min(energies):.4f}  median {np.median(energies):.4f}  "
              f"max {max(energies):.4f}  spread {max(energies)-min(energies):.4f} eV")

    (sd / "structures.json").write_text(json.dumps(records))
    print(f"Updated {sd / 'structures.json'}")


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(nw)
