"""xtb GFN-FF runner with PBC support for periodic crystals.

xtb GFN2 does NOT support PBC ("Multipoles not available with PBC"); GFN-FF
is the universal force field that handles 3D periodic cells via xtb 6.x.
We use POSCAR (VASP) input format because xtb auto-detects PBC from it.

For perovskites and heavy-element periodic systems where DFTB+ Slater-Koster
files are missing, this is the fastest available reference engine.

API mirrors `oce.xtb_runner.xtb_energy`:

    E_eV, atoms_out = gfnff_energy(atoms, optimize=False, threads=1)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

XTB_BIN = "/home/xtb/bin/xtb"
EH_TO_EV = 27.211386245988


def _write_poscar(atoms: Atoms, path: Path) -> None:
    """ASE write VASP-style POSCAR (xtb auto-detects PBC from this format)."""
    ase_write(str(path), atoms, format="vasp", direct=True, sort=True)


def gfnff_energy(atoms: Atoms, optimize: bool = False,
                  threads: int = 1,
                  keep_dir: Path | None = None) -> tuple[float, Atoms]:
    """Run xtb GFN-FF on a periodic structure and return (E_total_eV, atoms).

    Forces PBC = (T,T,T) on atoms before serialising; if the input has no
    cell, GFN-FF treats it as cluster (also fine).
    """
    workdir = Path(tempfile.mkdtemp(prefix="gfnff_"))
    try:
        poscar = workdir / "POSCAR"
        _write_poscar(atoms, poscar)
        cmd = [XTB_BIN, "POSCAR", "--gfnff"]
        cmd.append("--opt" if optimize else "--sp")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        res = subprocess.run(cmd, cwd=workdir, capture_output=True,
                              text=True, timeout=180, env=env)
        if res.returncode != 0:
            raise RuntimeError(
                f"xtb GFN-FF failed:\n{res.stdout[-1500:]}\n"
                f"stderr:\n{res.stderr[-500:]}")
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", res.stdout)
        if not m:
            raise RuntimeError("could not parse TOTAL ENERGY from xtb stdout")
        E_eh = float(m.group(1))
        if optimize:
            opt_path = workdir / "xtbopt.poscar"
            if opt_path.exists():
                opt_atoms = ase_read(str(opt_path))
            else:
                opt_atoms = atoms.copy()
        else:
            opt_atoms = atoms.copy()
        return E_eh * EH_TO_EV, opt_atoms
    finally:
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in workdir.iterdir():
                shutil.copy(f, keep_dir)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    # Smoke test on cubic CsPbI3
    a = 6.39
    ats = Atoms(
        symbols=["Cs", "Pb", "I", "I", "I"],
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5),
                           (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=[a, a, a], pbc=True,
    )
    E, _ = gfnff_energy(ats, optimize=False)
    print(f"CsPbI3 cubic primitive: E = {E:+.4f} eV  ({E/len(ats):+.4f} eV/atom)")
