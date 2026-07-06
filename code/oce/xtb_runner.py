"""Subprocess wrapper around xtb for ground-state energies."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write

XTB_BIN = "/home/xtb/bin/xtb"
EH_TO_EV = 27.211386245988


import os as _os

def xtb_energy(atoms: Atoms, optimize: bool = False,
               gfn: int = 2, charge: int = 0, uhf: int = 0,
               keep_dir: Path | None = None,
               constraints: list[dict] | None = None,
               force_constant: float = 1.0,
               threads: int | None = None) -> tuple[float, Atoms]:
    """Run xtb on an ASE Atoms object. Return (E_total_eV, optimized_atoms).

    If optimize=False, returns single-point energy and the input geometry.

    `constraints`: optional list of dicts to fix internal coordinates during
    optimization.  Each dict must have:
        type: "dihedral" | "angle" | "distance"
        atoms: 1-based atom indices (length 4 for dihedral, 3 for angle,
               2 for distance)
        value: target value (degrees for dihedral/angle, Å for distance)
    Example: [{"type":"dihedral","atoms":[2,4,5,7],"value":-85.0}, …]
    """
    workdir = Path(tempfile.mkdtemp(prefix="xtb_"))
    try:
        xyz = workdir / "mol.xyz"
        ase_write(str(xyz), atoms, format="xyz")
        cmd = [XTB_BIN, "mol.xyz", "--gfn", str(gfn),
               "--chrg", str(charge), "--uhf", str(uhf)]
        if optimize:
            cmd.append("--opt")
        else:
            cmd.append("--sp")
        if constraints:
            inp = workdir / "xcontrol.inp"
            lines = ["$constrain", f"   force constant={force_constant}"]
            for c in constraints:
                t = c["type"]
                ats = ",".join(str(int(a)) for a in c["atoms"])
                v = c["value"]
                lines.append(f"   {t}: {ats},{v}")
            lines.append("$end")
            inp.write_text("\n".join(lines) + "\n")
            cmd += ["--input", "xcontrol.inp"]
        env = _os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            env["OPENBLAS_NUM_THREADS"] = str(threads)
        res = subprocess.run(cmd, cwd=workdir, capture_output=True,
                             text=True, timeout=300, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"xtb failed:\n{res.stderr[-2000:]}")
        m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", res.stdout)
        if not m:
            raise RuntimeError("could not parse xtb total energy")
        E_eh = float(m.group(1))

        from ase.io import read as ase_read
        if optimize:
            opt_xyz = workdir / "xtbopt.xyz"
            if opt_xyz.exists():
                opt_atoms = ase_read(str(opt_xyz))
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
    from ase.build import molecule
    for name in ["H2O", "CH4", "C2H6"]:
        atoms = molecule(name)
        E_sp, _ = xtb_energy(atoms, optimize=False)
        E_opt, opt = xtb_energy(atoms, optimize=True)
        print(f"{name:6s}  E_sp = {E_sp:+12.4f} eV   "
              f"E_opt = {E_opt:+12.4f} eV   "
              f"ΔE_opt = {E_opt-E_sp:+.4f} eV")
