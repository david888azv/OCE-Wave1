"""xtb GFN2 single-point and optimisation for radical carbon clusters.

We do NOT terminate dangling bonds.  Instead, we let xtb treat the cluster
as a high-spin radical molecule by passing `--uhf` = parity of the total
electron count (so xtb finds the lowest-energy state of the correct
spin-parity).

Note: GFN2 with the radical on every edge atom can struggle to converge.
We retry with progressively larger uhf and looser SCF if needed.
"""
from __future__ import annotations

import re
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write, read as ase_read

from percolation import PercoCluster

XTB_BIN = "/home/xtb/bin/xtb"
EH_TO_EV = 27.211386245988
VALENCE_E = {"H": 1, "C": 4, "N": 5, "O": 6}


def cluster_to_atoms(c: PercoCluster) -> Atoms:
    """Build an ASE Atoms object from a percolation cluster (all carbon)."""
    return Atoms(symbols=["C"] * c.n_atoms, positions=c.positions)


def total_electrons(atoms: Atoms) -> int:
    """Sum of valence electrons (GFN2 uses valence-only)."""
    return sum(VALENCE_E[s] for s in atoms.get_chemical_symbols())


def xtb_radical_energy(atoms: Atoms,
                       optimize: bool = False,
                       max_uhf: int = 8,
                       timeout: int = 1800,
                       threads: int | None = None) -> dict:
    """Run xtb GFN2 on a (possibly high-spin) radical cluster.

    Strategy: try uhf = parity(electrons) first.  If xtb fails to converge,
    retry with uhf increased by 2 (next-higher-spin state), up to max_uhf.

    Returns a dict with keys:
        E_eV          float  total energy
        uhf           int    spin state actually used
        n_atoms       int
        n_electrons   int
        opt_atoms     Atoms (= input if optimize=False)
        wall_time_s   float
        converged     bool
    """
    import time
    n_e = total_electrons(atoms)
    parity = n_e % 2
    uhf_to_try = list(range(parity, max_uhf + 1, 2))

    workdir = Path(tempfile.mkdtemp(prefix="xtb_perco_"))
    last_err = ""
    try:
        xyz = workdir / "mol.xyz"
        ase_write(str(xyz), atoms, format="xyz")
        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)
            env["MKL_NUM_THREADS"] = str(threads)
            env["OPENBLAS_NUM_THREADS"] = str(threads)

        for uhf in uhf_to_try:
            cmd = [XTB_BIN, "mol.xyz",
                   "--gfn", "2",
                   "--chrg", "0",
                   "--uhf", str(uhf),
                   "--iterations", "500",
                   "--acc", "1.0"]
            cmd.append("--opt" if optimize else "--sp")
            t0 = time.perf_counter()
            try:
                res = subprocess.run(cmd, cwd=workdir, capture_output=True,
                                     text=True, timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                last_err = f"timeout (uhf={uhf})"
                continue
            dt = time.perf_counter() - t0

            if res.returncode != 0:
                last_err = f"uhf={uhf} returncode={res.returncode}\n{res.stderr[-1500:]}"
                continue
            m = re.search(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh", res.stdout)
            if not m:
                last_err = f"uhf={uhf} no TOTAL ENERGY parsed"
                continue
            E_eV = float(m.group(1)) * EH_TO_EV

            opt_atoms = atoms.copy()
            if optimize and (workdir / "xtbopt.xyz").exists():
                opt_atoms = ase_read(str(workdir / "xtbopt.xyz"))
            return dict(
                E_eV=E_eV,
                uhf=uhf,
                n_atoms=len(atoms),
                n_electrons=n_e,
                opt_atoms=opt_atoms,
                wall_time_s=dt,
                converged=True,
            )
        # All uhf attempts failed
        return dict(
            E_eV=float("nan"),
            uhf=-1,
            n_atoms=len(atoms),
            n_electrons=n_e,
            opt_atoms=atoms.copy(),
            wall_time_s=0.0,
            converged=False,
            error=last_err,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    from lattices import build_honeycomb, P_C
    from percolation import sample_percolation
    lat = build_honeycomb(4)
    cl = sample_percolation(lat, P_C["honeycomb"], seed=0)
    atoms = cluster_to_atoms(cl)
    print(f"Cluster: {cl.n_atoms} atoms, {cl.n_bonds} bonds, "
          f"sum dangling = {sum(cl.n_dangling)}")
    res = xtb_radical_energy(atoms, optimize=False)
    print(res)
