"""SIESTA DFT-PBE single-point runner for free C / Si clusters.

Strategy: spin-polarised DZP basis with PBE GGA on a non-periodic
simulation cell.  The cell is set with ≥ 12 Å vacuum on all sides so that
the cluster does not interact with its periodic images.

Pseudopotentials (PSF / PSML) live in `pseudos/`.

Net charge is zero (default); spin polarisation is enabled so that SIESTA
can populate the highest occupied levels self-consistently for radical
clusters with odd electron count.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from ase import Atoms

SIESTA_BIN = "/home/david/siesta54/bin/siesta"
PSEUDOS_DIR = Path(__file__).resolve().parent / "pseudos"

INPUT_TEMPLATE = """\
SystemName        cluster_{tag}
SystemLabel       cluster_{tag}

NumberOfAtoms     {n_atoms}
NumberOfSpecies   1

%block ChemicalSpeciesLabel
 1  {atomic_number}  {element}
%endblock ChemicalSpeciesLabel

PAO.BasisSize       SZP
PAO.EnergyShift     0.05 Ry

XC.functional       GGA
XC.authors          PBE

# Vacuum-padded cell — non-periodic cluster
LatticeConstant 1.0 Ang
%block LatticeVectors
{box_x:.6f}  0.000000  0.000000
0.000000  {box_y:.6f}  0.000000
0.000000  0.000000  {box_z:.6f}
%endblock LatticeVectors

AtomicCoordinatesFormat        Ang
%block AtomicCoordinatesAndAtomicSpecies
{coords}
%endblock AtomicCoordinatesAndAtomicSpecies

MeshCutoff          120.0 Ry
DM.MixingWeight     0.02
DM.NumberPulay      6
MaxSCFIterations    500
DM.Tolerance        1.0e-3
SCF.Mix             density

# Spin-unrestricted (paramagnetic radicals OK)
SpinPolarized       .false.
NetCharge           0.0

ElectronicTemperature  500 K
SolutionMethod         diagon

WriteForces            .false.
SaveTotalCharge        .false.
WriteCoorStep          .false.
SaveHS                 .false.
SaveRho                .false.
WriteMullikenPop       0
"""

ISOLATED_ATOM_TEMPLATE = """\
SystemName        atom_{element}
SystemLabel       atom_{element}

NumberOfAtoms     1
NumberOfSpecies   1

%block ChemicalSpeciesLabel
 1  {atomic_number}  {element}
%endblock ChemicalSpeciesLabel

PAO.BasisSize       SZP
PAO.EnergyShift     0.05 Ry

XC.functional       GGA
XC.authors          PBE

LatticeConstant 1.0 Ang
%block LatticeVectors
20.0   0.0   0.0
 0.0  20.0   0.0
 0.0   0.0  20.0
%endblock LatticeVectors

AtomicCoordinatesFormat        Ang
%block AtomicCoordinatesAndAtomicSpecies
   10.0  10.0  10.0  1
%endblock AtomicCoordinatesAndAtomicSpecies

MeshCutoff          120.0 Ry
DM.MixingWeight     0.02
DM.NumberPulay      6
MaxSCFIterations    500
DM.Tolerance        1.0e-3

SpinPolarized       .true.
NetCharge           0.0

ElectronicTemperature  500 K
SolutionMethod         diagon

WriteForces            .false.
"""


def _make_input(atoms: Atoms, tag: str, vacuum: float = 12.0) -> tuple[str, dict]:
    """Build SIESTA input file content.  Returns (text, info)."""
    syms = atoms.get_chemical_symbols()
    if len(set(syms)) != 1:
        raise ValueError(f"Mixed-element cluster not supported: {set(syms)}")
    el = syms[0]
    z = {"C": 6, "Si": 14, "Ge": 32}[el]
    pos = atoms.get_positions()
    lo = pos.min(axis=0); hi = pos.max(axis=0)
    box = (hi - lo) + 2 * vacuum
    # centre the cluster within the box
    centred = pos - lo + vacuum
    coords_lines = "\n".join(
        f" {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}  1"
        for p in centred
    )
    text = INPUT_TEMPLATE.format(
        tag=tag,
        n_atoms=len(atoms),
        atomic_number=z,
        element=el,
        box_x=box[0], box_y=box[1], box_z=box[2],
        coords=coords_lines,
    )
    return text, dict(element=el, box=box.tolist(), atomic_number=z)


def siesta_energy(atoms: Atoms,
                   tag: str = "perco",
                   timeout: int = 7200,
                   threads: int = 8,
                   keep_dir: Path | None = None,
                   mpi_procs: int = 1) -> dict:
    """Run SIESTA single-point on the input atoms.  Return energy + walltime.

    For mpi_procs > 1, runs `mpirun -np N siesta`.  Threads sets
    OMP_NUM_THREADS for the subprocess.
    """
    workdir = Path(tempfile.mkdtemp(prefix="siesta_perco_"))
    try:
        text, info = _make_input(atoms, tag)
        (workdir / "input.fdf").write_text(text)
        # link pseudopotential
        el = info["element"]
        if el == "C":
            shutil.copy(PSEUDOS_DIR / "C.psf", workdir / "C.psf")
        elif el == "Si":
            shutil.copy(PSEUDOS_DIR / "Si.psml", workdir / "Si.psml")
        elif el == "Ge":
            shutil.copy(PSEUDOS_DIR / "Ge.psml", workdir / "Ge.psml")
        else:
            raise ValueError(f"No pseudopotential configured for {el}")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)

        if mpi_procs > 1:
            from runners_lammps import MPIRUN_BIN
            cmd = [MPIRUN_BIN, "-np", str(mpi_procs), "--bind-to", "none",
                   SIESTA_BIN]
        else:
            cmd = [SIESTA_BIN]

        t0 = time.perf_counter()
        try:
            with open(workdir / "input.fdf", "r") as fin, \
                 open(workdir / "siesta.out", "w") as fout:
                res = subprocess.run(cmd, cwd=workdir, stdin=fin,
                                      stdout=fout, stderr=subprocess.PIPE,
                                      timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return dict(converged=False, error="timeout",
                        E_eV=float("nan"), wall_time_s=timeout,
                        n_atoms=len(atoms))
        dt = time.perf_counter() - t0

        out = (workdir / "siesta.out").read_text() if (workdir / "siesta.out").exists() \
              else ""
        if res.returncode != 0 and "Job completed" not in out:
            return dict(converged=False,
                        error=f"returncode={res.returncode}\n"
                               f"stderr={res.stderr.decode()[-1500:]}\n"
                               f"out_tail={out[-1500:]}",
                        E_eV=float("nan"), wall_time_s=dt,
                        n_atoms=len(atoms))

        # SIESTA prints "siesta:         Total =" near the end
        m = re.search(r"siesta:\s+Total\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            # try "Total energy"
            m = re.search(r"Total energy\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            return dict(converged=False,
                        error=f"no Total energy parsed; tail:\n{out[-2000:]}",
                        E_eV=float("nan"), wall_time_s=dt,
                        n_atoms=len(atoms))
        E_eV = float(m.group(1))   # SIESTA reports in eV by default
        # check SCF convergence
        scf_ok = "SCF cycle converged" in out or "converged" in out.lower()
        return dict(converged=True, E_eV=E_eV, wall_time_s=dt,
                    n_atoms=len(atoms),
                    scf_converged=scf_ok)
    finally:
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in workdir.iterdir():
                shutil.copy(f, keep_dir)
        shutil.rmtree(workdir, ignore_errors=True)


_ATOM_REF_CACHE: dict = {}


def siesta_isolated_atom_energy(element: str,
                                  threads: int = 4,
                                  mpi_procs: int = 1,
                                  timeout: int = 600) -> float:
    """Compute SIESTA total energy of an isolated `element` atom (eV).

    Cached in-memory and on-disk so repeated calls are cheap.  Used to
    build per-element cohesive-energy reference.
    """
    if element in _ATOM_REF_CACHE:
        return _ATOM_REF_CACHE[element]
    cache_path = PSEUDOS_DIR.parent / "data" / "siesta_atom_refs.json"
    if cache_path.exists():
        import json as _json
        d = _json.loads(cache_path.read_text())
        if element in d:
            _ATOM_REF_CACHE[element] = float(d[element])
            return _ATOM_REF_CACHE[element]
    workdir = Path(tempfile.mkdtemp(prefix="siesta_atom_"))
    try:
        z = {"C": 6, "Si": 14, "Ge": 32}[element]
        text = ISOLATED_ATOM_TEMPLATE.format(element=element, atomic_number=z)
        (workdir / "input.fdf").write_text(text)
        if element == "C":
            shutil.copy(PSEUDOS_DIR / "C.psf", workdir / "C.psf")
        elif element == "Si":
            shutil.copy(PSEUDOS_DIR / "Si.psml", workdir / "Si.psml")
        elif element == "Ge":
            shutil.copy(PSEUDOS_DIR / "Ge.psml", workdir / "Ge.psml")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)
        if mpi_procs > 1:
            from runners_lammps import MPIRUN_BIN
            cmd = [MPIRUN_BIN, "-np", str(mpi_procs), "--bind-to", "none",
                   SIESTA_BIN]
        else:
            cmd = [SIESTA_BIN]
        with open(workdir / "input.fdf", "r") as fin, \
             open(workdir / "siesta.out", "w") as fout:
            subprocess.run(cmd, cwd=workdir, stdin=fin, stdout=fout,
                           stderr=subprocess.PIPE,
                           timeout=timeout, env=env)
        out = (workdir / "siesta.out").read_text()
        m = re.search(r"siesta:\s+Total\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            m = re.search(r"Total energy\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            raise RuntimeError(f"isolated-atom SIESTA failed for {element}\n"
                                f"tail:\n{out[-2000:]}")
        E_eV = float(m.group(1))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    # cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    d = {}
    if cache_path.exists():
        d = _json.loads(cache_path.read_text())
    d[element] = E_eV
    cache_path.write_text(_json.dumps(d, indent=2))
    _ATOM_REF_CACHE[element] = E_eV
    return E_eV


if __name__ == "__main__":
    # Tiny smoke test: 4-atom C cluster
    a = Atoms(symbols=["C"] * 4,
               positions=[[0, 0, 0], [1.42, 0, 0],
                          [0, 1.42, 0], [1.42, 1.42, 0]])
    print("Running SIESTA smoke test on 4-C cluster...")
    r = siesta_energy(a, tag="smoke", timeout=600, threads=4)
    print(r)
