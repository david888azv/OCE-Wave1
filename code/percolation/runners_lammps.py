"""LAMMPS + AIREBO runner for carbon percolation clusters.

Cheap, well-parameterised reactive force-field for pure-C clusters with
arbitrary connectivity (handles dangling bonds gracefully — AIREBO was
designed for radical hydrocarbons).  Single-point and minimisation modes.

Output: total potential energy in eV and the relaxed positions.
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
from ase.io import write as ase_write

LMP_BIN = "/usr/local/bin/lmp_mpi"   # works in serial too
MPIRUN_BIN = "/usr/lib64/mpi/gcc/openmpi5/bin/mpirun"
AIREBO_POT = "/usr/share/lammps/potentials/CH.airebo"
SI_TERSOFF_POT = "/usr/share/lammps/potentials/Si.tersoff"
GE_TERSOFF_POT = "/usr/share/lammps/potentials/Ge.tersoff"


INPUT_TEMPLATE_SP = """\
# AIREBO single-point on a free C cluster
units           metal
boundary        p p p
atom_style      atomic
read_data       data.in

pair_style      airebo 3.0 1 0
pair_coeff      * * {pot} C

run             0
variable        epot equal pe
print           "FINAL_ENERGY ${{epot}} eV"
"""

INPUT_TEMPLATE_OPT = """\
# AIREBO minimisation of a free C cluster
units           metal
boundary        p p p
atom_style      atomic
read_data       data.in

pair_style      airebo 3.0 1 0
pair_coeff      * * {pot} C

dump            d1 all custom 100 dump.last id x y z
dump_modify     d1 sort id

# Relax atomic positions — fixed simulation cell, no PBC
fix             1 all nve
min_style       cg
minimize        1.0e-8 1.0e-10 5000 50000

write_data      data.relaxed
variable        epot equal pe
print           "FINAL_ENERGY ${{epot}} eV"
"""


SI_INPUT_TEMPLATE_SP = """\
# Tersoff single-point on a free Si cluster
units           metal
boundary        p p p
atom_style      atomic
read_data       data.in

mass            1 28.0855
pair_style      tersoff
pair_coeff      * * {pot} Si

run             0
variable        epot equal pe
print           "FINAL_ENERGY ${{epot}} eV"
"""

SI_INPUT_TEMPLATE_OPT = """\
# Tersoff minimisation of a free Si cluster
units           metal
boundary        p p p
atom_style      atomic
read_data       data.in

mass            1 28.0855
pair_style      tersoff
pair_coeff      * * {pot} Si

dump            d1 all custom 100 dump.last id x y z
dump_modify     d1 sort id

fix             1 all nve
min_style       cg
minimize        1.0e-8 1.0e-10 5000 50000

write_data      data.relaxed
variable        epot equal pe
print           "FINAL_ENERGY ${{epot}} eV"
"""


def _write_data_file(atoms: Atoms, path: Path,
                       pad: float = 25.0,
                       mass: float = 12.011,
                       header: str = "carbon cluster") -> None:
    """LAMMPS 'read_data'-format file for a single-element type, free boundary.

    pad: vacuum padding around the cluster in Å.  Must exceed the master
    neighbour-list cutoff (~12 Å for AIREBO, ~5 Å for Si Tersoff) to
    avoid 'box << cutoff' bin errors.
    """
    pos = atoms.get_positions()
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    n = len(atoms)
    lines = [
        header,
        "",
        f"{n} atoms",
        "1 atom types",
        "",
        f"{lo[0]:.6f} {hi[0]:.6f} xlo xhi",
        f"{lo[1]:.6f} {hi[1]:.6f} ylo yhi",
        f"{lo[2]:.6f} {hi[2]:.6f} zlo zhi",
        "",
        "Masses",
        "",
        f"1 {mass}",
        "",
        "Atoms",
        "",
    ]
    for i, p in enumerate(pos, start=1):
        lines.append(f"{i} 1 {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def _read_relaxed_positions(path: Path, n_atoms: int) -> np.ndarray:
    """Read positions from a LAMMPS dump 'custom id x y z' file."""
    txt = path.read_text().splitlines()
    # find last frame
    starts = [i for i, line in enumerate(txt) if line.startswith("ITEM: TIMESTEP")]
    if not starts:
        raise RuntimeError("no ITEM: TIMESTEP in dump")
    s = starts[-1]
    # ITEM: ATOMS id x y z is at s + 8 (5 header lines + 3 box) for orth
    # find the ATOMS header
    for i in range(s, len(txt)):
        if txt[i].startswith("ITEM: ATOMS"):
            atoms_start = i + 1
            break
    else:
        raise RuntimeError("no ITEM: ATOMS header in dump")
    pos = np.zeros((n_atoms, 3))
    for line in txt[atoms_start:atoms_start + n_atoms]:
        toks = line.split()
        i = int(toks[0]) - 1
        pos[i] = [float(toks[1]), float(toks[2]), float(toks[3])]
    return pos


def lammps_airebo(atoms: Atoms,
                   optimize: bool = False,
                   timeout: int = 600,
                   threads: int | None = None,
                   mpi_procs: int = 1) -> dict:
    """Run LAMMPS + AIREBO on a free carbon cluster.

    mpi_procs: if >1, run with `mpirun -np mpi_procs lmp_mpi ...` (Open MPI).
    threads:   if not None, sets OMP_NUM_THREADS for the subprocess.
    """
    workdir = Path(tempfile.mkdtemp(prefix="lmp_perco_"))
    try:
        _write_data_file(atoms, workdir / "data.in")
        tmpl = INPUT_TEMPLATE_OPT if optimize else INPUT_TEMPLATE_SP
        (workdir / "in.lmp").write_text(tmpl.format(pot=AIREBO_POT))

        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)
        if mpi_procs > 1:
            cmd = [MPIRUN_BIN, "-np", str(mpi_procs),
                   "--bind-to", "none",
                   LMP_BIN, "-in", "in.lmp",
                   "-screen", "none", "-log", "log.lammps"]
        else:
            cmd = [LMP_BIN, "-in", "in.lmp",
                   "-screen", "none", "-log", "log.lammps"]
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, cwd=workdir, capture_output=True,
                                 text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return dict(converged=False, error="timeout",
                        E_eV=float("nan"), opt_atoms=atoms.copy(),
                        wall_time_s=timeout, n_atoms=len(atoms))
        dt = time.perf_counter() - t0

        log = (workdir / "log.lammps").read_text() if (workdir / "log.lammps").exists() \
              else (res.stdout + res.stderr)
        m = re.search(r"FINAL_ENERGY\s+(-?\d+\.\d+)", log)
        if m is None:
            return dict(converged=False,
                        error=f"no FINAL_ENERGY parsed; tail:\n{log[-1500:]}",
                        E_eV=float("nan"), opt_atoms=atoms.copy(),
                        wall_time_s=dt, n_atoms=len(atoms))
        E_eV = float(m.group(1))

        opt_atoms = atoms.copy()
        if optimize and (workdir / "dump.last").exists():
            try:
                pos = _read_relaxed_positions(workdir / "dump.last", len(atoms))
                opt_atoms.set_positions(pos)
            except Exception as e:
                return dict(converged=False,
                            error=f"dump-parse failed: {e}",
                            E_eV=E_eV, opt_atoms=atoms.copy(),
                            wall_time_s=dt, n_atoms=len(atoms))
        return dict(converged=True, E_eV=E_eV, opt_atoms=opt_atoms,
                    wall_time_s=dt, n_atoms=len(atoms))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


GE_INPUT_TEMPLATE_SP = """\
# Tersoff single-point on a free Ge cluster
units           metal
boundary        p p p
atom_style      atomic
read_data       data.in

mass            1 72.6300
pair_style      tersoff
pair_coeff      * * {pot} Ge

run             0
variable        epot equal pe
print           "FINAL_ENERGY ${{epot}} eV"
"""

GE_INPUT_TEMPLATE_OPT = """\
# Tersoff minimisation of a free Ge cluster
units           metal
boundary        p p p
atom_style      atomic
read_data       data.in

mass            1 72.6300
pair_style      tersoff
pair_coeff      * * {pot} Ge

dump            d1 all custom 100 dump.last id x y z
dump_modify     d1 sort id

fix             1 all nve
min_style       cg
minimize        1.0e-8 1.0e-10 5000 50000

write_data      data.relaxed
variable        epot equal pe
print           "FINAL_ENERGY ${{epot}} eV"
"""


def lammps_ge_tersoff(atoms: Atoms,
                       optimize: bool = False,
                       timeout: int = 600,
                       threads: int | None = None,
                       mpi_procs: int = 1) -> dict:
    """LAMMPS + Tersoff (Tersoff PRB 1989) on a free Ge cluster."""
    workdir = Path(tempfile.mkdtemp(prefix="lmp_perco_ge_"))
    try:
        _write_data_file(atoms, workdir / "data.in",
                          mass=72.6300, header="Ge Tersoff cluster",
                          pad=20.0)
        tmpl = GE_INPUT_TEMPLATE_OPT if optimize else GE_INPUT_TEMPLATE_SP
        (workdir / "in.lmp").write_text(tmpl.format(pot=GE_TERSOFF_POT))
        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)
        if mpi_procs > 1:
            cmd = [MPIRUN_BIN, "-np", str(mpi_procs), "--bind-to", "none",
                   LMP_BIN, "-in", "in.lmp", "-screen", "none", "-log", "log.lammps"]
        else:
            cmd = [LMP_BIN, "-in", "in.lmp", "-screen", "none", "-log", "log.lammps"]
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, cwd=workdir, capture_output=True,
                                 text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return dict(converged=False, error="timeout",
                        E_eV=float("nan"), opt_atoms=atoms.copy(),
                        wall_time_s=timeout, n_atoms=len(atoms))
        dt = time.perf_counter() - t0
        log = (workdir / "log.lammps").read_text() if (workdir / "log.lammps").exists() \
              else (res.stdout + res.stderr)
        m = re.search(r"FINAL_ENERGY\s+(-?\d+\.\d+)", log)
        if m is None:
            return dict(converged=False,
                        error=f"no FINAL_ENERGY parsed; tail:\n{log[-1500:]}",
                        E_eV=float("nan"), opt_atoms=atoms.copy(),
                        wall_time_s=dt, n_atoms=len(atoms))
        E_eV = float(m.group(1))
        opt_atoms = atoms.copy()
        if optimize and (workdir / "dump.last").exists():
            try:
                pos = _read_relaxed_positions(workdir / "dump.last", len(atoms))
                opt_atoms.set_positions(pos)
            except Exception as e:
                return dict(converged=False,
                            error=f"dump-parse failed: {e}",
                            E_eV=E_eV, opt_atoms=atoms.copy(),
                            wall_time_s=dt, n_atoms=len(atoms))
        return dict(converged=True, E_eV=E_eV, opt_atoms=opt_atoms,
                    wall_time_s=dt, n_atoms=len(atoms))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def lammps_si_tersoff(atoms: Atoms,
                        optimize: bool = False,
                        timeout: int = 600,
                        threads: int | None = None,
                        mpi_procs: int = 1) -> dict:
    """Run LAMMPS + Tersoff (Tersoff 1988 Si parameters) on a free Si cluster.

    Same protocol as lammps_airebo but with the Si.tersoff parameter file
    and Si atomic mass (28.0855 amu).
    """
    workdir = Path(tempfile.mkdtemp(prefix="lmp_perco_si_"))
    try:
        _write_data_file(atoms, workdir / "data.in",
                          mass=28.0855, header="Si Tersoff cluster",
                          pad=20.0)
        tmpl = SI_INPUT_TEMPLATE_OPT if optimize else SI_INPUT_TEMPLATE_SP
        (workdir / "in.lmp").write_text(tmpl.format(pot=SI_TERSOFF_POT))

        env = os.environ.copy()
        if threads is not None:
            env["OMP_NUM_THREADS"] = str(threads)
        if mpi_procs > 1:
            cmd = [MPIRUN_BIN, "-np", str(mpi_procs),
                   "--bind-to", "none",
                   LMP_BIN, "-in", "in.lmp",
                   "-screen", "none", "-log", "log.lammps"]
        else:
            cmd = [LMP_BIN, "-in", "in.lmp",
                   "-screen", "none", "-log", "log.lammps"]
        t0 = time.perf_counter()
        try:
            res = subprocess.run(cmd, cwd=workdir, capture_output=True,
                                 text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return dict(converged=False, error="timeout",
                        E_eV=float("nan"), opt_atoms=atoms.copy(),
                        wall_time_s=timeout, n_atoms=len(atoms))
        dt = time.perf_counter() - t0
        log = (workdir / "log.lammps").read_text() if (workdir / "log.lammps").exists() \
              else (res.stdout + res.stderr)
        m = re.search(r"FINAL_ENERGY\s+(-?\d+\.\d+)", log)
        if m is None:
            return dict(converged=False,
                        error=f"no FINAL_ENERGY parsed; tail:\n{log[-1500:]}",
                        E_eV=float("nan"), opt_atoms=atoms.copy(),
                        wall_time_s=dt, n_atoms=len(atoms))
        E_eV = float(m.group(1))

        opt_atoms = atoms.copy()
        if optimize and (workdir / "dump.last").exists():
            try:
                pos = _read_relaxed_positions(workdir / "dump.last", len(atoms))
                opt_atoms.set_positions(pos)
            except Exception as e:
                return dict(converged=False,
                            error=f"dump-parse failed: {e}",
                            E_eV=E_eV, opt_atoms=atoms.copy(),
                            wall_time_s=dt, n_atoms=len(atoms))
        return dict(converged=True, E_eV=E_eV, opt_atoms=opt_atoms,
                    wall_time_s=dt, n_atoms=len(atoms))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    from lattices import build_honeycomb, P_C
    from percolation import sample_percolation
    lat = build_honeycomb(6)
    cl = sample_percolation(lat, P_C["honeycomb"], seed=0)
    a = Atoms(symbols=["C"] * cl.n_atoms, positions=cl.positions)
    sp = lammps_airebo(a, optimize=False)
    print(f"sp:  N={cl.n_atoms} E={sp['E_eV']:+.3f} eV  ({sp['wall_time_s']:.2f}s)")
    opt = lammps_airebo(a, optimize=True)
    print(f"opt: N={cl.n_atoms} E={opt['E_eV']:+.3f} eV  ({opt['wall_time_s']:.2f}s)")
    if opt["converged"]:
        rms = float(np.sqrt(np.mean(np.sum(
            (opt["opt_atoms"].get_positions() - cl.positions) ** 2, axis=1))))
        print(f"  ⟨|Δr|⟩={rms:.2f} Å  ΔE_relax={sp['E_eV']-opt['E_eV']:+.3f} eV")
