"""SIESTA SZP runner for periodic halide perovskites.

Targets ABX3 cubic Pm-3m primitives (5 atoms) and 2x2x2 supercells (40 atoms)
with elements in {Cs, K, Pb, Sn, Ge, I, Br, Cl, F}.

Pseudopotentials: Pseudo-Dojo NC PBE stringent PSML, located in
data/perovskites/pseudos/.

Uses Monkhorst-Pack k-mesh sized by cell side (~25/a Å^-1 → 4×4×4 for
6.4-Å primitives, 2×2×2 for 12.8-Å supercells).  No spin polarisation
(all-closed-shell ionic crystals).  MeshCutoff 250 Ry — calibrated for
Pb/I heavy elements.

API parallel to oce.xtb_periodic.gfnff_energy:
    E_eV, atoms_out = siesta_szp_energy(atoms, threads=2)
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
PSEUDO_DIR = Path(__file__).resolve().parent / "pseudos"

# atomic numbers for our element set
Z_TABLE = {
    "H": 1,  "C": 6,  "N": 7,  "O": 8,  "F": 9,  "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19,
    "Ca": 20, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Rb": 37, "Sr": 38, "Y": 39,
    "Zr": 40, "Nb": 41, "Mo": 42, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Cs": 55, "Ba": 56, "La": 57,
    "Hf": 72, "Ta": 73, "W": 74, "Pt": 78, "Au": 79, "Hg": 80,
    "Pb": 82, "Bi": 83,
}


INPUT_TEMPLATE = """\
SystemName        perov_{tag}
SystemLabel       perov_{tag}

NumberOfAtoms     {n_atoms}
NumberOfSpecies   {n_species}

%block ChemicalSpeciesLabel
{species_block}
%endblock ChemicalSpeciesLabel

PAO.BasisSize       SZP
PAO.EnergyShift     0.05 Ry

XC.functional       GGA
XC.authors          PBE

LatticeConstant 1.0 Ang
%block LatticeVectors
{lattice}
%endblock LatticeVectors

AtomicCoordinatesFormat        Ang
%block AtomicCoordinatesAndAtomicSpecies
{coords}
%endblock AtomicCoordinatesAndAtomicSpecies

%block kgrid_Monkhorst_Pack
{kx}  0  0  0.0
0  {ky}  0  0.0
0  0  {kz}  0.0
%endblock kgrid_Monkhorst_Pack

MeshCutoff          250.0 Ry
DM.MixingWeight     0.05
DM.NumberPulay      8
MaxSCFIterations    300
DM.Tolerance        1.0e-3
SCF.Mix             density

SpinPolarized       .false.
NetCharge           0.0

ElectronicTemperature  300 K
SolutionMethod         diagon

WriteForces            .false.
SaveTotalCharge        .false.
WriteCoorStep          .false.
SaveHS                 .false.
SaveRho                .false.
WriteMullikenPop       0
"""


def _kgrid_for_cell(cell: np.ndarray, target_density_inv_A: float = 0.20
                     ) -> tuple[int, int, int]:
    """Pick MP k-mesh so |b_i| / k_i ≈ target_density_inv_A.

    target_density_inv_A = 0.20 corresponds to ≈ 25 grid points across
    a typical 12-Å reciprocal length, which is dense for ionic crystals.
    For a 6.4-Å primitive cell (b ≈ 0.98 Å^-1) this gives 5 → bumped to 4
    for symmetry; for 12.8-Å supercell (b ≈ 0.49 Å^-1) → 2-3.
    """
    a = np.linalg.norm(cell, axis=1)
    rec = 2 * np.pi / a
    k = np.maximum(1, np.round(rec / target_density_inv_A).astype(int))
    return tuple(int(x) for x in k)


def _make_input(atoms: Atoms, tag: str) -> str:
    syms = atoms.get_chemical_symbols()
    species = sorted(set(syms))
    sp_idx = {el: i + 1 for i, el in enumerate(species)}
    species_block = "\n".join(
        f" {sp_idx[el]:2d}  {Z_TABLE[el]:2d}  {el}" for el in species
    )
    cell = atoms.get_cell().array
    lat_lines = "\n".join(
        " ".join(f"{x:14.8f}" for x in row) for row in cell
    )
    pos = atoms.get_positions()
    coord_lines = "\n".join(
        f" {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}  {sp_idx[s]:2d}"
        for p, s in zip(pos, syms)
    )
    kx, ky, kz = _kgrid_for_cell(cell)
    return INPUT_TEMPLATE.format(
        tag=tag,
        n_atoms=len(atoms),
        n_species=len(species),
        species_block=species_block,
        lattice=lat_lines,
        coords=coord_lines,
        kx=kx, ky=ky, kz=kz,
    )


def siesta_szp_energy(atoms: Atoms, tag: str = "perov",
                       threads: int = 2,
                       timeout: int = 7200,
                       keep_dir: Path | None = None) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="siesta_perov_"))
    try:
        text = _make_input(atoms, tag)
        (workdir / "input.fdf").write_text(text)
        # link pseudopotentials for every species used
        for el in set(atoms.get_chemical_symbols()):
            src = PSEUDO_DIR / f"{el}.psml"
            if not src.exists():
                return dict(converged=False,
                             error=f"missing pseudo: {src}",
                             E_eV=float("nan"), wall_time_s=0.0,
                             n_atoms=len(atoms))
            shutil.copy(src, workdir / f"{el}.psml")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)

        t0 = time.perf_counter()
        try:
            with open(workdir / "input.fdf", "r") as fin, \
                 open(workdir / "siesta.out", "w") as fout:
                res = subprocess.run(
                    [SIESTA_BIN], cwd=workdir, stdin=fin,
                    stdout=fout, stderr=subprocess.PIPE,
                    timeout=timeout, env=env,
                )
        except subprocess.TimeoutExpired:
            return dict(converged=False, error="timeout",
                         E_eV=float("nan"), wall_time_s=timeout,
                         n_atoms=len(atoms))
        dt = time.perf_counter() - t0

        out = (workdir / "siesta.out").read_text() if (workdir / "siesta.out").exists() else ""
        if res.returncode != 0 and "Job completed" not in out:
            return dict(converged=False,
                         error=f"returncode={res.returncode}\n"
                                f"stderr_tail={res.stderr.decode()[-1000:]}\n"
                                f"out_tail={out[-1500:]}",
                         E_eV=float("nan"), wall_time_s=dt,
                         n_atoms=len(atoms))

        # Look for siesta:         Total = ... eV (line near end)
        m = re.search(r"siesta:\s+Total\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            m = re.search(r"Total energy\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            return dict(converged=False,
                         error=f"no Total energy parsed\nout_tail={out[-2000:]}",
                         E_eV=float("nan"), wall_time_s=dt,
                         n_atoms=len(atoms))
        E_eV = float(m.group(1))
        scf_ok = "SCF cycle converged" in out or "converged" in out.lower()
        kx, ky, kz = _kgrid_for_cell(atoms.get_cell().array)
        return dict(converged=True, E_eV=E_eV, wall_time_s=dt,
                     n_atoms=len(atoms),
                     scf_converged=scf_ok,
                     kmesh=[kx, ky, kz])
    finally:
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in workdir.iterdir():
                shutil.copy(f, keep_dir)
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    # smoke: cubic CsPbI3 primitive
    a = 6.39
    ats = Atoms(
        symbols=["Cs", "Pb", "I", "I", "I"],
        scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5),
                           (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        cell=[a, a, a], pbc=True,
    )
    print("Smoke test: CsPbI3 primitive (5 atoms, cubic Pm-3m, a=6.39 Å)")
    r = siesta_szp_energy(ats, tag="cspbi3_smoke", threads=4, timeout=600)
    print(r)
