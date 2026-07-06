"""SIESTA isolated-atom references for the 9 perovskite elements.

Each atom is placed alone in a 20-A cubic vacuum box, spin-polarised, with
the same SZP basis and MeshCutoff 250 Ry used in the perovskite production
run.  The unpaired-electron count comes from the GROUND_STATE_UHF dict in
oce.atomic_table (xtb-derived but standard textbook).

Output: data/perovskites/atom_refs.json
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from oce.atomic_table import GROUND_STATE_UHF
from data.perovskites.siesta_perov import SIESTA_BIN, PSEUDO_DIR, Z_TABLE

ATOM_FDF = """\
SystemName        atom_{el}
SystemLabel       atom_{el}

NumberOfAtoms     1
NumberOfSpecies   1

%block ChemicalSpeciesLabel
 1  {z}  {el}
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

%block kgrid_Monkhorst_Pack
1  0  0  0.0
0  1  0  0.0
0  0  1  0.0
%endblock kgrid_Monkhorst_Pack

MeshCutoff          250.0 Ry
DM.MixingWeight     0.05
DM.NumberPulay      8
MaxSCFIterations    300
DM.Tolerance        1.0e-3
SCF.Mix             density

SpinPolarized       .true.
NetCharge           0.0

ElectronicTemperature  300 K
SolutionMethod         diagon

WriteForces            .false.

# Initial spin guess from atomic ground-state unpaired electrons
%block DM.InitSpin
 1  +{uhf_half}
%endblock DM.InitSpin
"""


def run_atom(element: str, threads: int = 2, timeout: int = 1200,
              keep_dir: Path | None = None) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix=f"siesta_atom_{element}_"))
    try:
        if element not in GROUND_STATE_UHF:
            return dict(element=element, error="not in GROUND_STATE_UHF",
                        E_eV=float("nan"))
        if element not in Z_TABLE:
            return dict(element=element, error="not in Z_TABLE",
                        E_eV=float("nan"))
        uhf = GROUND_STATE_UHF[element]
        z = Z_TABLE[element]
        text = ATOM_FDF.format(el=element, z=z, uhf_half=uhf / 2.0)
        (workdir / "input.fdf").write_text(text)
        # link pseudo
        src = PSEUDO_DIR / f"{element}.psml"
        if not src.exists():
            return dict(element=element, error=f"missing pseudo {src}",
                        E_eV=float("nan"))
        shutil.copy(src, workdir / f"{element}.psml")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)

        t0 = time.perf_counter()
        with open(workdir / "input.fdf", "r") as fin, \
             open(workdir / "siesta.out", "w") as fout:
            res = subprocess.run([SIESTA_BIN], cwd=workdir, stdin=fin,
                                  stdout=fout, stderr=subprocess.PIPE,
                                  timeout=timeout, env=env)
        dt = time.perf_counter() - t0

        out = (workdir / "siesta.out").read_text() if (workdir / "siesta.out").exists() else ""
        if res.returncode != 0 and "Job completed" not in out:
            return dict(element=element, converged=False,
                         error=f"returncode={res.returncode}\n"
                                f"out_tail={out[-1500:]}",
                         E_eV=float("nan"), wall_time_s=dt)

        m = re.search(r"siesta:\s+Total\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            m = re.search(r"Total energy\s+=\s+(-?\d+\.\d+)", out)
        if m is None:
            return dict(element=element, converged=False,
                         error=f"no energy parsed; tail:\n{out[-2000:]}",
                         E_eV=float("nan"), wall_time_s=dt)
        E_eV = float(m.group(1))
        scf_ok = "SCF cycle converged" in out or "converged" in out.lower()
        return dict(element=element, converged=True, E_eV=E_eV,
                     uhf=uhf, wall_time_s=dt, scf_converged=scf_ok)
    finally:
        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in workdir.iterdir():
                shutil.copy(f, keep_dir)
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    elements = ["Cs", "K", "Pb", "Sn", "Ge", "I", "Br", "Cl", "F"]
    out_path = ROOT / "data" / "perovskites" / "atom_refs.json"
    print(f"Running SIESTA atom refs for {elements}")
    t0 = time.time()
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_atom, el, threads=2): el for el in elements}
        for fut in as_completed(futs):
            r = fut.result()
            el = r["element"]
            results[el] = r
            ok = "OK" if r.get("converged") else "FAIL"
            print(f"  {el:>3s}  E={r.get('E_eV', 'nan'):>14}  "
                  f"wall={r.get('wall_time_s', 0):.1f}s  {ok}  "
                  f"uhf={r.get('uhf', '?')}")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")
    print(f"Total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
