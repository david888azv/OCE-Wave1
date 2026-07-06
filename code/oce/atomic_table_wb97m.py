"""Atomic orbital eigenenergy table at ωB97M-D3BJ / def2-TZVPPD.

DFT analogue of `atomic_table.py` (which uses GFN2-xTB single atoms).
Built with PySCF UKS:

  - functional: base ωB97M semilocal XC (libxc `wb97m_v` with the VV10
    NLC switched OFF).  ωB97M-D3BJ uses the *same* semilocal XC as
    ωB97M-V but replaces the VV10 nonlocal term with a D3BJ dispersion
    tail; for an isolated atom D3BJ contributes exactly zero energy and
    nothing to the orbital eigenvalues, so the eigenvalues produced here
    are the ωB97M-D3BJ atomic orbital energies (SPICE level of theory).
  - basis: def2-TZVPPD (def2-ECP for I).

To keep the OCE feature space *identical* to the xtb run (same figure
keys), we mirror the xtb table's shell labels and occupations for every
element, replacing ONLY the orbital eigenenergy ε (and the total energy).
That makes the swap a clean controlled experiment: the design-matrix
sparsity structure is unchanged; only the numeric ε values differ.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyscf import gto, dft

HARTREE_EV = 27.211386245988

# Ground-state spin = number of unpaired electrons (2S).
SPIN = {
    "H": 1, "Li": 1, "Na": 1, "K": 1, "Ca": 0, "Mg": 0,
    "C": 2, "N": 3, "O": 2, "F": 1,
    "P": 3, "S": 2, "Cl": 1, "Br": 1, "I": 1,
}
# Elements that require an ECP in def2-TZVPPD (Z > Kr).
NEEDS_ECP = {"I"}

L_NAME = {0: "s", 1: "p", 2: "d", 3: "f"}


def _l_of_label(lbl) -> str:
    # ao_labels(fmt=None) -> (atom_id, symbol, nl, m); nl like '2p'
    nl = lbl[2]
    return nl[-1]


def _run_atom(element: str):
    """UKS ωB97M(no NLC)/def2-TZVPPD on the isolated atom.

    Returns (E_total_Eh, mo_energy_eV list, l_char list, occ list) merged
    over alpha+beta spin channels.
    """
    kw = dict(atom=f"{element} 0 0 0", basis="def2-tzvppd",
              spin=SPIN[element], charge=0, verbose=0)
    if element in NEEDS_ECP:
        kw["ecp"] = "def2-tzvppd"
    mol = gto.M(**kw)
    mf = dft.UKS(mol)
    mf.xc = "wb97m_v"
    mf.nlc = ""           # D3BJ variant: no VV10 nonlocal correlation
    mf.grids.level = 4
    e = mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        e = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF not converged for {element}")

    labels = mol.ao_labels(fmt=None)
    l_idx = np.array([{"s": 0, "p": 1, "d": 2, "f": 3}.get(_l_of_label(l), 9)
                      for l in labels])
    ovlp = mol.intor("int1e_ovlp")

    rows = []  # (energy_eV, l_name, occ)
    for spin_ch in (0, 1):
        mo_e = mf.mo_energy[spin_ch]
        mo_c = mf.mo_coeff[spin_ch]
        mo_o = mf.mo_occ[spin_ch]
        for k in range(len(mo_e)):
            if mo_o[k] < 0.5:
                continue
            col = mo_c[:, k]
            pops = col * (ovlp @ col)
            by = {lq: pops[l_idx == lq].sum() for lq in set(l_idx.tolist())}
            dom = max(by, key=by.get)
            rows.append((mo_e[k] * HARTREE_EV, L_NAME.get(dom, "?"), float(mo_o[k])))
    return float(e), rows


def _valence_eps(rows, label: str, window_eV: float = 8.0) -> float:
    """Mean energy of the *valence* occupied orbitals of a given l-character.

    Valence = the highest-energy cluster of that character (within
    `window_eV` of the topmost occupied orbital of that character).  This
    excludes deep core shells (e.g. 2p of P at ~-200 eV) while keeping the
    spin-split valence set together.
    """
    es = [e for (e, l, _o) in rows if l == label]
    if not es:
        raise RuntimeError(f"no occupied {label} orbital found")
    top = max(es)
    val = [e for e in es if e >= top - window_eV]
    return float(np.mean(val))


def build(elements, xtb_table_path: Path, out_path: Path):
    xtb = json.loads(xtb_table_path.read_text())
    table = {}
    for el in elements:
        e_tot_Eh, rows = _run_atom(el)
        # mirror xtb shells/occupations, swap epsilon for DFT valence eps
        shells = []
        for sh in xtb[el]["shells"]:
            lbl = sh["label"]
            eps = _valence_eps(rows, lbl)
            shells.append({"label": lbl,
                           "epsilon_eV": eps,
                           "occupation": sh["occupation"]})
        table[el] = {"element": el,
                     "total_energy_Eh": e_tot_Eh,
                     "shells": shells}
        sh_str = "  ".join(f"{s['label']}(ε={s['epsilon_eV']:+.2f},occ={s['occupation']:.1f})"
                           for s in shells)
        print(f"{el:3s} Etot={e_tot_Eh:+.5f} Eh | {sh_str}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2))
    print(f"\nWrote {out_path}  ({len(table)} elements)")


if __name__ == "__main__":
    REPO = Path(__file__).resolve().parents[1]
    elements = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
                "Li", "Na", "K", "Ca", "Mg"]
    build(elements,
          xtb_table_path=REPO / "data" / "atoms" / "atomic_table.json",
          out_path=REPO / "data" / "atoms" / "atomic_table_wb97m.json")
