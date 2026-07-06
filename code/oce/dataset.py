"""Build train/test molecule datasets from SMILES, optimize with xtb,
and cache the resulting geometries + total energies as JSON.

Focus: small C/H/O molecules with a mix of bond orders, plus isomer sets
that share the same molecular formula — those are the key validation case
for hierarchization (same atoms, different topology, ranking by energy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from oce.xtb_runner import xtb_energy


# Training set: diverse small molecules covering single/double/triple bonds
# and the C/H/O chemical space.
TRAIN_SMILES = {
    # alkanes (linear + branched, n=1..5)
    "methane":          "C",
    "ethane":           "CC",
    "propane":          "CCC",
    "butane":           "CCCC",
    "isobutane":        "CC(C)C",
    "pentane":          "CCCCC",
    "isopentane":       "CCC(C)C",
    "neopentane":       "CC(C)(C)C",
    # alkenes
    "ethylene":         "C=C",
    "propene":          "CC=C",
    "1-butene":         "CCC=C",
    "2-butene":         "CC=CC",
    "isobutylene":      "CC(=C)C",
    "1,3-butadiene":    "C=CC=C",
    # alkynes
    "acetylene":        "C#C",
    "propyne":          "CC#C",
    "2-butyne":         "CC#CC",
    # alcohols (longer + branched, no C2H6O / C3H8O test duplicates beyond ethanol)
    "methanol":         "CO",
    "ethanol":          "CCO",
    "n-butanol":        "CCCCO",
    "tert-butanol":     "CC(C)(C)O",
    "ethylene-glycol":  "OCCO",
    # carbonyls (extend beyond C2H4O / C3H6O test duplicates)
    "formaldehyde":     "C=O",
    "acetaldehyde":     "CC=O",
    "acetone":          "CC(=O)C",
    "butanal":          "CCCC=O",
    "2-butanone":       "CCC(=O)C",
    "glyoxal":          "O=CC=O",
    # acids
    "formic-acid":      "OC=O",
    "acetic-acid":      "CC(=O)O",
    "propanoic-acid":   "CCC(=O)O",
    # ethers
    "dimethyl-ether":   "COC",
    "diethyl-ether":    "CCOCC",
    "methyl-vinyl-ether":"COC=C",
    # rings (3,4,5,6-membered, with and without O — ring sizes must cover
    # everything we expect at inference time, since ring_size is part of
    # the 3-fig key)
    "cyclopropane":     "C1CC1",
    "cyclobutane":      "C1CCC1",
    "cyclopentane":     "C1CCCC1",
    "cyclohexane":      "C1CCCCC1",
    "cyclopropene":     "C1=CC1",
    "cyclobutene":      "C1=CCC1",
    "oxirane":          "C1CO1",
    "oxetane":          "C1COC1",
    "tetrahydrofuran":  "C1CCOC1",
    "tetrahydropyran":  "C1CCCOC1",
    "1,3-dioxolane":    "O1CCOC1",
    "1,4-dioxane":      "O1CCOCC1",
    # peroxides
    "hydrogen-peroxide":"OO",
    "methyl-peroxide":  "COO",
    # references
    "water":            "O",
    "co2":              "O=C=O",
}


# Transfer test: larger isomer sets, all heavy-atom counts > anything in
# training.  Tests whether OCE J_F learned on small molecules transfers to
# bigger ones (additivity of the cluster expansion).
TRANSFER_SMILES = {
    # C6H14 isomers (5) — classic textbook benchmark; experimental order
    # most-stable → least-stable: 2,3-DMB > 2,2-DMB > 3-MP > 2-MP > n-hexane
    "n-hexane":         "CCCCCC",
    "2-methylpentane":  "CCCC(C)C",
    "3-methylpentane":  "CCC(C)CC",
    "2,2-dimethylbutane":"CC(C)(C)CC",
    "2,3-dimethylbutane":"CC(C)C(C)C",
    # C5H12O isomers (4)
    "1-pentanol":       "CCCCCO",
    "2-pentanol":       "CCCC(C)O",
    "3-pentanol":       "CCC(O)CC",
    "neo-pentanol":     "CC(C)(C)CO",
    # C6H12O isomers (4) — includes a 6-membered ring + larger carbonyls
    "hexanal":          "CCCCCC=O",
    "2-hexanone":       "CCCCC(=O)C",
    "cyclohexanol":     "C1CCCCC1O",
    "tetrahydropyran-2-methanol":"OCC1CCCCO1",
}


# Test set: isomers of the same molecular formula — exact same atoms,
# different topology. Hierarchization test.
TEST_SMILES = {
    # C2H6O isomers
    "ethanol_iso":          "CCO",
    "dimethyl-ether_iso":   "COC",
    # C3H8O isomers
    "1-propanol":           "CCCO",
    "2-propanol":           "CC(O)C",
    "methyl-ethyl-ether":   "COCC",
    # C2H4O isomers
    "acetaldehyde_iso":     "CC=O",
    "ethylene-oxide":       "C1CO1",
    "vinyl-alcohol":        "C=CO",
    # C3H6O isomers
    "acetone_iso":           "CC(=O)C",
    "propanal":              "CCC=O",
    "propylene-oxide":       "CC1CO1",
    "allyl-alcohol":         "C=CCO",
    # other small molecules outside isomer groups
    "propane_iso":           "CCC",
    "1-butyne":              "CCC#C",
}


@dataclass
class MolEntry:
    name: str
    smiles: str
    formula: str
    n_atoms: int
    elements: list[str]
    positions: list[list[float]]
    energy_eV: float


def _smiles_to_atoms(smiles: str) -> Atoms:
    """Generate a 3D conformer from SMILES using RDKit (MMFF-pre-relaxed)."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    conf = mol.GetConformer()
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [[conf.GetAtomPosition(i).x,
                  conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z]
                 for i in range(mol.GetNumAtoms())]
    return Atoms(symbols=elements, positions=positions)


def build_dataset(smiles_dict: dict[str, str], cache: Path) -> list[MolEntry]:
    """Generate geometries (RDKit + xtb opt), record energy, write to cache."""
    entries: list[MolEntry] = []
    for name, smi in smiles_dict.items():
        atoms = _smiles_to_atoms(smi)
        E_eV, opt_atoms = xtb_energy(atoms, optimize=True)
        entries.append(MolEntry(
            name=name, smiles=smi,
            formula=opt_atoms.get_chemical_formula(),
            n_atoms=len(opt_atoms),
            elements=opt_atoms.get_chemical_symbols(),
            positions=opt_atoms.get_positions().tolist(),
            energy_eV=E_eV,
        ))
        print(f"  {name:25s} {entries[-1].formula:10s}  "
              f"E = {E_eV:+12.4f} eV")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([e.__dict__ for e in entries], indent=2))
    return entries


def load_dataset(cache: Path) -> list[MolEntry]:
    raw = json.loads(cache.read_text())
    return [MolEntry(**e) for e in raw]


def entry_to_atoms(entry: MolEntry) -> Atoms:
    return Atoms(symbols=entry.elements, positions=entry.positions)


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parents[1] / "data" / "molecules"
    only = sys.argv[1] if len(sys.argv) > 1 else "all"
    if only in ("all", "train"):
        print("=== TRAIN SET ===")
        build_dataset(TRAIN_SMILES, base / "train.json")
    if only in ("all", "test"):
        print("\n=== TEST SET ===")
        build_dataset(TEST_SMILES, base / "test.json")
    if only in ("all", "transfer"):
        print("\n=== TRANSFER SET ===")
        build_dataset(TRANSFER_SMILES, base / "transfer.json")
    print(f"\nDatasets cached under {base}")
