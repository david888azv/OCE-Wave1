"""Build Wave-1.5 feature matrix for perovskite candidate pool (n=1784).

Mirrors data/mofs_qmof/build_wave15.py: adaptive CT2F + Madelung-CT2F + CT3F
on each perovskite supercell, with charges taken from `formal_charges`
(ionic Cs+/K+, Pb+2/Sn+2/Ge+2, I-/Br-/Cl-/F-).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"


def _record_to_atoms(rec: dict) -> Atoms:
    return Atoms(
        symbols=rec["symbols"],
        positions=rec["positions"],
        cell=rec["cell"],
        pbc=rec.get("pbc", [True, True, True]),
    )


def _process_one(args):
    import sys
    sys.path.insert(0, str(ROOT))
    from oce_carbon.features.wave15 import (
        enumerate_ct2f_adaptive, enumerate_madct2f, enumerate_ct3f
    )
    idx, rec_path = args
    rec = json.loads(Path(rec_path).read_text())
    atoms = _record_to_atoms(rec)
    bag: dict[tuple, float] = {}

    a_bag = enumerate_ct2f_adaptive(atoms)
    for k, v in a_bag.items():
        bag[k] = float(v)

    charges = rec.get("formal_charges", None)
    if charges is not None and len(charges) == len(atoms):
        m_bag = enumerate_madct2f(atoms, charges=charges)
        for k, v in m_bag.items():
            bag[k] = float(v)

    c_bag = enumerate_ct3f(atoms)
    for k, v in c_bag.items():
        bag[k] = float(v)

    return idx, dict(bag), len(atoms), rec.get("name", f"#{idx}"), \
        rec.get("family", "")


def build():
    structures = json.loads((SD / "structures.json").read_text())
    n = len(structures)
    print(f"[perov-wave15] {n} structures")

    tmp_dir = SD / ".tmp_wave15"
    tmp_dir.mkdir(exist_ok=True)
    paths = []
    for i, r in enumerate(structures):
        p = tmp_dir / f"r{i:06d}.json"
        p.write_text(json.dumps(r))
        paths.append(p)

    t0 = time.time()
    bags: list[dict] = [None] * n
    sizes: list[int] = [0] * n
    names: list[str] = [""] * n
    families: list[str] = [""] * n
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_process_one, (i, str(paths[i]))) for i in range(n)]
        done = 0
        for fut in as_completed(futs):
            idx, bag, sz, nm, fam = fut.result()
            bags[idx] = bag
            sizes[idx] = sz
            names[idx] = nm
            families[idx] = fam
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                el = time.time() - t0
                print(f"  {done}/{n}   {el:.1f} s   {el/done*1000:.1f} ms/struct")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    support = Counter()
    for bag in bags:
        for k in bag:
            support[k] += 1
    thr = int(np.ceil(0.01 * n))
    kept = sorted([k for k, s in support.items() if s >= thr])
    print(f"[perov-wave15] raw keys: {len(support)}, kept (>={thr}): {len(kept)}")

    cls_counts = Counter(k[0] for k in kept)
    for cls, c in sorted(cls_counts.items()):
        print(f"   {cls:<12s} {c:>4d} features")

    key_idx = {k: j for j, k in enumerate(kept)}
    X = np.zeros((n, len(kept)), dtype=np.float64)
    for i, bag in enumerate(bags):
        for k, v in bag.items():
            j = key_idx.get(k)
            if j is not None:
                X[i, j] = v

    np.savez(SD / "features_wave15.npz",
              X=X,
              names=np.asarray(names, dtype=object),
              sizes=np.asarray(sizes, dtype=np.int32),
              families=np.asarray(families, dtype=object))
    (SD / "feature_index_wave15.json").write_text(
        json.dumps([{"key": list(k), "support": support[k]} for k in kept],
                    indent=2))
    print(f"[perov-wave15] features_wave15.npz  X={X.shape}")


if __name__ == "__main__":
    build()
