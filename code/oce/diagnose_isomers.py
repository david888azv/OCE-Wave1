"""Diagnose which isomer group fails for the best selection methods."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from oce.atomic_table import load_table
from oce.correlations import correlations_for_molecule
from oce.dataset import entry_to_atoms, load_dataset
from oce.figures import enumerate_figures
from oce.fit import build_design_matrix
from oce.selection import (METHODS, gc_weights, fit_ridge_gc,
                           fit_sa_gc, fit_sa_multiseed)


def report_groups(yhat, y, entries, label):
    print(f"\n=== {label} ===")
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[e.formula].append(i)
    for formula, idxs in groups.items():
        if len(idxs) < 2:
            continue
        true_order = sorted(idxs, key=lambda i: y[i])
        pred_order = sorted(idxs, key=lambda i: yhat[i])
        ok = "OK" if true_order == pred_order else "FAIL"
        print(f"\n  [{ok}] {formula} ({len(idxs)} isomers)")
        for rank, i in enumerate(true_order):
            pos = pred_order.index(i) + 1
            mark = "" if pos == rank + 1 else f"  ← pred-rank #{pos}"
            print(f"    true #{rank+1}: {entries[i].name:24s}"
                  f" y={y[i]:+10.3f}  ŷ={yhat[i]:+10.3f}  Δ={yhat[i]-y[i]:+6.3f}{mark}")


def main():
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    test = load_dataset(base / "data" / "molecules" / "test.json")

    Xtr, ytr, keys, _ = build_design_matrix(train, table)
    formulas_tr = [e.formula for e in train]
    w_gc = gc_weights(ytr, formulas_tr, T=0.5)

    test_vecs = []
    for e in test:
        atoms = entry_to_atoms(e)
        one, two, three, four = enumerate_figures(atoms, table)
        test_vecs.append(correlations_for_molecule(one, two, table,
                                                    three_figs=three))
    Xte = np.array([v.as_array(keys) for v in test_vecs])
    yte = np.array([e.energy_eV for e in test])

    for name in ["ga", "sa"]:
        result = METHODS[name](Xtr, ytr)
        yhat = result.predict(Xte)
        report_groups(yhat, yte, test, f"{name} (n_feat={result.n_features})")

    r_sgc = fit_sa_gc(Xtr, ytr, sample_weight=w_gc)
    yhat = r_sgc.predict(Xte)
    report_groups(yhat, yte, test, f"sa_gc (n_feat={r_sgc.n_features})")


if __name__ == "__main__":
    main()
