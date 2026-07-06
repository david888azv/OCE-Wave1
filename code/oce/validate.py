"""Validate OCE on test set: per-isomer-group hierarchization + global metrics."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

from oce.atomic_table import load_table
from oce.dataset import load_dataset
from oce.fit import build_design_matrix, fit_model, report_fit


def evaluate(model, X, y, entries, label="TEST"):
    yhat = X @ np.array(model.J) + model.intercept
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rho, _ = spearmanr(y, yhat)
    tau, _ = kendalltau(y, yhat)

    print(f"\n=== {label}  ({model.method}) ===")
    print(f"  n = {len(y)}    RMSE = {rmse:.4f} eV    MAE = {mae:.4f} eV")
    print(f"  Spearman ρ = {rho:.4f}    Kendall τ = {tau:.4f}")
    print(f"  per-molecule:")
    for i, e in enumerate(entries):
        print(f"    {e.name:25s} {e.formula:8s}  "
              f"y={y[i]:+10.3f}  ŷ={yhat[i]:+10.3f}  Δ={err[i]:+7.3f} eV")

    # Per-isomer-group ranking
    groups: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        groups[e.formula].append(i)
    print(f"\n  --- isomer-group rankings ---")
    correct, total = 0, 0
    for formula, idxs in groups.items():
        if len(idxs) < 2:
            continue
        true_order = sorted(idxs, key=lambda i: y[i])
        pred_order = sorted(idxs, key=lambda i: yhat[i])
        ok = true_order == pred_order
        correct += int(ok)
        total += 1
        print(f"\n  {formula} ({len(idxs)} isomers)  ranking match: {ok}")
        for rank, i in enumerate(true_order):
            pos_pred = pred_order.index(i) + 1
            print(f"    true #{rank+1}: {entries[i].name:25s}"
                  f"   y={y[i]:+10.3f}  ŷ={yhat[i]:+10.3f}  pred-rank=#{pos_pred}")
    print(f"\n  Isomer groups with correct ranking: {correct}/{total}")
    return dict(rmse=rmse, mae=mae, spearman=rho, kendall=tau,
                isomer_correct=correct, isomer_total=total)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    test  = load_dataset(base / "data" / "molecules" / "test.json")

    # Build design matrices (test must use the SAME feature_keys order as train)
    Xtr, ytr, keys, _ = build_design_matrix(train, table)

    # Re-build test using the train keys
    from oce.figures import enumerate_figures
    from oce.correlations import correlations_for_molecule
    from oce.dataset import entry_to_atoms
    test_vecs = []
    for e in test:
        atoms = entry_to_atoms(e)
        one, two, three, four = enumerate_figures(atoms, table)
        test_vecs.append(correlations_for_molecule(one, two, table,
                                                    three_figs=three))
    Xte = np.array([v.as_array(keys) for v in test_vecs])
    yte = np.array([e.energy_eV for e in test])

    # Sanity check: any test feature not in train?
    new_feats = set()
    for v in test_vecs:
        for k in v.pi:
            if k not in keys:
                new_feats.add(k)
    if new_feats:
        print(f"WARNING: {len(new_feats)} feature(s) in test absent in train:")
        for k in new_feats:
            print(f"   {k}")

    for alpha in [0.0, 1e-3, 1e-2, 1e-1, 1.0]:
        model = fit_model(Xtr, ytr, keys, alpha=alpha)
        report_fit(model, Xtr, ytr, train, label=f"TRAIN α={alpha}")
        evaluate(model, Xte, yte, test, label=f"TEST α={alpha}")
        print("="*72)
