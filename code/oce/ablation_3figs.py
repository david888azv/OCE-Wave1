"""Ablation: does dropping 3-figures recover 4/4 isomer ranking on the
expanded training set?  The C3H6O group (propanal vs propylene-oxide,
ΔE = 93 meV) is the failure mode of the full model."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from oce.atomic_table import load_table
from oce.correlations import correlations_for_molecule
from oce.dataset import entry_to_atoms, load_dataset
from oce.figures import enumerate_figures
from oce.fit import build_design_matrix
from oce.selection import (METHODS, gc_weights, fit_ridge_gc,
                           fit_sa_gc)


def evaluate(yhat, y, entries):
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rho, _ = spearmanr(y, yhat)
    tau, _ = kendalltau(y, yhat)
    groups = defaultdict(list)
    for k, e in enumerate(entries):
        groups[e.formula].append(k)
    correct = 0
    total = 0
    for fml, idxs in groups.items():
        if len(idxs) < 2:
            continue
        true_ord = sorted(idxs, key=lambda i: y[i])
        pred_ord = sorted(idxs, key=lambda i: yhat[i])
        correct += int(true_ord == pred_ord)
        total += 1
    return rmse, mae, float(rho), float(tau), correct, total


def run(include_angles: bool, label: str):
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    test = load_dataset(base / "data" / "molecules" / "test.json")

    Xtr, ytr, keys, _ = build_design_matrix(train, table,
                                             include_angles=include_angles)
    formulas_tr = [e.formula for e in train]
    w_gc = gc_weights(ytr, formulas_tr, T=0.5)

    test_vecs = []
    for e in test:
        atoms = entry_to_atoms(e)
        one, two, three, four = enumerate_figures(atoms, table,
                                             include_angles=include_angles)
        three_arg = three if include_angles else None
        test_vecs.append(correlations_for_molecule(one, two, table,
                                                    three_figs=three_arg))
    Xte = np.array([v.as_array(keys) for v in test_vecs])
    yte = np.array([e.energy_eV for e in test])

    print(f"\n=== {label}  (Xtr={Xtr.shape}, Xte={Xte.shape}) ===")

    methods = list(METHODS.items()) + [
        ("ridge_gc", lambda X, y: fit_ridge_gc(X, y, sample_weight=w_gc)),
        ("sa_gc",    lambda X, y: fit_sa_gc(X, y, sample_weight=w_gc)),
    ]
    for name, fn in methods:
        try:
            r = fn(Xtr, ytr)
            yhat = r.predict(Xte)
            rmse, mae, rho, tau, ok, tot = evaluate(yhat, yte, test)
            print(f"  {name:12s}  n_feat={r.n_features:3d}  "
                  f"RMSE={rmse:.3f}  MAE={mae:.3f}  ρ={rho:.3f}  "
                  f"τ={tau:.3f}  iso={ok}/{tot}  t={r.fit_time_s:.2f}s")
        except Exception as e:
            print(f"  {name:12s}  FAILED: {e}")


if __name__ == "__main__":
    run(include_angles=False, label="1F+2F only (no angles)")
    run(include_angles=True,  label="1F+2F+3F (with angles)")
