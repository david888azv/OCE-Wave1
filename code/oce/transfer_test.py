"""Transferability test: train OCE on small (≤5 heavy atoms) molecules,
predict on larger (6+ heavy atom) isomers.  This tests the additivity
property of the cluster expansion — whether J_F learned locally still
ranks bigger structures correctly."""
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
from oce.selection import (METHODS, gc_weights, fit_ridge_gc, fit_sa_gc,
                           fit_sa_multiseed)


def build_X(entries, keys, table):
    vecs = []
    new_keys = set()
    for e in entries:
        atoms = entry_to_atoms(e)
        one, two, three, four = enumerate_figures(atoms, table)
        cv = correlations_for_molecule(one, two, table, three_figs=three)
        vecs.append(cv)
        for k in cv.pi:
            if k not in keys:
                new_keys.add(k)
    X = np.array([v.as_array(keys) for v in vecs])
    return X, new_keys


def report_groups(yhat, y, entries, label):
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rho, _ = spearmanr(y, yhat)
    tau, _ = kendalltau(y, yhat)
    print(f"\n=== {label} ===")
    print(f"  RMSE = {rmse:.3f} eV   MAE = {mae:.3f} eV   "
          f"Spearman ρ = {rho:.3f}   Kendall τ = {tau:.3f}")

    groups: dict[str, list[int]] = defaultdict(list)
    for k, e in enumerate(entries):
        groups[e.formula].append(k)
    correct, total = 0, 0
    for fml, idxs in groups.items():
        if len(idxs) < 2:
            continue
        true_ord = sorted(idxs, key=lambda i: y[i])
        pred_ord = sorted(idxs, key=lambda i: yhat[i])
        ok = true_ord == pred_ord
        correct += int(ok)
        total += 1
        print(f"\n  [{'OK' if ok else 'FAIL'}] {fml} ({len(idxs)} isomers)")
        for rank, i in enumerate(true_ord):
            pos = pred_ord.index(i) + 1
            mark = "" if pos == rank + 1 else f"  ← pred-rank #{pos}"
            print(f"    true #{rank+1}: {entries[i].name:30s}"
                  f" y={y[i]:+10.3f}  ŷ={yhat[i]:+10.3f}  Δ={yhat[i]-y[i]:+6.3f}{mark}")
    print(f"\n  Isomer groups correct: {correct}/{total}")
    return correct, total


def main():
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    transfer = load_dataset(base / "data" / "molecules" / "transfer.json")

    Xtr, ytr, keys, _ = build_design_matrix(train, table)
    formulas_tr = [e.formula for e in train]
    w_gc = gc_weights(ytr, formulas_tr, T=0.5)

    Xtf, new_keys = build_X(transfer, keys, table)
    ytf = np.array([e.energy_eV for e in transfer])

    print(f"Train: {Xtr.shape},  Transfer: {Xtf.shape}")
    if new_keys:
        print(f"\nWARNING: {len(new_keys)} feature class(es) appear in transfer "
              f"set but NOT in train (will be ignored / set to 0):")
        for k in sorted(new_keys)[:15]:
            print(f"   {k}")
        if len(new_keys) > 15:
            print(f"   ... and {len(new_keys) - 15} more")

    # Fit several methods, evaluate on transfer
    summary = []
    for name in ["ridge", "sa", "ga"]:
        r = METHODS[name](Xtr, ytr)
        yhat = r.predict(Xtf)
        ok, tot = report_groups(yhat, ytf, transfer,
                                  f"{name} (n_feat={r.n_features})")
        summary.append((name, ok, tot, r.n_features))

    r_rgc = fit_ridge_gc(Xtr, ytr, sample_weight=w_gc)
    yhat = r_rgc.predict(Xtf)
    ok, tot = report_groups(yhat, ytf, transfer,
                             f"ridge_gc (n_feat={r_rgc.n_features})")
    summary.append(("ridge_gc", ok, tot, r_rgc.n_features))

    r_sgc, _ = fit_sa_multiseed(Xtr, ytr, n_seeds=5, sample_weight=w_gc)
    yhat = r_sgc.predict(Xtf)
    ok, tot = report_groups(yhat, ytf, transfer,
                             f"sa_gc[5x] (n_feat={r_sgc.n_features})")
    summary.append(("sa_gc[5x]", ok, tot, r_sgc.n_features))

    print("\n=== TRANSFER SUMMARY ===")
    print(f"{'method':12s} {'n_feat':>6s} {'iso':>6s}")
    for name, ok, tot, nf in summary:
        print(f"{name:12s} {nf:>6d}    {ok}/{tot}")


if __name__ == "__main__":
    main()
