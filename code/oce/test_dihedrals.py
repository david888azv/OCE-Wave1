"""Quick test: does adding 4-figures (dihedrals) help in-domain isomer
ranking and the C6H14 transfer ranking that's the current OCE limit?"""
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
                           fit_sa_multiseed)


def evaluate(yhat, y, entries, label):
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rho, _ = spearmanr(y, yhat)
    tau, _ = kendalltau(y, yhat)
    groups = defaultdict(list)
    for k, e in enumerate(entries):
        groups[e.formula].append(k)
    correct, total = 0, 0
    fails = []
    for fml, idxs in groups.items():
        if len(idxs) < 2: continue
        true_ord = sorted(idxs, key=lambda i: y[i])
        pred_ord = sorted(idxs, key=lambda i: yhat[i])
        ok = true_ord == pred_ord
        correct += int(ok); total += 1
        if not ok:
            fails.append(fml)
    return dict(rmse=rmse, mae=mae, spearman=float(rho), kendall=float(tau),
                iso=correct, total=total, fails=fails, label=label)


def run(include_dihedrals: bool, label: str):
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    test = load_dataset(base / "data" / "molecules" / "test.json")
    transfer = load_dataset(base / "data" / "molecules" / "transfer.json")

    Xtr, ytr, keys, _ = build_design_matrix(train, table,
                                             include_dihedrals=include_dihedrals)
    formulas_tr = [e.formula for e in train]
    w_gc = gc_weights(ytr, formulas_tr, T=0.5)

    def project(entries):
        vecs = []
        for e in entries:
            atoms = entry_to_atoms(e)
            one, two, three, four = enumerate_figures(
                atoms, table, include_dihedrals=include_dihedrals)
            four_arg = four if include_dihedrals else None
            vecs.append(correlations_for_molecule(
                one, two, table, three_figs=three, four_figs=four_arg))
        X = np.array([v.as_array(keys) for v in vecs])
        y = np.array([e.energy_eV for e in entries])
        return X, y

    Xte, yte = project(test)
    Xtf, ytf = project(transfer)

    print(f"\n{'=' * 72}\n{label}: train {Xtr.shape}, test {Xte.shape}, transfer {Xtf.shape}\n")

    print(f"{'method':12s} {'n_feat':>6s} {'TEST iso':>9s} {'TEST RMSE':>9s} "
          f"{'TRANSFER iso':>13s} {'TRANSFER RMSE':>14s} {'TEST fails':<20s} {'TRANSFER fails':<20s}")

    for name, fn in [("ridge", METHODS["ridge"]),
                     ("ga", METHODS["ga"]),
                     ("sa_gc[5x]", lambda X, y: fit_sa_multiseed(X, y,
                                                  n_seeds=5,
                                                  sample_weight=w_gc)[0])]:
        try:
            r = fn(Xtr, ytr)
            mtest = evaluate(r.predict(Xte), yte, test, "test")
            mtrans = evaluate(r.predict(Xtf), ytf, transfer, "transfer")
            print(f"{name:12s} {r.n_features:>6d} "
                  f"{mtest['iso']}/{mtest['total']:>6d} "
                  f"{mtest['rmse']:>9.3f} "
                  f"{mtrans['iso']}/{mtrans['total']:>11d} "
                  f"{mtrans['rmse']:>14.3f} "
                  f"{','.join(mtest['fails']):<20s} "
                  f"{','.join(mtrans['fails']):<20s}")
        except Exception as e:
            print(f"{name:12s}  FAILED: {e}")


if __name__ == "__main__":
    run(include_dihedrals=False, label="WITHOUT dihedrals (1F+2F+3F)")
    run(include_dihedrals=True,  label="WITH dihedrals (1F+2F+3F+4F)")
