"""Test OCE on the configurational benzene H/F dataset.

We split the 64 configs into train / test by substitution count and
report:
- isomer ordering within each fluorine count (the standard ortho/meta/para
  test on di-F, the three tri-F isomers, etc.)
- equivariance: predictions for D6h-equivalent masks should match (this
  is a basis-symmetry check, not just an energy-ranking check).
- transferability: train on low- and high-fluorine configs, predict the
  middle (trisubstituted) compositions — direct test of the configurational
  CE additivity claim.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from oce.atomic_table import load_table
from oce.configurational import (entry_to_atoms as conf_to_atoms,
                                  load_dataset as load_conf,
                                  canonical_orbit_label)
from oce.correlations import correlations_for_molecule, collect_feature_keys
from oce.figures import enumerate_figures
from oce.selection import (METHODS, gc_weights, fit_ridge_gc,
                           fit_sa_multiseed)


def build_X(entries, table, keys=None, include_dihedrals=False):
    vecs = []
    for e in entries:
        atoms = conf_to_atoms(e)
        one, two, three, four = enumerate_figures(
            atoms, table, include_dihedrals=include_dihedrals)
        four_arg = four if include_dihedrals else None
        vecs.append(correlations_for_molecule(
            one, two, table, three_figs=three, four_figs=four_arg))
    if keys is None:
        keys = collect_feature_keys(vecs)
    X = np.array([v.as_array(keys) for v in vecs])
    return X, keys


def evaluate_orbit_invariance(yhat, entries):
    """Check that D6h-equivalent masks produce the same prediction."""
    orbits: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        orbits[canonical_orbit_label(tuple(e.mask))].append(i)
    spreads = []
    for orb, idxs in orbits.items():
        if len(idxs) < 2:
            continue
        vals = yhat[idxs]
        spreads.append(float(vals.max() - vals.min()))
    return float(np.mean(spreads)) if spreads else 0.0, len(orbits)


def evaluate_isomer_orderings(yhat, y, entries):
    """For each fluorine count, check that the unique-isomer ordering is
    correctly predicted. Returns dict {n_F: (correct, total)}."""
    by_nF: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(entries):
        orb = canonical_orbit_label(tuple(e.mask))
        by_nF[e.n_F][orb].append(i)

    results = {}
    fail_details = {}
    for nF, orbits in sorted(by_nF.items()):
        unique_isomers = list(orbits.keys())
        if len(unique_isomers) < 2:
            continue
        # take the median predicted value within each orbit for stability
        rep_pred = {orb: float(np.median(yhat[orbits[orb]])) for orb in unique_isomers}
        rep_true = {orb: float(np.median(y[orbits[orb]])) for orb in unique_isomers}
        true_order = sorted(unique_isomers, key=lambda o: rep_true[o])
        pred_order = sorted(unique_isomers, key=lambda o: rep_pred[o])
        ok = (true_order == pred_order)
        results[nF] = (1 if ok else 0, 1)
        if not ok:
            fail_details[nF] = (true_order, pred_order, rep_true, rep_pred)
    return results, fail_details


def main():
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    entries = load_conf(base / "data" / "configurational" / "benzene_hf.json")
    print(f"Total configs: {len(entries)}")

    # Build full feature matrix (no dihedrals; benzene ring locks geometry)
    X_all, keys = build_X(entries, table)
    y_all = np.array([e.energy_eV for e in entries])
    print(f"Feature matrix: {X_all.shape}")

    # ---- Splits ----
    # SPLIT A: train on configs with nF ∉ {2, 3, 4}, test on all
    # (i.e. train only on benzene + monoF + pentaF + C6F6 — 1+6+6+1 = 14 configs)
    # This is the brutal test: predict the middle of the chemical space
    # from only the boundaries.
    train_idx_A = [i for i, e in enumerate(entries)
                    if e.n_F in (0, 1, 5, 6)]
    test_idx_A = list(range(len(entries)))

    # SPLIT B: train on 80% random, test on 20%
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(entries))
    cut = int(0.8 * len(entries))
    train_idx_B = sorted(perm[:cut].tolist())
    test_idx_B = sorted(perm[cut:].tolist())

    formulas = [f"C6H{6-e.n_F}F{e.n_F}" for e in entries]

    for split_name, train_idx, test_idx in [
        ("A: train={0,1,5,6}-F, test=all (extreme transfer)", train_idx_A, test_idx_A),
        ("B: random 80/20", train_idx_B, test_idx_B),
    ]:
        Xtr = X_all[train_idx]
        ytr = y_all[train_idx]
        Xte = X_all[test_idx]
        yte = y_all[test_idx]
        train_entries = [entries[i] for i in train_idx]
        test_entries = [entries[i] for i in test_idx]
        formulas_tr = [formulas[i] for i in train_idx]
        w_gc = gc_weights(ytr, formulas_tr, T=0.5)

        print(f"\n{'=' * 80}\nSPLIT {split_name}")
        print(f"  train: {Xtr.shape}, test: {Xte.shape}")

        for name, fn in [
            ("ridge", METHODS["ridge"]),
            ("ga", METHODS["ga"]),
            ("ridge_gc", lambda X, y: fit_ridge_gc(X, y, sample_weight=w_gc)),
            ("sa_gc[5x]", lambda X, y: fit_sa_multiseed(X, y, n_seeds=5,
                                                          sample_weight=w_gc)[0]),
        ]:
            try:
                r = fn(Xtr, ytr)
                yhat = r.predict(Xte)
                err = yhat - yte
                rmse = float(np.sqrt(np.mean(err ** 2)))
                mae = float(np.mean(np.abs(err)))
                rho, _ = spearmanr(yte, yhat)
                tau, _ = kendalltau(yte, yhat)
                spread, n_orb = evaluate_orbit_invariance(yhat, test_entries)
                iso_results, fails = evaluate_isomer_orderings(yhat, yte, test_entries)
                iso_ok = sum(c for c, _ in iso_results.values())
                iso_tot = sum(t for _, t in iso_results.values())
                detail = " ".join(f"{nF}F={'OK' if c else 'FAIL'}"
                                   for nF, (c, _) in sorted(iso_results.items()))
                print(f"  {name:12s} n_feat={r.n_features:>4d}  "
                      f"RMSE={rmse:.3f}  MAE={mae:.3f}  ρ={rho:.3f}  "
                      f"iso={iso_ok}/{iso_tot}  "
                      f"orbit-spread={spread*1000:.1f}meV  ({detail})")
                for nF, (true_o, pred_o, rt, rp) in fails.items():
                    print(f"     [FAIL nF={nF}]")
                    print(f"        true:  " + " < ".join(
                        f"{o[:15]}({rt[o]:+.3f})" for o in true_o))
                    print(f"        pred:  " + " < ".join(
                        f"{o[:15]}({rp[o]:+.3f})" for o in pred_o))
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  {name:12s}  FAILED: {e}")


if __name__ == "__main__":
    main()
