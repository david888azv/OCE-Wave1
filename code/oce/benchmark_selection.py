"""End-to-end comparison of selection methods on the OCE pipeline.

For each method:
    1. Fit on the training design matrix (with 1+2+3 figures).
    2. Predict on the test design matrix (re-projected onto train feature keys).
    3. Report: n_features, CV-RMSE on train, RMSE/MAE on test, Spearman ρ,
       Kendall τ, isomer-group ranking accuracy, fit time.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from oce.atomic_table import load_table
from oce.correlations import correlations_for_molecule
from oce.dataset import entry_to_atoms, load_dataset
from oce.figures import enumerate_figures
from oce.fit import build_design_matrix
from oce.selection import (METHODS, gc_weights, fit_ridge_gc,
                           fit_sa_gc, fit_sa_multiseed,
                           fit_rank_loss, fit_sa_ensemble_rank)


def evaluate_on_test(result, X_test, y_test, entries):
    yhat = result.predict(X_test)
    err = yhat - y_test
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    rho, _ = spearmanr(y_test, yhat)
    tau, _ = kendalltau(y_test, yhat)

    # isomer ranking accuracy
    groups = defaultdict(list)
    for k, e in enumerate(entries):
        groups[e.formula].append(k)
    correct = 0
    total = 0
    for fml, idxs in groups.items():
        if len(idxs) < 2:
            continue
        true_ord = sorted(idxs, key=lambda i: y_test[i])
        pred_ord = sorted(idxs, key=lambda i: yhat[i])
        correct += int(true_ord == pred_ord)
        total += 1
    return dict(rmse=rmse, mae=mae, spearman=float(rho),
                kendall=float(tau), iso_correct=correct, iso_total=total)


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

    print(f"Train: {Xtr.shape},  Test: {Xte.shape}")
    print(f"GC weights: min={w_gc.min():.3g}  max={w_gc.max():.3g}  "
          f"mean={w_gc.mean():.3g}  (n_isomer_groups>1 = "
          f"{sum(1 for f in set(formulas_tr) if formulas_tr.count(f) > 1)})\n")

    rows = []

    # --- baseline methods (unweighted) ---
    for name, fn in METHODS.items():
        try:
            result = fn(Xtr, ytr)
            metrics = evaluate_on_test(result, Xte, yte, test)
            rows.append((name, result, metrics))
            print(f"{name:12s}  n_feat={result.n_features:4d}  "
                  f"CV-RMSE={result.cv_rmse:.3f}  "
                  f"TEST RMSE={metrics['rmse']:.3f}  "
                  f"MAE={metrics['mae']:.3f}  "
                  f"ρ={metrics['spearman']:.3f}  "
                  f"τ={metrics['kendall']:.3f}  "
                  f"iso={metrics['iso_correct']}/{metrics['iso_total']}  "
                  f"t={result.fit_time_s:.2f}s")
        except Exception as e:
            print(f"{name:12s}  FAILED: {e}")

    # --- Garbulsky-Ceder weighted methods ---
    print()
    try:
        r_rgc = fit_ridge_gc(Xtr, ytr, sample_weight=w_gc)
        m_rgc = evaluate_on_test(r_rgc, Xte, yte, test)
        rows.append(("ridge_gc", r_rgc, m_rgc))
        print(f"{'ridge_gc':12s}  n_feat={r_rgc.n_features:4d}  "
              f"CV-RMSE={r_rgc.cv_rmse:.3f}  "
              f"TEST RMSE={m_rgc['rmse']:.3f}  MAE={m_rgc['mae']:.3f}  "
              f"ρ={m_rgc['spearman']:.3f}  τ={m_rgc['kendall']:.3f}  "
              f"iso={m_rgc['iso_correct']}/{m_rgc['iso_total']}  "
              f"t={r_rgc.fit_time_s:.2f}s")
    except Exception as e:
        print(f"{'ridge_gc':12s}  FAILED: {e}")

    try:
        # multi-seed SA-GC: pick best by CV-RMSE (single seed is unstable
        # because Jaccard across seeds is only ~0.2)
        r_sgc, _ = fit_sa_multiseed(Xtr, ytr, n_seeds=5, sample_weight=w_gc)
        m_sgc = evaluate_on_test(r_sgc, Xte, yte, test)
        rows.append(("sa_gc", r_sgc, m_sgc))
        print(f"{'sa_gc[5x]':12s}  n_feat={r_sgc.n_features:4d}  "
              f"CV-RMSE={r_sgc.cv_rmse:.3f}  "
              f"TEST RMSE={m_sgc['rmse']:.3f}  MAE={m_sgc['mae']:.3f}  "
              f"ρ={m_sgc['spearman']:.3f}  τ={m_sgc['kendall']:.3f}  "
              f"iso={m_sgc['iso_correct']}/{m_sgc['iso_total']}  "
              f"t={r_sgc.fit_time_s:.2f}s")
    except Exception as e:
        print(f"{'sa_gc':12s}  FAILED: {e}")

    # --- rank-loss methods (pairwise hinge on isomer groups) ---
    print()
    try:
        r_rk = fit_rank_loss(Xtr, ytr, formulas_tr,
                             alpha=1e-2, lambda_rank=5.0)
        m_rk = evaluate_on_test(r_rk, Xte, yte, test)
        rows.append(("rank_full", r_rk, m_rk))
        print(f"{'rank_full':12s}  n_feat={r_rk.n_features:4d}  "
              f"CV-RMSE={r_rk.cv_rmse:.3f}  "
              f"TEST RMSE={m_rk['rmse']:.3f}  MAE={m_rk['mae']:.3f}  "
              f"ρ={m_rk['spearman']:.3f}  τ={m_rk['kendall']:.3f}  "
              f"iso={m_rk['iso_correct']}/{m_rk['iso_total']}  "
              f"t={r_rk.fit_time_s:.2f}s")
    except Exception as e:
        print(f"{'rank_full':12s}  FAILED: {e}")

    try:
        # SA selects features, then rank loss refits on that subset
        sa_res = METHODS["sa"](Xtr, ytr)
        r_sk = fit_rank_loss(Xtr, ytr, formulas_tr,
                             alpha=1e-2, lambda_rank=5.0,
                             selected_idx=sa_res.selected_idx)
        m_sk = evaluate_on_test(r_sk, Xte, yte, test)
        rows.append(("sa+rank", r_sk, m_sk))
        print(f"{'sa+rank':12s}  n_feat={r_sk.n_features:4d}  "
              f"CV-RMSE={r_sk.cv_rmse:.3f}  "
              f"TEST RMSE={m_sk['rmse']:.3f}  MAE={m_sk['mae']:.3f}  "
              f"ρ={m_sk['spearman']:.3f}  τ={m_sk['kendall']:.3f}  "
              f"iso={m_sk['iso_correct']}/{m_sk['iso_total']}  "
              f"t={r_sk.fit_time_s:.2f}s")
    except Exception as e:
        print(f"{'sa+rank':12s}  FAILED: {e}")

    try:
        r_en = fit_sa_ensemble_rank(Xtr, ytr, formulas_tr, n_seeds=5,
                                     lambda_rank=5.0, combine="union")
        m_en = evaluate_on_test(r_en, Xte, yte, test)
        rows.append(("sa5+rank", r_en, m_en))
        print(f"{'sa5+rank':12s}  n_feat={r_en.n_features:4d}  "
              f"CV-RMSE={r_en.cv_rmse:.3f}  "
              f"TEST RMSE={m_en['rmse']:.3f}  MAE={m_en['mae']:.3f}  "
              f"ρ={m_en['spearman']:.3f}  τ={m_en['kendall']:.3f}  "
              f"iso={m_en['iso_correct']}/{m_en['iso_total']}  "
              f"t={r_en.fit_time_s:.2f}s")
    except Exception as e:
        print(f"{'sa5+rank':12s}  FAILED: {e}")

    # --- multi-seed SA stability (no GC weights) ---
    print()
    print("--- SA stability across 5 seeds (unweighted) ---")
    try:
        best, stats = fit_sa_multiseed(Xtr, ytr, n_seeds=5)
        m_best = evaluate_on_test(best, Xte, yte, test)
        print(f"  best seed:    n_feat={best.n_features}  "
              f"CV-RMSE={best.cv_rmse:.3f}  TEST RMSE={m_best['rmse']:.3f}  "
              f"iso={m_best['iso_correct']}/{m_best['iso_total']}")
        print(f"  across seeds: n_feat = {stats['n_feat_mean']:.1f} ± "
              f"{stats['n_feat_std']:.1f}")
        print(f"               CV-RMSE = {stats['cv_rmse_mean']:.3f} ± "
              f"{stats['cv_rmse_std']:.3f}")
        print(f"               Jaccard(selected) = "
              f"{stats['jaccard_mean']:.2f} ± {stats['jaccard_std']:.2f}")
    except Exception as e:
        print(f"  multi-seed SA FAILED: {e}")

    print("\n--- SA-GC stability across 5 seeds (Boltzmann-weighted) ---")
    try:
        best_gc, stats_gc = fit_sa_multiseed(Xtr, ytr, n_seeds=5,
                                              sample_weight=w_gc)
        m_best_gc = evaluate_on_test(best_gc, Xte, yte, test)
        print(f"  best seed:    n_feat={best_gc.n_features}  "
              f"CV-RMSE={best_gc.cv_rmse:.3f}  "
              f"TEST RMSE={m_best_gc['rmse']:.3f}  "
              f"iso={m_best_gc['iso_correct']}/{m_best_gc['iso_total']}")
        print(f"  across seeds: n_feat = {stats_gc['n_feat_mean']:.1f} ± "
              f"{stats_gc['n_feat_std']:.1f}")
        print(f"               CV-RMSE = {stats_gc['cv_rmse_mean']:.3f} ± "
              f"{stats_gc['cv_rmse_std']:.3f}")
        print(f"               Jaccard(selected) = "
              f"{stats_gc['jaccard_mean']:.2f} ± {stats_gc['jaccard_std']:.2f}")
    except Exception as e:
        print(f"  multi-seed SA-GC FAILED: {e}")

    # Bar plot of test RMSE and isomer accuracy
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = [r[0] for r in rows]
    rmse = [r[2]["rmse"] for r in rows]
    mae = [r[2]["mae"] for r in rows]
    iso = [r[2]["iso_correct"] / max(1, r[2]["iso_total"]) for r in rows]
    n_feat = [r[1].n_features for r in rows]
    times = [r[1].fit_time_s for r in rows]

    ax = axes[0]
    ax.bar(names, rmse, color="steelblue", label="test RMSE")
    ax.bar(names, mae, color="orange", alpha=0.7, label="test MAE")
    ax.set_ylabel("error [eV]")
    ax.set_title("Test set error")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    bars = ax.bar(names, iso, color="seagreen")
    for b, v, r in zip(bars, iso, rows):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{r[2]['iso_correct']}/{r[2]['iso_total']}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("fraction of isomer groups ranked correctly")
    ax.set_title("Isomer hierarchization accuracy")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar(names, n_feat, color="indianred")
    ax.set_ylabel("# selected features")
    ax.set_title("Sparsity")
    ax.tick_params(axis="x", rotation=35)
    for n, t in zip(names, times):
        i = names.index(n)
        ax.text(i, n_feat[i] + 1, f"{t:.1f}s", ha="center", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"OCE feature-selection benchmark — train n={Xtr.shape[0]} "
                 f"({Xtr.shape[1]} feats), test n={Xte.shape[0]}")
    fig.tight_layout()
    out = base / "results" / "selection_benchmark.png"
    fig.savefig(out, dpi=140)
    print(f"\nFigure saved at {out}")


if __name__ == "__main__":
    main()
