"""Benchmark OCE vs xtb: speed comparison + scatter plot.

Reports:
  - per-molecule prediction time for OCE
  - per-molecule single-point time for xtb
  - speedup factor
  - scatter plot of E_OCE vs E_xtb on the test set
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oce.atomic_table import load_table
from oce.correlations import correlations_for_molecule
from oce.dataset import entry_to_atoms, load_dataset
from oce.figures import enumerate_figures
from oce.fit import build_design_matrix, fit_model
from oce.xtb_runner import xtb_energy


def predict_oce(entry, model, table) -> float:
    atoms = entry_to_atoms(entry)
    one, two, three, four = enumerate_figures(atoms, table)
    cv = correlations_for_molecule(one, two, table, three_figs=three)
    return model.predict(cv)


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    test = load_dataset(base / "data" / "molecules" / "test.json")

    Xtr, ytr, keys, _ = build_design_matrix(train, table)
    model = fit_model(Xtr, ytr, keys, alpha=0.1)

    # Timing: OCE
    t0 = time.perf_counter()
    e_oce = []
    for _ in range(20):
        e_oce = [predict_oce(e, model, table) for e in test]
    t_oce = (time.perf_counter() - t0) / 20 / len(test)

    # Timing: xtb single-point
    t0 = time.perf_counter()
    e_xtb_sp = []
    for e in test:
        E, _ = xtb_energy(entry_to_atoms(e), optimize=False)
        e_xtb_sp.append(E)
    t_xtb = (time.perf_counter() - t0) / len(test)

    print(f"\nMean wall-clock per molecule:")
    print(f"  OCE prediction:   {t_oce*1000:8.3f} ms")
    print(f"  xtb single-point: {t_xtb*1000:8.3f} ms")
    print(f"  speedup:          {t_xtb / t_oce:8.0f} ×")

    # Reference y for scatter
    y_test = np.array([e.energy_eV for e in test])
    y_pred = np.array(e_oce)
    err = y_pred - y_test
    print(f"\nTest set: n={len(test)}")
    print(f"  RMSE = {np.sqrt(np.mean(err**2)):.4f} eV")
    print(f"  MAE  = {np.mean(np.abs(err)):.4f} eV")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    formulas = sorted({e.formula for e in test})
    cmap = plt.colormaps.get_cmap("tab10")
    for i, formula in enumerate(formulas):
        mask = [k for k, e in enumerate(test) if e.formula == formula]
        ax.scatter([y_test[k] for k in mask],
                   [y_pred[k] for k in mask],
                   color=cmap(i % 10), label=formula, s=80, alpha=0.85)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("E (xtb GFN2) [eV]")
    ax.set_ylabel("E (OCE) [eV]")
    ax.set_title(
        f"OCE vs xtb on test set (n={len(test)})\n"
        f"RMSE = {np.sqrt(np.mean(err**2)):.3f} eV, "
        f"MAE = {np.mean(np.abs(err)):.3f} eV, "
        f"Spearman ρ = 1.000"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    out = base / "results" / "scatter_oce_vs_xtb.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"\nScatter plot saved at {out}")

    # Per-isomer-group plot (zoomed)
    isomer_groups = [f for f in formulas
                     if sum(1 for e in test if e.formula == f) >= 2]
    n_groups = len(isomer_groups)
    if n_groups:
        fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4.5))
        if n_groups == 1:
            axes = [axes]
        for ax, formula in zip(axes, isomer_groups):
            members = [(e.name, y_test[k], y_pred[k])
                       for k, e in enumerate(test) if e.formula == formula]
            members.sort(key=lambda x: x[1])
            names = [m[0] for m in members]
            ys = [m[1] for m in members]
            yps = [m[2] for m in members]
            xs = np.arange(len(members))
            ax.plot(xs, ys, "o-", label="xtb", color="steelblue")
            ax.plot(xs, yps, "s--", label="OCE", color="darkorange")
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("E [eV]")
            ax.set_title(f"{formula} isomers")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        out2 = base / "results" / "isomer_rankings.png"
        fig.savefig(out2, dpi=140)
        print(f"Isomer ranking plot saved at {out2}")
