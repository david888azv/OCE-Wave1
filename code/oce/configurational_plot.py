"""Plot OCE prediction vs xtb truth on the benzene H/F configurational set.

Generates a parity plot colored by fluorine count, with all 64 configs
predicted from a model trained on only the {0,1,5,6}-F subset (14 of 64).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oce.atomic_table import load_table
from oce.configurational import load_dataset as load_conf
from oce.configurational_test import build_X
from oce.selection import METHODS, gc_weights, fit_sa_multiseed


def main():
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    entries = load_conf(base / "data" / "configurational" / "benzene_hf.json")

    X_all, keys = build_X(entries, table)
    y_all = np.array([e.energy_eV for e in entries])
    n_F = np.array([e.n_F for e in entries])

    train_idx = [i for i, e in enumerate(entries) if e.n_F in (0, 1, 5, 6)]
    Xtr = X_all[train_idx]; ytr = y_all[train_idx]
    formulas_tr = [f"C6H{6-entries[i].n_F}F{entries[i].n_F}" for i in train_idx]
    w_gc = gc_weights(ytr, formulas_tr, T=0.5)

    methods = {
        "Ridge (n=14 feats)": METHODS["ridge"](Xtr, ytr),
        "GA (sparse)": METHODS["ga"](Xtr, ytr),
        "SA-GC[5x] (sparse + GC)": fit_sa_multiseed(Xtr, ytr, n_seeds=5,
                                                     sample_weight=w_gc)[0],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (name, model) in zip(axes, methods.items()):
        yhat = model.predict(X_all)
        for nF in range(7):
            mask = n_F == nF
            ax.scatter(y_all[mask], yhat[mask],
                       c=[cmap(nF / 6.0)], s=60, edgecolors="k",
                       linewidths=0.5, label=f"n_F={nF}", zorder=3)
        lo = min(y_all.min(), yhat.min()) - 5
        hi = max(y_all.max(), yhat.max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, zorder=1)
        rmse = float(np.sqrt(np.mean((yhat - y_all) ** 2)))
        ax.set_title(f"{name}\nRMSE = {rmse*1000:.1f} meV   "
                     f"n_used = {model.n_features}/{X_all.shape[1]}")
        ax.set_xlabel("xtb energy [eV]")
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("OCE prediction [eV]")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("OCE configurational expansion on C6H_(6-n)F_n  "
                 "(train: only n_F ∈ {0,1,5,6} = 14 configs;  "
                 "predict all 64)")
    fig.tight_layout()
    out = base / "results" / "configurational_parity.png"
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")

    # ---- Second plot: within-n_F isomer energy differences (relative to min)
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (name, model) in zip(axes2, methods.items()):
        yhat = model.predict(X_all)
        for nF in range(2, 5):  # only multi-isomer counts
            mask = n_F == nF
            yh_rel = (yhat[mask] - yhat[mask].min()) * 1000  # to meV
            yt_rel = (y_all[mask] - y_all[mask].min()) * 1000
            ax.scatter(yt_rel, yh_rel, c=[cmap(nF / 6.0)], s=60,
                       edgecolors="k", linewidths=0.5,
                       label=f"n_F={nF}", zorder=3)
        ax.plot([-50, 350], [-50, 350], "k--", alpha=0.5, linewidth=1, zorder=1)
        ax.set_title(name)
        ax.set_xlabel("xtb ΔE within n_F group [meV]")
        ax.grid(alpha=0.3)
        if ax is axes2[0]:
            ax.set_ylabel("OCE ΔE within n_F group [meV]")
            ax.legend(fontsize=9, loc="upper left")
    fig2.suptitle("OCE isomer-resolution detail — relative energies inside each n_F class")
    fig2.tight_layout()
    out2 = base / "results" / "configurational_isomer_detail.png"
    fig2.savefig(out2, dpi=140)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
