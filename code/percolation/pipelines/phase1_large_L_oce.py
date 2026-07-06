"""Phase 1b — extend the per-atom power-law to L >> 16 using OCE alone.

We use the OCE J_F coefficients fitted in `phase1_full.py` (trained on
L ≤ 8, validated on L ∈ {12, 16} at 45 meV/atom).  The model is then
evaluated on much larger L where xtb would cost hours-to-days but OCE
finishes in seconds.

This is the regime in which the cluster expansion shines: linear in
N_atoms, no SCF, no diagonalisation.  Memory is dominated by storing
the largest cluster's atoms list — a few MB for L=512.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms

from lattices import build_honeycomb, P_C
from percolation import sample_many
from oce_predict import featurise, fit_ridge, predict
from analysis import fit_loglog, fit_per_atom_with_bulk


LATTICE = "honeycomb"
P = P_C[LATTICE]
TRAIN_L = [4, 6, 8, 12, 16]                # use the data already gathered
EXTRAPOLATE_L = [24, 32, 48, 64, 96, 128]  # OCE-only regime
N_TRAIN_PER_L = {4: 24, 6: 24, 8: 24, 12: 18, 16: 12}
N_EXTRAPOLATE = {24: 12, 32: 10, 48: 8, 64: 6, 96: 4, 128: 3}
RIDGE_ALPHA = 1e-3
RANDOM_SEED = 1234

DATA_DIR = ROOT / "data" / "phase1_large_L_oce"
RESULTS_DIR = ROOT / "results" / "phase1_large_L_oce"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CACHE = ROOT / "data" / "phase1_full" / "clusters_xtb.json"
EXTRAP_CACHE = DATA_DIR / "clusters_oce.json"


def main():
    if not TRAIN_CACHE.exists():
        print(f"[err] {TRAIN_CACHE} not found.  Run pipelines/phase1_full.py first.")
        return
    train_records = json.loads(TRAIN_CACHE.read_text())
    print(f"Loaded {len(train_records)} training clusters from {TRAIN_CACHE}")

    train_atoms = [
        Atoms(symbols=["C"] * r["n_atoms"], positions=np.array(r["positions"]))
        for r in train_records
    ]
    X_tr, keys = featurise(train_atoms)
    y_tr = np.array([r["E_xtb_eV"] for r in train_records])

    # Fit on ALL phase-1 clusters
    model = fit_ridge(X_tr, y_tr, alpha=RIDGE_ALPHA)
    print(f"OCE fit on all {len(train_records)} L∈{TRAIN_L} clusters: "
          f"train RMSE={model['train_rmse']:.3f} eV  R²={model['train_r2']:.6f}")

    # ---------- Extrapolate ----------
    rng = np.random.default_rng(RANDOM_SEED)
    extrap_records = []
    if EXTRAP_CACHE.exists():
        extrap_records = json.loads(EXTRAP_CACHE.read_text())
        print(f"Loaded {len(extrap_records)} cached extrapolation clusters")
    if not extrap_records:
        for L in EXTRAPOLATE_L:
            n = N_EXTRAPOLATE[L]
            print(f"  Sampling L={L}, {n} realisations...", flush=True)
            t0 = time.perf_counter()
            lat = build_honeycomb(L)
            t_lat = time.perf_counter() - t0
            t1 = time.perf_counter()
            clusters = sample_many(lat, P, n,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=4)
            t_perc = time.perf_counter() - t1
            t2 = time.perf_counter()
            atoms_list = [Atoms(symbols=["C"] * c.n_atoms, positions=c.positions)
                          for c in clusters]
            X_ex, _ = featurise(atoms_list)
            # restrict to the same key list as training (drop unseen keys, pad zeros)
            key_idx = {k: i for i, k in enumerate(keys)}
            X_aligned = np.zeros((len(atoms_list), len(keys)))
            from oce_predict import (atomic_table_mod, figures_mod,
                                       correlations_mod, ATOMIC_TABLE_PATH)
            table = atomic_table_mod.load_table(ATOMIC_TABLE_PATH)
            for ic, atoms in enumerate(atoms_list):
                one, two, three, four = figures_mod.enumerate_figures(
                    atoms, table, include_angles=True, include_dihedrals=False)
                cv = correlations_mod.correlations_for_molecule(
                    one, two, table, three_figs=three)
                for k, v in cv.pi.items():
                    if k in key_idx:
                        X_aligned[ic, key_idx[k]] = v
            yhat = predict(X_aligned, model)
            t_oce = time.perf_counter() - t2

            for ic, c in enumerate(clusters):
                extrap_records.append(dict(
                    L=int(L), realisation=int(ic), seed=int(c.seed),
                    n_atoms=int(c.n_atoms), n_bonds=int(c.n_bonds),
                    E_oce_eV=float(yhat[ic]),
                ))
            print(f"    L={L}: {len(clusters)} clusters  "
                  f"⟨N⟩={np.mean([c.n_atoms for c in clusters]):.1f}  "
                  f"lat={t_lat:.2f}s perco={t_perc:.2f}s oce={t_oce:.3f}s")
        EXTRAP_CACHE.write_text(json.dumps(extrap_records, indent=2))
        print(f"Wrote → {EXTRAP_CACHE}")

    # ---------- Combine train + extrapolation, fit power-law ----------
    all_L = sorted(set([r["L"] for r in train_records] +
                        [r["L"] for r in extrap_records]))
    e_pa_byL: dict[int, list[float]] = {L: [] for L in all_L}
    n_byL: dict[int, list[int]] = {L: [] for L in all_L}
    for r in train_records:
        e_pa_byL[r["L"]].append(r["E_xtb_eV"] / r["n_atoms"])
        n_byL[r["L"]].append(r["n_atoms"])
    for r in extrap_records:
        e_pa_byL[r["L"]].append(r["E_oce_eV"] / r["n_atoms"])
        n_byL[r["L"]].append(r["n_atoms"])

    arrL = np.array(all_L, dtype=float)
    e_pa = np.array([np.mean(e_pa_byL[int(L)]) for L in arrL])
    e_pa_std = np.array([np.std(e_pa_byL[int(L)]) for L in arrL])
    nbar = np.array([np.mean(n_byL[int(L)]) for L in arrL])

    fit_N = fit_loglog(arrL, nbar)
    fit_pa, eps_bulk = fit_per_atom_with_bulk(arrL, e_pa, label="all-L")

    print(f"\n=== Combined fit (xtb-trained OCE on L={TRAIN_L} + "
          f"OCE extrapolation L={EXTRAPOLATE_L}) ===")
    print(f"  ⟨N⟩(L) ~ L^{fit_N.exponent:.3f}    "
          f"(theory D_f = 91/48 ≈ 1.896)")
    print(f"  ε_bulk = {eps_bulk:+.4f} eV/atom    "
          f"α = {-fit_pa.exponent:+.4f}    log-RMSE = {fit_pa.rmse_log:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.loglog(arrL, nbar, "o-", label=rf"$\langle N\rangle(L) \sim L^{{{fit_N.exponent:.3f}}}$")
    ax.loglog(arrL, arrL ** (91 / 48), ":", color="red",
              label=r"theory $L^{91/48}$")
    ax.set_xlabel("L"); ax.set_ylabel("⟨N_atoms⟩")
    ax.set_title("Cluster mass scaling at p = p_c (honeycomb site)")
    ax.legend(fontsize=8)

    ax = axes[1]
    res = e_pa - eps_bulk
    ax.loglog(arrL, res, "o-",
              label=rf"data,  $\alpha = {-fit_pa.exponent:.3f}$")
    fitline = fit_pa.intercept * arrL ** fit_pa.exponent
    ax.loglog(arrL, fitline, "--", color="grey", label="fit")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\langle E/N\rangle - \varepsilon_{bulk}$  (eV)")
    ax.set_title(rf"Per-atom power-law,  $\varepsilon_{{bulk}} = {eps_bulk:.3f}$ eV")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_extrapolation.png", dpi=180)
    plt.close(fig)

    summary = dict(
        train_L=TRAIN_L, extrapolate_L=EXTRAPOLATE_L,
        n_train_clusters=len(train_records),
        n_extrap_clusters=len(extrap_records),
        eps_bulk_eV=float(eps_bulk),
        alpha=float(-fit_pa.exponent),
        D_f_empirical=float(fit_N.exponent),
        per_L=dict(
            L=[int(L) for L in arrL],
            n_atoms=[float(x) for x in nbar],
            e_per_atom=[float(x) for x in e_pa],
            e_per_atom_std=[float(x) for x in e_pa_std],
        ),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_extrapolation.png'}")


if __name__ == "__main__":
    main()
