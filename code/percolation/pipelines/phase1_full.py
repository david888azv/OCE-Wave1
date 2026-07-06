"""Phase 1 full: site-percolation honeycomb at p_c, L ∈ {4, 6, 8, 12, 16}.

This is the production version of pipelines/phase1_smoke.py.  It samples
more realisations, fits the OCE J_F coefficients with a parent-stratified
holdout (different L's on each side), saves matplotlib plots, and reports
the power-law exponent of  E/N − ε_bulk  vs  L.

Run-time scales as ⟨wall_xtb(L)⟩ × N_realisations(L) × |L_VALUES|.
With N_per_L = 24 and L up to 16 we expect ~5-15 minutes of xtb on this
machine.  Memory is dominated by xtb (≪1 GB at L≤16).
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

from lattices import build_honeycomb, P_C
from percolation import sample_many
from runners import xtb_radical_energy, cluster_to_atoms
from oce_predict import featurise, fit_ridge, predict, evaluate
from analysis import fit_loglog, fit_per_atom_with_bulk


LATTICE = "honeycomb"
P = P_C[LATTICE]
L_VALUES = [4, 6, 8, 12, 16]
N_PER_L = {4: 24, 6: 24, 8: 24, 12: 18, 16: 12}
RIDGE_ALPHA = 1e-3
RANDOM_SEED = 1234

DATA_DIR = ROOT / "data" / "phase1_full"
RESULTS_DIR = ROOT / "results" / "phase1_full"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE = DATA_DIR / "clusters_xtb.json"


def gather_xtb_clusters() -> list[dict]:
    """Sample + xtb singlepoint, with cache."""
    if CACHE.exists():
        records = json.loads(CACHE.read_text())
        print(f"Loaded {len(records)} cached records from {CACHE}")
        return records
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []
    t_global = time.perf_counter()
    for L in L_VALUES:
        lat = build_honeycomb(L)
        n = N_PER_L[L]
        clusters = sample_many(lat, P, n,
                                base_seed=int(rng.integers(2**31)),
                                min_size=4)
        for ic, cl in enumerate(clusters):
            atoms = cluster_to_atoms(cl)
            res = xtb_radical_energy(atoms, optimize=False, threads=1)
            ok = "OK" if res["converged"] else "FAIL"
            print(f"  L={L:2d}  rep={ic:2d}  N={cl.n_atoms:4d}  "
                  f"bonds={cl.n_bonds:4d}  uhf={res['uhf']:2d}  "
                  f"E={res['E_eV']:+11.3f} eV  "
                  f"({res['wall_time_s']:.2f}s) {ok}")
            if not res["converged"]:
                continue
            records.append(dict(
                L=int(L), realisation=int(ic), seed=int(cl.seed),
                n_atoms=int(cl.n_atoms), n_bonds=int(cl.n_bonds),
                n_dangling_total=int(sum(cl.n_dangling)),
                positions=cl.positions.tolist(),
                site_idx=cl.site_idx,
                E_xtb_eV=float(res["E_eV"]),
                uhf=int(res["uhf"]),
                wall_time_s=float(res["wall_time_s"]),
            ))
    print(f"\nTotal xtb wall: {time.perf_counter() - t_global:.1f}s "
          f"for {len(records)} clusters")
    CACHE.write_text(json.dumps(records, indent=2))
    return records


def main():
    print(f"=== Phase 1 full: {LATTICE} site-perc at p_c={P:.6f} ===")
    print(f"L_VALUES = {L_VALUES}")

    records = gather_xtb_clusters()

    # -------- Featurise (OCE v1.0.0) --------
    from ase import Atoms
    atoms_list = [Atoms(symbols=["C"] * r["n_atoms"],
                         positions=np.array(r["positions"]))
                  for r in records]
    X, keys = featurise(atoms_list, include_angles=True, include_dihedrals=False)
    y = np.array([r["E_xtb_eV"] for r in records])
    Ls = np.array([r["L"] for r in records])
    Ns = np.array([r["n_atoms"] for r in records])
    print(f"\nDesign matrix X{X.shape}, {len(keys)} OCE features")

    # -------- TRAIN: L≤8 ; TEST: L=12,16 (parent-stratified by L) --------
    train_mask = np.isin(Ls, [4, 6, 8])
    test_mask = np.isin(Ls, [12, 16])
    print(f"  TRAIN clusters: {train_mask.sum()}    "
          f"TEST clusters (L=12,16, never seen during training): "
          f"{test_mask.sum()}")

    model = fit_ridge(X[train_mask], y[train_mask], alpha=RIDGE_ALPHA)
    yhat = predict(X, model)
    yhat_tr = yhat[train_mask]
    yhat_te = yhat[test_mask]

    train_metrics = evaluate(yhat_tr, y[train_mask])
    test_metrics = evaluate(yhat_te, y[test_mask])
    rmse_per_atom_tr = float(np.sqrt(np.mean(((yhat_tr - y[train_mask])
                                              / Ns[train_mask]) ** 2))) * 1000
    rmse_per_atom_te = float(np.sqrt(np.mean(((yhat_te - y[test_mask])
                                              / Ns[test_mask]) ** 2))) * 1000

    print(f"\n--- OCE Ridge(α={RIDGE_ALPHA}) trained on L≤8 ---")
    print(f"  TRAIN n={train_mask.sum()}  RMSE={train_metrics['rmse']:.3f} eV  "
          f"R²={train_metrics['r2']:.5f}  ρ={train_metrics['spearman']:.5f}  "
          f"per-atom={rmse_per_atom_tr:.1f} meV/atom")
    print(f"  TEST  n={test_mask.sum()}  RMSE={test_metrics['rmse']:.3f} eV  "
          f"R²={test_metrics['r2']:.5f}  ρ={test_metrics['spearman']:.5f}  "
          f"per-atom={rmse_per_atom_te:.1f} meV/atom")

    # -------- E_total scaling: D_E from log-log --------
    # Use mean per-L (E_total at fixed L is well-defined as ⟨E_total⟩ over realisations)
    L_unique = sorted(set(Ls.tolist()))
    arrL = np.array(L_unique, dtype=float)
    e_tot_xtb_byL = np.array([y[Ls == L].mean() for L in L_unique])
    e_tot_oce_byL = np.array([yhat[Ls == L].mean() for L in L_unique])
    n_byL = np.array([Ns[Ls == L].mean() for L in L_unique])
    e_pa_xtb_byL = e_tot_xtb_byL / n_byL
    e_pa_oce_byL = e_tot_oce_byL / n_byL
    e_pa_xtb_std = np.array([(y[Ls == L] / Ns[Ls == L]).std() for L in L_unique])

    fit_E = fit_loglog(arrL, -e_tot_xtb_byL, label="xtb |E_total|")
    fit_N = fit_loglog(arrL, n_byL, label="N_atoms")
    print(f"\n--- Mass scaling N_atoms ~ L^D_f (theory: 91/48 = 1.896) ---")
    print(f"  empirical exponent D_f = {fit_N.exponent:+.3f}  "
          f"(intercept c = {fit_N.intercept:.3f}, log-RMSE = {fit_N.rmse_log:.3f})")
    print(f"\n--- Total energy E_total ~ L^D_E ---")
    print(f"  xtb : D_E = {fit_E.exponent:+.3f}    log-RMSE = {fit_E.rmse_log:.3f}")
    fit_E_oce = fit_loglog(arrL, -e_tot_oce_byL, label="oce |E_total|")
    print(f"  oce : D_E = {fit_E_oce.exponent:+.3f}    log-RMSE = {fit_E_oce.rmse_log:.3f}")

    # -------- Per-atom power-law: E/N − ε_bulk = c L^{-α} --------
    fit_xtb, eps_bulk_xtb = fit_per_atom_with_bulk(arrL, e_pa_xtb_byL, label="xtb")
    fit_oce, eps_bulk_oce = fit_per_atom_with_bulk(arrL, e_pa_oce_byL, label="oce")
    print(f"\n--- Per-atom power-law (E/N − ε_bulk) = c · L^{{-α}} ---")
    print(f"  xtb : ε_bulk = {eps_bulk_xtb:+.4f} eV/atom    "
          f"α = {-fit_xtb.exponent:+.4f}    log-RMSE = {fit_xtb.rmse_log:.3f}")
    print(f"  oce : ε_bulk = {eps_bulk_oce:+.4f} eV/atom    "
          f"α = {-fit_oce.exponent:+.4f}    log-RMSE = {fit_oce.rmse_log:.3f}")
    print(f"\nTheory: 2D incipient cluster boundary fractal:")
    print(f"  α between |D_h - D_f| = 19/48 ≈ 0.396  (hull-only)")
    print(f"  and       |D_e - D_f| =  7/48 ≈ 0.146  (full boundary, including holes)")

    # -------- Plots --------
    plot_dir = RESULTS_DIR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.scatter(y, yhat - y, c=Ls, cmap="viridis", s=22, alpha=0.85,
               edgecolor="k", linewidth=0.3)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("xtb total energy (eV)")
    ax.set_ylabel("E_OCE − E_xtb  (eV)")
    ax.set_title(f"OCE residuals  (TEST per-atom = {rmse_per_atom_te:.0f} meV/atom)")

    ax = axes[1]
    ax.errorbar(arrL, e_pa_xtb_byL, yerr=e_pa_xtb_std,
                fmt="o-", color="C0", label="xtb")
    ax.plot(arrL, e_pa_oce_byL, "x--", color="C3", label="OCE", markersize=8)
    ax.axhline(eps_bulk_xtb, color="grey", lw=0.5, ls=":",
               label=f"ε_bulk(xtb) = {eps_bulk_xtb:.3f} eV/atom")
    ax.set_xlabel("L")
    ax.set_ylabel("⟨E/N⟩  (eV / atom)")
    ax.set_title(f"Per-atom energy of percolation clusters")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_dir / "fig_residuals_and_perL.png", dpi=180)
    plt.close(fig)

    # Power-law plot (log-log)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.loglog(arrL, n_byL, "o-", label=rf"$\langle N\rangle$  fit $\sim L^{{{fit_N.exponent:.3f}}}$")
    ax.loglog(arrL, fit_N.intercept * arrL ** fit_N.exponent, "--",
              color="grey", label="fit")
    ax.loglog(arrL, arrL ** (91 / 48), ":", color="red",
              label=r"theory $L^{91/48}$")
    ax.set_xlabel("L")
    ax.set_ylabel("⟨N_atoms⟩")
    ax.legend(fontsize=8)
    ax.set_title("Mass scaling at p = p_c")

    ax = axes[1]
    res_xtb = e_pa_xtb_byL - eps_bulk_xtb
    res_oce = e_pa_oce_byL - eps_bulk_oce
    ax.loglog(arrL, res_xtb, "o-", color="C0",
              label=rf"xtb,  $\alpha={-fit_xtb.exponent:.3f}$, "
                     f"$\\varepsilon_{{bulk}}={eps_bulk_xtb:.3f}$")
    ax.loglog(arrL, res_oce, "x--", color="C3",
              label=rf"OCE,  $\alpha={-fit_oce.exponent:.3f}$, "
                     f"$\\varepsilon_{{bulk}}={eps_bulk_oce:.3f}$")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$\langle E/N\rangle - \varepsilon_{bulk}$  (eV)")
    ax.legend(fontsize=8)
    ax.set_title("Per-atom power-law decay")
    plt.tight_layout()
    fig.savefig(plot_dir / "fig_powerlaw.png", dpi=180)
    plt.close(fig)

    # -------- Summary --------
    summary = dict(
        lattice=LATTICE, p=float(P),
        L_values=[int(L) for L in L_unique],
        n_total=int(len(records)),
        n_train=int(train_mask.sum()),
        n_test=int(test_mask.sum()),
        n_features=int(len(keys)),
        ridge_alpha=RIDGE_ALPHA,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        rmse_per_atom_train_meV=rmse_per_atom_tr,
        rmse_per_atom_test_meV=rmse_per_atom_te,
        D_f_empirical=fit_N.exponent,
        D_f_theory=91 / 48,
        D_E_xtb=fit_E.exponent,
        D_E_oce=fit_E_oce.exponent,
        eps_bulk_xtb_eV=eps_bulk_xtb,
        eps_bulk_oce_eV=eps_bulk_oce,
        alpha_xtb=-fit_xtb.exponent,
        alpha_oce=-fit_oce.exponent,
        alpha_theory_window=[7 / 48, 19 / 48],
        per_L_means=dict(
            L=[int(x) for x in L_unique],
            N_atoms=[float(x) for x in n_byL],
            E_total_xtb_eV=[float(x) for x in e_tot_xtb_byL],
            E_total_oce_eV=[float(x) for x in e_tot_oce_byL],
            E_per_atom_xtb_eV=[float(x) for x in e_pa_xtb_byL],
            E_per_atom_oce_eV=[float(x) for x in e_pa_oce_byL],
        ),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written → {RESULTS_DIR / 'summary.json'}")
    print(f"Plots → {plot_dir}/fig_*.png")


if __name__ == "__main__":
    main()
