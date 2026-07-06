"""Phase 1 smoke pipeline: site-percolation honeycomb at p_c, L ≤ 8.

Steps (all unrelaxed, single-point energies only):

  1. For L ∈ {4, 6, 8}: sample N_realisations percolation clusters at p_c.
  2. Compute xtb GFN2 single-point energy of each cluster (radical, no H).
  3. Featurise each cluster with the OCE v1.0.0 basis (1F+2F+3F).
  4. Fit Ridge to the (X, y_xtb) pairs; report train R², RMSE.
  5. Evaluate OCE on a held-out subset; report parent-stratified accuracy.
  6. Save per-cluster records (energy, n_atoms, n_bonds, n_dangling) for
     downstream power-law analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# allow imports from package root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from lattices import build_honeycomb, P_C
from percolation import sample_many
from runners import xtb_radical_energy, cluster_to_atoms
from oce_predict import featurise, fit_ridge, predict, evaluate
from analysis import fit_loglog, fit_per_atom_with_bulk


# ---------- configuration ----------
LATTICE = "honeycomb"
P = P_C[LATTICE]                           # ≈ 0.6970402 (Suding & Ziff)
L_VALUES = [4, 6, 8]                       # smoke-test
N_REALISATIONS = 12                        # per L
RIDGE_ALPHA = 1e-3
RANDOM_SEED = 42
TEST_FRACTION = 0.30
DATA_DIR = ROOT / "data" / "phase1_smoke"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results" / "phase1_smoke"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print(f"=== Phase-1 smoke: {LATTICE} site-perc at p_c={P:.6f} ===")
    rng = np.random.default_rng(RANDOM_SEED)

    # ---------- 1. Sample + xtb ----------
    records: list[dict] = []
    for L in L_VALUES:
        lat = build_honeycomb(L)
        seed_base = int(rng.integers(2**31))
        clusters = sample_many(lat, P, N_REALISATIONS,
                                base_seed=seed_base, min_size=4)
        for ic, cl in enumerate(clusters):
            atoms = cluster_to_atoms(cl)
            res = xtb_radical_energy(atoms, optimize=False, threads=1)
            print(f"  L={L:2d}  realisation={ic:2d}  N={cl.n_atoms:3d}  "
                  f"bonds={cl.n_bonds:3d}  uhf={res['uhf']:2d}  "
                  f"E_xtb={res['E_eV']:+10.3f} eV  "
                  f"({res['wall_time_s']*1000:.0f} ms)  "
                  f"{'OK' if res['converged'] else 'FAIL'}")
            if not res["converged"]:
                continue
            records.append(dict(
                L=L,
                realisation=ic,
                seed=cl.seed,
                n_atoms=cl.n_atoms,
                n_bonds=cl.n_bonds,
                n_dangling_total=int(sum(cl.n_dangling)),
                positions=cl.positions.tolist(),
                site_idx=cl.site_idx,
                E_xtb_eV=res["E_eV"],
                uhf=res["uhf"],
                wall_time_s=res["wall_time_s"],
            ))

    raw_path = DATA_DIR / "clusters_xtb.json"
    raw_path.write_text(json.dumps(records, indent=2))
    print(f"\nSaved {len(records)} xtb records → {raw_path}")

    # ---------- 2. Featurise (OCE v1.0.0 basis) ----------
    from ase import Atoms
    atoms_list = [
        Atoms(symbols=["C"] * r["n_atoms"], positions=np.array(r["positions"]))
        for r in records
    ]
    X, keys = featurise(atoms_list, include_angles=True, include_dihedrals=False)
    y = np.array([r["E_xtb_eV"] for r in records])
    print(f"\nDesign matrix: X{X.shape}, {len(keys)} OCE feature keys")

    # ---------- 3. Train/test split ----------
    n = len(records)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    model = fit_ridge(X_tr, y_tr, alpha=RIDGE_ALPHA)
    yhat_tr = predict(X_tr, model)
    yhat_te = predict(X_te, model)
    train_metrics = evaluate(yhat_tr, y_tr)
    test_metrics = evaluate(yhat_te, y_te)

    print(f"\n--- OCE Ridge(α={RIDGE_ALPHA}) ---")
    print(f"  TRAIN  n={len(train_idx):3d}  RMSE={train_metrics['rmse']:.3f} eV  "
          f"R²={train_metrics['r2']:.5f}  ρ={train_metrics['spearman']:.5f}")
    print(f"  TEST   n={len(test_idx):3d}  RMSE={test_metrics['rmse']:.3f} eV  "
          f"R²={test_metrics['r2']:.5f}  ρ={test_metrics['spearman']:.5f}")

    yhat_all = predict(X, model)
    for r, ypred in zip(records, yhat_all):
        r["E_oce_eV"] = float(ypred)

    Path(DATA_DIR / "clusters_oce.json").write_text(json.dumps(records, indent=2))

    # ---------- 4. Per-atom OCE↔xtb agreement ----------
    err_per_atom = (yhat_all - y) / np.array([r["n_atoms"] for r in records])
    print(f"\nPer-atom RMS error |E_oce − E_xtb|/N = "
          f"{float(np.sqrt(np.mean(err_per_atom ** 2))) * 1000:.1f} meV/atom")

    # ---------- 5. Power-law E/N(L) ----------
    Ls = np.array([r["L"] for r in records])
    Ns = np.array([r["n_atoms"] for r in records])
    e_per_atom_xtb = y / Ns
    e_per_atom_oce = yhat_all / Ns

    # Per-L means (collapse the realisation axis)
    L_unique = sorted(set(Ls))
    e_xtb_byL = np.array([e_per_atom_xtb[Ls == L].mean() for L in L_unique])
    e_oce_byL = np.array([e_per_atom_oce[Ls == L].mean() for L in L_unique])
    e_xtb_std = np.array([e_per_atom_xtb[Ls == L].std() for L in L_unique])

    print(f"\n--- E/N vs L (xtb vs OCE) ---")
    print(f"  L     ⟨E/N⟩_xtb (eV)   σ        ⟨E/N⟩_oce (eV)   ⟨N⟩")
    for L, exb, sxb, eoc in zip(L_unique, e_xtb_byL, e_xtb_std, e_oce_byL):
        nbar = float(np.mean(Ns[Ls == L]))
        print(f"  {L:2d}    {exb:+10.4f}   {sxb:7.4f}   {eoc:+10.4f}   {nbar:6.1f}")

    # Power-law fit on (E/N − ε_bulk) via 3-par grid search (xtb only)
    fit_xtb, eps_bulk_xtb = fit_per_atom_with_bulk(
        np.array(L_unique, dtype=float), e_xtb_byL, label="xtb")
    fit_oce, eps_bulk_oce = fit_per_atom_with_bulk(
        np.array(L_unique, dtype=float), e_oce_byL, label="oce")

    summary = dict(
        lattice=LATTICE,
        p=P,
        L_values=[int(L) for L in L_unique],
        n_clusters=len(records),
        n_features=len(keys),
        ridge_alpha=RIDGE_ALPHA,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        per_atom_rmse_meV=float(np.sqrt(np.mean(err_per_atom ** 2))) * 1000,
        eps_bulk_xtb_eV=eps_bulk_xtb,
        eps_bulk_oce_eV=eps_bulk_oce,
        powerlaw_xtb=dict(exponent=fit_xtb.exponent,
                          intercept=fit_xtb.intercept,
                          rmse_log=fit_xtb.rmse_log,
                          n_points=fit_xtb.n_points),
        powerlaw_oce=dict(exponent=fit_oce.exponent,
                          intercept=fit_oce.intercept,
                          rmse_log=fit_oce.rmse_log,
                          n_points=fit_oce.n_points),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n--- Power-law E/N − ε_bulk = c · L^(-α) ---")
    print(f"  xtb : ε_bulk = {eps_bulk_xtb:+.3f} eV/atom   "
          f"α = {-fit_xtb.exponent:+.3f}   logRMSE={fit_xtb.rmse_log:.3f}")
    print(f"  oce : ε_bulk = {eps_bulk_oce:+.3f} eV/atom   "
          f"α = {-fit_oce.exponent:+.3f}   logRMSE={fit_oce.rmse_log:.3f}")
    print(f"\nSummary → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
