"""Phase 2 smoke — xtb optimisation + OCE on relaxed structures (L ≤ 6).

Workflow:
  1. Resample percolation clusters at p_c (small L).
  2. Run xtb single-point at the AS-BUILT lattice geometry → E_xtb_sp.
  3. Run xtb optimisation                             → E_xtb_relaxed.
  4. Compare ΔE_relax = E_xtb_sp − E_xtb_relaxed.
  5. Featurise BOTH the as-built and relaxed geometries with the OCE
     v1.0.0 basis; the OCE 1F/2F/3F values DO depend on geometry through
     bond distances and angles → ΔE_OCE_relax should track ΔE_xtb_relax.
  6. Save records and a small plot.

This is a smoke test only; full phase-2 production would need many more
realisations and L values, plus retraining OCE on relaxed structures.
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
from runners import xtb_radical_energy, cluster_to_atoms
from oce_predict import featurise, fit_ridge, predict


LATTICE = "honeycomb"
P = P_C[LATTICE]
L_VALUES = [4, 6]
N_PER_L = {4: 6, 6: 4}
RANDOM_SEED = 31415

DATA_DIR = ROOT / "data" / "phase2_smoke"
RESULTS_DIR = ROOT / "results" / "phase2_smoke"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE = DATA_DIR / "clusters_relax.json"


def main():
    print(f"=== Phase 2 smoke: relax percolation clusters L ≤ 6, "
          f"site-perc honeycomb at p_c={P:.4f} ===")
    rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict] = []
    if CACHE.exists():
        records = json.loads(CACHE.read_text())
        print(f"Loaded {len(records)} cached records")
    if not records:
        for L in L_VALUES:
            lat = build_honeycomb(L)
            n = N_PER_L[L]
            clusters = sample_many(lat, P, n,
                                    base_seed=int(rng.integers(2**31)),
                                    min_size=4)
            for ic, cl in enumerate(clusters):
                atoms = cluster_to_atoms(cl)
                t0 = time.perf_counter()
                sp = xtb_radical_energy(atoms, optimize=False, threads=1)
                opt = xtb_radical_energy(atoms, optimize=True, threads=1)
                if not (sp["converged"] and opt["converged"]):
                    print(f"  [skip] L={L} rep={ic} sp_ok={sp['converged']} "
                          f"opt_ok={opt['converged']}")
                    continue
                relax_dE = sp["E_eV"] - opt["E_eV"]
                relax_pos = opt["opt_atoms"].get_positions()
                rms_disp = float(np.sqrt(np.mean(np.sum(
                    (relax_pos - cl.positions) ** 2, axis=1))))
                print(f"  L={L:2d}  rep={ic:2d}  N={cl.n_atoms:3d}  "
                      f"E_sp={sp['E_eV']:+10.3f}  E_opt={opt['E_eV']:+10.3f}  "
                      f"ΔE_relax={relax_dE:+7.3f} eV  "
                      f"⟨|Δr|⟩={rms_disp:.2f} Å  "
                      f"({time.perf_counter() - t0:.1f}s)")
                records.append(dict(
                    L=int(L), realisation=int(ic), seed=int(cl.seed),
                    n_atoms=int(cl.n_atoms), n_bonds=int(cl.n_bonds),
                    positions_unrelaxed=cl.positions.tolist(),
                    positions_relaxed=relax_pos.tolist(),
                    E_xtb_sp_eV=float(sp["E_eV"]),
                    E_xtb_relaxed_eV=float(opt["E_eV"]),
                    relax_dE_eV=float(relax_dE),
                    rms_displacement_A=rms_disp,
                ))
        CACHE.write_text(json.dumps(records, indent=2))

    if not records:
        print("No converged structures.")
        return

    # ---------- Featurise both geometries ----------
    atoms_unrel = [Atoms(symbols=["C"] * r["n_atoms"],
                          positions=np.array(r["positions_unrelaxed"]))
                   for r in records]
    atoms_relax = [Atoms(symbols=["C"] * r["n_atoms"],
                          positions=np.array(r["positions_relaxed"]))
                   for r in records]
    # Featurise BOTH sets together so they share a common key list
    X_all, keys = featurise(atoms_unrel + atoms_relax)
    n_rec = len(records)
    X_unrel = X_all[:n_rec]
    X_relax = X_all[n_rec:]
    print(f"\nDesign matrices: X_unrel{X_unrel.shape}  X_relax{X_relax.shape}  "
          f"({len(keys)} unified features)")

    # ---------- Train OCE on UNRELAXED data, evaluate on RELAXED ----------
    y_sp = np.array([r["E_xtb_sp_eV"] for r in records])
    y_opt = np.array([r["E_xtb_relaxed_eV"] for r in records])

    model_un = fit_ridge(X_unrel, y_sp, alpha=1e-3)
    yhat_un = predict(X_unrel, model_un)
    yhat_relax_via_un = predict(X_relax, model_un)

    # Train a SECOND OCE on the relaxed geometries paired with relaxed energies
    model_rel = fit_ridge(X_relax, y_opt, alpha=1e-3)
    yhat_relax = predict(X_relax, model_rel)

    err_unrel_pa = (yhat_un - y_sp) / np.array([r["n_atoms"] for r in records])
    err_relax_pa = (yhat_relax - y_opt) / np.array([r["n_atoms"] for r in records])
    err_via_un_pa = (yhat_relax_via_un - y_opt) / np.array([r["n_atoms"] for r in records])

    print("\n--- Per-atom OCE↔xtb agreement ---")
    print(f"  unrelaxed  (train=eval same set)        : RMSE = "
          f"{np.sqrt(np.mean(err_unrel_pa ** 2)) * 1000:.1f} meV/atom")
    print(f"  relaxed    (OCE retrained on relaxed)   : RMSE = "
          f"{np.sqrt(np.mean(err_relax_pa ** 2)) * 1000:.1f} meV/atom")
    print(f"  relaxed    (OCE trained on UN-relaxed)  : RMSE = "
          f"{np.sqrt(np.mean(err_via_un_pa ** 2)) * 1000:.1f} meV/atom")
    print("    → if the latter ≈ second value, the OCE basis already")
    print("      sees enough geometric info to handle relaxation.")

    # ΔE_relax tracking
    relax_dE_xtb = y_sp - y_opt
    relax_dE_oce_un = yhat_un - yhat_relax_via_un
    relax_dE_oce_rel = yhat_un - yhat_relax
    print("\n--- ΔE_relax = E_unrelaxed − E_relaxed (each cluster) ---")
    print("  realisation  N    Δ_xtb (eV)   Δ_OCE_unfit (eV)   Δ_OCE_refit (eV)")
    for ic, r in enumerate(records):
        print(f"  L={r['L']:2d}/{r['realisation']:<2d}    "
              f"{r['n_atoms']:3d}     "
              f"{relax_dE_xtb[ic]:+7.3f}        "
              f"{relax_dE_oce_un[ic]:+7.3f}            "
              f"{relax_dE_oce_rel[ic]:+7.3f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(relax_dE_xtb, relax_dE_oce_un, label="OCE trained on unrelaxed",
               edgecolor="k", linewidth=0.3)
    ax.scatter(relax_dE_xtb, relax_dE_oce_rel, marker="x", color="C3",
               label="OCE retrained on relaxed")
    span = max(0.1, float(np.max(np.abs(relax_dE_xtb))) * 1.1)
    ax.plot([0, span], [0, span], "k--", lw=0.6)
    ax.set_xlabel(r"$\Delta E_{\rm relax}$ from xtb (eV)")
    ax.set_ylabel(r"$\Delta E_{\rm relax}$ predicted by OCE (eV)")
    ax.set_title("ΔE_relax tracking — phase-2 smoke")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_relax_dE.png", dpi=180)
    plt.close(fig)

    summary = dict(
        n_clusters=len(records),
        rmse_unrelaxed_meV=float(np.sqrt(np.mean(err_unrel_pa ** 2))) * 1000,
        rmse_relaxed_meV=float(np.sqrt(np.mean(err_relax_pa ** 2))) * 1000,
        rmse_relaxed_via_un_meV=float(np.sqrt(np.mean(err_via_un_pa ** 2))) * 1000,
        mean_relax_dE_xtb_eV=float(np.mean(relax_dE_xtb)),
        mean_rms_disp_A=float(np.mean([r["rms_displacement_A"] for r in records])),
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
