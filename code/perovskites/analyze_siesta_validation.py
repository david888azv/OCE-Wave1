"""Analyse the 93-perovskite SIESTA SZP validation run.

Three meaningful comparisons:
  (A) Ranking-only: GFN-FF vs SIESTA (Spearman/Kendall). RMSE in absolute eV
      is meaningless because the two engines have wildly different total-energy
      reference (SIESTA includes pseudo offsets, ~−3 keV/atom).
  (B) OCE (trained on GFN-FF) vs SIESTA — same caveat, ranking only.
  (C) OCE refitted on SIESTA labels with 5-fold CV — held-out per-atom RMSE
      against PBE-DFT reference.  This is the decisive validation.

Per-(A,B) family and per-kind breakdowns.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"


NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+?)(\d+)?(_.*)?$")


def _parse_kind(name: str) -> str:
    m = NAME_RE.match(name)
    if not m:
        return "?"
    kind = m.group(4)
    if kind.startswith("mix"):
        return kind[:4]   # mixX/mixA/mixB
    if kind == "prim":
        return "prim"
    return "strain"


def _ranking_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    return dict(
        spearman=float(spearmanr(a, b).correlation),
        kendall=float(kendalltau(a, b).correlation),
        pearson=float(pearsonr(a, b).statistic),
        n=int(len(a)),
    )


def _refit_oce_on_siesta(rows: list[dict]) -> dict:
    """5-fold CV refit of OCE ridge on SIESTA labels.

    Loads features.npz to get the design matrix, aligns by structure name.
    Returns held-out predictions and metrics.
    """
    npz = np.load(SD / "features.npz", allow_pickle=True)
    X_all = npz["X"]
    names = [str(n) for n in npz["names"]]
    sizes = npz["sizes"]
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Index rows in feature matrix
    idx = []
    y = []
    n_sup = []
    fams = []
    kinds = []
    for r in rows:
        i = name_to_idx.get(r["name"])
        if i is None:
            continue
        idx.append(i)
        y.append(r["E_siesta"])
        n_sup.append(sizes[i])
        fams.append(r["family"])
        kinds.append(r["kind"])
    idx = np.array(idx)
    y = np.array(y)
    n_sup = np.array(n_sup)
    X = X_all[idx]
    print(f"  refit pool: {len(idx)} (perovskites with both features and SIESTA energy)")
    print(f"  feature dim: {X.shape[1]}")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=20260508)
    pred = np.zeros_like(y)
    for itr, ite in kf.split(X):
        m = Ridge(alpha=1.0, fit_intercept=True).fit(X[itr], y[itr])
        pred[ite] = m.predict(X[ite])
    err = pred - y
    err_pa = err / n_sup
    return dict(
        n=int(len(y)),
        rmse_eV=float(np.sqrt(np.mean(err ** 2))),
        rmse_meV_per_atom=float(np.sqrt(np.mean(err_pa ** 2))) * 1000,
        mae_meV_per_atom=float(np.mean(np.abs(err_pa))) * 1000,
        spearman=float(spearmanr(y, pred).correlation),
        kendall=float(kendalltau(y, pred).correlation),
        r2=1.0 - float(np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        per_family={
            f: {
                "n": int((np.array(fams) == f).sum()),
                "rmse_meV_per_atom": (
                    float(np.sqrt(np.mean(err_pa[np.array(fams) == f] ** 2))) * 1000
                    if (np.array(fams) == f).any() else None
                ),
            }
            for f in sorted(set(fams))
        },
        per_kind={
            k: {
                "n": int((np.array(kinds) == k).sum()),
                "rmse_meV_per_atom": (
                    float(np.sqrt(np.mean(err_pa[np.array(kinds) == k] ** 2))) * 1000
                    if (np.array(kinds) == k).any() else None
                ),
            }
            for k in sorted(set(kinds))
        },
    )


def _load_oce_predictions_gfnff() -> dict[str, float]:
    """OCE model fitted on full GFN-FF labels — used to compare ranking."""
    npz = np.load(SD / "features.npz", allow_pickle=True)
    X = npz["X"]; names = npz["names"]
    structs = json.loads((SD / "structures.json").read_text())
    by_name = {r["name"]: r for r in structs}
    energies = np.array([
        by_name[str(n)].get("energy_total_eV") for n in names
    ], dtype=object)
    valid = np.array([e is not None for e in energies])
    X_v = X[valid]; y_v = np.array([float(e) for e in energies[valid]])
    model = Ridge(alpha=1.0, fit_intercept=True).fit(X_v, y_v)
    pred = model.predict(X)
    return {str(names[i]): float(pred[i]) for i in range(len(names))}


def main():
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    converged = [r for r in siesta if r.get("converged")]
    print(f"SIESTA records: {len(siesta)}  converged: {len(converged)}")

    oce_pred_gfnff = _load_oce_predictions_gfnff()

    rows = []
    for r in converged:
        nm = r["name"]
        if nm not in oce_pred_gfnff:
            continue
        rows.append({
            "name": nm,
            "family": r.get("family", ""),
            "kind": _parse_kind(nm),
            "n_atoms": r["n_atoms"],
            "E_siesta": r["E_eV"],
            "E_gfnff": r.get("E_gfnff_eV"),
            "E_oce_pred_gfnff_trained": oce_pred_gfnff[nm],
        })
    n = len(rows)
    sizes = np.array([r["n_atoms"] for r in rows])
    e_siesta = np.array([r["E_siesta"] for r in rows])
    e_gfnff  = np.array([r["E_gfnff"] for r in rows])
    e_oce_g  = np.array([r["E_oce_pred_gfnff_trained"] for r in rows])

    print(f"Aligned set: {n}\n")

    # ===== Block A: ranking comparisons =====
    print("=" * 78)
    print("Ranking comparisons (Spearman/Kendall — scale-invariant)")
    print("=" * 78)
    pairs = {
        "GFN-FF  vs SIESTA": (e_gfnff, e_siesta),
        "OCE(GFNFF-trained) vs SIESTA": (e_oce_g, e_siesta),
        "OCE(GFNFF-trained) vs GFN-FF": (e_oce_g, e_gfnff),
    }
    rank_metrics = {}
    for label, (a, b) in pairs.items():
        m = _ranking_metrics(a, b)
        rank_metrics[label] = m
        print(f"  {label:<35s}  ρ={m['spearman']:6.3f}  τ={m['kendall']:6.3f}  "
              f"r={m['pearson']:6.3f}  n={m['n']}")

    # ===== Block B: per-atom RMSE on absolute scale =====
    # OCE vs GFN-FF is the only meaningful absolute-scale comparison
    err_oce_gfnff = e_oce_g - e_gfnff
    err_pa = err_oce_gfnff / sizes
    print(f"\n  OCE vs GFN-FF on the GFN-FF energy scale:")
    print(f"    RMSE = {np.sqrt(np.mean(err_oce_gfnff**2)):.3f} eV   "
          f"({np.sqrt(np.mean(err_pa**2))*1000:.1f} meV/atom)")
    print(f"    MAE  = {np.mean(np.abs(err_oce_gfnff)):.3f} eV   "
          f"({np.mean(np.abs(err_pa))*1000:.1f} meV/atom)")

    # ===== Block C: re-fit OCE on SIESTA labels (5-fold CV) =====
    print("\n" + "=" * 78)
    print("OCE refit on SIESTA labels (5-fold CV, ridge α=1.0)")
    print("=" * 78)
    refit = _refit_oce_on_siesta(rows)
    print(f"  n_train_total = {refit['n']}")
    print(f"  CV held-out RMSE  = {refit['rmse_eV']:.3f} eV   "
          f"({refit['rmse_meV_per_atom']:.1f} meV/atom)")
    print(f"  CV held-out MAE   = {refit['mae_meV_per_atom']:.1f} meV/atom")
    print(f"  R² = {refit['r2']:.4f}   ρ = {refit['spearman']:.4f}   "
          f"τ = {refit['kendall']:.4f}")
    print("\n  Per (A,B) family:")
    for fam, m in refit["per_family"].items():
        if m["rmse_meV_per_atom"] is None: continue
        print(f"    {fam:>6s}  n={m['n']:>3d}  "
              f"RMSE={m['rmse_meV_per_atom']:>6.1f} meV/atom")
    print("\n  Per kind:")
    for kind, m in refit["per_kind"].items():
        if m["rmse_meV_per_atom"] is None: continue
        print(f"    {kind:>6s}  n={m['n']:>3d}  "
              f"RMSE={m['rmse_meV_per_atom']:>6.1f} meV/atom")

    # ===== Per-family ranking table =====
    print("\n" + "=" * 78)
    print("Per-family ranking: GFN-FF vs SIESTA  vs  OCE(GFN-trained) vs SIESTA")
    print("=" * 78)
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    fam_table = {}
    for fam, items in sorted(by_fam.items()):
        if len(items) < 2: continue
        a = np.array([x["E_gfnff"]                    for x in items])
        b = np.array([x["E_siesta"]                    for x in items])
        c = np.array([x["E_oce_pred_gfnff_trained"]    for x in items])
        ρ_gs = float(spearmanr(a, b).correlation)
        ρ_os = float(spearmanr(c, b).correlation)
        fam_table[fam] = dict(n=len(items), rho_gfnff_siesta=ρ_gs,
                               rho_oce_siesta=ρ_os)
        print(f"  {fam:>6s}  n={len(items):>3d}  "
              f"ρ(GFN-FF vs SIESTA)={ρ_gs:6.3f}  "
              f"ρ(OCE vs SIESTA)={ρ_os:6.3f}")

    # Wall stats
    walls = [r["wall_time_s"] for r in converged]
    print(f"\nSIESTA wall-time: total {sum(walls):.0f}s ({sum(walls)/60:.1f}min), "
          f"median {np.median(walls):.1f}s, max {max(walls):.1f}s")

    out = {
        "n_total": n,
        "ranking_pairs": rank_metrics,
        "oce_vs_gfnff_per_atom": {
            "rmse_meV_per_atom": float(np.sqrt(np.mean(err_pa**2))) * 1000,
        },
        "refit_on_siesta_5fold": refit,
        "per_family_ranking": fam_table,
        "wall_seconds": float(sum(walls)),
    }
    out_path = SD / "siesta_validation_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
