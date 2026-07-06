"""Fit OCE on perovskite or MOF feature matrices and report headline metrics.

Reads:
  data/<subset>/features.npz        — X (n,p), names, sizes, families
  data/<subset>/feature_index.json  — column metadata
  data/<subset>/structures.json     — energy_total_eV (when present)

Splits parent-stratified by `family` (topology for MOFs, ABX for perovskites).
Fits ridge with alpha in a logspace and picks the best by 5-fold CV.

Reports:
  inter-structure RMSE / MAE / Spearman / Kendall (held-out)
  per-atom RMSE in meV/atom
  per-family MAE breakdown
  ablation: 1F only, +2F, +3F, +Madelung

Usage:
  python data/fit_subset.py mofs_qmof
  python data/fit_subset.py perovskites --energy-source xtb
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]


def _load(subset: str):
    sd = ROOT / "data" / subset
    npz = np.load(sd / "features.npz", allow_pickle=True)
    X = npz["X"]
    names = npz["names"]
    sizes = npz["sizes"]
    families = npz["families"]
    feat_index = json.loads((sd / "feature_index.json").read_text())
    feat_keys = [tuple(e["key"]) for e in feat_index]
    structs = json.loads((sd / "structures.json").read_text())
    by_name = {r["name"]: r for r in structs}
    energies = np.full(len(names), np.nan)
    for i, nm in enumerate(names):
        rec = by_name.get(str(nm))
        if rec is None:
            continue
        e = rec.get("energy_total_eV", rec.get("energy_eV"))
        if e is not None:
            energies[i] = float(e)
    return sd, X, names, sizes, families, feat_keys, energies


def _split_stratified(families: np.ndarray, frac: float = 0.10,
                       seed: int = 20260508
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Family-stratified holdout: drop `frac` of every family to test."""
    rng = np.random.default_rng(seed)
    idx_train: list[int] = []
    idx_test:  list[int] = []
    for fam in sorted(set(families.tolist())):
        sel = np.where(families == fam)[0]
        rng.shuffle(sel)
        n_test = max(1, int(np.ceil(len(sel) * frac)))
        idx_test.extend(sel[:n_test])
        idx_train.extend(sel[n_test:])
    return np.array(sorted(idx_train)), np.array(sorted(idx_test))


def _ridge_cv(X: np.ndarray, y: np.ndarray, sample_w: np.ndarray | None,
              alphas: list[float], k: int = 5,
              seed: int = 20260508) -> tuple[float, float]:
    """Pick alpha by k-fold CV minimising RMSE.  Returns (best_alpha, cv_rmse)."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    best_alpha = alphas[0]; best_rmse = np.inf
    for a in alphas:
        errs = []
        for itr, ite in kf.split(X):
            m = Ridge(alpha=a, fit_intercept=True)
            m.fit(X[itr], y[itr],
                   sample_weight=sample_w[itr] if sample_w is not None else None)
            errs.append(m.predict(X[ite]) - y[ite])
        rmse = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        if rmse < best_rmse:
            best_alpha, best_rmse = a, rmse
    return best_alpha, best_rmse


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, sizes: np.ndarray
              ) -> dict:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    err_pa = err / sizes
    rmse_pa = float(np.sqrt(np.mean(err_pa ** 2))) * 1000  # meV/atom
    mae_pa = float(np.mean(np.abs(err_pa))) * 1000
    rho = float(spearmanr(y_true, y_pred).correlation) if len(y_true) > 1 else float("nan")
    tau = float(kendalltau(y_true, y_pred).correlation) if len(y_true) > 1 else float("nan")
    r2 = 1.0 - float(np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse_eV": rmse, "mae_eV": mae,
        "rmse_meV_per_atom": rmse_pa, "mae_meV_per_atom": mae_pa,
        "spearman": rho, "kendall": tau, "r2": r2,
    }


def _column_class(key: tuple) -> str:
    if key == ("MAD",):
        return "MAD"
    return key[0]


def _fit_and_eval(X: np.ndarray, y: np.ndarray, sizes: np.ndarray,
                   families: np.ndarray, idx_tr: np.ndarray, idx_te: np.ndarray,
                   alpha: float | None = None,
                   alphas: list[float] | None = None) -> dict:
    Xt, yt, st = X[idx_tr], y[idx_tr], sizes[idx_tr]
    Xv, yv, sv = X[idx_te], y[idx_te], sizes[idx_te]
    fv = families[idx_te]

    if alpha is None:
        alphas = alphas or [1e-6, 1e-4, 1e-2, 1.0, 100.0]
        alpha, cv_rmse = _ridge_cv(Xt, yt, None, alphas)
    else:
        cv_rmse = float("nan")

    m = Ridge(alpha=alpha, fit_intercept=True).fit(Xt, yt)
    pred = m.predict(Xv)
    base = _metrics(yv, pred, sv)
    by_fam = defaultdict(list)
    for i, f in enumerate(fv):
        by_fam[str(f)].append((yv[i], pred[i], sv[i]))
    fam_metrics = {}
    for f, rows in sorted(by_fam.items()):
        ya = np.array([r[0] for r in rows])
        pa = np.array([r[1] for r in rows])
        sa = np.array([r[2] for r in rows])
        if len(ya) >= 2:
            fam_metrics[f] = _metrics(ya, pa, sa) | {"n": int(len(ya))}
        else:
            fam_metrics[f] = {"n": int(len(ya))}
    return {
        "alpha": alpha, "cv_rmse_eV": cv_rmse, "n_features": int(X.shape[1]),
        "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "test": base, "per_family": fam_metrics,
    }


def main(subset: str, seed: int = 20260508):
    sd, X, names, sizes, families, feat_keys, energies = _load(subset)
    n = len(names)
    valid = ~np.isnan(energies)
    print(f"Loaded {n} structures  ({int(valid.sum())} with energy)")
    if valid.sum() < n:
        print(f"  dropping {n - int(valid.sum())} without energy reference")
    X = X[valid]; sizes = sizes[valid]; families = families[valid]
    energies = energies[valid]; names = names[valid]
    n = len(names)

    # Strata: families with <2 samples after split fail; coalesce them
    fam_count = Counter(families.tolist())
    families_clean = np.array([
        f if fam_count[f] >= 4 else "_misc" for f in families
    ])
    idx_tr, idx_te = _split_stratified(families_clean, frac=0.10, seed=seed)
    print(f"split: {len(idx_tr)} train  /  {len(idx_te)} test  "
          f"(stratified by family)")

    feat_class = np.array([_column_class(k) for k in feat_keys])

    # Ablations: subsets of feature columns
    cls_to_cols = defaultdict(list)
    for j, c in enumerate(feat_class):
        cls_to_cols[c].append(j)
    available = sorted(cls_to_cols.keys())
    print(f"feature classes available: {available}")

    cumulative = []
    cum_cols: list[int] = []
    # order matters: 1F first, then 2F, then 3F, then MAD
    order = ["1F", "2F", "3F", "MAD"]
    order = [c for c in order if c in cls_to_cols]
    runs: dict[str, dict] = {}
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]

    for c in order:
        cum_cols.extend(cls_to_cols[c])
        cum_cols.sort()
        Xc = X[:, cum_cols]
        label = "+".join(order[:order.index(c) + 1])
        result = _fit_and_eval(
            Xc, energies, sizes, families_clean, idx_tr, idx_te,
            alphas=alphas,
        )
        runs[label] = result
        print(
            f"  {label:>16s}  "
            f"p={result['n_features']:>4d}  "
            f"alpha={result['alpha']:.1e}  "
            f"RMSE_test={result['test']['rmse_eV']:7.3f} eV  "
            f"({result['test']['rmse_meV_per_atom']:6.1f} meV/atom)  "
            f"ρ={result['test']['spearman']:6.3f}  "
            f"τ={result['test']['kendall']:6.3f}"
        )

    out = {
        "subset": subset, "n": int(n),
        "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "feature_classes": available,
        "ablation": runs,
    }
    (sd / "fit_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {sd / 'fit_summary.json'}")

    # Top + bottom families by MAE (best ablation)
    last = order[-1]
    fam_metrics = runs["+".join(order)]["per_family"]
    fam_with_mae = [
        (f, m["mae_meV_per_atom"], m["n"])
        for f, m in fam_metrics.items() if "mae_meV_per_atom" in m
    ]
    if fam_with_mae:
        fam_with_mae.sort(key=lambda x: x[1])
        print("\nPer-family MAE (full ablation):")
        for f, m, n in fam_with_mae:
            print(f"  {f:>10s}  n={n:>3d}  MAE={m:7.1f} meV/atom")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", help="data/<subset>/")
    args = ap.parse_args()
    main(args.subset)
