"""Fit OCE coefficients J_F by linear regression on xtb training energies."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

from oce.atomic_table import load_table
from oce.correlations import (CorrelationVector,
                              collect_feature_keys,
                              correlations_for_molecule)
from oce.dataset import MolEntry, entry_to_atoms, load_dataset
from oce.figures import enumerate_figures


@dataclass
class OCEModel:
    feature_keys: list[tuple]
    J: list[float]
    intercept: float
    method: str

    def predict(self, vec: CorrelationVector) -> float:
        x = vec.as_array(self.feature_keys)
        return float(np.dot(x, self.J) + self.intercept)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "feature_keys": [list(k) for k in self.feature_keys],
            "J": list(self.J),
            "intercept": self.intercept,
            "method": self.method,
        }, default=str, indent=2))


def build_design_matrix(entries: list[MolEntry], atomic_table,
                        include_angles: bool = True,
                        include_dihedrals: bool = False) \
        -> tuple[np.ndarray, np.ndarray, list[tuple], list[CorrelationVector]]:
    vectors: list[CorrelationVector] = []
    for e in entries:
        atoms = entry_to_atoms(e)
        one, two, three, four = enumerate_figures(
            atoms, atomic_table,
            include_angles=include_angles,
            include_dihedrals=include_dihedrals)
        three_arg = three if include_angles else None
        four_arg = four if include_dihedrals else None
        vectors.append(correlations_for_molecule(
            one, two, atomic_table,
            three_figs=three_arg, four_figs=four_arg))
    keys = collect_feature_keys(vectors)
    X = np.array([v.as_array(keys) for v in vectors])
    y = np.array([e.energy_eV for e in entries])
    return X, y, keys, vectors


def fit_model(X: np.ndarray, y: np.ndarray, keys: list[tuple],
              alpha: float = 0.0) -> OCEModel:
    if alpha > 0:
        reg = Ridge(alpha=alpha, fit_intercept=True)
        method = f"Ridge(alpha={alpha})"
    else:
        reg = LinearRegression(fit_intercept=True)
        method = "OLS"
    reg.fit(X, y)
    return OCEModel(
        feature_keys=keys,
        J=reg.coef_.tolist(),
        intercept=float(reg.intercept_),
        method=method,
    )


def report_fit(model: OCEModel, X: np.ndarray, y: np.ndarray,
               entries: list[MolEntry], label: str = "fit") -> None:
    yhat = X @ np.array(model.J) + model.intercept
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print(f"\n--- {label}: {model.method} ---")
    print(f"  n_molecules = {len(y)},  n_features = {len(model.J)}")
    print(f"  R^2 = {r2:.6f}    RMSE = {rmse:.4f} eV    MAE = {mae:.4f} eV")
    print(f"  largest |J_F| coefficients:")
    abs_j = sorted(zip(model.feature_keys, model.J), key=lambda kv: -abs(kv[1]))
    for k, j in abs_j[:8]:
        print(f"    J = {j:+10.5f}    {k}")
    print(f"  per-molecule residuals (first 10):")
    for i, e in enumerate(entries[:10]):
        print(f"    {e.name:24s}  y={y[i]:+10.3f}  ŷ={yhat[i]:+10.3f}  Δ={err[i]:+7.3f} eV")


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    table = load_table(base / "data" / "atoms" / "atomic_table.json")
    train = load_dataset(base / "data" / "molecules" / "train.json")
    Xtr, ytr, keys, _ = build_design_matrix(train, table)

    print(f"Design matrix: X shape = {Xtr.shape}")
    print(f"Feature keys ({len(keys)}):")
    for k in keys:
        print(f"  {k}")

    # OLS (no regularization)
    m_ols = fit_model(Xtr, ytr, keys, alpha=0.0)
    report_fit(m_ols, Xtr, ytr, train, "TRAIN (OLS)")
    m_ols.to_json(base / "results" / "model_ols.json")

    # Ridge for comparison
    m_ridge = fit_model(Xtr, ytr, keys, alpha=1e-2)
    report_fit(m_ridge, Xtr, ytr, train, "TRAIN (Ridge)")
    m_ridge.to_json(base / "results" / "model_ridge.json")
