"""Power-law analysis for E_total(L) and E/N(L) on percolation clusters.

Two regimes are interesting:
  (a) Total energy:    E_total ~ L^{D_E}  with D_E ≈ D_f = 91/48 ≈ 1.896
       (extensive part of the energy ∝ cluster mass).
  (b) Per-atom energy: E/N − ε_bulk ~ L^{−α}
       with α expected in [|D_h − D_f|, |D_e − D_f|] ⊂ [7/48, 19/48]
       ≈ [0.146, 0.396]   (boundary-fractal corrections).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PowerLawFit:
    label: str
    exponent: float
    intercept: float
    rmse_log: float
    n_points: int


def fit_loglog(x: np.ndarray, y: np.ndarray, label: str = "") -> PowerLawFit:
    """Fit y = c x^exp via log-log linear regression."""
    mask = (x > 0) & (y > 0) & np.isfinite(y)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    exp_, log_c = sol
    pred = A @ sol
    rmse = float(np.sqrt(np.mean((pred - ly) ** 2)))
    return PowerLawFit(label=label, exponent=float(exp_),
                        intercept=float(np.exp(log_c)),
                        rmse_log=rmse, n_points=int(mask.sum()))


def fit_per_atom_with_bulk(L: np.ndarray, e_per_atom: np.ndarray,
                            label: str = "E/N − ε_bulk") -> tuple[PowerLawFit, float]:
    """Estimate ε_bulk by extrapolating E/N to L→∞ via a 3-parameter fit:

        E/N(L) = ε_bulk + c · L^{-α}.

    Strategy: grid-search ε_bulk so that log(E/N − ε_bulk) is most linear in
    log L (minimum log-log RMSE).  Returns (fit, ε_bulk_eV).
    """
    if len(L) < 3:
        return PowerLawFit(label=label, exponent=float("nan"),
                           intercept=float("nan"), rmse_log=float("nan"),
                           n_points=len(L)), float("nan")
    e_min = float(e_per_atom.min())
    # search below the minimum measured value (asymptote should be more negative)
    grid = np.linspace(e_min - 5.0, e_min - 1e-3, 401)
    best = None
    for eps in grid:
        residual = e_per_atom - eps   # should be > 0 for valid power-law
        if np.any(residual <= 0):
            continue
        f = fit_loglog(L.astype(float), residual)
        if best is None or f.rmse_log < best[0].rmse_log:
            best = (f, eps)
    if best is None:
        return PowerLawFit(label=label, exponent=float("nan"),
                           intercept=float("nan"), rmse_log=float("nan"),
                           n_points=len(L)), float("nan")
    fit, eps_bulk = best
    fit = PowerLawFit(label=label, exponent=fit.exponent,
                       intercept=fit.intercept, rmse_log=fit.rmse_log,
                       n_points=fit.n_points)
    return fit, float(eps_bulk)
