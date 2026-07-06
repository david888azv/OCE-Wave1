"""Train and evaluate the OCE v1.0.0 (JPCL submission) on percolation clusters.

We re-use the design-matrix code from `oce_v1.0.0` *unchanged* — only the
INPUT differs (carbon-only clusters with arbitrary connectivity instead of
small organic molecules).  This is the cleanest possible application of the
published method.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from sklearn.linear_model import Ridge

# Make the v1.0.0 OCE importable as `oce.*`
HERE = Path(__file__).resolve().parent
NEW_METHODS = HERE.parent
V1 = NEW_METHODS / "versions" / "oce_v1.0.0"
if str(NEW_METHODS) not in sys.path:
    # the package import structure expects 'oce' on sys.path
    sys.path.insert(0, str(NEW_METHODS))
# point oce -> versions/oce_v1.0.0
import importlib
import importlib.util as _ilu


def _load_v100_module(name: str):
    """Load a module file from versions/oce_v1.0.0 as oce.<name>.

    Register the module in sys.modules BEFORE exec_module so dataclass
    introspection (which looks the module up in sys.modules) works.
    """
    qualname = f"oce.{name}"
    path = V1 / f"{name}.py"
    spec = _ilu.spec_from_file_location(qualname, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


# Bootstrap: register a fake `oce` package pointing at v1.0.0
import types
oce_pkg = types.ModuleType("oce")
oce_pkg.__path__ = [str(V1)]
sys.modules["oce"] = oce_pkg

atomic_table_mod = _load_v100_module("atomic_table")
figures_mod = _load_v100_module("figures")
correlations_mod = _load_v100_module("correlations")


load_table = atomic_table_mod.load_table
enumerate_figures = figures_mod.enumerate_figures
correlations_for_molecule = correlations_mod.correlations_for_molecule
collect_feature_keys = correlations_mod.collect_feature_keys

ATOMIC_TABLE_PATH = NEW_METHODS / "data" / "atoms" / "atomic_table.json"


def featurise(atoms_list: list[Atoms],
              include_angles: bool = True,
              include_dihedrals: bool = False) -> tuple[np.ndarray, list[tuple]]:
    """Build the OCE design matrix for a list of ASE Atoms."""
    table = load_table(ATOMIC_TABLE_PATH)
    vectors = []
    for atoms in atoms_list:
        one, two, three, four = enumerate_figures(
            atoms, table,
            include_angles=include_angles,
            include_dihedrals=include_dihedrals,
        )
        three_arg = three if include_angles else None
        four_arg = four if include_dihedrals else None
        vectors.append(correlations_for_molecule(
            one, two, table,
            three_figs=three_arg, four_figs=four_arg,
        ))
    keys = collect_feature_keys(vectors)
    X = np.array([v.as_array(keys) for v in vectors])
    return X, keys


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> dict:
    """Fit Ridge with given alpha, return coefficients + intercept + diagnostics."""
    t0 = time.perf_counter()
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X, y)
    yhat = reg.predict(X)
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return dict(
        coef=reg.coef_.copy(),
        intercept=float(reg.intercept_),
        alpha=alpha,
        train_rmse=rmse,
        train_mae=mae,
        train_r2=r2,
        fit_time_s=time.perf_counter() - t0,
    )


def predict(X: np.ndarray, model: dict) -> np.ndarray:
    return X @ model["coef"] + model["intercept"]


def evaluate(yhat: np.ndarray, y: np.ndarray) -> dict:
    err = yhat - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    from scipy.stats import spearmanr, pearsonr
    rho, _ = spearmanr(y, yhat)
    r, _ = pearsonr(y, yhat)
    return dict(rmse=float(rmse), mae=float(mae), r2=float(r2),
                spearman=float(rho), pearson=float(r))


if __name__ == "__main__":
    from ase.build import molecule as _amol
    from runners import xtb_radical_energy
    # tiny smoke test: featurise a few real molecules to confirm import
    mols = [_amol("CH4"), _amol("C2H6"), _amol("C2H4"), _amol("C2H2")]
    X, keys = featurise(mols)
    print(f"Featurised {len(mols)} molecules → X{X.shape} with {len(keys)} keys")
    for k in keys[:6]:
        print("   key:", k)
