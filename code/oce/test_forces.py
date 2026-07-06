"""Numerical-vs-analytic gradient tests for :mod:`oce.forces`.

These tests fix the *interface* and the success criterion before the
actual derivatives are implemented.  In v0.1.0 the analytic gradient
routines return zero (mock); the cross-checks against central
differences are therefore expected to fail and are decorated with
``@xfail``.  As each feature class gets a real gradient (2F → 3F →
4F), remove the corresponding ``@xfail``.

The file runs both as a pytest module **and** as a plain script.
``pytest`` is optional; if it is not installed, ``python
oce/test_forces.py`` runs the same checks via a hand-rolled mini-
runner that prints PASS/FAIL/XFAIL.

Test cases
----------
* ``test_numerical_gradient_helper_runs``    — sanity on the helper
* ``test_one_figure_gradient_is_zero``       — passes today
* ``test_calculator_returns_energy_and_forces_shape``  — ASE wiring
* ``test_analytic_matches_numerical[mol]``   — XFAIL until v1.1.0
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule

# --------------------------------------------------------------------- #
# Optional pytest                                                        #
# --------------------------------------------------------------------- #
try:
    import pytest                                          # type: ignore
    _HAS_PYTEST = True
except ImportError:                                          # pragma: no cover
    _HAS_PYTEST = False

    class _XFailMarker:
        """No-op stand-in so @pytest.mark.xfail still parses."""
        def __init__(self, *_a, **_kw): pass
        def __call__(self, fn): fn._xfail = True; return fn

    class _MarkNamespace:
        def xfail(self, *a, **kw): return _XFailMarker(*a, **kw)

    class _PytestStub:
        mark = _MarkNamespace()

    pytest = _PytestStub()                                  # type: ignore


# --------------------------------------------------------------------- #
# Imports under test                                                     #
# --------------------------------------------------------------------- #
from oce.atomic_table import load_table
from oce.calculator import OCECalculator
from oce.correlations import correlations_for_molecule
from oce.figures import enumerate_figures
from oce.fit import OCEModel
from oce.forces import (compute_forces, freeze_topology, grad_one_figure)


# --------------------------------------------------------------------- #
# Fixtures (also reusable as plain functions)                            #
# --------------------------------------------------------------------- #
_ATOMIC_TABLE_PATH = (Path(__file__).resolve().parents[1]
                      / "data" / "atoms" / "atomic_table.json")


def _load_table():
    return load_table(_ATOMIC_TABLE_PATH)


def _toy_model_for_atoms(atoms: Atoms, atomic_table) -> OCEModel:
    """Build an OCEModel with J=1.0 on every feature key seen in
    ``atoms`` and intercept zero. Decoupled from real training data
    so the tests stay deterministic and self-contained."""
    one, two, three, _four = enumerate_figures(
        atoms, atomic_table,
        include_angles=True, include_dihedrals=False)
    cv = correlations_for_molecule(
        one, two, atomic_table, three_figs=three)
    keys = list(cv.feature_keys)
    return OCEModel(
        feature_keys=keys,
        J=[1.0] * len(keys),
        intercept=0.0,
        method="toy(J=1)",
    )


def _numerical_forces(atoms: Atoms, atomic_table, model: OCEModel,
                      h: float = 1e-4) -> np.ndarray:
    """Central-difference forces from energy alone (3·N·2 evals)."""
    n = len(atoms)
    F = np.zeros((n, 3))
    pos0 = atoms.get_positions().copy()
    try:
        for a in range(n):
            for d in range(3):
                pos = pos0.copy(); pos[a, d] += h
                atoms.set_positions(pos)
                E_p = _energy(atoms, atomic_table, model)

                pos = pos0.copy(); pos[a, d] -= h
                atoms.set_positions(pos)
                E_m = _energy(atoms, atomic_table, model)

                F[a, d] = -(E_p - E_m) / (2.0 * h)
    finally:
        atoms.set_positions(pos0)
    return F


def _energy(atoms: Atoms, atomic_table, model: OCEModel) -> float:
    one, two, three, _four = enumerate_figures(
        atoms, atomic_table,
        include_angles=True, include_dihedrals=False)
    cv = correlations_for_molecule(
        one, two, atomic_table, three_figs=three)
    return float(model.predict(cv))


_TEST_MOLECULES = ["H2O", "CH4", "C2H4"]


# --------------------------------------------------------------------- #
# Tests                                                                  #
# --------------------------------------------------------------------- #

def test_numerical_gradient_helper_runs():
    """Sanity: ``_numerical_forces`` returns a finite (N, 3) array."""
    table = _load_table()
    for name in _TEST_MOLECULES:
        atoms = molecule(name)
        model = _toy_model_for_atoms(atoms, table)
        F_num = _numerical_forces(atoms, table, model)
        assert F_num.shape == (len(atoms), 3), (
            f"{name}: got shape {F_num.shape}")
        assert np.all(np.isfinite(F_num)), f"{name}: non-finite entries"


def test_one_figure_gradient_is_zero():
    """1F has no position dependence → analytic grad is identically zero."""
    table = _load_table()
    for name in _TEST_MOLECULES:
        atoms = molecule(name)
        one, _two, _three, _four = enumerate_figures(atoms, table)
        for fig in one:
            g = grad_one_figure(fig, len(atoms), table)
            assert g.shape == (len(atoms), 3)
            assert np.array_equal(g, np.zeros_like(g)), (
                f"{name}: 1F {fig.key()} produced nonzero grad")


def test_calculator_returns_energy_and_forces_shape():
    """OCECalculator exposes ``energy`` (scalar) and ``forces`` (N, 3)."""
    table = _load_table()
    for name in _TEST_MOLECULES:
        atoms = molecule(name)
        model = _toy_model_for_atoms(atoms, table)
        atoms.calc = OCECalculator(model=model, atomic_table=table)
        E = atoms.get_potential_energy()
        F = atoms.get_forces()
        assert isinstance(E, float), f"{name}: energy is {type(E)}"
        assert F.shape == (len(atoms), 3), (
            f"{name}: forces shape {F.shape}")
        assert np.all(np.isfinite(F)), f"{name}: non-finite forces"


def test_calculator_forces_nonzero_when_perturbed():
    """Regression: a previous bug caused get_forces() to return cached
    zeros when get_potential_energy() had been called first. Verify
    forces are nonzero on a perturbed (non-symmetric) geometry and
    independent of the order energy/forces are queried."""
    table = _load_table()
    rng = np.random.default_rng(7)
    for name in _TEST_MOLECULES:
        atoms = molecule(name)
        atoms.set_positions(atoms.get_positions()
                            + rng.uniform(-0.05, 0.05, atoms.positions.shape))
        model = _toy_model_for_atoms(atoms, table)

        # Order A: energy first, then forces (the failure mode).
        a = atoms.copy()
        a.calc = OCECalculator(model=model, atomic_table=table)
        _ = a.get_potential_energy()
        F_after_E = np.asarray(a.get_forces())

        # Order B: forces first.
        b = atoms.copy()
        b.calc = OCECalculator(model=model, atomic_table=table)
        F_first = np.asarray(b.get_forces())

        assert np.linalg.norm(F_after_E) > 1e-3, (
            f"{name}: forces zero after energy-first call (caching bug)")
        np.testing.assert_allclose(F_after_E, F_first, atol=1e-10,
                                    err_msg=f"{name}: order dependence")


def test_analytic_matches_numerical_H2O():
    _check_matches("H2O")


def test_analytic_matches_numerical_CH4():
    _check_matches("CH4")


def test_analytic_matches_numerical_C2H4():
    _check_matches("C2H4")


def _check_matches(name: str, atol: float = 1e-3, rtol: float = 1e-3,
                   perturb: float = 0.0, seed: int = 0) -> None:
    """Numeric-vs-analytic gradient check.

    A nonzero ``perturb`` displaces every atom by a uniform random
    vector of magnitude ``perturb`` Å — useful to leave the symmetric
    equilibrium where many gradient components are accidentally zero.
    """
    table = _load_table()
    atoms = molecule(name)
    if perturb > 0.0:
        rng = np.random.default_rng(seed)
        delta = rng.uniform(-perturb, perturb, size=(len(atoms), 3))
        atoms.set_positions(atoms.get_positions() + delta)
    model = _toy_model_for_atoms(atoms, table)
    topo = freeze_topology(atoms, table)
    F_anal = compute_forces(atoms, table, model, topology=topo)
    F_num = _numerical_forces(atoms, table, model)
    err = np.abs(F_anal - F_num).max()
    assert err < max(atol, rtol * np.abs(F_num).max() + 1e-12), (
        f"{name}: max |F_anal-F_num| = {err:.3e} eV/Å "
        f"(F_num range = {np.abs(F_num).max():.3e})")


def test_analytic_matches_numerical_H2O_perturbed():
    """H2O at a displaced (non-symmetric) geometry."""
    _check_matches("H2O", perturb=0.05, seed=1)


def test_analytic_matches_numerical_CH4_perturbed():
    _check_matches("CH4", perturb=0.05, seed=2)


def test_analytic_matches_numerical_C2H4_perturbed():
    _check_matches("C2H4", perturb=0.05, seed=3)


# --------------------------------------------------------------------- #
# Plain-script runner (no pytest dep)                                    #
# --------------------------------------------------------------------- #

def _run_one(fn) -> tuple[str, str | None]:
    """Run a test function. Return (status, message).

    status ∈ {"PASS", "FAIL", "XPASS", "XFAIL"}.
    """
    is_xfail = bool(getattr(fn, "_xfail", False))
    try:
        fn()
    except AssertionError as exc:
        return ("XFAIL" if is_xfail else "FAIL", str(exc).splitlines()[0])
    except Exception as exc:
        if is_xfail:
            return ("XFAIL", f"{type(exc).__name__}: {exc}")
        return ("FAIL",
                f"{type(exc).__name__}: {exc}\n"
                + traceback.format_exc(limit=3))
    return ("XPASS" if is_xfail else "PASS", None)


def _main() -> int:
    tests = [
        test_numerical_gradient_helper_runs,
        test_one_figure_gradient_is_zero,
        test_calculator_returns_energy_and_forces_shape,
        test_calculator_forces_nonzero_when_perturbed,
        test_analytic_matches_numerical_H2O,
        test_analytic_matches_numerical_CH4,
        test_analytic_matches_numerical_C2H4,
        test_analytic_matches_numerical_H2O_perturbed,
        test_analytic_matches_numerical_CH4_perturbed,
        test_analytic_matches_numerical_C2H4_perturbed,
    ]
    width = max(len(t.__name__) for t in tests)
    counts = {"PASS": 0, "FAIL": 0, "XFAIL": 0, "XPASS": 0}
    print(f"\noce/test_forces.py — running {len(tests)} tests "
          f"(pytest {'available' if _HAS_PYTEST else 'NOT installed; using script runner'})\n")
    for fn in tests:
        status, msg = _run_one(fn)
        counts[status] += 1
        line = f"  [{status:5s}] {fn.__name__:<{width}}"
        if msg:
            line += f"  — {msg[:100]}"
        print(line)
    print(f"\nSummary: PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
          f"XFAIL={counts['XFAIL']}  XPASS={counts['XPASS']}")
    # Exit non-zero only on unexpected outcomes (FAIL or XPASS).
    return 0 if (counts["FAIL"] == 0 and counts["XPASS"] == 0) else 1


if __name__ == "__main__":
    sys.exit(_main())
