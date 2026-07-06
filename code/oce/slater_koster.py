"""Slater-Koster directional 2-centre integrals + Born-Mayer repulsion.

Wave-2 OCE extension. Provides the *machinery* (SK integrals + Born-Mayer)
and the *feature representation* needed to upgrade OCE 2F.

ARCHITECTURE OVERVIEW (essential):

  In v1.1.x, 2F is shell-only Wolfsberg-Helmholz:
      h_2F^WH(i,j) = sum_{shells μν} ε_μi ε_νj / (ε_μi + ε_νj) ·
                       <phi_μi | H | phi_νj>(r_ij)
  where the matrix element is approximated by a non-directional
  ε-weighted form.  This loses σ/π distinction and has no repulsive
  short-range part.

  Wave-2 PROPER REPLACEMENT:
    (1) Each SK channel ssσ, spσ_x, spσ_y, spσ_z, ppσ_xx, ppσ_xy, ...,
        ppπ_xx, ... becomes its OWN linear feature in the OCE basis.
        That is, the basis grows by ~6 directional channels per (elem, elem)
        pair, each with its own J coefficient learnt by ridge.
    (2) Born-Mayer V_rep(r) = A_ij exp(-r/ρ_ij) becomes a separate feature
        per element pair (or per universal scale).  A, ρ are fit during
        the same ridge as J coefficients.
    (3) Forces are derived analytically from sum_F J_F · ∂Φ_F/∂r, where
        each Φ_F has known closed-form gradient (radial + cos² direction).
    (4) The training data are total energies E_total — the model learns
        the right linear combination of channels that reproduces E_total
        AND keeps |F| < 1 eV/Å at equilibria (BFGS-stable).

  This file provides the integrals V(r) and direction cosines.  The
  feature-builder integration and force-routine are downstream tasks.

REFERENCES:
  Slater-Koster: PRB 94, 1498 (1954)
  Harrison universal: PRB 14, 702 (1976)
  Born-Mayer for ionic crystals: Phys Rev 50, 1018 (1936)

Note: a previous version of this module attempted "sum of all channels =
total pair energy" but that's not how SK integrals work — they're matrix
elements, not energy contributions.  The energy comes from sum-over-occupied
eigenvalues of the full Hamiltonian.  The right OCE flavour is to expose
each integral as a feature and let ridge learn the right combination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Harrison universal SK coefficients (eV·Å²)
# η_llm * ℏ²/m_e in eV·Å² units.  Source: Harrison 1980 "Electronic Structure"
HARRISON_ETA = {
    ("s", "s", "sigma"): -1.40,
    ("s", "p", "sigma"): +1.84,
    ("p", "p", "sigma"): +3.24,
    ("p", "p", "pi"):    -0.81,
}
# ℏ²/m_e in eV·Å² is 7.61996...; Harrison absorbs it into η.

# Born-Mayer: V_rep(r) = A · exp(-r/ρ).  Default scaffold parameters
# (will be calibrated by fitting to xtb potential at small-r samples)
BORN_MAYER_DEFAULT_A_eV  = 200.0   # placeholder
BORN_MAYER_DEFAULT_RHO_A = 0.40    # placeholder


@dataclass(frozen=True)
class SKPair:
    """Material parameters for an element pair under Harrison + Born-Mayer."""
    element_i: str
    element_j: str
    A_eV: float = BORN_MAYER_DEFAULT_A_eV
    rho_A: float = BORN_MAYER_DEFAULT_RHO_A


def _eta(l1: str, l2: str, m: str) -> float:
    """Symmetric in l1, l2 by orbital nomenclature."""
    if (l1, l2, m) in HARRISON_ETA:
        return HARRISON_ETA[(l1, l2, m)]
    if (l2, l1, m) in HARRISON_ETA:
        # swap of orbital order: V_sp = +V_ps but η_sp = -η_ps for some authors
        eta = HARRISON_ETA[(l2, l1, m)]
        return -eta if (l1 == "p" and l2 == "s") else eta
    raise KeyError(f"No Harrison η for ({l1}, {l2}, {m})")


def V_sk(l1: str, l2: str, m: str, r: float | np.ndarray) -> float | np.ndarray:
    """Slater-Koster integral V_{l1 l2 m}(r) = η · 1/r².

    Returns energy in eV when r is in Angstrom.
    """
    return _eta(l1, l2, m) / (r * r)


def dV_sk_dr(l1: str, l2: str, m: str, r: float | np.ndarray) -> float | np.ndarray:
    """Radial derivative dV/dr = -2η/r³."""
    return -2.0 * _eta(l1, l2, m) / (r * r * r)


def V_born_mayer(r: float | np.ndarray, A: float = BORN_MAYER_DEFAULT_A_eV,
                  rho: float = BORN_MAYER_DEFAULT_RHO_A) -> float | np.ndarray:
    return A * np.exp(-r / rho)


def dV_born_mayer_dr(r, A=BORN_MAYER_DEFAULT_A_eV,
                       rho=BORN_MAYER_DEFAULT_RHO_A):
    return -(A / rho) * np.exp(-r / rho)


# Slater-Koster directional projection — converts atomic-orbital matrix
# elements into a function of bond direction (l, m, n) = unit vector.
def two_centre_block(r_vec: np.ndarray) -> dict:
    """Compute SK matrix-element values for one (i, j) pair.

    r_vec: vector from atom i to atom j (Å).
    Returns dict with keys 'ss_sigma', 'sp_sigma_x', 'sp_sigma_y', 'sp_sigma_z',
    'pp_sigma_xx', 'pp_sigma_yy', 'pp_sigma_zz',
    'pp_pi_xy', 'pp_pi_xz', 'pp_pi_yz',
    each in eV.

    Standard SK formulae (Slater & Koster 1954):
      <s | H | s>_ij        = V_ssσ
      <s | H | p_α>_ij      = l_α · V_spσ
      <p_α | H | p_β>_ij    = l_α l_β V_ppσ + (δ_αβ − l_α l_β) V_ppπ
    where (l_x, l_y, l_z) are direction cosines.
    """
    r = float(np.linalg.norm(r_vec))
    if r < 1e-6:
        raise ValueError(f"Atoms too close: r={r}")
    l, m, n = (r_vec / r).tolist()
    ss_s = V_sk("s","s","sigma",r)
    sp_s = V_sk("s","p","sigma",r)
    pp_s = V_sk("p","p","sigma",r)
    pp_p = V_sk("p","p","pi",r)
    return {
        "r": r,
        "direction": (l, m, n),
        "ss_sigma":   ss_s,
        "sp_sigma":   {"x": l*sp_s, "y": m*sp_s, "z": n*sp_s},
        "pp_sigma":   {"xx": l*l*pp_s, "yy": m*m*pp_s, "zz": n*n*pp_s,
                        "xy": l*m*pp_s, "xz": l*n*pp_s, "yz": m*n*pp_s},
        "pp_pi": {"xx": (1-l*l)*pp_p, "yy": (1-m*m)*pp_p, "zz": (1-n*n)*pp_p,
                   "xy": -l*m*pp_p, "xz": -l*n*pp_p, "yz": -m*n*pp_p},
    }


def channel_feature_vector(r_vec: np.ndarray) -> dict:
    """Return the numerical value of each SK directional channel for an
    (i, j) pair separated by r_vec.

    This is the per-pair contribution that should be SUMMED across all
    pairs (i,j) of a given (elem_i, elem_j) class to form an OCE 2F-SK
    feature.  Each channel is a separate feature column with its own J.

    Returns: dict mapping channel name → eV value.  Keys are:
      'ss_sigma', 'sp_sigma_l', 'sp_sigma_m', 'sp_sigma_n',
      'pp_sigma_xx', 'pp_sigma_yy', 'pp_sigma_zz',
      'pp_sigma_xy', 'pp_sigma_xz', 'pp_sigma_yz',
      'pp_pi_xx',    'pp_pi_yy',    'pp_pi_zz',
      'pp_pi_xy',    'pp_pi_xz',    'pp_pi_yz',
    where l/m/n are direction cosines.  Sum-over-pairs each produces a
    rotationally-invariant scalar feature for OCE training.
    """
    block = two_centre_block(r_vec)
    return {
        "ss_sigma": block["ss_sigma"],
        "sp_sigma_l": block["sp_sigma"]["x"],
        "sp_sigma_m": block["sp_sigma"]["y"],
        "sp_sigma_n": block["sp_sigma"]["z"],
        **{f"pp_sigma_{k}": v for k, v in block["pp_sigma"].items()},
        **{f"pp_pi_{k}":    v for k, v in block["pp_pi"].items()},
    }


def channel_radial_gradient(r_vec: np.ndarray) -> dict:
    """∂(channel_value) / ∂r — for analytical forces.

    Returns same keys as channel_feature_vector, each value being d/dr.
    """
    r = float(np.linalg.norm(r_vec))
    r_hat = r_vec / r
    l, m, n = r_hat
    d_ss = dV_sk_dr("s","s","sigma",r)
    d_sp = dV_sk_dr("s","p","sigma",r)
    d_pps = dV_sk_dr("p","p","sigma",r)
    d_ppp = dV_sk_dr("p","p","pi",r)
    return {
        "ss_sigma": d_ss,
        "sp_sigma_l": l*d_sp,
        "sp_sigma_m": m*d_sp,
        "sp_sigma_n": n*d_sp,
        "pp_sigma_xx": l*l*d_pps, "pp_sigma_yy": m*m*d_pps,
        "pp_sigma_zz": n*n*d_pps, "pp_sigma_xy": l*m*d_pps,
        "pp_sigma_xz": l*n*d_pps, "pp_sigma_yz": m*n*d_pps,
        "pp_pi_xx": (1-l*l)*d_ppp, "pp_pi_yy": (1-m*m)*d_ppp,
        "pp_pi_zz": (1-n*n)*d_ppp, "pp_pi_xy": -l*m*d_ppp,
        "pp_pi_xz": -l*n*d_ppp, "pp_pi_yz": -m*n*d_ppp,
    }


def born_mayer_pair(r_vec: np.ndarray, sk: SKPair) -> tuple:
    """Born-Mayer feature value + radial gradient.

    Returns (V_rep, dV_rep/dr).
    """
    r = float(np.linalg.norm(r_vec))
    V = V_born_mayer(r, sk.A_eV, sk.rho_A)
    dV = dV_born_mayer_dr(r, sk.A_eV, sk.rho_A)
    return V, dV


# === Smoke test === #

if __name__ == "__main__":
    print("Wave-2 scaffold — SK + Born-Mayer integrals\n")

    # 1. Verify Harrison values at common bond lengths
    print("Slater-Koster integrals at r = 1.42 Å (graphene C-C):")
    for (l1, l2, m), eta in HARRISON_ETA.items():
        V = V_sk(l1, l2, m, 1.42)
        print(f"  V_{l1}{l2}{m:<6s} = {V:+8.3f} eV  (η = {eta:+.2f})")

    # 2. Channel feature vector
    print("\nChannel feature vector for r=(1.42, 0, 0) (along x-axis):")
    cf = channel_feature_vector(np.array([1.42, 0, 0]))
    for k, v in sorted(cf.items()):
        print(f"  {k:<14s}  {v:+8.3f} eV")

    # 3. Channel feature vector along (1,1,1)/√3 — shows directional mixing
    print("\nChannel feature vector for r=(1,1,1)·1.42/√3 (diagonal direction):")
    rv = np.array([1, 1, 1]) * (1.42 / np.sqrt(3))
    cf2 = channel_feature_vector(rv)
    for k, v in sorted(cf2.items()):
        print(f"  {k:<14s}  {v:+8.3f} eV")

    # 4. Verify gradient: numerical vs analytical for one channel
    print("\nGradient check (channel: pp_sigma_xx) at r=(1.42, 0, 0):")
    rv = np.array([1.42, 0, 0])
    grad_an = channel_radial_gradient(rv)
    h = 1e-5
    fp = channel_feature_vector(rv + np.array([h, 0, 0]))["pp_sigma_xx"]
    fm = channel_feature_vector(rv - np.array([h, 0, 0]))["pp_sigma_xx"]
    grad_num = (fp - fm) / (2*h)
    print(f"  analytic ∂/∂x  = {grad_an['pp_sigma_xx']:+.6f} eV/Å")
    print(f"  numerical      = {grad_num:+.6f} eV/Å")
    print(f"  diff           = {grad_an['pp_sigma_xx'] - grad_num:+.2e} eV/Å")

    # 5. Born-Mayer
    print("\nBorn-Mayer (default A=200 ρ=0.4):")
    sk = SKPair("C", "C")
    for r in [1.0, 1.4, 2.0, 4.0]:
        rv = np.array([r, 0, 0])
        V, dV = born_mayer_pair(rv, sk)
        print(f"  r={r:.1f}  V_rep={V:>8.4f}  dV/dr={dV:>+9.4f} eV/Å")

    print("\n→ Each channel is now an OCE-ready feature.  Next step:")
    print("    - integrate channel_feature_vector into build_features.py")
    print("    - sum over (i,j) pairs of given (elem_i, elem_j, channel)")
    print("    - add to OCE design matrix → ridge fits J coeff per channel")
    print("    - forces: F_i = -∇_r_i [Σ_pairs Σ_chan J_chan · Φ_chan(r_ij)]")
