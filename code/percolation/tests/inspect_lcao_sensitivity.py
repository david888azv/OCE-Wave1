"""How sensitive is the OCE 2F LCAO bonding eigenvalue to bond length?

The 2F figure value uses
    h(r) = h0 · exp(-α (r - r0)),  h0 = -8.0 eV, α = 1.5 Å⁻¹, r0 = 1.5 Å.
Plot λ⁻(r) over the typical C–C bond range.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from oce_predict import correlations_mod, atomic_table_mod, ATOMIC_TABLE_PATH

table = atomic_table_mod.load_table(ATOMIC_TABLE_PATH)
eps_C_p = next(s.epsilon_eV for s in table["C"].shells if s.label == "p")
eps_C_s = next(s.epsilon_eV for s in table["C"].shells if s.label == "s")

print(f"Free-atom ε_C(2p) = {eps_C_p:.3f} eV   ε_C(2s) = {eps_C_s:.3f} eV\n")

print(f"  r (Å)   h(r) (eV)   λ⁻_pp(r) (eV)   λ⁻_ss(r) (eV)")
for r in [1.20, 1.30, 1.35, 1.42, 1.46, 1.54, 1.60, 1.80]:
    lam_pp = correlations_mod.lcao_pair_energy(eps_C_p, eps_C_p, r)
    lam_ss = correlations_mod.lcao_pair_energy(eps_C_s, eps_C_s, r)
    h0, alpha, r0 = (correlations_mod.H0_DEFAULT,
                     correlations_mod.ALPHA_DEFAULT,
                     correlations_mod.R0_DEFAULT)
    h = h0 * np.exp(-alpha * (r - r0))
    print(f"  {r:.2f}    {h:+8.3f}    {lam_pp:+9.3f}      {lam_ss:+9.3f}")

print(f"\nReference: graphene equilibrium r=1.42 Å; relaxed clusters average ~1.36 Å;")
print(f"bond-order classification cutoff is at r/r_cov ≈ 0.93 → r ≈ 1.41 Å.")

print(f"\nFractional sensitivity of λ⁻_pp:")
lam_142 = correlations_mod.lcao_pair_energy(eps_C_p, eps_C_p, 1.42)
for r in [1.30, 1.35, 1.40, 1.46, 1.54]:
    lam = correlations_mod.lcao_pair_energy(eps_C_p, eps_C_p, r)
    print(f"  r={r:.2f}: λ⁻_pp = {lam:+.3f} eV   "
          f"Δ vs r=1.42  =  {lam-lam_142:+.3f} eV   "
          f"({100*(lam-lam_142)/abs(lam_142):+.1f}%)")
