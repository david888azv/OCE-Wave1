"""Compute SIESTA-PBE decomposition energy for the 18 F-free endmembers
and recompute cross-correlation against Mannodi PBE / HSE06 decomp.

E_decomp(ABX3) = E_coh(ABX3) - E_coh(AX) - E_coh(BX2)

with each E_coh in eV per formula unit (each primitive cell here is 1 f.u.).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

SD = Path(__file__).resolve().parent


def main():
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD/"atom_refs.json").read_text()).items()
                  if r.get("converged")}
    binaries = json.loads((SD/"siesta_binaries.json").read_text())

    # E_coh per formula unit (the cell may contain multiple f.u.)
    # AX (B2/B1): 2 atoms / cell = 1 f.u. → /1
    # BX2 (CdI2): 3 atoms / cell = 1 f.u. → /1
    # BX2 (cotunnite PNMA): 12 atoms / cell = 4 f.u. → /4
    binary_coh = {}
    for name, r in binaries.items():
        if not r.get("converged"):
            continue
        symbols = r["symbols"]
        n_atoms = len(symbols)
        ref_sum = sum(atom_refs[s] for s in symbols)
        E_coh_cell = r["E_eV"] - ref_sum
        # detect f.u. count: AX (2 atoms = 1 f.u.), BX2 (3 atoms = 1 f.u.,
        # 12 atoms = 4 f.u. cotunnite)
        if n_atoms == 2:
            n_fu = 1
        elif n_atoms == 3:
            n_fu = 1
        elif n_atoms == 12:
            n_fu = 4
        else:
            n_fu = 1
        binary_coh[name] = E_coh_cell / n_fu
    print(f"Binary E_coh (eV per f.u.):")
    for n in sorted(binary_coh):
        print(f"  {n:>6s}  E_coh = {binary_coh[n]:>9.4f} eV")

    # Build E_decomp for each user endmember
    endmembers = json.loads((SD/"hse_endmember_validation.json").read_text())

    rows = []
    for e in endmembers:
        A, B, X = e["A"], e["B"], e["X"]
        ax = f"{A}{X}"           # CsCl, KI, ...
        bx2 = f"{B}{X}2"         # PbI2, SnBr2, ...
        if ax not in binary_coh or bx2 not in binary_coh:
            print(f"  SKIP {e['name']}: missing binaries {ax}, {bx2}")
            continue
        # User's SIESTA E_coh for the perovskite is per supercell (5-atom prim = 1 f.u.)
        e_perov_coh = e["E_coh_eV"]
        e_decomp_siesta = e_perov_coh - binary_coh[ax] - binary_coh[bx2]
        rows.append({**e,
                      "binary_AX": ax,
                      "binary_BX2": bx2,
                      "E_coh_AX_eV": binary_coh[ax],
                      "E_coh_BX2_eV": binary_coh[bx2],
                      "E_decomp_siesta_eV": e_decomp_siesta})

    print()
    print(f"{'Endmember':<22s}  {'SIESTA_decomp':>13s}  {'Mannodi_PBE':>11s}  {'Mannodi_HSE':>11s}  {'Δ_SI-MAN_pbe':>13s}")
    print("-" * 90)
    for r in sorted(rows, key=lambda x: x["family"]+x["X"]):
        s_d = r["E_decomp_siesta_eV"]
        m_p = r["mannodi_pbe_decomp"]
        m_h = r["mannodi_hse_decomp"]
        delta = s_d - m_p
        print(f"  {r['name']:<20s}  {s_d:>+13.4f}  {m_p:>+11.4f}  {m_h:>+11.4f}  {delta:>+13.4f}")

    s_decomp = np.array([r["E_decomp_siesta_eV"] for r in rows])
    m_pbe = np.array([r["mannodi_pbe_decomp"] for r in rows])
    m_hse = np.array([r["mannodi_hse_decomp"] for r in rows])

    print()
    print("Cross-correlation matrix (n=18 endmembers, F-free):")
    def corr(a, b, label):
        rho = spearmanr(a, b).correlation
        tau = kendalltau(a, b).correlation
        r = pearsonr(a, b).statistic
        return f"  {label:<45s}  ρ={rho:>+.4f}  τ={tau:>+.4f}  r={r:>+.4f}"
    print(corr(s_decomp, m_pbe, "SIESTA-PBE decomp vs Mannodi-VASP-PBE decomp"))
    print(corr(s_decomp, m_hse, "SIESTA-PBE decomp vs Mannodi-VASP-HSE decomp"))
    print(corr(m_pbe,    m_hse, "Mannodi PBE vs Mannodi HSE (intra-code)"))

    # Linear fit for the cross-functional shift
    print()
    print("Linear fit SIESTA-PBE → Mannodi-HSE06:")
    A = np.vstack([s_decomp, np.ones_like(s_decomp)]).T
    a, b = np.linalg.lstsq(A, m_hse, rcond=None)[0]
    pred = a*s_decomp + b
    resid_rmse = np.sqrt(np.mean((pred - m_hse)**2)) * 1000  # meV/f.u.
    print(f"  Mannodi_HSE ≈ {a:.4f} · SIESTA_decomp + {b:.4f}")
    print(f"  Linear fit RMSE = {resid_rmse:.1f} meV/f.u.")

    # save
    out = {"endmembers": rows,
            "binary_coh_eV": binary_coh,
            "correlations": {
                "siesta_decomp_vs_mannodi_pbe":  {"rho": float(spearmanr(s_decomp,m_pbe).correlation), "tau": float(kendalltau(s_decomp,m_pbe).correlation), "r": float(pearsonr(s_decomp,m_pbe).statistic)},
                "siesta_decomp_vs_mannodi_hse":  {"rho": float(spearmanr(s_decomp,m_hse).correlation), "tau": float(kendalltau(s_decomp,m_hse).correlation), "r": float(pearsonr(s_decomp,m_hse).statistic)},
                "mannodi_pbe_vs_mannodi_hse":    {"rho": float(spearmanr(m_pbe,m_hse).correlation),    "tau": float(kendalltau(m_pbe,m_hse).correlation),    "r": float(pearsonr(m_pbe,m_hse).statistic)},
            },
            "linear_fit_siesta_to_mannodi_hse": {"slope": float(a), "intercept": float(b),
                                                  "rmse_meV": float(resid_rmse)}}
    (SD/"siesta_decomp_validation.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved → data/perovskites/siesta_decomp_validation.json")


if __name__ == "__main__":
    main()
