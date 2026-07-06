"""Phase 11 — theoretical decomposition of ε_V*(bond) / ε_V*(site) ≈ 1.25.

We observed empirically that the ratio of asymptotic energy density between
bond and site percolation is approximately 1.25 ± 0.03 for both C and Si
on all 3 lattices.  Here we decompose this into two physical factors:

    ε_V*  =  (−E/N)  ×  (N/V)
    -------    -------    -------
    energy     cohesive     atomic
    density   energy/atom    density

So the ratio decomposes:

    ε_V*(bond)     (E/N)_bond    (N/V)_bond
    ----------  =  -----------  × ----------
    ε_V*(site)     (E/N)_site    (N/V)_site

We compute each factor from the cached data and ask:

  (a) Which factor dominates the 1.25 enhancement of bond over site?
  (b) Are the per-factor ratios universal across lattices?
  (c) Do the percolation-theory predictions for "active site density"
      ρ_active = p_c(site) [site] vs 1−(1−p_c(bond))^z [bond] explain
      the (N/V) ratio?
  (d) Does the average bond-density per atom predict the (E/N) ratio?

Theoretical predictions:
  Active site fraction (i.e. fraction of lattice sites that are in any
  occupied component or connected via any occupied bond):
       site:  ρ_a^site = p_c^site
       bond:  ρ_a^bond = 1 − (1 − p_c^bond)^z

  Average bond density per active site (post-percolation, pre-relaxation):
       site:  ⟨z⟩_lat^site ≈ z · p_c^site            (each occupied site
                                                       has z·p_c neighbours
                                                       that are ALSO occupied)
       bond:  ⟨z⟩_lat^bond ≈ z · p_c^bond / ρ_a^bond  (renormalised by
                                                        active fraction)

After AIREBO/Tersoff relaxation atoms move and ⟨z⟩ may change; the
'lattice prediction' is just the starting connectivity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lattices import P_C, P_C_BOND


SITE_C   = ROOT / "data" / "phase5_three_lattices"
SITE_SI  = ROOT / "data" / "phase6_silicon"
BOND     = ROOT / "data" / "phase8_bond"
LATTICES = ("honeycomb", "square", "triangular")
Z_LAT = {"honeycomb": 3, "square": 4, "triangular": 6}

RESULTS_DIR = ROOT / "results" / "phase11_theory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def per_L_means(records, key, L_target=128) -> tuple[float, float]:
    sub = [r[key] for r in records if r["L"] == L_target]
    if not sub:
        # fall back to largest available L
        Ls = sorted({r["L"] for r in records})
        sub = [r[key] for r in records if r["L"] == Ls[-1]]
    return float(np.mean(sub)), float(np.std(sub))


def load_records(directory: Path, lattice: str, material: str) -> list[dict]:
    if material is None:
        # phase 5 / phase 6 each lattice has its own json
        path = directory / f"{lattice}_clusters.json"
    else:
        # phase 8: indexed by material
        path = directory / f"{material}_clusters.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def collect(L_target: int = 128) -> list[dict]:
    """Build a per-(material, lattice, mode) summary at L = L_target."""
    rows = []
    for material in ("C", "Si"):
        # SITE
        site_dir = SITE_C if material == "C" else SITE_SI
        e_key = "E_airebo_relaxed_eV" if material == "C" else "E_tersoff_relaxed_eV"
        for lat in LATTICES:
            recs = load_records(site_dir, lat, material=None)
            if not recs:
                continue
            recs_lat = [r for r in recs if r.get("lattice", lat) == lat] or recs
            n, _ = per_L_means(recs_lat, "n_atoms", L_target)
            E, _ = per_L_means(recs_lat, e_key, L_target)
            V_mc, _ = per_L_means(recs_lat, "V_mc_A3", L_target)
            E_per_N = E / n
            V_per_N = V_mc / n
            rows.append(dict(
                material=material, lattice=lat, mode="site",
                z_lattice=Z_LAT[lat], p=P_C[lat],
                n_realisations=int(sum(1 for r in recs_lat if r["L"] == L_target)),
                N=n, E=E, V_mc=V_mc,
                E_per_N=E_per_N, V_per_N=V_per_N,
                eps_V_star=-E / V_mc,
            ))

        # BOND
        recs_bond = load_records(BOND, None, material=material)
        for lat in LATTICES:
            recs_lat = [r for r in recs_bond if r["lattice"] == lat]
            if not recs_lat:
                continue
            n, _ = per_L_means(recs_lat, "n_atoms", L_target)
            E, _ = per_L_means(recs_lat, e_key, L_target)
            V_mc, _ = per_L_means(recs_lat, "V_mc_A3", L_target)
            E_per_N = E / n
            V_per_N = V_mc / n
            rows.append(dict(
                material=material, lattice=lat, mode="bond",
                z_lattice=Z_LAT[lat], p=P_C_BOND[lat],
                n_realisations=int(sum(1 for r in recs_lat if r["L"] == L_target)),
                N=n, E=E, V_mc=V_mc,
                E_per_N=E_per_N, V_per_N=V_per_N,
                eps_V_star=-E / V_mc,
            ))
    return rows


def main():
    print("=== Phase 11 — theoretical decomposition of bond/site ratio ===\n")
    rows = collect(L_target=128)
    if not rows:
        print("No data found.")
        return

    # Build a lookup
    table = {(r["material"], r["lattice"], r["mode"]): r for r in rows}

    # ---------- Decomposition table ----------
    print(f"{'='*116}")
    print(f"            DECOMPOSITION ε_V*(bond) / ε_V*(site) at L=128")
    print(f"{'='*116}")
    print(f"  {'mat':3s}  {'lattice':10s}  z_lat  "
          f"{'(−E/N)_site':12s} {'(−E/N)_bond':12s} {'ratio E':8s}   "
          f"{'(V/N)_site':12s} {'(V/N)_bond':12s} {'ratio V⁻¹':9s}   "
          f"{'ε_V*(s)':9s}  {'ε_V*(b)':9s} {'ratio_ε':8s}  {'product':8s}")

    decomp_rows = []
    for material in ("C", "Si"):
        for lat in LATTICES:
            site = table.get((material, lat, "site"))
            bond = table.get((material, lat, "bond"))
            if site is None or bond is None:
                continue
            E_s = -site["E_per_N"]
            E_b = -bond["E_per_N"]
            V_s = site["V_per_N"]
            V_b = bond["V_per_N"]
            ratio_E = E_b / E_s
            ratio_Vinv = V_s / V_b
            ratio_eps = bond["eps_V_star"] / site["eps_V_star"]
            product = ratio_E * ratio_Vinv
            print(f"   {material:2s}  {lat:10s}  {Z_LAT[lat]}    "
                  f"{E_s:8.3f}    {E_b:8.3f}    {ratio_E:.3f}     "
                  f"{V_s:8.3f}    {V_b:8.3f}    {ratio_Vinv:.3f}      "
                  f"{site['eps_V_star']:.4f}   {bond['eps_V_star']:.4f}  "
                  f"{ratio_eps:.3f}    {product:.3f}")
            decomp_rows.append(dict(
                material=material, lattice=lat, z_lattice=Z_LAT[lat],
                E_per_N_site=E_s, E_per_N_bond=E_b, ratio_E=ratio_E,
                V_per_N_site=V_s, V_per_N_bond=V_b, ratio_Vinv=ratio_Vinv,
                eps_V_site=site["eps_V_star"], eps_V_bond=bond["eps_V_star"],
                ratio_eps=ratio_eps, product=product,
                p_site=P_C[lat], p_bond=P_C_BOND[lat],
            ))

    # Aggregate
    arr = np.array([(r["ratio_E"], r["ratio_Vinv"], r["ratio_eps"]) for r in decomp_rows])
    print(f"\n  ----- aggregate over 6 (material, lattice) pairs -----")
    print(f"  ratio_E       = {arr[:, 0].mean():.4f} ± {arr[:, 0].std():.4f}")
    print(f"  ratio_V⁻¹     = {arr[:, 1].mean():.4f} ± {arr[:, 1].std():.4f}")
    print(f"  ratio_ε_V*    = {arr[:, 2].mean():.4f} ± {arr[:, 2].std():.4f}")

    # ---------- Percolation-theory predictions ----------
    print(f"\n{'='*100}")
    print(f"            PERCOLATION-THEORY PREDICTION  vs  EMPIRICAL")
    print(f"{'='*100}")
    print(f"  Active fraction:")
    print(f"     site mode:   ρ_a^site = p_c^site")
    print(f"     bond mode:   ρ_a^bond = 1 − (1 − p_c^bond)^z")
    print(f"  Bond-per-active-site (lattice connectivity):")
    print(f"     site mode:   ⟨k⟩_site = z · p_c^site")
    print(f"     bond mode:   ⟨k⟩_bond = z · p_c^bond / ρ_a^bond")
    print(f"\n  {'lat':10s}  {'z':3s}  "
          f"{'ρ_site':8s} {'ρ_bond':8s} {'ρ_b/ρ_s':9s}    "
          f"{'⟨k⟩_site':9s} {'⟨k⟩_bond':9s} {'k_b/k_s':9s}    "
          f"{'predicted ratio_eps':16s}  {'observed':10s}")

    theory_rows = []
    for lat in LATTICES:
        ps = P_C[lat]
        pb = P_C_BOND[lat]
        z = Z_LAT[lat]
        rho_s = ps
        rho_b = 1.0 - (1.0 - pb) ** z
        k_s = z * ps
        k_b = z * pb / rho_b
        # geometric prediction: density ratio (N/V)_b/(N/V)_s = ρ_b/ρ_s
        # energetic prediction: (E/N)_b/(E/N)_s ≈ k_b/k_s
        # combined: ρ_b/ρ_s × k_b/k_s
        pred_geom = rho_b / rho_s
        pred_eng = k_b / k_s
        pred_eps = pred_geom * pred_eng

        # average observed ratio across materials
        obs_mat = []
        for material in ("C", "Si"):
            r = next((d for d in decomp_rows
                      if d["material"] == material and d["lattice"] == lat), None)
            if r is not None:
                obs_mat.append(r["ratio_eps"])
        obs_mean = float(np.mean(obs_mat)) if obs_mat else float("nan")
        obs_std = float(np.std(obs_mat)) if obs_mat else 0.0

        print(f"  {lat:10s}  {z}    "
              f"{rho_s:.4f}   {rho_b:.4f}   {pred_geom:.3f}        "
              f"{k_s:.3f}     {k_b:.3f}     {pred_eng:.3f}        "
              f"{pred_eps:.3f}              "
              f"{obs_mean:.3f} ± {obs_std:.3f}")
        theory_rows.append(dict(
            lattice=lat, z=z, p_site=ps, p_bond=pb,
            rho_active_site=rho_s, rho_active_bond=rho_b,
            ratio_rho=pred_geom,
            k_avg_site=k_s, k_avg_bond=k_b, ratio_k=pred_eng,
            predicted_eps_ratio=pred_eps,
            observed_eps_ratio_mean=obs_mean,
            observed_eps_ratio_std=obs_std,
        ))

    # ---------- Hypothesis check: does (V/N)_s/(V/N)_b ≈ ρ_b/ρ_s? ----------
    print(f"\n{'='*92}")
    print(f"            FINER CHECK — does (V/N) ratio TRACK active-site density ratio?")
    print(f"{'='*92}")
    print(f"  {'mat':3s}  {'lat':10s}  {'(V/N)_s/(V/N)_b':16s} {'ρ_b/ρ_s':10s}  "
          f"{'(E/N)_b/(E/N)_s':16s} {'⟨k⟩_b/⟨k⟩_s':14s}")
    for r in decomp_rows:
        lat = r["lattice"]
        t = next(t for t in theory_rows if t["lattice"] == lat)
        print(f"   {r['material']:2s}  {lat:10s}    "
              f"{r['ratio_Vinv']:.3f}             {t['ratio_rho']:.3f}     "
              f"     {r['ratio_E']:.3f}            {t['ratio_k']:.3f}")

    # ---------- Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # plot observed ratios vs theory
    obs_eps = np.array([r["ratio_eps"] for r in decomp_rows])
    obs_E = np.array([r["ratio_E"] for r in decomp_rows])
    obs_Vinv = np.array([r["ratio_Vinv"] for r in decomp_rows])
    pred_eps_per_mat = []
    pred_rho = []
    pred_k = []
    for r in decomp_rows:
        t = next(t for t in theory_rows if t["lattice"] == r["lattice"])
        pred_eps_per_mat.append(t["predicted_eps_ratio"])
        pred_rho.append(t["ratio_rho"])
        pred_k.append(t["ratio_k"])
    pred_eps_per_mat = np.array(pred_eps_per_mat)
    pred_rho = np.array(pred_rho)
    pred_k = np.array(pred_k)
    colors = {"C": "C0", "Si": "C1"}
    markers = {"honeycomb": "v", "square": "s", "triangular": "^"}
    for i, r in enumerate(decomp_rows):
        c = colors[r["material"]]
        m = markers[r["lattice"]]
        axes[0].scatter(pred_eps_per_mat[i], obs_eps[i], color=c, marker=m,
                         s=100, edgecolor="k", linewidth=0.4)
        axes[1].scatter(pred_rho[i], obs_Vinv[i], color=c, marker=m,
                         s=80, edgecolor="k", linewidth=0.3,
                         label="(V/N)_s/(V/N)_b" if i == 0 else None)
        axes[1].scatter(pred_k[i], obs_E[i], color=c, marker=m,
                         s=80, edgecolor="r", linewidth=0.4,
                         label="(E/N)_b/(E/N)_s" if i == 0 else None)
    lo = min(pred_eps_per_mat.min(), obs_eps.min()) * 0.95
    hi = max(pred_eps_per_mat.max(), obs_eps.max()) * 1.05
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.6)
    axes[0].set_xlabel(r"theoretical prediction  $\rho_b/\rho_s \times \langle k\rangle_b/\langle k\rangle_s$")
    axes[0].set_ylabel(r"observed  $\varepsilon_V^*(b)/\varepsilon_V^*(s)$")
    axes[0].set_title(r"$\varepsilon_V^*$ ratio: theory vs observation")

    lo2 = 0.9; hi2 = 2.0
    axes[1].plot([lo2, hi2], [lo2, hi2], "k--", lw=0.6)
    axes[1].set_xlabel(r"theory ratio (active-site density or coordination)")
    axes[1].set_ylabel(r"observed ratio (V/N or E/N)")
    axes[1].legend(fontsize=8)
    axes[1].set_title(r"Decomposition: each factor separately")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_theory_vs_obs.png", dpi=180)
    plt.close(fig)

    summary = dict(decomp_rows=decomp_rows, theory_rows=theory_rows,
                   aggregate=dict(
                       ratio_E_mean=float(arr[:, 0].mean()),
                       ratio_Vinv_mean=float(arr[:, 1].mean()),
                       ratio_eps_mean=float(arr[:, 2].mean()),
                   ))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_theory_vs_obs.png'}")


if __name__ == "__main__":
    main()
