"""Cross-element variance analysis for C, Si, Ge phase17 SZP campaigns.

Tests the "carbon's privileged status" hypothesis: among group-14 elements,
does the AIREBO/Tersoff vs DFT cross-lattice spread depend systematically
on bonding flexibility (sp²/sp³ in C vs sp³ only in Si/Ge)?

Inputs combined:
  C  : phase 12 + 15 + 16 + 17 (n=203 total, 29/lat)
  Si : phase 17 (n=168, 24/lat)
  Ge : phase 17 (n=133, 4-24/lat — diamond_3d undersampled)

Outputs:
  - per-element ANOVA F, p
  - per-element cross-lattice spread + 95% CI
  - per-lattice z-coordinate vs ratio scatter
  - JSON summary
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def load_records(material: str) -> list[dict]:
    if material == "C":
        paths = [
            ROOT/"data"/"phase12_siesta_validation"/"siesta_validation.json",
            ROOT/"data"/"phase15_dft_c_3d"/"dft_3d.json",
            ROOT/"data"/"phase16_dft_strengthen"/"more_reps_szp.json",
            ROOT/"data"/"phase17_dft_more_reps"/"more_reps_szp.json",
        ]
    else:
        paths = [ROOT/"data"/f"phase17_{material}_dft_more_reps"/"more_reps_szp.json"]
    out = []
    for p in paths:
        if not p.exists(): continue
        for r in json.loads(p.read_text()):
            mat = r.get("material", "C")
            basis = r.get("basis", "SZP")
            if mat == material and basis == "SZP":
                out.append(r)
    return out


def per_lattice_stats(records, label):
    by_lat = defaultdict(list)
    for r in records:
        by_lat[r["lattice"]].append(r["eps_V_siesta"]/r["eps_V_classical"])
    print(f"\n{label} ({len(records)} records, {len(by_lat)} lattices):")
    print(f"  {'lat':>12s}  {'n':>3s}  {'⟨ratio⟩':>9s}  {'std':>7s}  {'sem':>7s}  {'95%CI':>11s}")
    means = []; samples = []
    for lat in sorted(by_lat):
        a = np.array(by_lat[lat]); means.append(a.mean()); samples.append(a)
        sem = a.std(ddof=1)/np.sqrt(len(a))
        ci = 1.96 * sem
        print(f"  {lat:>12s}  {len(a):>3d}  {a.mean():>9.4f}  {a.std(ddof=1):>7.4f}  "
              f"{sem:>7.4f}  ±{ci:>6.4f}")
    means = np.array(means)
    spread = means.std(ddof=1)/means.mean()*100
    F, p = stats.f_oneway(*samples)
    sigma_within = np.sqrt(np.mean([np.var(s, ddof=1) for s in samples]))
    print(f"  Cross-lattice spread = {spread:.2f}%   "
          f"(σ_within = {sigma_within:.4f}, σ_between = {means.std(ddof=1):.4f})")
    print(f"  ANOVA F = {F:.2f}, p = {p:.2e}")
    return dict(
        material=label, n_records=len(records),
        per_lattice={lat: dict(n=len(by_lat[lat]),
                                mean=float(np.mean(by_lat[lat])),
                                std=float(np.std(by_lat[lat], ddof=1)) if len(by_lat[lat])>1 else 0.0,
                                sem=float(np.std(by_lat[lat], ddof=1)/np.sqrt(len(by_lat[lat]))) if len(by_lat[lat])>1 else float('nan'))
                     for lat in sorted(by_lat)},
        cross_lattice_spread_pct=float(spread),
        sigma_within=float(sigma_within),
        sigma_between=float(means.std(ddof=1)),
        anova_F=float(F), anova_p=float(p),
        mean_of_means=float(means.mean()),
    )


def correlation_with_z(records, label):
    z_lat = {"honeycomb":3, "square":4, "triangular":6,
             "diamond_3d":4, "cubic_3d":6, "bcc_3d":8, "fcc_3d":12}
    by_lat = defaultdict(list)
    for r in records:
        by_lat[r["lattice"]].append(r["eps_V_siesta"]/r["eps_V_classical"])
    zs = np.array([z_lat[lat] for lat in sorted(by_lat)])
    means = np.array([np.mean(by_lat[lat]) for lat in sorted(by_lat)])
    r_pear, p_pear = stats.pearsonr(zs, means)
    print(f"  ratio vs z (Pearson): r={r_pear:+.4f}, p={p_pear:.4f}")
    # Linear fit
    slope, intercept, _, _, stderr = stats.linregress(zs, means)
    print(f"  fit: ratio = {intercept:.4f} + {slope:+.5f}·z   (slope std={stderr:.5f})")
    return dict(pearson_r=float(r_pear), pearson_p=float(p_pear),
                slope=float(slope), intercept=float(intercept),
                slope_stderr=float(stderr))


def main():
    print("=" * 78)
    print("Cross-element variance analysis — C, Si, Ge phase17")
    print("=" * 78)
    summaries = {}
    z_corrs = {}
    for mat in ["C", "Si", "Ge"]:
        recs = load_records(mat)
        if not recs:
            print(f"\n{mat}: no records")
            continue
        s = per_lattice_stats(recs, mat)
        summaries[mat] = s
        z_corrs[mat] = correlation_with_z(recs, mat)

    print("\n" + "=" * 78)
    print("HEADLINE — cross-element comparison")
    print("=" * 78)
    print(f"  {'Element':>8s}  {'n':>4s}  {'spread%':>8s}  "
          f"{'mean':>8s}  {'σ_within':>9s}  {'σ_between':>10s}  "
          f"{'F':>7s}  {'p':>9s}  {'corr(z)':>9s}")
    for mat in ["C", "Si", "Ge"]:
        if mat not in summaries: continue
        s = summaries[mat]; z = z_corrs[mat]
        print(f"  {mat:>8s}  {s['n_records']:>4d}  "
              f"{s['cross_lattice_spread_pct']:>7.2f}%  "
              f"{s['mean_of_means']:>8.4f}  "
              f"{s['sigma_within']:>9.4f}  {s['sigma_between']:>10.4f}  "
              f"{s['anova_F']:>7.1f}  {s['anova_p']:>9.1e}  "
              f"{z['pearson_r']:>+9.3f}")

    out_path = ROOT / "results" / "cross_element_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(per_element=summaries,
                                          z_correlations=z_corrs),
                                     indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
