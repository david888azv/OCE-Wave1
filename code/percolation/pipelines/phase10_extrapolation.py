"""Phase 10 — finite-size correction-to-scaling extrapolation.

Re-analyse the cached data from phases 5 (C site), 6 (Si site), 8 (bond)
with a 3-parameter fit:

    ⟨E/V⟩(L) = ε_V*(∞) + A · L^{-ω}

Universal prediction: ω ≈ 0.5 for 2D percolation (Levy & Aharony 1986
correction-to-scaling; Aharony & Stauffer "Introduction to Percolation
Theory", chapter on corrections).

We do BOTH:
  (a) fix ω = 0.5 and extract a 2-parameter (ε_V*∞, A) fit
  (b) free 3-parameter fit (ω, ε_V*∞, A); compare the resulting ω with 0.5.

Output: tighter ε_V*(∞) per material × lattice × percolation mode, plus
plots showing data − ε_V*(∞) vs L^{-ω} (should be linear if the Ansatz
is correct).
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
from scipy.optimize import curve_fit


SOURCES = {
    "C-site":   (ROOT / "results" / "phase5_three_lattices" / "summary.json", "eov_mc"),
    "Si-site":  (ROOT / "results" / "phase6_silicon" / "summary.json", "eov_mc"),
    "C-bond":   (ROOT / "results" / "phase8_bond" / "summary.json", None),
    "Si-bond":  (ROOT / "results" / "phase8_bond" / "summary.json", None),
}
LATTICES = ("honeycomb", "square", "triangular")

RESULTS_DIR = ROOT / "results" / "phase10_extrapolation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def fit_omega_fixed(L: np.ndarray, y: np.ndarray, omega: float = 0.5) -> dict:
    """Fit y = a + b·L^(-ω) at fixed ω."""
    x = L ** (-omega)
    A = np.vstack([np.ones_like(x), x]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = sol
    pred = A @ sol
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return dict(eps_V_inf=float(a), A=float(b), omega=float(omega),
                rmse=rmse)


def fit_omega_free(L: np.ndarray, y: np.ndarray) -> dict:
    """3-parameter nonlinear fit y = a + b·L^(-ω)."""
    def model(L, a, b, w):
        return a + b * L ** (-w)
    p0 = (y[-1], y[0] - y[-1], 0.5)
    try:
        popt, pcov = curve_fit(model, L, y, p0=p0, maxfev=20000)
        rmse = float(np.sqrt(np.mean((model(L, *popt) - y) ** 2)))
        return dict(eps_V_inf=float(popt[0]), A=float(popt[1]),
                    omega=float(popt[2]), rmse=rmse)
    except Exception as e:
        return dict(eps_V_inf=float("nan"), A=float("nan"),
                    omega=float("nan"), rmse=float("nan"),
                    error=str(e))


def load_site_summary(path: Path) -> dict:
    """Parse phase5/phase6 summary.json into {lattice: (L_array, -eov_mc_array)}."""
    raw = json.loads(path.read_text())
    out = {}
    for lat in LATTICES:
        if lat not in raw:
            continue
        L = np.array(raw[lat]["L"], dtype=float)
        y = -np.array(raw[lat]["eov_mc"])  # use −E/V (positive)
        out[lat] = (L, y)
    return out


def load_bond_summary(path: Path, material: str) -> dict:
    """Parse phase8 summary.json (nested by material) into {lattice: (L, -eov)}."""
    raw = json.loads(path.read_text())
    out = {}
    if material not in raw:
        return out
    for lat in LATTICES:
        if lat not in raw[material]:
            continue
        L = np.array(raw[material][lat]["L"], dtype=float)
        y = -np.array(raw[material][lat]["eov_mc"])
        out[lat] = (L, y)
    return out


def main():
    print("=== Phase 10 — finite-size correction-to-scaling extrapolation ===")
    print("  Ansatz:  −⟨E/V⟩(L) = ε_V*(∞) + A · L^(-ω)")
    print("  Universal 2D-percolation prediction: ω ≈ 0.5\n")

    all_results = {}
    for label, (path, _) in SOURCES.items():
        if not path.exists():
            print(f"  [skip] {label}: no cache at {path}")
            continue
        if "site" in label:
            data = load_site_summary(path)
        else:
            material = label.split("-")[0]
            data = load_bond_summary(path, material)
        if not data:
            continue
        all_results[label] = {}
        for lat, (L, y) in data.items():
            fit_05 = fit_omega_fixed(L, y, omega=0.5)
            fit_fr = fit_omega_free(L, y)
            all_results[label][lat] = dict(
                L=L.tolist(), y_data=y.tolist(),
                fit_omega_05=fit_05,
                fit_omega_free=fit_fr,
            )

    print(f"{'='*92}")
    print(f"            FIT TABLE — ε_V*(∞) and ω across data sources")
    print(f"{'='*92}")
    print(f"  {'source':10s}  {'lattice':10s}  "
          f"{'ε_V*(∞,ω=½)':12s}  {'ε_V*(∞,ω free)':14s}  {'ω fit':6s}  "
          f"{'data L_max':10s}")
    for label, lats in all_results.items():
        for lat, r in lats.items():
            f5 = r["fit_omega_05"]
            ff = r["fit_omega_free"]
            print(f"   {label:10s}  {lat:10s}  "
                  f"{f5['eps_V_inf']:.4f}        "
                  f"{ff['eps_V_inf']:.4f}          "
                  f"{ff['omega']:+.3f}   "
                  f"{r['y_data'][-1]:.4f} (L=128)")

    # ---------- Cross-source universal-ε_V* table ----------
    print(f"\n{'='*86}")
    print(f"            ASYMPTOTIC ε_V*(∞) — material universality across percolation modes")
    print(f"{'='*86}")
    print(f"  {'source':10s}  honeycomb   square    triangular   mean    spread")
    for label in ("C-site", "C-bond", "Si-site", "Si-bond"):
        if label not in all_results:
            continue
        vals = []
        out = f"  {label:10s} "
        for lat in LATTICES:
            v = all_results[label].get(lat, {}).get("fit_omega_05", {}) \
                                .get("eps_V_inf", float("nan"))
            vals.append(v)
            out += f"  {v:.4f}    "
        a = np.array(vals)
        finite = a[np.isfinite(a)]
        m = float(finite.mean()) if len(finite) else float("nan")
        sp = (finite.max() - finite.min()) / max(abs(m), 1e-9) * 100 if len(finite) else 0
        print(f"{out}  {m:.4f}   {sp:.1f}%")

    # Plots: data vs L^{-ω=0.5} should be linear if the Ansatz is right
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors_lat = {"honeycomb": "C0", "square": "C1", "triangular": "C2"}
    for ax, mat in zip(axes, ("C", "Si")):
        for lat in LATTICES:
            for source in (f"{mat}-site", f"{mat}-bond"):
                if source not in all_results or lat not in all_results[source]:
                    continue
                r = all_results[source][lat]
                L = np.array(r["L"])
                y = np.array(r["y_data"])
                style = "-" if "site" in source else "--"
                ax.plot(L ** (-0.5), y, style + "o", color=colors_lat[lat],
                         label=f"{lat} {source.split('-')[1]}", markersize=4)
        ax.set_xlabel(r"$L^{-1/2}$"); ax.set_ylabel(r"$-\langle E/V\rangle$ (eV/Å³)")
        ax.set_title(f"{mat} — data vs $L^{{-\\omega}}$ (ω=0.5)")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_extrapolation.png", dpi=180)
    plt.close(fig)

    (RESULTS_DIR / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plot     → {RESULTS_DIR / 'fig_extrapolation.png'}")


if __name__ == "__main__":
    main()
