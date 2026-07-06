"""Plot the SIESTA-PBE vs Mannodi-VASP-HSE06 decomposition-energy parity
on the 18 F-free endmembers, partitioned by A-cation (Cs vs K).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau

SD = Path(__file__).resolve().parent
FIGDIR = SD.parents[1] / "submissions" / "oce-jctc" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

data = json.loads((SD / "siesta_decomp_validation.json").read_text())
rows = data["endmembers"]

# Build arrays
A_label  = np.array([r["A"] for r in rows])
B_label  = np.array([r["B"] for r in rows])
X_label  = np.array([r["X"] for r in rows])
fam      = np.array([r["family"] for r in rows])
s_decomp = np.array([r["E_decomp_siesta_eV"] for r in rows])
m_pbe    = np.array([r["mannodi_pbe_decomp"] for r in rows])
m_hse    = np.array([r["mannodi_hse_decomp"] for r in rows])
names    = [r["name"].replace("_prim", "") for r in rows]

# Marker style by halide, colour by A-cation
shape = {"I": "o", "Br": "s", "Cl": "^"}
color = {"Cs": "#3b82f6", "K": "#e11d48"}

fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

# Left: SIESTA decomp vs Mannodi HSE decomp
ax = axes[0]
for i in range(len(rows)):
    ax.scatter(s_decomp[i], m_hse[i],
                marker=shape[X_label[i]],
                color=color[A_label[i]],
                s=90, edgecolor="black", linewidth=0.5,
                alpha=0.9, zorder=5)

# Per-cation regression lines
for A, c in color.items():
    mask = A_label == A
    s = s_decomp[mask]
    h = m_hse[mask]
    rho = spearmanr(s, h).correlation
    tau = kendalltau(s, h).correlation
    # linear fit
    p = np.polyfit(s, h, 1)
    xfit = np.linspace(s.min() - 0.5, s.max() + 0.5, 50)
    ax.plot(xfit, np.polyval(p, xfit), "--",
            color=c, alpha=0.7, lw=1.5,
            label=f"{A}-perovskite (n=9)\n  ρ={rho:.2f}, τ={tau:.2f}")

# Pooled stats
rho_all = spearmanr(s_decomp, m_hse).correlation
tau_all = kendalltau(s_decomp, m_hse).correlation
ax.set_xlabel(r"SIESTA-PBE $E_\mathrm{decomp}$ (eV / f.u.)")
ax.set_ylabel(r"Mannodi HSE06 $E_\mathrm{decomp}$ (eV / f.u.)")
ax.set_title(
    f"Cross-functional decomposition energy\n"
    f"pooled: ρ={rho_all:.2f}, τ={tau_all:.2f} (n=18 F-free endmembers)"
)
ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
ax.axhline(0, color="black", lw=0.4, ls=":")
ax.axvline(0, color="black", lw=0.4, ls=":")
ax.grid(alpha=0.3)

# Right: per-cation z-scored parity (within-A normalization shows ranking quality)
ax = axes[1]
for A, c in color.items():
    mask = A_label == A
    s = s_decomp[mask]
    h = m_hse[mask]
    sz = (s - s.mean()) / s.std()
    hz = (h - h.mean()) / h.std()
    for i, idx in enumerate(np.where(mask)[0]):
        ax.scatter(sz[i], hz[i],
                    marker=shape[X_label[idx]],
                    color=c, s=90, edgecolor="black", linewidth=0.5,
                    alpha=0.9, zorder=5)
        # annotate name
        ax.annotate(names[idx], (sz[i], hz[i]),
                     fontsize=7, alpha=0.6,
                     xytext=(3, 3), textcoords="offset points")
ax.plot([-2, 2], [-2, 2], "k--", alpha=0.4)
ax.set_xlabel("SIESTA-PBE (per-cation z-score)")
ax.set_ylabel("Mannodi HSE06 (per-cation z-score)")
ax.set_title("Per-cation normalised: Cs ρ=0.25, K ρ=0.87")
ax.grid(alpha=0.3)
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)

# Halide legend
from matplotlib.lines import Line2D
halide_handles = [Line2D([], [], marker=shape[X], color="gray",
                          markeredgecolor="black", linestyle="",
                          markersize=8, label=f"X = {X}") for X in ["I","Br","Cl"]]
ax.legend(handles=halide_handles, loc="lower right", fontsize=9, framealpha=0.85)

fig.suptitle("SIESTA-PBE → Mannodi-HSE06 cross-functional validation on 18 F-free ABX$_3$ endmembers",
              fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_pdf = FIGDIR / "fig5_hse_endmembers.pdf"
out_png = FIGDIR / "fig5_hse_endmembers.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")
