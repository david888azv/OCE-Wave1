"""Generate learning-curve plot for the AL-OCE-SIESTA pipeline.

Produces data/perovskites/learning_curve.png with two panels:
  (A) Overall RMSE & Spearman ρ vs n_train (non-strain pool, 1F+2F basis)
  (B) Per-family RMSE curves

Also produces data/perovskites/parity_at_n230.png — scatter of
predicted vs SIESTA E_coh at the largest pool size, color-coded by family.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"
NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+?)(\d+)?(_.*)?$")


def parse_family(name: str) -> str:
    m = NAME_RE.match(name)
    return f"{m.group(1)}{m.group(2)}" if m else "?"


def parse_kind(name: str) -> str:
    m = NAME_RE.match(name)
    if not m: return "?"
    k = m.group(4)
    return k[:4] if k.startswith("mix") else ("prim" if k == "prim" else "strain")


def build_pool(max_iter: int, exclude_strain: bool):
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD / "atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    val100 = {r["name"]: r for r in
              json.loads((SD / "validation_100.json").read_text())}
    by_name = dict(val100)
    for fname in ("active_learning_iter1_selection.json",
                   "active_learning_iter2_selection.json",
                   "active_learning_iter3_selection.json"):
        p = SD / fname
        if p.exists():
            by_name.update({r["name"]: r for r in json.loads(p.read_text())})

    npz = np.load(SD / "features.npz", allow_pickle=True)
    names = [str(n) for n in npz["names"]]; sizes = npz["sizes"]
    n2i = {n: i for i, n in enumerate(names)}

    idx, y, sz, kinds, fams = [], [], [], [], []
    for r in siesta:
        if not r.get("converged"): continue
        nm = r["name"]
        it = int(r.get("al_iteration") or 0)
        if it > max_iter: continue
        ki = parse_kind(nm)
        if exclude_strain and ki == "strain": continue
        rec = by_name.get(nm)
        if rec is None: continue
        ref_sum = sum(atom_refs.get(s, 0.0) for s in rec["symbols"])
        i = n2i.get(nm)
        if i is None: continue
        idx.append(i); y.append(r["E_eV"] - ref_sum)
        sz.append(sizes[i]); kinds.append(ki); fams.append(parse_family(nm))
    return (np.array(idx), np.array(y), np.array(sz),
             np.array(kinds), np.array(fams))


def cv_metrics(idx, y, sz, alpha=10.0):
    npz = np.load(SD / "features.npz", allow_pickle=True)
    X_all = npz["X"]
    feat_index = json.loads((SD / "feature_index.json").read_text())
    feat_keys = [tuple(e["key"]) for e in feat_index]
    cols_12 = sorted([j for j, k in enumerate(feat_keys)
                       if k != ("MAD",) and k[0] in ("1F", "2F")])
    X = X_all[idx][:, cols_12]
    kf = KFold(5, shuffle=True, random_state=20260508)
    pred = np.zeros_like(y)
    for itr, ite in kf.split(X):
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X[itr], y[itr])
        pred[ite] = m.predict(X[ite])
    err_pa = (pred - y) / sz
    rmse_pa = float(np.sqrt(np.mean(err_pa ** 2))) * 1000
    rho = float(spearmanr(y, pred).correlation)
    return rmse_pa, rho, pred


def main():
    iters = [0, 1, 2, 3]
    series = []
    fam_series = defaultdict(list)
    for it in iters:
        idx, y, sz, kinds, fams = build_pool(it, exclude_strain=True)
        if len(idx) == 0: continue
        rmse, rho, pred = cv_metrics(idx, y, sz)
        n = len(idx)
        series.append((it, n, rmse, rho, idx, y, sz, fams, pred))
        # per-family
        err_pa = (pred - y) / sz
        for fam in sorted(set(fams.tolist())):
            sel = fams == fam
            if sel.sum() < 2: continue
            r_f = float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000
            fam_series[fam].append((n, r_f))

    # --- learning curve plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ns = [s[1] for s in series]
    rmses = [s[2] for s in series]
    rhos = [s[3] for s in series]
    ax = axes[0]
    ax.plot(ns, rmses, "o-", color="#0072B2", lw=2, ms=8, label="RMSE")
    ax.set_xlabel("Training set size  n", fontsize=11)
    ax.set_ylabel("Held-out RMSE (meV / atom)", color="#0072B2", fontsize=11)
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ns, rhos, "s--", color="#D55E00", lw=1.6, ms=7, label="Spearman ρ")
    ax2.set_ylabel("Spearman ρ", color="#D55E00", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#D55E00")
    ax2.set_ylim(0.92, 1.0)
    for n, r, rho in zip(ns, rmses, rhos):
        ax.annotate(f"n={n}\n{r:.0f}", xy=(n, r),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8.5, color="#0072B2")
    ax.set_title("(A) AL-OCE learning curve  (1F+2F, ridge α=10, non-strain)",
                  fontsize=11)

    ax = axes[1]
    fam_colors = {
        "CsGe": "#0072B2", "CsPb": "#D55E00", "CsSn": "#009E73",
        "KGe":  "#CC79A7", "KPb":  "#56B4E9", "KSn":  "#E69F00",
    }
    for fam, pts in sorted(fam_series.items()):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=fam_colors.get(fam, "k"),
                 lw=1.6, ms=6, label=fam)
    ax.set_xlabel("Training set size  n", fontsize=11)
    ax.set_ylabel("Held-out RMSE (meV / atom)", fontsize=11)
    ax.set_title("(B) Per-family learning curves", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_lc = SD / "learning_curve.png"
    fig.savefig(out_lc, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_lc}")

    # --- parity plot at largest n ---
    if series:
        last = series[-1]
        it, n, rmse, rho, idx, y, sz, fams, pred = last
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for fam in sorted(set(fams.tolist())):
            sel = fams == fam
            ax.scatter(y[sel], pred[sel], s=28, alpha=0.85,
                       color=fam_colors.get(fam, "k"), label=fam,
                       edgecolors="white", linewidths=0.4)
        lo = min(y.min(), pred.min()); hi = max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
        ax.set_xlabel("SIESTA cohesive E (eV)", fontsize=11)
        ax.set_ylabel("OCE prediction (eV)", fontsize=11)
        ax.set_title(f"OCE 1F+2F vs SIESTA-PBE  (n={n}, α=10)\n"
                      f"RMSE = {rmse:.0f} meV/atom, ρ = {rho:.3f}",
                      fontsize=11)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", "datalim")
        plt.tight_layout()
        out_par = SD / f"parity_at_n{n}.png"
        fig.savefig(out_par, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_par}")

    print("\nLearning curve points:")
    for it, n, rmse, rho, *_ in series:
        print(f"  iter={it}  n={n:>3d}  RMSE={rmse:>6.1f} meV/atom  ρ={rho:.3f}")


if __name__ == "__main__":
    main()
