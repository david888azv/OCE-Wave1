"""Refit OCE on SIESTA cohesive energies (E_coh = E_total - Σ E_atom_ref).

This removes the per-element pseudopotential offset from SIESTA's absolute
total energy, giving per-atom RMSE in interpretable meV/atom.

Pipeline:
  1. Load atom_refs.json + siesta_szp_results.json
  2. Compute E_coh per perovskite
  3. Load features.npz (X, names) and align with the 93 SIESTA records
  4. Ablation 1F / +2F / +3F / +Madelung in 5-fold CV
  5. Per (A,B) family + per kind
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"

NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+?)(\d+)?(_.*)?$")


def parse_kind(name: str) -> str:
    m = NAME_RE.match(name)
    if not m: return "?"
    k = m.group(4)
    return k[:4] if k.startswith("mix") else ("prim" if k == "prim" else "strain")


def main():
    atom_refs = json.loads((SD / "atom_refs.json").read_text())
    E_atom = {el: r["E_eV"] for el, r in atom_refs.items() if r.get("converged")}
    print(f"Atom refs loaded: {sorted(E_atom)}")
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    converged = [r for r in siesta if r.get("converged")]
    print(f"SIESTA records converged: {len(converged)}")

    # Compute E_coh per record
    rows = []
    for r in converged:
        nm = r["name"]
        # Need atom symbols — pull from validation_100.json
        # Or re-derive from features.npz alignment later
        rows.append({
            "name": nm, "n_atoms": r["n_atoms"],
            "E_total_siesta": r["E_eV"],
            "family": r.get("family", ""),
            "kind": parse_kind(nm),
        })
    # Pull symbols
    by_name = {r["name"]: r for r in
               json.loads((SD / "validation_100.json").read_text())}
    for row in rows:
        rec = by_name.get(row["name"])
        if rec is None:
            row["E_coh"] = None; continue
        ref_sum = sum(E_atom.get(s, 0.0) for s in rec["symbols"])
        row["E_coh"] = row["E_total_siesta"] - ref_sum
        row["symbols"] = rec["symbols"]
    print(f"Cohesive energy computed for {sum(1 for r in rows if r['E_coh'] is not None)}/{len(rows)}")

    # Sanity check — E_coh per atom
    epn = [r["E_coh"] / r["n_atoms"] for r in rows if r["E_coh"] is not None]
    print(f"\nE_coh per atom (eV/atom):")
    print(f"  min  = {min(epn):.3f}")
    print(f"  mean = {np.mean(epn):.3f}")
    print(f"  med  = {np.median(epn):.3f}")
    print(f"  max  = {max(epn):.3f}")
    print(f"  spread = {max(epn) - min(epn):.3f} eV/atom")

    # === Ablation refit ===
    npz = np.load(SD / "features.npz", allow_pickle=True)
    X_all = npz["X"]
    names = [str(n) for n in npz["names"]]
    sizes = npz["sizes"]
    feat_index = json.loads((SD / "feature_index.json").read_text())
    feat_keys = [tuple(e["key"]) for e in feat_index]
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Align rows with feature matrix
    idx, y, sz, fams, kinds, names_aligned = [], [], [], [], [], []
    for r in rows:
        i = name_to_idx.get(r["name"])
        if i is None or r["E_coh"] is None: continue
        idx.append(i); y.append(r["E_coh"])
        sz.append(sizes[i]); fams.append(r["family"])
        kinds.append(r["kind"]); names_aligned.append(r["name"])
    idx = np.array(idx); y = np.array(y); sz = np.array(sz)
    fams = np.array(fams); kinds = np.array(kinds)
    print(f"\nAligned to features: n={len(idx)}, p={X_all.shape[1]}")

    # Feature-class column groups
    cls_cols = defaultdict(list)
    for j, k in enumerate(feat_keys):
        c = "MAD" if k == ("MAD",) else k[0]
        cls_cols[c].append(j)
    print("Feature classes:", {c: len(v) for c, v in cls_cols.items()})

    order = ["1F", "2F", "3F", "MAD"]
    order = [c for c in order if c in cls_cols]
    cum_cols: list[int] = []
    runs: dict[str, dict] = {}
    print("\n" + "=" * 78)
    print("Ablation: OCE refit on E_coh (5-fold CV, ridge α=1.0)")
    print("=" * 78)
    print(f"  {'Subset':<16s}  {'p':>4s}  {'RMSE (meV/atom)':>16s}  "
          f"{'MAE':>10s}  {'ρ':>6s}  {'τ':>6s}  {'R²':>6s}")
    for c in order:
        cum_cols.extend(cls_cols[c]); cum_cols.sort()
        Xc = X_all[idx][:, cum_cols]
        kf = KFold(n_splits=5, shuffle=True, random_state=20260508)
        pred = np.zeros_like(y)
        for itr, ite in kf.split(Xc):
            m = Ridge(alpha=1.0, fit_intercept=True).fit(Xc[itr], y[itr])
            pred[ite] = m.predict(Xc[ite])
        err = pred - y
        err_pa = err / sz
        rmse_pa = float(np.sqrt(np.mean(err_pa ** 2))) * 1000
        mae_pa  = float(np.mean(np.abs(err_pa))) * 1000
        rho = float(spearmanr(y, pred).correlation)
        tau = float(kendalltau(y, pred).correlation)
        r2 = 1.0 - float(np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
        label = "+".join(order[:order.index(c) + 1])
        runs[label] = dict(p=Xc.shape[1], rmse_meV_per_atom=rmse_pa,
                            mae_meV_per_atom=mae_pa, spearman=rho,
                            kendall=tau, r2=r2)
        print(f"  {label:<16s}  {Xc.shape[1]:>4d}  "
              f"{rmse_pa:>16.1f}  {mae_pa:>10.1f}  "
              f"{rho:>6.3f}  {tau:>6.3f}  {r2:>6.3f}")

    # Per-family on full ablation
    full_label = "+".join(order)
    print(f"\nPer (A,B) family — {full_label}:")
    Xf = X_all[idx]
    kf = KFold(n_splits=5, shuffle=True, random_state=20260508)
    pred_full = np.zeros_like(y)
    for itr, ite in kf.split(Xf):
        m = Ridge(alpha=1.0, fit_intercept=True).fit(Xf[itr], y[itr])
        pred_full[ite] = m.predict(Xf[ite])
    err_full = pred_full - y
    err_pa = err_full / sz
    fam_rows = []
    for fam in sorted(set(fams)):
        sel = fams == fam
        if sel.sum() < 2: continue
        rmse_p = float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000
        mae_p  = float(np.mean(np.abs(err_pa[sel]))) * 1000
        rho_p  = float(spearmanr(y[sel], pred_full[sel]).correlation)
        n_f = int(sel.sum())
        fam_rows.append((fam, n_f, rmse_p, mae_p, rho_p))
        print(f"  {fam:>6s}  n={n_f:>3d}  "
              f"RMSE={rmse_p:>6.1f} meV/atom  MAE={mae_p:>5.1f}  "
              f"ρ={rho_p:>6.3f}")

    print(f"\nPer kind — {full_label}:")
    for kind in sorted(set(kinds)):
        sel = kinds == kind
        if sel.sum() < 2: continue
        rmse_p = float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000
        rho_p  = float(spearmanr(y[sel], pred_full[sel]).correlation)
        print(f"  {kind:>6s}  n={int(sel.sum()):>3d}  "
              f"RMSE={rmse_p:>6.1f} meV/atom  ρ={rho_p:>6.3f}")

    # Save
    out = {
        "n_perovskites": int(len(idx)),
        "atom_refs_eV": E_atom,
        "epn_stats_eV_per_atom": {
            "min": float(min(epn)), "mean": float(np.mean(epn)),
            "median": float(np.median(epn)), "max": float(max(epn)),
            "spread": float(max(epn) - min(epn)),
        },
        "ablation": runs,
        "per_family_full": [
            {"family": f, "n": n, "rmse_meV_per_atom": r,
              "mae_meV_per_atom": ma, "spearman": rho}
            for f, n, r, ma, rho in fam_rows
        ],
    }
    out_path = SD / "siesta_cohesive_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
