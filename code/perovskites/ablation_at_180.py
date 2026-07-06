"""Ablation 1F/+2F/+3F/+Madelung at the n=180 non-strain pool.

Decisive test of the hypothesis that the optimal OCE basis depth scales with
n/p ratio.  At n=93 we observed 1F+2F won and full ablation (n/p≈0.06) lost.
At n=180, n/p ratio rises to 0.107 for the full basis — does +3F now win?

Sweeps α ∈ {1, 10, 100, 1000} for each ablation level (since the optimal α
shifts with feature count).
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


def parse_family(name: str) -> str:
    m = NAME_RE.match(name)
    return f"{m.group(1)}{m.group(2)}" if m else "?"


def main():
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD / "atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    val100 = {r["name"]: r for r in
              json.loads((SD / "validation_100.json").read_text())}
    al1 = json.loads((SD / "active_learning_iter1_selection.json").read_text())
    al2 = json.loads((SD / "active_learning_iter2_selection.json").read_text())
    by_name = dict(val100)
    by_name.update({r["name"]: r for r in al1})
    by_name.update({r["name"]: r for r in al2})

    npz = np.load(SD / "features.npz", allow_pickle=True)
    X_all = npz["X"]; names = [str(n) for n in npz["names"]]; sizes = npz["sizes"]
    n2i = {n: i for i, n in enumerate(names)}
    feat_index = json.loads((SD / "feature_index.json").read_text())
    feat_keys = [tuple(e["key"]) for e in feat_index]
    cls_cols = defaultdict(list)
    for j, k in enumerate(feat_keys):
        c = "MAD" if k == ("MAD",) else k[0]
        cls_cols[c].append(j)

    # Build n=180 non-strain pool
    idx, y, sz, kinds, fams = [], [], [], [], []
    for r in siesta:
        if not r.get("converged"): continue
        nm = r["name"]
        if parse_kind(nm) == "strain": continue
        rec = by_name.get(nm)
        if rec is None: continue
        ref_sum = sum(atom_refs.get(s, 0.0) for s in rec["symbols"])
        i = n2i.get(nm)
        if i is None: continue
        idx.append(i); y.append(r["E_eV"] - ref_sum)
        sz.append(sizes[i]); kinds.append(parse_kind(nm))
        fams.append(parse_family(nm))
    idx = np.array(idx); y = np.array(y); sz = np.array(sz)
    kinds = np.array(kinds); fams = np.array(fams)
    print(f"Non-strain pool: n={len(idx)}, full-basis p={X_all.shape[1]}, "
          f"n/p ratio = {len(idx)/X_all.shape[1]:.3f}")

    # Ablation order
    order = ["1F", "2F", "3F", "MAD"]
    order = [c for c in order if c in cls_cols]
    cum_cols: list[int] = []
    print()
    print("=" * 80)
    print(f"Ablation at n={len(idx)} (non-strain), α-sweep over [1, 10, 100, 1000]")
    print("=" * 80)
    print(f"  {'subset':<14s}  {'p':>5s}  {'n/p':>6s}  {'best α':>8s}  "
          f"{'RMSE':>8s}  {'MAE':>8s}  {'ρ':>6s}  {'τ':>6s}  {'R²':>6s}")
    runs: dict[str, dict] = {}
    for c in order:
        cum_cols.extend(cls_cols[c]); cum_cols.sort()
        Xc = X_all[idx][:, cum_cols]
        # alpha sweep
        best = None
        all_alphas: list[dict] = []
        for alpha in [1.0, 10.0, 100.0, 1000.0]:
            kf = KFold(5, shuffle=True, random_state=20260508)
            pred = np.zeros_like(y)
            for itr, ite in kf.split(Xc):
                m = Ridge(alpha=alpha, fit_intercept=True).fit(Xc[itr], y[itr])
                pred[ite] = m.predict(Xc[ite])
            err = pred - y
            err_pa = err / sz
            r = dict(
                alpha=alpha, p=Xc.shape[1],
                rmse_meV_per_atom=float(np.sqrt(np.mean(err_pa**2))) * 1000,
                mae_meV_per_atom=float(np.mean(np.abs(err_pa))) * 1000,
                spearman=float(spearmanr(y, pred).correlation),
                kendall=float(kendalltau(y, pred).correlation),
                r2=1.0 - float(np.sum(err**2) / np.sum((y - y.mean())**2)),
            )
            all_alphas.append(r)
            if best is None or r["rmse_meV_per_atom"] < best["rmse_meV_per_atom"]:
                best = r
        label = "+".join(order[:order.index(c) + 1])
        runs[label] = dict(best=best, all_alphas=all_alphas)
        print(f"  {label:<14s}  {Xc.shape[1]:>5d}  "
              f"{len(idx)/Xc.shape[1]:>6.3f}  "
              f"{best['alpha']:>8.0f}  "
              f"{best['rmse_meV_per_atom']:>8.1f}  "
              f"{best['mae_meV_per_atom']:>8.1f}  "
              f"{best['spearman']:>6.3f}  {best['kendall']:>6.3f}  "
              f"{best['r2']:>6.3f}")

    # Compare with n=93 baseline
    print()
    print("Reference: n=93 baseline ablation (from siesta_cohesive_summary.json)")
    try:
        prev = json.loads((SD / "siesta_cohesive_summary.json").read_text())
        for label, m in prev["ablation"].items():
            print(f"  {label:<14s}  p={m.get('p','?'):>5}  "
                  f"RMSE_meV_per_atom={m['rmse_meV_per_atom']:>7.1f}  "
                  f"ρ={m['spearman']:.3f}")
    except Exception as e:
        print(f"  (could not load: {e})")

    # Save
    out = {
        "n": int(len(idx)), "n_features_full": int(X_all.shape[1]),
        "ablation_at_180": runs,
    }
    out_path = SD / "ablation_at_180.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    # Per-kind for the BEST ablation
    best_label = min(runs, key=lambda L: runs[L]["best"]["rmse_meV_per_atom"])
    print(f"\nBest ablation: {best_label}  "
          f"(α={runs[best_label]['best']['alpha']}, "
          f"RMSE={runs[best_label]['best']['rmse_meV_per_atom']:.1f} meV/atom, "
          f"ρ={runs[best_label]['best']['spearman']:.3f})")


if __name__ == "__main__":
    main()
