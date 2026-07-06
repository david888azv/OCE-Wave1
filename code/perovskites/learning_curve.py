"""3-state learning-curve comparison: n=93 → n=143 → n=193.

For each pool we report 5-fold CV metrics on:
  (i)  the full pool (with strain if present)
  (ii) the non-strain subset (the fair, basis-suitable comparison)

This is the canonical AL learning curve for the perovskite OCE workflow.
"""
from __future__ import annotations

import json
import re
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


def cv_metrics(idx, y, sz, kinds, fams, alpha=10.0):
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
    err = pred - y
    err_pa = err / sz
    return dict(
        n=int(len(y)),
        rmse_meV_per_atom=float(np.sqrt(np.mean(err_pa ** 2))) * 1000,
        mae_meV_per_atom=float(np.mean(np.abs(err_pa))) * 1000,
        spearman=float(spearmanr(y, pred).correlation),
        kendall=float(kendalltau(y, pred).correlation),
        r2=1.0 - float(np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    ), pred


def build_pool(max_iter: int, exclude_strain: bool):
    """max_iter: 0=baseline, 1=after iter1, 2=after iter2."""
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD / "atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    val100 = {r["name"]: r for r in
              json.loads((SD / "validation_100.json").read_text())}
    al1 = json.loads((SD / "active_learning_iter1_selection.json").read_text())
    al2_path = SD / "active_learning_iter2_selection.json"
    al2 = json.loads(al2_path.read_text()) if al2_path.exists() else []
    by_name = dict(val100)
    by_name.update({r["name"]: r for r in al1})
    by_name.update({r["name"]: r for r in al2})

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
        idx.append(i)
        y.append(r["E_eV"] - ref_sum)
        sz.append(sizes[i])
        kinds.append(ki)
        fams.append(parse_family(nm))
    return (np.array(idx), np.array(y), np.array(sz),
             np.array(kinds), np.array(fams))


def main():
    print("=" * 78)
    print("Learning curve: n=93 → n=143 → n=193")
    print("=" * 78)

    pools = []
    for max_iter, exclude_strain in [
        (0, False), (1, False), (2, False),
        (0, True),  (1, True),  (2, True),
    ]:
        idx, y, sz, kinds, fams = build_pool(max_iter, exclude_strain)
        if len(idx) == 0: continue
        m, _ = cv_metrics(idx, y, sz, kinds, fams)
        label = (f"iter={max_iter}, n={m['n']}, "
                 f"strain={'no' if exclude_strain else 'yes'}")
        pools.append((label, m, max_iter, exclude_strain))

    print(f"\n  {'pool':<40s}  {'n':>4s}  {'RMSE(meV/atom)':>15s}  "
          f"{'MAE':>7s}  {'ρ':>6s}  {'τ':>6s}  {'R²':>6s}")
    for label, m, _, _ in pools:
        print(f"  {label:<40s}  {m['n']:>4d}  "
              f"{m['rmse_meV_per_atom']:>15.1f}  "
              f"{m['mae_meV_per_atom']:>7.1f}  "
              f"{m['spearman']:>6.3f}  {m['kendall']:>6.3f}  {m['r2']:>6.3f}")

    # Per-kind learning curve (only for non-strain pools)
    print("\nPer-kind on non-strain pools:")
    print(f"  {'pool':<28s}  {'kind':>6s}  {'n':>3s}  "
          f"{'RMSE':>7s}  {'ρ':>6s}")
    for max_iter in [0, 1, 2]:
        idx, y, sz, kinds, fams = build_pool(max_iter, exclude_strain=True)
        if len(idx) == 0: continue
        m_full, pred = cv_metrics(idx, y, sz, kinds, fams)
        err_pa = (pred - y) / sz
        for k_ in sorted(set(kinds.tolist())):
            sel = kinds == k_
            if sel.sum() < 2: continue
            rmse_k = float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000
            rho_k = float(spearmanr(y[sel], pred[sel]).correlation)
            print(f"  iter={max_iter}, n={m_full['n']:>3d} non-strain    "
                  f"{k_:>6s}  {int(sel.sum()):>3d}  "
                  f"{rmse_k:>7.1f}  {rho_k:>6.3f}")

    # Per-family on non-strain
    print("\nPer-family on non-strain pools:")
    print(f"  {'pool':<28s}  {'fam':>6s}  {'n':>3s}  "
          f"{'RMSE':>7s}  {'ρ':>6s}")
    for max_iter in [0, 1, 2]:
        idx, y, sz, kinds, fams = build_pool(max_iter, exclude_strain=True)
        if len(idx) == 0: continue
        m_full, pred = cv_metrics(idx, y, sz, kinds, fams)
        err_pa = (pred - y) / sz
        for f_ in sorted(set(fams.tolist())):
            sel = fams == f_
            if sel.sum() < 2: continue
            rmse_f = float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000
            rho_f = float(spearmanr(y[sel], pred[sel]).correlation)
            print(f"  iter={max_iter}, n={m_full['n']:>3d} non-strain    "
                  f"{f_:>6s}  {int(sel.sum()):>3d}  "
                  f"{rmse_f:>7.1f}  {rho_f:>6.3f}")

    out_path = SD / "learning_curve_summary.json"
    out_path.write_text(json.dumps({"learning_curve": [
        {"label": l, "max_iter": mi, "exclude_strain": es, "metrics": m}
        for l, m, mi, es in pools
    ]}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
