"""Compare OCE refit on n=93 (initial) vs n=143 (after AL iteration 1).

Both use 1F+2F basis, 5-fold CV, ridge α=10 (shown to be optimal earlier).
Reports:
  - Held-out RMSE / MAE / ρ / R² for each pool
  - Reduction in RMSE per added training point (learning rate)
  - Per-family / per-kind breakdown
  - Specifically: how the strain-kind RMSE drops (we picked 13 strained in AL
    and the n=93 set had zero strained, so this is the cleanest test of AL gain)
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


def parse_family(name: str) -> str:
    m = NAME_RE.match(name)
    return f"{m.group(1)}{m.group(2)}" if m else "?"


def parse_kind(name: str) -> str:
    m = NAME_RE.match(name)
    if not m: return "?"
    k = m.group(4)
    return k[:4] if k.startswith("mix") else ("prim" if k == "prim" else "strain")


def cv_metrics(X: np.ndarray, y: np.ndarray, sizes: np.ndarray,
                fams: np.ndarray, kinds: np.ndarray, alpha: float = 10.0):
    kf = KFold(n_splits=5, shuffle=True, random_state=20260508)
    pred = np.zeros_like(y)
    for itr, ite in kf.split(X):
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X[itr], y[itr])
        pred[ite] = m.predict(X[ite])
    err = pred - y
    err_pa = err / sizes
    out = dict(
        n=int(len(y)), p=int(X.shape[1]),
        rmse_eV=float(np.sqrt(np.mean(err ** 2))),
        rmse_meV_per_atom=float(np.sqrt(np.mean(err_pa ** 2))) * 1000,
        mae_meV_per_atom=float(np.mean(np.abs(err_pa))) * 1000,
        spearman=float(spearmanr(y, pred).correlation),
        kendall=float(kendalltau(y, pred).correlation),
        r2=1.0 - float(np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    )
    fam_rows = {}
    for f in sorted(set(fams)):
        sel = fams == f
        if sel.sum() < 2: continue
        fam_rows[f] = dict(
            n=int(sel.sum()),
            rmse_meV_per_atom=float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000,
            spearman=float(spearmanr(y[sel], pred[sel]).correlation),
        )
    kind_rows = {}
    for k in sorted(set(kinds)):
        sel = kinds == k
        if sel.sum() < 2: continue
        kind_rows[k] = dict(
            n=int(sel.sum()),
            rmse_meV_per_atom=float(np.sqrt(np.mean(err_pa[sel] ** 2))) * 1000,
            spearman=float(spearmanr(y[sel], pred[sel]).correlation),
        )
    return out, fam_rows, kind_rows


def build_pool(filter_pred=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD / "atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    converged = [r for r in siesta if r.get("converged")
                 and (filter_pred is None or filter_pred(r))]
    by_name = {r["name"]: r for r in
               json.loads((SD / "validation_100.json").read_text())}
    al_recs = json.loads((SD / "active_learning_iter1_selection.json").read_text())
    by_name.update({r["name"]: r for r in al_recs})

    npz = np.load(SD / "features.npz", allow_pickle=True)
    X_all = npz["X"]; names = [str(n) for n in npz["names"]]
    sizes_all = npz["sizes"]
    n2i = {n: i for i, n in enumerate(names)}
    feat_index = json.loads((SD / "feature_index.json").read_text())
    feat_keys = [tuple(e["key"]) for e in feat_index]
    cols_12 = sorted([j for j, k in enumerate(feat_keys)
                       if k != ("MAD",) and k[0] in ("1F", "2F")])

    idx, y, sizes, fams, kinds, names_aligned = [], [], [], [], [], []
    for r in converged:
        nm = r["name"]
        rec = by_name.get(nm)
        if rec is None: continue
        ref_sum = sum(atom_refs.get(s, 0.0) for s in rec["symbols"])
        E_coh = r["E_eV"] - ref_sum
        i = n2i.get(nm)
        if i is None: continue
        idx.append(i); y.append(E_coh)
        sizes.append(sizes_all[i])
        fams.append(parse_family(nm))
        kinds.append(parse_kind(nm))
        names_aligned.append(nm)
    idx = np.array(idx); y = np.array(y); sizes = np.array(sizes)
    fams = np.array(fams); kinds = np.array(kinds)
    X = X_all[idx][:, cols_12]
    return X, y, sizes, fams, kinds, names_aligned


def main():
    # Pool A: only the original 93 (al_iteration is None or 0)
    XA, yA, szA, fA, kA, nA = build_pool(
        filter_pred=lambda r: not r.get("al_iteration")
    )
    # Pool B: all 143 (initial + AL iter 1)
    XB, yB, szB, fB, kB, nB = build_pool()

    print(f"Pool A (n=93 baseline):    {XA.shape}")
    print(f"Pool B (n=143, AL iter 1): {XB.shape}")
    print()

    print("=" * 78)
    print("Headline: 1F+2F basis, ridge α=10, 5-fold CV")
    print("=" * 78)
    mA, fA_per, kA_per = cv_metrics(XA, yA, szA, fA, kA, alpha=10.0)
    mB, fB_per, kB_per = cv_metrics(XB, yB, szB, fB, kB, alpha=10.0)
    print(f"             {'n':>3s}  {'p':>4s}  "
          f"{'RMSE(meV/atom)':>15s}  {'ρ':>6s}  {'τ':>6s}  {'R²':>6s}")
    print(f"  baseline   {mA['n']:>3d}  {mA['p']:>4d}  "
          f"{mA['rmse_meV_per_atom']:>15.1f}  "
          f"{mA['spearman']:>6.3f}  {mA['kendall']:>6.3f}  {mA['r2']:>6.3f}")
    print(f"  AL iter 1  {mB['n']:>3d}  {mB['p']:>4d}  "
          f"{mB['rmse_meV_per_atom']:>15.1f}  "
          f"{mB['spearman']:>6.3f}  {mB['kendall']:>6.3f}  {mB['r2']:>6.3f}")
    delta_rmse = mA['rmse_meV_per_atom'] - mB['rmse_meV_per_atom']
    print(f"\n  ΔRMSE = {delta_rmse:+.1f} meV/atom  ({delta_rmse/mA['rmse_meV_per_atom']*100:+.1f}%)")
    n_added = mB['n'] - mA['n']
    if delta_rmse > 0:
        print(f"  Per added sample: {delta_rmse/n_added:.1f} meV/atom reduction")

    print("\nPer-family:")
    print(f"  {'fam':>6s}  {'n_A':>4s}  {'n_B':>4s}  "
          f"{'RMSE_A':>7s}  {'RMSE_B':>7s}  {'ρ_A':>6s}  {'ρ_B':>6s}")
    for f in sorted(set(fA_per) | set(fB_per)):
        a = fA_per.get(f, {})
        b = fB_per.get(f, {})
        print(f"  {f:>6s}  {a.get('n', '-'):>4}  {b.get('n', '-'):>4}  "
              f"{a.get('rmse_meV_per_atom', float('nan')):>7.1f}  "
              f"{b.get('rmse_meV_per_atom', float('nan')):>7.1f}  "
              f"{a.get('spearman', float('nan')):>6.3f}  "
              f"{b.get('spearman', float('nan')):>6.3f}")

    print("\nPer-kind (where AL strain-pickup matters):")
    print(f"  {'kind':>6s}  {'n_A':>4s}  {'n_B':>4s}  "
          f"{'RMSE_A':>7s}  {'RMSE_B':>7s}  {'ρ_A':>6s}  {'ρ_B':>6s}")
    for k in sorted(set(kA_per) | set(kB_per)):
        a = kA_per.get(k, {})
        b = kB_per.get(k, {})
        print(f"  {k:>6s}  {a.get('n', '-'):>4}  {b.get('n', '-'):>4}  "
              f"{a.get('rmse_meV_per_atom', float('nan')):>7.1f}  "
              f"{b.get('rmse_meV_per_atom', float('nan')):>7.1f}  "
              f"{a.get('spearman', float('nan')):>6.3f}  "
              f"{b.get('spearman', float('nan')):>6.3f}")

    out = {
        "baseline_n93": dict(metrics=mA, per_family=fA_per, per_kind=kA_per),
        "after_al_iter1_n143": dict(metrics=mB, per_family=fB_per, per_kind=kB_per),
        "delta_rmse_meV_per_atom": float(delta_rmse),
        "delta_relative_pct": float(delta_rmse/mA['rmse_meV_per_atom']*100),
    }
    out_path = SD / "active_learning_iter1_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
