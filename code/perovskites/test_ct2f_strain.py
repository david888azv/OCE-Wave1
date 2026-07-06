"""Compare 1F+2F (baseline) vs +CT2F on perovskite SIESTA pool (n=193).

Specifically targets the strain regime where 1F+2F got ρ=0.29 due to
topology-only blindness.  CT2F is geometric-distance-binned and should
break this degeneracy.
"""
from __future__ import annotations

import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"
NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+?)(\d+)?(_.*)?$")


def parse_kind(name):
    m = NAME_RE.match(name); k = m.group(4) if m else ""
    return k[:4] if k.startswith("mix") else ("prim" if k=="prim" else "strain")

def parse_family(name):
    m = NAME_RE.match(name); return f"{m.group(1)}{m.group(2)}" if m else "?"


def load_pool(npz_name, idx_name):
    npz = np.load(SD / npz_name, allow_pickle=True)
    X = npz["X"]; names = [str(n) for n in npz["names"]]; sizes = npz["sizes"]
    idx_path = SD / idx_name
    feat_keys = [tuple(e["key"]) for e in json.loads(idx_path.read_text())]
    return X, names, sizes, feat_keys


def get_cols(feat_keys, classes):
    return sorted([j for j, k in enumerate(feat_keys)
                    if (k == ("MAD",) and "MAD" in classes)
                    or (k != ("MAD",) and k[0] in classes)])


def cv5_metrics(X, y, sz, alpha=10.0):
    kf = KFold(5, shuffle=True, random_state=20260508)
    pred = np.zeros_like(y)
    for itr, ite in kf.split(X):
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X[itr], y[itr])
        pred[ite] = m.predict(X[ite])
    err_pa = (pred - y) / sz
    return dict(
        n=len(y), p=X.shape[1],
        rmse_meV=float(np.sqrt(np.mean(err_pa**2)))*1000,
        mae_meV=float(np.mean(np.abs(err_pa)))*1000,
        spearman=float(spearmanr(y, pred).correlation),
        kendall=float(kendalltau(y, pred).correlation),
        r2=1.0 - float(np.sum((pred-y)**2)/np.sum((y-y.mean())**2)),
    ), pred


def main():
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD/"atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD/"siesta_szp_results.json").read_text())
    val100 = {r["name"]: r for r in
              json.loads((SD/"validation_100.json").read_text())}
    for f in ("active_learning_iter1_selection.json",
               "active_learning_iter2_selection.json",
               "active_learning_iter3_selection.json"):
        p = SD / f
        if p.exists():
            val100.update({r["name"]: r for r in json.loads(p.read_text())})

    # Build aligned arrays for both pools
    X_old, names_old, sizes_old, keys_old = load_pool("features.npz",
                                                       "feature_index.json")
    X_new, names_new, sizes_new, keys_new = load_pool("features_ct2f.npz",
                                                       "feature_index_ct2f.json")
    n2i_old = {n: i for i, n in enumerate(names_old)}
    n2i_new = {n: i for i, n in enumerate(names_new)}

    rows = []
    for r in siesta:
        if not r.get("converged"): continue
        nm = r["name"]
        rec = val100.get(nm)
        if rec is None: continue
        ref_sum = sum(atom_refs.get(s, 0.0) for s in rec["symbols"])
        E_coh = r["E_eV"] - ref_sum
        i_old = n2i_old.get(nm); i_new = n2i_new.get(nm)
        if i_old is None or i_new is None: continue
        rows.append(dict(name=nm, i_old=i_old, i_new=i_new,
                          y=E_coh, n=r["n_atoms"],
                          kind=parse_kind(nm), family=parse_family(nm)))

    idx_old = np.array([r["i_old"] for r in rows])
    idx_new = np.array([r["i_new"] for r in rows])
    y = np.array([r["y"] for r in rows])
    sz = np.array([r["n"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    fams = np.array([r["family"] for r in rows])

    print(f"Pool: n={len(rows)}, kinds={dict((k, int((kinds==k).sum())) for k in sorted(set(kinds)))}")

    # 4 model variants
    variants = [
        ("1F",                  ["1F"]),
        ("1F+2F",               ["1F","2F"]),
        ("1F+2F+CT2F",          ["1F","2F","CT2F"]),
        ("1F+2F+CT2F+MAD",      ["1F","2F","CT2F","MAD"]),
    ]
    print()
    print("=" * 90)
    print(f"{'Model':<20s}  {'p':>5s}  {'RMSE (meV/atom)':>15s}  {'MAE':>7s}  {'ρ':>6s}  {'τ':>6s}  {'R²':>6s}")
    print("=" * 90)
    runs = {}
    for label, classes in variants:
        if "CT2F" in classes:
            cols = get_cols(keys_new, set(classes))
            X = X_new[idx_new][:, cols]
        else:
            cols = get_cols(keys_old, set(classes))
            X = X_old[idx_old][:, cols]
        m, pred = cv5_metrics(X, y, sz, alpha=10.0)
        runs[label] = dict(metrics=m, pred=pred.tolist())
        print(f"  {label:<20s}  {m['p']:>5d}  "
              f"{m['rmse_meV']:>15.1f}  {m['mae_meV']:>7.1f}  "
              f"{m['spearman']:>6.3f}  {m['kendall']:>6.3f}  {m['r2']:>6.3f}")

    print()
    print("=" * 90)
    print(f"Per-kind breakdown (1F+2F vs 1F+2F+CT2F, full pool n={len(rows)})")
    print("=" * 90)
    pred_old = np.array(runs["1F+2F"]["pred"])
    pred_new = np.array(runs["1F+2F+CT2F"]["pred"])
    err_old = (pred_old - y) / sz
    err_new = (pred_new - y) / sz
    print(f"  {'kind':>6s}  {'n':>3s}  {'RMSE_1F2F':>10s}  {'ρ_1F2F':>8s}  "
          f"{'RMSE_+CT2F':>11s}  {'ρ_+CT2F':>9s}  {'Δ%':>6s}")
    for k_ in sorted(set(kinds)):
        sel = kinds == k_
        if sel.sum() < 2: continue
        rmse_a = float(np.sqrt(np.mean(err_old[sel]**2))) * 1000
        rmse_b = float(np.sqrt(np.mean(err_new[sel]**2))) * 1000
        rho_a = float(spearmanr(y[sel], pred_old[sel]).correlation)
        rho_b = float(spearmanr(y[sel], pred_new[sel]).correlation)
        delta = (rmse_b - rmse_a) / rmse_a * 100
        print(f"  {k_:>6s}  {int(sel.sum()):>3d}  {rmse_a:>10.1f}  {rho_a:>8.3f}  "
              f"{rmse_b:>11.1f}  {rho_b:>9.3f}  {delta:>+5.1f}%")

    print()
    print("Per-family (full ablation +CT2F):")
    for f_ in sorted(set(fams)):
        sel = fams == f_
        rmse_a = float(np.sqrt(np.mean(err_old[sel]**2))) * 1000
        rmse_b = float(np.sqrt(np.mean(err_new[sel]**2))) * 1000
        rho_b = float(spearmanr(y[sel], pred_new[sel]).correlation)
        print(f"  {f_:>6s}  n={int(sel.sum()):>3d}  "
              f"RMSE_1F2F={rmse_a:>6.1f}  RMSE_+CT2F={rmse_b:>6.1f}  ρ={rho_b:>6.3f}")

    out = {"variants": {k: v["metrics"] for k, v in runs.items()}}
    (SD / "ct2f_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {SD/'ct2f_comparison.json'}")


if __name__ == "__main__":
    main()
