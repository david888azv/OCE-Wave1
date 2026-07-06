"""Wave-1.5 OCE on perovskite SIESTA pool — full ablation.

Compares baseline (1F+2F), CT2F (Wave-1), and Wave-1.5 (CT2F_A + MADCT2F + CT3F)
variants on the n=243 SIESTA-PBE labelled pool, with per-kind and per-family
breakdown. Strain regime is the key target.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"
NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+?)(\d+)?(_.*)?$")


def parse_kind(name):
    m = NAME_RE.match(name)
    k = m.group(4) if m else ""
    return k[:4] if k.startswith("mix") else ("prim" if k == "prim" else "strain")


def parse_family(name):
    m = NAME_RE.match(name)
    return f"{m.group(1)}{m.group(2)}" if m else "?"


def load_pool(npz_name, idx_name):
    npz = np.load(SD / npz_name, allow_pickle=True)
    X = npz["X"]
    names = [str(n) for n in npz["names"]]
    sizes = npz["sizes"]
    keys = [tuple(e["key"]) for e in
             json.loads((SD / idx_name).read_text())]
    return X, names, sizes, keys


def get_cols(keys, classes):
    return sorted([j for j, k in enumerate(keys)
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
        n=len(y), p=int(X.shape[1]),
        rmse_meV=float(np.sqrt(np.mean(err_pa ** 2))) * 1000,
        mae_meV=float(np.mean(np.abs(err_pa))) * 1000,
        spearman=float(spearmanr(y, pred).correlation),
        kendall=float(kendalltau(y, pred).correlation),
        r2=1.0 - float(np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)),
    ), pred


def alpha_sweep_cv5(X, y, sz):
    best_alpha = 10.0
    best_cv = float("inf")
    for a in [1e-2, 1.0, 10.0, 100.0, 1000.0]:
        m, _ = cv5_metrics(X, y, sz, alpha=a)
        if m["rmse_meV"] < best_cv:
            best_alpha, best_cv = a, m["rmse_meV"]
    return best_alpha


def main():
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD / "atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    val100 = {r["name"]: r for r in
              json.loads((SD / "validation_100.json").read_text())}
    for f in ("active_learning_iter1_selection.json",
               "active_learning_iter2_selection.json",
               "active_learning_iter3_selection.json"):
        p = SD / f
        if p.exists():
            val100.update({r["name"]: r for r in json.loads(p.read_text())})

    X_old, names_old, sizes_old, keys_old = load_pool(
        "features.npz", "feature_index.json")
    X_ct, names_ct, sizes_ct, keys_ct = load_pool(
        "features_ct2f.npz", "feature_index_ct2f.json")
    X_wv, names_wv, sizes_wv, keys_wv = load_pool(
        "features_wave15.npz", "feature_index_wave15.json")

    n2i_old = {n: i for i, n in enumerate(names_old)}
    n2i_ct = {n: i for i, n in enumerate(names_ct)}
    n2i_wv = {n: i for i, n in enumerate(names_wv)}

    rows = []
    for r in siesta:
        if not r.get("converged"):
            continue
        nm = r["name"]
        rec = val100.get(nm)
        if rec is None:
            continue
        ref_sum = sum(atom_refs.get(s, 0.0) for s in rec["symbols"])
        E_coh = r["E_eV"] - ref_sum
        i_old = n2i_old.get(nm)
        i_ct = n2i_ct.get(nm)
        i_wv = n2i_wv.get(nm)
        if i_old is None or i_ct is None or i_wv is None:
            continue
        rows.append(dict(name=nm, i_old=i_old, i_ct=i_ct, i_wv=i_wv,
                          y=E_coh, n=r["n_atoms"],
                          kind=parse_kind(nm), family=parse_family(nm)))

    y = np.array([r["y"] for r in rows])
    sz = np.array([r["n"] for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    fams = np.array([r["family"] for r in rows])
    idx_old = np.array([r["i_old"] for r in rows])
    idx_ct = np.array([r["i_ct"] for r in rows])
    idx_wv = np.array([r["i_wv"] for r in rows])

    print(f"Pool: n={len(rows)}")
    print(f"  kinds = {dict((k, int((kinds==k).sum())) for k in sorted(set(kinds)))}")
    print(f"  families = {dict((k, int((fams==k).sum())) for k in sorted(set(fams)))}\n")

    def get_X(classes_set):
        """Assemble feature matrix combining columns from the three pools."""
        cols_old = get_cols(keys_old, classes_set & {"1F", "2F", "3F", "MAD"})
        cols_ct = get_cols(keys_ct, classes_set & {"CT2F"})
        cols_wv = get_cols(keys_wv, classes_set & {"CT2F_A", "MADCT2F", "CT3F"})
        parts = []
        if cols_old:
            parts.append(X_old[idx_old][:, cols_old])
        if cols_ct:
            parts.append(X_ct[idx_ct][:, cols_ct])
        if cols_wv:
            parts.append(X_wv[idx_wv][:, cols_wv])
        if not parts:
            return np.zeros((len(rows), 0))
        return np.hstack(parts)

    variants = [
        ("1F+2F (baseline)",            {"1F", "2F"}),
        ("1F+2F+CT2F (Wave-1)",         {"1F", "2F", "CT2F"}),
        ("1F+2F+CT2F+MAD",              {"1F", "2F", "CT2F", "MAD"}),
        ("1F+2F+CT2F_A (adaptive)",     {"1F", "2F", "CT2F_A"}),
        ("1F+2F+CT2F+MADCT2F",          {"1F", "2F", "CT2F", "MADCT2F"}),
        ("1F+2F+CT2F+CT3F",             {"1F", "2F", "CT2F", "CT3F"}),
        ("1F+2F+CT2F_A+MADCT2F",        {"1F", "2F", "CT2F_A", "MADCT2F"}),
        ("Wave-1.5 (CT2F_A+MADCT2F+CT3F)",
                                          {"1F", "2F", "CT2F_A", "MADCT2F", "CT3F"}),
        ("Wave-1.5 + CT2F (full)",      {"1F", "2F", "CT2F", "CT2F_A",
                                          "MADCT2F", "CT3F"}),
    ]

    print("=" * 110)
    print("Held-out 5-fold CV on SIESTA-PBE cohesive energy (n=243), per-atom meV")
    print("=" * 110)
    print(f"{'Model':<35s}  {'p':>5s}  {'alpha':>7s}  "
          f"{'RMSE meV':>9s}  {'MAE':>6s}  {'ρ':>6s}  {'τ':>6s}  {'R²':>6s}")
    print("-" * 110)
    runs = {}
    preds = {}
    for label, classes in variants:
        X = get_X(classes)
        if X.shape[1] == 0:
            continue
        a = alpha_sweep_cv5(X, y, sz)
        m, pred = cv5_metrics(X, y, sz, alpha=a)
        runs[label] = dict(metrics=m, alpha=a)
        preds[label] = pred
        print(f"  {label:<35s}  {m['p']:>5d}  {a:>7.1e}  "
              f"{m['rmse_meV']:>9.1f}  {m['mae_meV']:>6.1f}  "
              f"{m['spearman']:>6.3f}  {m['kendall']:>6.3f}  {m['r2']:>6.3f}")

    # ------------------------------------------------------------------
    # Per-kind breakdown: focus on strain regime
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("Per-kind RMSE (meV/atom) — baseline vs Wave-1 CT2F vs Wave-1.5")
    print("=" * 110)
    bench_labels = [
        "1F+2F (baseline)",
        "1F+2F+CT2F (Wave-1)",
        "1F+2F+CT2F+MADCT2F",
        "1F+2F+CT2F+CT3F",
        "Wave-1.5 + CT2F (full)",
    ]
    err_by = {lbl: ((preds[lbl] - y) / sz) * 1000 for lbl in bench_labels}
    header_cols = "  ".join(f"{lbl[:18]:>18s}" for lbl in bench_labels)
    print(f"  {'kind':>6s}  {'n':>3s}  {header_cols}")
    for k_ in sorted(set(kinds)):
        sel = kinds == k_
        if sel.sum() < 2:
            continue
        vals = []
        for lbl in bench_labels:
            rmse = float(np.sqrt(np.mean(err_by[lbl][sel] ** 2)))
            rho = float(spearmanr(y[sel], preds[lbl][sel]).correlation)
            vals.append(f"{rmse:>10.1f}|ρ={rho:>4.2f}")
        print(f"  {k_:>6s}  {int(sel.sum()):>3d}  {'  '.join(vals)}")

    # ------------------------------------------------------------------
    # Per-family breakdown: Cs vs K disparity
    # ------------------------------------------------------------------
    print()
    print("=" * 110)
    print("Per-family RMSE (meV/atom) — focus on Cs* vs K* disparity")
    print("=" * 110)
    print(f"  {'family':>6s}  {'n':>3s}  {'baseline':>8s}  {'+CT2F':>6s}  "
          f"{'+CT2F+MADCT2F':>13s}  {'+CT2F+CT3F':>10s}  {'Wave-1.5 full':>13s}")
    for f_ in sorted(set(fams)):
        sel = fams == f_
        if sel.sum() < 2:
            continue
        vals = []
        for lbl in bench_labels:
            rmse = float(np.sqrt(np.mean(err_by[lbl][sel] ** 2)))
            vals.append(rmse)
        print(f"  {f_:>6s}  {int(sel.sum()):>3d}  "
              f"{vals[0]:>8.1f}  {vals[1]:>6.1f}  "
              f"{vals[2]:>13.1f}  {vals[3]:>10.1f}  {vals[4]:>13.1f}")

    out = {"variants": {k: v["metrics"] for k, v in runs.items()},
            "alpha":   {k: v["alpha"]   for k, v in runs.items()}}
    (SD / "wave15_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {SD/'wave15_comparison.json'}")


if __name__ == "__main__":
    main()
