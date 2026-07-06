"""Active learning iteration 1 — select 50 high-variance perovskites.

Strategy: bootstrap aggregation of Ridge regressions on the 93 SIESTA-labelled
cohesive energies (1F+2F basis, α=10).  For each candidate (1532 GFN-FF-OK
records minus the 93 already-SIESTA), predict with K=50 bootstrap models and
take the prediction-std as uncertainty.

Stratified pick: 8 per (A,B) family (6 families = 48) + top-2 by overall std.
Output: active_learning_iter1_selection.json with the 50 selected records
ready for the SIESTA pipeline.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "data" / "perovskites"

NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+?)(\d+)?(_.*)?$")
RNG = np.random.default_rng(20260508)


def parse_family(name: str) -> str:
    m = NAME_RE.match(name)
    return f"{m.group(1)}{m.group(2)}" if m else "?"


def parse_kind(name: str) -> str:
    m = NAME_RE.match(name)
    if not m: return "?"
    k = m.group(4)
    return k[:4] if k.startswith("mix") else ("prim" if k == "prim" else "strain")


def main(K_boot: int = 50, n_target: int = 50, alpha: float = 10.0):
    # ----- inputs -----
    atom_refs = {el: r["E_eV"] for el, r in
                  json.loads((SD / "atom_refs.json").read_text()).items()
                  if r.get("converged")}
    siesta = json.loads((SD / "siesta_szp_results.json").read_text())
    siesta_ok = {r["name"]: r for r in siesta if r.get("converged")}

    structs = json.loads((SD / "structures.json").read_text())
    by_name = {r["name"]: r for r in structs}

    npz = np.load(SD / "features.npz", allow_pickle=True)
    X_all = npz["X"]; names = [str(n) for n in npz["names"]]
    sizes = npz["sizes"]
    feat_index = json.loads((SD / "feature_index.json").read_text())
    feat_keys = [tuple(e["key"]) for e in feat_index]
    n2i = {n: i for i, n in enumerate(names)}

    # 1F+2F columns only
    cols_12 = sorted([j for j, k in enumerate(feat_keys)
                       if k != ("MAD",) and k[0] in ("1F", "2F")])
    print(f"Using 1F+2F basis: {len(cols_12)} columns")

    # ----- training set: 93 SIESTA-labelled with E_coh -----
    train_idx = []; y_train = []
    for nm, r in siesta_ok.items():
        rec = by_name.get(nm)
        if rec is None: continue
        ref_sum = sum(atom_refs.get(s, 0.0) for s in rec["symbols"])
        E_coh = r["E_eV"] - ref_sum
        i = n2i.get(nm)
        if i is None: continue
        train_idx.append(i); y_train.append(E_coh)
    train_idx = np.array(train_idx); y_train = np.array(y_train)
    X_train = X_all[train_idx][:, cols_12]
    print(f"Training set (SIESTA E_coh): n={len(train_idx)}, p={X_train.shape[1]}")

    # ----- candidate pool: GFN-FF-OK minus the 93 already-SIESTA -----
    siesta_set = set(siesta_ok.keys())
    cand_records = []
    for r in structs:
        if r["name"] in siesta_set: continue
        if r.get("energy_total_eV") is None: continue
        i = n2i.get(r["name"])
        if i is None: continue
        cand_records.append((i, r))
    cand_idx = np.array([c[0] for c in cand_records])
    X_cand = X_all[cand_idx][:, cols_12]
    print(f"Candidate pool: {len(cand_idx)}")

    # ----- bootstrap variance -----
    n_train = len(train_idx)
    boot_preds = np.zeros((K_boot, len(cand_idx)))
    for k in range(K_boot):
        sel = RNG.choice(n_train, size=n_train, replace=True)
        m = Ridge(alpha=alpha, fit_intercept=True).fit(X_train[sel], y_train[sel])
        boot_preds[k] = m.predict(X_cand)
        if (k + 1) % 10 == 0:
            print(f"  bootstrap {k+1}/{K_boot}")
    pred_mean = boot_preds.mean(axis=0)
    pred_std = boot_preds.std(axis=0, ddof=1)
    print(f"Predictive std: min={pred_std.min():.3f}  med={np.median(pred_std):.3f}  "
          f"max={pred_std.max():.3f}  (eV total)")

    # Per-atom uncertainty (more meaningful for stratification)
    cand_sizes = sizes[cand_idx]
    pred_std_pa = pred_std / cand_sizes  # eV/atom
    print(f"Predictive std per atom (eV/atom): min={pred_std_pa.min():.4f} "
          f"med={np.median(pred_std_pa):.4f} max={pred_std_pa.max():.4f}")

    # ----- stratified selection -----
    fam_groups = defaultdict(list)
    for k, (i, rec) in enumerate(cand_records):
        fam = parse_family(rec["name"])
        fam_groups[fam].append((k, rec, float(pred_std_pa[k]), float(pred_std[k])))

    # Sort each family by per-atom uncertainty desc
    n_per_fam = max(1, n_target // max(1, len(fam_groups)))
    selected: list[tuple[int, dict, float, float]] = []
    used = set()
    print(f"\nFamilies: {sorted(fam_groups)}  picking ≤{n_per_fam} each")
    for fam, items in sorted(fam_groups.items()):
        items.sort(key=lambda x: -x[2])  # desc by per-atom std
        for tup in items[:n_per_fam]:
            selected.append(tup); used.add(tup[0])

    # Top up by overall std until we reach n_target
    all_sorted = []
    for fam, items in fam_groups.items():
        all_sorted.extend(items)
    all_sorted.sort(key=lambda x: -x[2])
    for tup in all_sorted:
        if len(selected) >= n_target: break
        if tup[0] in used: continue
        selected.append(tup); used.add(tup[0])

    selected = selected[:n_target]
    print(f"Selected {len(selected)} perovskites")

    # ----- save selection (records ready for SIESTA pipeline) -----
    out_records = []
    fam_count = defaultdict(int)
    kind_count = defaultdict(int)
    for k, rec, std_pa, std_total in selected:
        fam = parse_family(rec["name"])
        kind = parse_kind(rec["name"])
        fam_count[fam] += 1; kind_count[kind] += 1
        r2 = dict(rec)
        r2["pred_std_per_atom"] = std_pa
        r2["pred_std_total"]   = std_total
        r2["family"] = fam
        r2["kind"]   = kind
        out_records.append(r2)
    out_path = SD / "active_learning_iter1_selection.json"
    out_path.write_text(json.dumps(out_records, indent=2))
    print(f"\nWrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")
    print(f"  Per-family: {dict(fam_count)}")
    print(f"  Per-kind:   {dict(kind_count)}")
    print(f"  Mean pred_std (eV/atom): "
          f"{np.mean([r['pred_std_per_atom'] for r in out_records]):.4f}")
    print(f"  Top-1 pred_std (eV/atom): "
          f"{max(r['pred_std_per_atom'] for r in out_records):.4f}")


if __name__ == "__main__":
    main()
