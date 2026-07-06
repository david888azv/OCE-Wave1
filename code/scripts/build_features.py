"""Feature builder for perovskite + MOF subsets.

Walks data/<subset>/structures.json, enumerates 1F+2F+3F figures via the PBC
pipeline, bins 2F (distance) and 3F (angle), prunes low-support columns,
and writes:
  data/<subset>/features.npz  — compressed X matrix + feature index
  data/<subset>/features_summary.json — per-structure and per-class stats

Usage:
  python data/build_features.py perovskites [--no-prune] [--min-support 0.01]
  python data/build_features.py mofs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase import Atoms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oce.atomic_table import load_table
from oce_carbon.features.figures_pbc import enumerate_figures_pbc
from oce_carbon.features.cutoff_2f import enumerate_cutoff_2f, bin_distance
from oce_carbon.features.cutoff_2f_jit import enumerate_cutoff_2f_jit_counter

# Standard discretisation
N_DIST_BINS = 5
DIST_RANGE = (1.5, 6.0)   # Angstrom
N_ANGLE_BINS = 8
ANGLE_RANGE = (0.0, np.pi)
# Cutoff-2F (Wave-1 extension)
CT2F_CUTOFF = 5.0
CT2F_RMIN   = 1.0
CT2F_NBINS  = 6


def _record_to_atoms(rec: dict) -> Atoms:
    return Atoms(
        symbols=rec["symbols"],
        positions=rec["positions"],
        cell=rec["cell"],
        pbc=rec["pbc"],
    )


def _bin(value: float, lo: float, hi: float, nbins: int) -> int:
    """Clamp value to [lo, hi] then bin into nbins equal slots."""
    v = max(lo, min(hi - 1e-12, value))
    return int((v - lo) / (hi - lo) * nbins)


def _structure_keys(atoms: Atoms, table: dict, charges: list[float] | None,
                     include_ct2f: bool = False,
                     use_jit: bool = False) -> Counter:
    """Return Counter mapping bucketed feature keys to integer counts.

    1F key:   ("1F", element, shell)
    2F key:   ("2F", el_a, sh_a, el_b, sh_b, BO, dist_bin)
    3F key:   ("3F", el_center, sh_center, el_arm_a, sh_arm_a, el_arm_b,
               sh_arm_b, ring_size, BO_a, BO_b, angle_bin)
    CT2F key: ("CT2F", el_a, el_b, dist_bin)  — geometric cutoff pair count
    Madelung: ("MAD",) — sum of q_i q_j / r_ij over all bonded pairs (signed)
    """
    one, two, three, _four = enumerate_figures_pbc(atoms, table)
    bag: Counter = Counter()
    for f in one:
        bag[("1F", f.element, f.shell_label)] += 1
    for f in two:
        b = _bin(f.distance, *DIST_RANGE, N_DIST_BINS)
        a = (f.element_i, f.shell_i)
        c = (f.element_j, f.shell_j)
        a, c = sorted([a, c])
        bag[("2F", a[0], a[1], c[0], c[1], int(f.bond_order), b)] += 1
    for f in three:
        ang_bin = _bin(f.angle_rad, *ANGLE_RANGE, N_ANGLE_BINS)
        arm_a = (f.element_i, f.shell_i, int(f.bond_order_ij))
        arm_b = (f.element_k, f.shell_k, int(f.bond_order_jk))
        arm_a, arm_b = sorted([arm_a, arm_b])
        center = (f.element_j, f.shell_j)
        bag[("3F", center[0], center[1],
              arm_a[0], arm_a[1], arm_a[2],
              arm_b[0], arm_b[1], arm_b[2],
              int(f.ring_size), ang_bin)] += 1
    # Madelung: sum q_i q_j / r_ij over bonded pairs only (cheap proxy)
    if charges is not None:
        mad = 0.0
        for f in two:
            qi = charges[f.i]
            qj = charges[f.j]
            r = max(f.distance, 1e-3)
            mad += qi * qj / r
        bag[("MAD",)] = float(mad)  # store as float — handled separately later

    # Cutoff-2F (Wave 1 — strain-aware geometric pair counts)
    if include_ct2f:
        if use_jit:
            ct_bag = enumerate_cutoff_2f_jit_counter(
                atoms, r_cutoff=CT2F_CUTOFF, r_min=CT2F_RMIN, n_bins=CT2F_NBINS,
            )
            for k, v in ct_bag.items():
                bag[k] += v
        else:
            ct = enumerate_cutoff_2f(atoms, r_cutoff=CT2F_CUTOFF, r_min=CT2F_RMIN)
            for f in ct:
                b = bin_distance(f.distance, CT2F_RMIN, CT2F_CUTOFF, CT2F_NBINS)
                bag[f.key(b)] += 1
    return bag


def _process_one(args):
    idx, rec_path, table_path, include_ct2f, use_jit = args
    table = load_table(Path(table_path))
    rec = json.loads(Path(rec_path).read_text())
    atoms = _record_to_atoms(rec)
    charges = rec.get("formal_charges")
    bag = _structure_keys(atoms, table, charges,
                           include_ct2f=include_ct2f, use_jit=use_jit)
    return idx, dict(bag), len(atoms), rec.get("name", f"#{idx}"), \
        rec.get("family", "")


def build_features(subset_dir: Path, n_workers: int = 8,
                    min_support: float = 0.01,
                    do_prune: bool = True,
                    include_ct2f: bool = False,
                    use_jit: bool = False,
                    out_suffix: str = "") -> dict:
    table_path = ROOT / "data" / "atoms" / "atomic_table.json"
    structures = json.loads((subset_dir / "structures.json").read_text())
    n = len(structures)
    print(f"[{subset_dir.name}] {n} structures, "
          f"{sum(len(r['symbols']) for r in structures)} atoms total")

    # Stage records to per-row tmp files so workers don't share huge JSON
    tmp_dir = subset_dir / ".tmp_records"
    tmp_dir.mkdir(exist_ok=True)
    paths = []
    for i, r in enumerate(structures):
        p = tmp_dir / f"r{i:06d}.json"
        p.write_text(json.dumps(r))
        paths.append(p)

    t0 = time.time()
    bags: list[dict] = [None] * n
    sizes: list[int] = [0] * n
    names: list[str] = [""] * n
    families: list[str] = [""] * n
    print(f"[{subset_dir.name}] enumerating figures with {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_process_one, (i, str(paths[i]), str(table_path),
                                          include_ct2f, use_jit))
                for i in range(n)]
        done = 0
        for fut in as_completed(futs):
            idx, bag, sz, nm, fam = fut.result()
            bags[idx] = bag
            sizes[idx] = sz
            names[idx] = nm
            families[idx] = fam
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                el = time.time() - t0
                print(f"  {done}/{n}   {el:.1f} s   "
                      f"{el / done * 1000:.1f} ms/struct")
    t_enum = time.time() - t0

    # Cleanup tmp (robust against leftover files from prior crashed runs)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Aggregate global feature-key support
    support = Counter()
    for bag in bags:
        for k in bag:
            if k == ("MAD",):
                continue
            support[k] += 1

    n_raw = len(support)

    # Pruning
    if do_prune:
        thr = int(np.ceil(min_support * n))
        kept = [k for k, s in support.items() if s >= thr]
    else:
        kept = list(support.keys())
    kept = sorted(kept)
    has_mad = any(("MAD",) in bag for bag in bags)
    if has_mad:
        kept_with_mad = [("MAD",)] + kept
    else:
        kept_with_mad = kept

    key_to_col = {k: i for i, k in enumerate(kept_with_mad)}
    p = len(kept_with_mad)

    # Build X
    X = np.zeros((n, p), dtype=np.float64)
    for i, bag in enumerate(bags):
        for k, v in bag.items():
            col = key_to_col.get(k)
            if col is None:
                continue
            X[i, col] = float(v)

    # Save
    feat_index = [{"col": i, "key": list(k)} for i, k in enumerate(kept_with_mad)]
    npz_name = f"features{out_suffix}.npz"
    idx_name = f"feature_index{out_suffix}.json"
    np.savez_compressed(subset_dir / npz_name,
                         X=X, names=np.array(names), sizes=np.array(sizes),
                         families=np.array(families))
    (subset_dir / idx_name).write_text(json.dumps(feat_index, indent=1))

    # Per-class stats
    summary = {
        "subset": subset_dir.name,
        "n_structures": n,
        "n_atoms_total": int(np.sum(sizes)),
        "atoms_per_structure": {
            "min": int(min(sizes)), "mean": float(np.mean(sizes)),
            "max": int(max(sizes)),
        },
        "n_features_raw": n_raw,
        "n_features_kept": p,
        "min_support_frac": min_support if do_prune else 0.0,
        "n_features_by_class": {
            "1F":   sum(1 for k in kept_with_mad
                        if isinstance(k, tuple) and len(k) >= 1 and k[0] == "1F"),
            "2F":   sum(1 for k in kept_with_mad
                        if isinstance(k, tuple) and len(k) >= 1 and k[0] == "2F"),
            "3F":   sum(1 for k in kept_with_mad
                        if isinstance(k, tuple) and len(k) >= 1 and k[0] == "3F"),
            "CT2F": sum(1 for k in kept_with_mad
                        if isinstance(k, tuple) and len(k) >= 1 and k[0] == "CT2F"),
            "MAD":  int(has_mad),
        },
        "wall_time_seconds": t_enum,
        "throughput_ms_per_structure": t_enum / max(n, 1) * 1000,
        "compute": {
            "n_workers": n_workers,
        },
        "discretisation": {
            "dist_bins": N_DIST_BINS, "dist_range_A": list(DIST_RANGE),
            "angle_bins": N_ANGLE_BINS, "angle_range_rad": list(ANGLE_RANGE),
        },
        "per_family": {
            f: {
                "n": int((np.array(families) == f).sum()),
                "atoms_mean": float(np.array(sizes)[np.array(families) == f].mean())
                              if (np.array(families) == f).any() else 0.0,
            }
            for f in sorted(set(families))
        },
    }
    (subset_dir / f"features_summary{out_suffix}.json").write_text(
        json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subset", help="data/<subset>/ directory containing structures.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-support", type=float, default=0.01)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--ct2f", action="store_true",
                     help="Include cutoff-2F geometric features (Wave-1)")
    ap.add_argument("--jit", action="store_true",
                     help="Use Numba JIT for hot loops (CT2F kernel)")
    ap.add_argument("--suffix", default="",
                     help="Output suffix (e.g. '_ct2f')")
    args = ap.parse_args()
    sd = ROOT / "data" / args.subset
    if not (sd / "structures.json").exists():
        raise SystemExit(f"No structures.json in {sd}")
    suffix = args.suffix or ("_ct2f" if args.ct2f else "")
    summary = build_features(
        subset_dir=sd, n_workers=args.workers,
        min_support=args.min_support, do_prune=not args.no_prune,
        include_ct2f=args.ct2f, use_jit=args.jit, out_suffix=suffix,
    )
    print()
    print(json.dumps(summary, indent=2))
