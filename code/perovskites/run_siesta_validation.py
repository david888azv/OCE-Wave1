"""Run SIESTA SZP single-points on the 93-perovskite validation subset.

Parallelism: 6 jobs × 2 OMP threads each (= 12 cores total).  Each SIESTA
run produces (E_eV, scf_converged, wall_time_s, kmesh).  We persist
incrementally so a partial run can be inspected and resumed.

Usage:
  python data/perovskites/run_siesta_validation.py [--workers N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ase import Atoms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.perovskites.siesta_perov import siesta_szp_energy


def _process_one(idx_rec):
    idx, rec = idx_rec
    ats = Atoms(symbols=rec["symbols"], positions=rec["positions"],
                 cell=rec["cell"], pbc=rec["pbc"])
    r = siesta_szp_energy(ats, tag=rec["name"], threads=2, timeout=2400)
    r["name"] = rec["name"]
    r["original_index"] = rec.get("original_index", idx)
    r["family"] = rec.get("family", "")
    r["kind"]   = rec.get("kind", "")
    r["E_gfnff_eV"] = rec.get("energy_total_eV")
    return idx, r


def main(workers: int = 6, out_path: Path | None = None):
    sd = ROOT / "data" / "perovskites"
    subset = json.loads((sd / "validation_100.json").read_text())
    n = len(subset)
    out_path = out_path or (sd / "siesta_szp_results.json")

    # Resume: load any prior results
    prior = {}
    if out_path.exists():
        prior_recs = json.loads(out_path.read_text())
        for r in prior_recs:
            prior[r["name"]] = r

    queue = [(i, r) for i, r in enumerate(subset) if r["name"] not in prior]
    done = list(prior.values())
    print(f"Subset: {n}  already done: {len(prior)}  to run: {len(queue)}  "
          f"workers: {workers} (×2 OMP each)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_one, (i, r)) for i, r in queue]
        n_ok = sum(1 for r in done if r.get("converged"))
        n_fail = sum(1 for r in done if not r.get("converged"))
        for fut in as_completed(futs):
            idx, r = fut.result()
            done.append(r)
            if r.get("converged"):
                n_ok += 1
            else:
                n_fail += 1
            n_so_far = len(done)
            elapsed = time.time() - t0
            avg = elapsed / max(1, n_so_far - len(prior))
            remaining = (n - n_so_far) * avg
            tag = r.get("name", "?")
            wt = r.get("wall_time_s", 0)
            print(f"  [{n_so_far}/{n}]  {tag:<35s}  "
                  f"E={r.get('E_eV', float('nan')):.2f} eV  "
                  f"wall={wt:.1f}s  conv={r.get('converged')}  "
                  f"OK/FAIL={n_ok}/{n_fail}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={remaining/60:.1f}min")
            # incremental save
            out_path.write_text(json.dumps(done, indent=2))
    print(f"\nFinal: {n_ok} converged / {n_fail} failed")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    main(workers=args.workers)
