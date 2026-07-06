"""Run SIESTA SZP on the 50 active-learning iter-3 (non-strain) picks."""
from __future__ import annotations

import argparse
import json
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
    r["pred_std_per_atom"] = rec.get("pred_std_per_atom")
    r["al_iteration"] = 3
    return idx, r


def main(workers: int = 6):
    sd = ROOT / "data" / "perovskites"
    selection = json.loads((sd / "active_learning_iter3_selection.json").read_text())
    out_path = sd / "siesta_szp_results.json"
    prior = json.loads(out_path.read_text()) if out_path.exists() else []
    prior_names = {r["name"] for r in prior}
    queue = [(i, r) for i, r in enumerate(selection)
              if r["name"] not in prior_names]
    print(f"AL iter-3 batch: {len(selection)}  to run: {len(queue)}  "
          f"workers: {workers}×2 OMP")

    done = list(prior); n_ok = 0; n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_process_one, item) for item in queue]
        for fut in as_completed(futs):
            idx, r = fut.result()
            done.append(r)
            if r.get("converged"): n_ok += 1
            else: n_fail += 1
            elapsed = time.time() - t0
            n_so_far = n_ok + n_fail
            avg = elapsed / max(1, n_so_far)
            remaining = (len(queue) - n_so_far) * avg
            tag = r.get("name", "?"); wt = r.get("wall_time_s", 0)
            std_pa = r.get("pred_std_per_atom", 0)
            print(f"  [{n_so_far}/{len(queue)}]  {tag:<35s}  "
                  f"E={r.get('E_eV', float('nan')):.1f}  "
                  f"wall={wt:.1f}s  conv={r.get('converged')}  "
                  f"σ_pa={std_pa:.3f}  "
                  f"elapsed={elapsed/60:.1f}min  ETA={remaining/60:.1f}min")
            out_path.write_text(json.dumps(done, indent=2))
    print(f"\nAL iter 3 done: {n_ok} converged / {n_fail} failed")
    print(f"Total in file: {len(done)} (was {len(prior)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    main(workers=args.workers)
