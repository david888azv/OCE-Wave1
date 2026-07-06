"""Pick 100 perovskites for SIESTA validation, stratified.

Composition target:
  24 endmembers (5-atom primitives, all (A,B,X) combinations)
  76 supercells (40 atom 2x2x2) — proportional across kind × (A,B)
       split: 40 mixed-X / 18 mixed-A / 12 mixed-B / 6 strained

Only includes records that succeeded under GFN-FF.  Output:
  data/perovskites/validation_100.json — same record format + index/order.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


NAME_RE = re.compile(r"^(Cs|K)(Pb|Sn|Ge)(I|Br|Cl|F)3_(\w+)")


def _key(rec: dict) -> tuple:
    m = NAME_RE.match(rec["name"])
    if not m:
        return ("?", "?", "?", "?")
    a, b, x, kind = m.groups()
    kind = "prim" if kind == "prim" else (
        kind[:4] if kind.startswith(("mixX", "mixA", "mixB"))
        else "strain"
    )
    return (a, b, x, kind)


def select(records: list[dict], rng: random.Random,
            n_target: int = 100) -> list[int]:
    # Filter: only those with energy
    cand = [(i, r) for i, r in enumerate(records)
             if r.get("energy_total_eV") is not None]
    print(f"GFN-FF success pool: {len(cand)}")

    # Endmembers: prefer all 24 (Cs/K × Pb/Sn/Ge × I/Br/Cl/F)
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, r in cand:
        by_key[_key(r)].append(i)

    chosen: list[int] = []
    # 1) endmembers (24 ABX combos × prim)
    abx = [(a, b, x) for a in ["Cs", "K"] for b in ["Pb", "Sn", "Ge"]
            for x in ["I", "Br", "Cl", "F"]]
    for a, b, x in abx:
        ids = by_key.get((a, b, x, "prim"), [])
        if ids:
            chosen.append(rng.choice(ids))
    print(f"endmembers picked: {len(chosen)} (max 24)")

    # 2) supercells, by kind quota
    quotas = {"mixX": 40, "mixA": 18, "mixB": 12, "stra": 6}
    for kind, n_kind in quotas.items():
        # collect by (A,B,X,kind) and round-robin
        sub_keys = [k for k in by_key if k[3] == kind]
        rng.shuffle(sub_keys)
        per_key = max(1, n_kind // max(1, len(sub_keys)))
        picks = []
        for k in sub_keys:
            picks.extend(rng.sample(by_key[k], min(per_key, len(by_key[k]))))
        # if still under quota, top up randomly from same kind
        while len(picks) < n_kind:
            extras = [i for k in sub_keys for i in by_key[k] if i not in picks]
            if not extras: break
            picks.append(rng.choice(extras))
        # prune over quota
        picks = picks[:n_kind]
        chosen.extend(picks)
        print(f"{kind:>5s}: target {n_kind}, picked {len(picks)}")

    if len(chosen) > n_target:
        chosen = rng.sample(chosen, n_target)

    return sorted(set(chosen))


def main(n_target: int = 100, seed: int = 20260508):
    rng = random.Random(seed)
    records = json.loads((ROOT / "data" / "perovskites" / "structures.json").read_text())
    idxs = select(records, rng, n_target=n_target)
    out = []
    for i in idxs:
        r = dict(records[i])
        r["original_index"] = int(i)
        out.append(r)
    summary = Counter(_key(r)[3] for r in out)
    print(f"\nFinal subset: {len(out)} structures, kinds: {dict(summary)}")
    print(f"family (A,B): {Counter((_key(r)[0], _key(r)[1]) for r in out)}")
    out_path = ROOT / "data" / "perovskites" / "validation_100.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(n)
