"""Engine-agnostic parallel batch wrapper for energy engines.

Maps any `(atoms, **kw) → (E_eV, atoms_opt)` callable over a list of Atoms
in parallel.  Originally written for `oce.xtb_runner.xtb_energy`; now also
used for `oce_carbon.runners.dftbplus.dftbplus_energy`.

Both engines (xtb / DFTB+) are CPU-bound subprocess calls, so we use a
ThreadPoolExecutor: Python threads release the GIL while waiting for the
external process, giving true parallelism without fork/pickle overhead of
multiprocessing.  Each worker forces OMP_NUM_THREADS=1 inside the engine
to avoid CPU oversubscription.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable

from ase import Atoms

from oce.xtb_runner import xtb_energy


def _engine_one(args):
    """Worker: takes (idx, atoms, kwargs, engine_fn) and returns
    (idx, E_eV, opt_atoms, error_or_None)."""
    idx, atoms, kwargs, engine_fn = args
    kwargs.setdefault("threads", 1)
    try:
        E_eV, opt = engine_fn(atoms, **kwargs)
        return (idx, E_eV, opt, None)
    except Exception as exc:
        return (idx, None, None, str(exc))


def energy_batch(atoms_list: list[Atoms],
                 engine: Callable = xtb_energy,
                 kwargs_list: list[dict] | dict | None = None,
                 n_workers: int | None = None,
                 progress: bool = True,
                 executor: str = "thread") -> list[tuple[float, Atoms]]:
    """Engine-agnostic parallel batch: run any energy engine over many atoms.

    Args:
        atoms_list: list of ASE Atoms to evaluate
        engine: callable that takes (atoms, **kwargs) and returns
                (E_eV, atoms_opt).  Defaults to oce.xtb_runner.xtb_energy
                for backward compatibility; also accepts dftbplus_energy.
        kwargs_list: dict (broadcast) or list of dicts (per-call) of kwargs
                to pass to the engine
        n_workers: number of concurrent workers (default os.cpu_count())
        executor: "thread" (default) or "process"

    Returns: list of (E_eV, atoms_opt) in INPUT order.  Failed entries
             return (None, None).
    """
    n = len(atoms_list)
    if n == 0:
        return []
    if kwargs_list is None:
        kwargs_list = [{} for _ in range(n)]
    elif isinstance(kwargs_list, dict):
        kwargs_list = [dict(kwargs_list) for _ in range(n)]
    if len(kwargs_list) != n:
        raise ValueError("kwargs_list length must match atoms_list")

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = min(n_workers, n)

    results: list[tuple[float, Atoms] | tuple[None, None]] = [
        (None, None)] * n
    failed = []

    t0 = time.perf_counter()
    if progress:
        print(f"  [parallel {engine.__name__}] {n} tasks across "
              f"{n_workers} workers ...")

    args = [(i, atoms_list[i], kwargs_list[i], engine) for i in range(n)]
    done = 0
    pool_cls = ThreadPoolExecutor if executor == "thread" else ProcessPoolExecutor
    with pool_cls(max_workers=n_workers) as pool:
        for fut in as_completed([pool.submit(_engine_one, a) for a in args]):
            idx, E, opt, err = fut.result()
            done += 1
            if err is not None:
                failed.append((idx, err))
                if progress:
                    print(f"    [{done}/{n}] FAILED idx={idx}: {err[:80]}")
            else:
                results[idx] = (E, opt)
                if progress and (done % max(1, n // 20) == 0 or done == n):
                    elapsed = time.perf_counter() - t0
                    rate = done / max(elapsed, 1e-9)
                    eta = (n - done) / max(rate, 1e-9)
                    print(f"    [{done}/{n}]  {elapsed:6.1f}s elapsed,"
                          f" {rate:.1f}/s, ETA {eta:5.1f}s")

    elapsed = time.perf_counter() - t0
    n_ok = sum(1 for r in results if r[0] is not None)
    if progress:
        print(f"  [parallel {engine.__name__}] done in {elapsed:.1f}s — "
              f"{n_ok}/{n} succeeded, {len(failed)} failed")
    return results


# Backward-compatible alias
def xtb_energy_batch(atoms_list, kwargs_list=None, n_workers=None,
                      progress=True, executor="thread"):
    """Deprecated alias — use energy_batch(engine=xtb_energy, ...)."""
    return energy_batch(atoms_list, xtb_energy, kwargs_list,
                         n_workers, progress, executor)


if __name__ == "__main__":
    """Demo with two scales:
       (a) tiny G2 molecules (5-15 atoms) — Pool overhead dominates,
           speedup capped at ~4×
       (b) peptide-scale molecules (30-100 atoms) — true compute regime,
           speedup approaches 16-32×
    """
    from pathlib import Path
    from time import perf_counter

    base = Path(__file__).resolve().parents[1]

    # ---- (b) peptide-scale benchmark with FRESH RDKit geometries
    # so xtb actually does work (cached geometries are pre-optimised
    # and converge in 1 step, hiding the parallelism benefit).
    from oce_protein.peptides import (load_peptides, seq_to_smiles,
                                        _smiles_to_atoms)
    short = load_peptides(base / "data" / "protein" / "peptides_short.json")
    real = load_peptides(base / "data" / "protein" / "peptides_real.json")
    pep_atoms = []
    for p in (short + real):
        try:
            pep_atoms.append(_smiles_to_atoms(p.smiles, seed=42))
        except Exception:
            pass
    print(f"--- peptide-scale benchmark ({len(pep_atoms)} mols, "
          f"avg {sum(len(a) for a in pep_atoms)/len(pep_atoms):.0f} atoms,"
          f"  fresh RDKit geometries) ---")

    # ---- serial with OMP=1 (single-thread, cleanest baseline) ----
    print("\n[serial, xtb OMP=1 — true single-thread baseline]")
    t0 = perf_counter()
    serial1 = []
    for a in pep_atoms:
        E, opt = xtb_energy(a, optimize=True, threads=1)
        serial1.append((E, opt))
    t_s1 = perf_counter() - t0
    print(f"  serial(OMP=1): {t_s1:.2f}s  ({t_s1/len(pep_atoms)*1000:.0f}ms/mol)")

    # ---- serial with OMP=auto (the default xtb uses) ----
    print("\n[serial, xtb OMP=auto — what 'naive serial' gives today]")
    t0 = perf_counter()
    serial_auto = []
    for a in pep_atoms:
        E, opt = xtb_energy(a, optimize=True)
        serial_auto.append((E, opt))
    t_sa = perf_counter() - t0
    print(f"  serial(OMP=auto): {t_sa:.2f}s  ({t_sa/len(pep_atoms)*1000:.0f}ms/mol)")

    # ---- parallel, each worker pinned to OMP=1 ----
    print("\n[parallel, ThreadPool, each xtb at OMP=1, N workers]")
    print(f"  {'workers':>8s}  {'time':>8s}  {'speedup':>8s}")
    for nw in (4, 8, 16, 32):
        t0 = perf_counter()
        batch = xtb_energy_batch(pep_atoms,
                                  kwargs_list={"optimize": True},
                                  n_workers=nw, progress=False,
                                  executor="thread")
        t_par = perf_counter() - t0
        valid = [(a, b) for a, b in zip(serial1, batch)
                  if a[0] is not None and b[0] is not None]
        diffs = [abs(a[0] - b[0]) for a, b in valid]
        max_diff = max(diffs) if diffs else 0
        print(f"  {nw:>8d}  {t_par:>7.2f}s  {t_s1/t_par:>7.1f}×  "
              f"(max ΔE = {max_diff:.1e} eV)")

    print("\n[parallel, ProcessPool (for comparison), each xtb at OMP=1]")
    print(f"  {'workers':>8s}  {'time':>8s}  {'speedup':>8s}")
    for nw in (32,):
        t0 = perf_counter()
        batch = xtb_energy_batch(pep_atoms,
                                  kwargs_list={"optimize": True},
                                  n_workers=nw, progress=False,
                                  executor="process")
        t_par = perf_counter() - t0
        print(f"  {nw:>8d}  {t_par:>7.2f}s  {t_s1/t_par:>7.1f}×")
