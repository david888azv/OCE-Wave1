"""Phase 6 — silicon Tersoff cross-lattice universality test.

Replica of phase5_three_lattices.py with Si replacing C:

  potential        : Tersoff (Si.tersoff, Tersoff PRB 1988)
  bond length a₀   : 2.35 Å (Si–Si in diamond Si)
  vdW radius r     : 2.10 Å (Bondi)
  atomic mass      : 28.0855 amu

Same 3 lattices, same L grid {16, 24, 32, 48, 64, 96, 128}, same MC volume
estimator.  The question we want to answer: is the law

    ⟨E⟩/⟨V⟩(L) → ε_V*  with γ = 0,  independent of starting lattice

still observed for Si — and is ε_V*(Si) ≠ ε_V*(C) (so that ε_V* is genuinely
a material property, not a numerical coincidence)?

Naïve estimate from bulk cohesion + atomic volume:
    ε_V*(Si)_predicted ≈ E_coh(Si) / V_mc/N(Si)
                       ≈ 4.6 eV / (0.6 × 4πr_vdW³/3 ≈ 23.3 Å³)
                       ≈ 0.20 eV/Å³        (≈ 3× smaller than C)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms

from lattices import build_honeycomb, build_square, build_triangular, P_C
from percolation import sample_many
from runners_lammps import lammps_si_tersoff
from volume import (volume_atomic, volume_inflated_hull, volume_obb_pad,
                     planar_area_pad, volume_mc_vdw, R_VDW_SI)
from analysis import fit_loglog


L_VALUES = [16, 24, 32, 48, 64, 96, 128]
N_PER_L = {16: 18, 24: 14, 32: 12, 48: 10, 64: 8, 96: 6, 128: 5}
A_SI = 2.35   # Å, Si-Si bond length
LATTICES = {
    "honeycomb":  ((lambda L: build_honeycomb(L, a=A_SI)),  P_C["honeycomb"]),
    "square":     ((lambda L: build_square(L, a=A_SI)),      P_C["square"]),
    "triangular": ((lambda L: build_triangular(L, a=A_SI)), P_C["triangular"]),
}
RANDOM_SEED = 20260511
MC_SAMPLES = 300_000
MPI_PROCS = 8
PARALLEL_REPLICAS = 2

DATA_DIR = ROOT / "data" / "phase6_silicon"
RESULTS_DIR = ROOT / "results" / "phase6_silicon"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _process_one(args) -> dict | None:
    name, L, ic, cl_pos, cl_seed, cl_n, cl_bonds, mpi_procs = args
    atoms = Atoms(symbols=["Si"] * cl_n, positions=cl_pos)
    sp = lammps_si_tersoff(atoms, optimize=False, timeout=1800,
                            mpi_procs=mpi_procs)
    opt = lammps_si_tersoff(atoms, optimize=True, timeout=3600,
                             mpi_procs=mpi_procs)
    if not (sp["converged"] and opt["converged"]):
        return dict(_skip=True, name=name, L=L, ic=ic, n=cl_n,
                    err=sp.get("error", "") + " | " + opt.get("error", ""))
    pos_rel = opt["opt_atoms"].get_positions()
    v_atomic = volume_atomic(cl_n, r=R_VDW_SI)
    v_obb = volume_obb_pad(pos_rel, r=R_VDW_SI)
    v_hull = volume_inflated_hull(pos_rel, r=R_VDW_SI)
    _, v_planar = planar_area_pad(pos_rel, r=R_VDW_SI)
    v_mc = volume_mc_vdw(pos_rel, r=R_VDW_SI,
                          n_samples=MC_SAMPLES, seed=cl_seed)
    rms = float(np.sqrt(np.mean(np.sum((pos_rel - cl_pos) ** 2, axis=1))))
    return dict(
        lattice=name, L=int(L), realisation=int(ic),
        seed=int(cl_seed), n_atoms=int(cl_n), n_bonds=int(cl_bonds),
        E_tersoff_relaxed_eV=float(opt["E_eV"]),
        E_tersoff_sp_eV=float(sp["E_eV"]),
        rms_displacement_A=rms,
        V_atomic_A3=float(v_atomic),
        V_obb_A3=float(v_obb), V_hull_A3=float(v_hull),
        V_planar_A3=float(v_planar), V_mc_A3=float(v_mc),
        wall_sp_s=float(sp["wall_time_s"]),
        wall_opt_s=float(opt["wall_time_s"]),
    )


def gather_lattice(name: str, builder, p: float) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = DATA_DIR / f"{name}_clusters.json"
    if cache.exists():
        recs = json.loads(cache.read_text())
        print(f"  [{name}] loaded {len(recs)} cached")
        return recs
    rng = np.random.default_rng(RANDOM_SEED + abs(hash(name)) % 10_000)
    records: list[dict] = []
    t0 = time.perf_counter()
    tasks = []
    for L in L_VALUES:
        lat = builder(L)
        n = N_PER_L[L]
        clusters = sample_many(lat, p, n,
                                base_seed=int(rng.integers(2**31)),
                                min_size=4)
        for ic, cl in enumerate(clusters):
            tasks.append((name, L, ic, cl.positions, cl.seed,
                          cl.n_atoms, cl.n_bonds, MPI_PROCS))
    with ThreadPoolExecutor(max_workers=PARALLEL_REPLICAS) as ex:
        futures = {ex.submit(_process_one, t): t for t in tasks}
        for f in as_completed(futures):
            t = futures[f]
            try:
                rec = f.result()
            except Exception as e:
                print(f"  [exc] {name} L={t[1]} ic={t[2]}: {e}")
                continue
            if rec.get("_skip"):
                print(f"  [skip] {name} L={rec['L']} ic={rec['ic']}  "
                      f"N={rec['n']}  err={rec['err'][:120]}")
                continue
            tt = (rec["wall_sp_s"] + rec["wall_opt_s"])
            print(f"  [{name}] L={rec['L']:3d} rep={rec['realisation']:2d} "
                  f"N={rec['n_atoms']:5d}  "
                  f"E={rec['E_tersoff_relaxed_eV']:+10.2f}  "
                  f"V_mc={rec['V_mc_A3']:8.0f}  "
                  f"E/V={rec['E_tersoff_relaxed_eV']/rec['V_mc_A3']:+.4f}  "
                  f"({tt:.1f}s LMP)", flush=True)
            records.append(rec)
            cache.write_text(json.dumps(sorted(
                records, key=lambda r: (r["L"], r["realisation"])
            ), indent=2))
    print(f"  [{name}] {len(records)} clusters in "
          f"{time.perf_counter() - t0:.1f}s\n")
    cache.write_text(json.dumps(sorted(
        records, key=lambda r: (r["L"], r["realisation"])
    ), indent=2))
    return records


def per_L_means(records, key) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ls = sorted({r["L"] for r in records})
    arrL = np.array(Ls, dtype=float)
    mean = np.array([np.mean([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    std = np.array([np.std([r[key] for r in records if r["L"] == L])
                     for L in Ls])
    return arrL, mean, std


def analyse(name: str, records: list[dict]) -> dict:
    arrL, n_mean, _ = per_L_means(records, "n_atoms")
    arrL, e_mean, e_std = per_L_means(records, "E_tersoff_relaxed_eV")
    _, v_mc_mean, v_mc_std = per_L_means(records, "V_mc_A3")
    _, v_hull_mean, _ = per_L_means(records, "V_hull_A3")
    _, v_atomic_mean, _ = per_L_means(records, "V_atomic_A3")

    eov_mc = e_mean / v_mc_mean
    eov_hull = e_mean / v_hull_mean
    eov_atomic = e_mean / v_atomic_mean
    epn = e_mean / n_mean

    fit_N = fit_loglog(arrL, n_mean)
    fit_E = fit_loglog(arrL, -e_mean)
    fit_V_mc = fit_loglog(arrL, v_mc_mean)
    fit_V_hull = fit_loglog(arrL, v_hull_mean)
    fit_V_atomic = fit_loglog(arrL, v_atomic_mean)
    fit_eov_mc = fit_loglog(arrL, -eov_mc)
    fit_eov_hull = fit_loglog(arrL, -eov_hull)
    fit_eov_atomic = fit_loglog(arrL, -eov_atomic)
    fit_epn = fit_loglog(arrL, -epn)

    return dict(
        name=name, L=[int(L) for L in arrL],
        N_atoms=[float(x) for x in n_mean],
        E_relaxed=[float(x) for x in e_mean],
        E_relaxed_std=[float(x) for x in e_std],
        V_mc=[float(x) for x in v_mc_mean],
        V_mc_std=[float(x) for x in v_mc_std],
        V_hull=[float(x) for x in v_hull_mean],
        V_atomic=[float(x) for x in v_atomic_mean],
        eov_mc=[float(x) for x in eov_mc],
        eov_hull=[float(x) for x in eov_hull],
        eov_atomic=[float(x) for x in eov_atomic],
        epn=[float(x) for x in epn],
        D_f=float(fit_N.exponent),
        D_E=float(fit_E.exponent),
        D_V_mc=float(fit_V_mc.exponent),
        D_V_hull=float(fit_V_hull.exponent),
        D_V_atomic=float(fit_V_atomic.exponent),
        gamma_eov_mc=float(fit_eov_mc.exponent),
        gamma_eov_hull=float(fit_eov_hull.exponent),
        gamma_eov_atomic=float(fit_eov_atomic.exponent),
        alpha_epn=float(fit_epn.exponent),
        eps_V_star_mc=float(-eov_mc[-1]),
        eps_V_star_hull=float(-eov_hull[-1]),
        eps_V_star_atomic=float(-eov_atomic[-1]),
        eps_per_atom_at_Lmax=float(-epn[-1]),
    )


def main():
    print(f"=== Phase 6: Silicon Tersoff, 3 lattices × L = {L_VALUES} ===\n"
          f"  bond length a₀ = {A_SI} Å,  r_vdW = {R_VDW_SI} Å\n")
    summaries = {}
    for name, (builder, p) in LATTICES.items():
        recs = gather_lattice(name, builder, p)
        summaries[name] = analyse(name, recs)

    print(f"\n{'='*78}")
    print(f"            MASTER TABLE — Si Tersoff (log–log, 7 L points)")
    print(f"{'='*78}")
    keys = ("D_f", "D_E", "D_V_mc", "D_V_hull", "D_V_atomic",
            "gamma_eov_mc", "gamma_eov_hull", "gamma_eov_atomic",
            "alpha_epn")
    print(f"  {'':18s} " + "   ".join(f"{n:8s}" for n in summaries))
    for k in keys:
        print(f"  {k:18s} " + "   ".join(
            f"{summaries[n][k]:+.3f}" for n in summaries))

    print(f"\n{'='*78}")
    print(f"            ASYMPTOTIC CONSTANTS at L_max = {L_VALUES[-1]}")
    print(f"{'='*78}")
    print(f"  {'':22s} " + "   ".join(f"{n:10s}" for n in summaries))
    for k in ("eps_V_star_mc", "eps_V_star_hull", "eps_V_star_atomic",
              "eps_per_atom_at_Lmax"):
        vals = [summaries[n][k] for n in summaries]
        print(f"  {k:22s} " + "   ".join(f"{v:.4f}" for v in vals))

    print(f"\n--- Cross-lattice spread (max−min)/mean ---")
    for k in ("eps_V_star_mc", "eps_V_star_hull", "eps_V_star_atomic",
              "eps_per_atom_at_Lmax"):
        v = np.array([summaries[n][k] for n in summaries])
        print(f"  {k:24s} mean={v.mean():.4f}  "
              f"spread={(v.max() - v.min()) / v.mean() * 100:.2f}%")

    # Compare to phase 5 (carbon AIREBO) if available
    c_summary_path = ROOT / "results" / "phase5_three_lattices" / "summary.json"
    if c_summary_path.exists():
        c_summ = json.loads(c_summary_path.read_text())
        print(f"\n--- Material comparison: Si vs C ---")
        for est in ("mc", "atomic"):
            si_v = np.mean([summaries[n][f"eps_V_star_{est}"] for n in summaries])
            c_v = np.mean([c_summ[n][f"eps_V_star_{est}"] for n in summaries])
            print(f"  ε_V*  via V_{est:6s}   Si={si_v:.4f}  C={c_v:.4f}  "
                  f"ratio C/Si = {c_v / si_v:.2f}")
        si_e = np.mean([summaries[n]["eps_per_atom_at_Lmax"] for n in summaries])
        c_e = np.mean([c_summ[n]["eps_per_atom_at_Lmax"] for n in summaries])
        print(f"  −E/N (eV/atom)    Si={si_e:.3f}  C={c_e:.3f}  "
              f"ratio C/Si = {c_e / si_e:.2f}")

    # Plots
    colors = {"honeycomb": "C0", "square": "C1", "triangular": "C2"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for name, s in summaries.items():
        arrL = np.array(s["L"], dtype=float)
        col = colors[name]
        axes[0, 0].plot(arrL, s["N_atoms"], "o-", color=col,
                        label=f"{name}  D_f={s['D_f']:+.3f}")
        axes[0, 1].plot(arrL, [-x for x in s["E_relaxed"]], "o-",
                         color=col, label=f"{name}  D_E={s['D_E']:+.3f}")
        axes[1, 0].plot(arrL, s["V_mc"], "o-", color=col,
                         label=f"{name}  D_V_mc={s['D_V_mc']:+.3f}")
        axes[1, 1].plot(arrL, [-x for x in s["eov_mc"]], "o-",
                         color=col, label=f"{name}  γ={s['gamma_eov_mc']:+.3f}")
    for ax in axes.ravel():
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("L")
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("⟨N_atoms⟩")
    axes[0, 0].set_title("Si cluster mass scaling")
    axes[0, 1].set_ylabel("−⟨E⟩  (eV)")
    axes[0, 1].set_title("Si total energy scaling")
    axes[1, 0].set_ylabel("⟨V_mc⟩  (Å³)")
    axes[1, 0].set_title("Si volume (MC vdW union)")
    axes[1, 1].set_ylabel(r"$-\langle E\rangle / \langle V_{mc}\rangle$  (eV/Å³)")
    axes[1, 1].set_title("Si energy density")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_master.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, s in summaries.items():
        arrL = np.array(s["L"], dtype=float)
        ax.plot(arrL, [-x for x in s["eov_mc"]], "o-",
                 color=colors[name],
                 label=f"{name}  ε_V*={-s['eov_mc'][-1]:.3f}")
    ax.set_xlabel("L")
    ax.set_ylabel(r"$-\langle E\rangle / \langle V_{mc}\rangle$  (eV/Å³)")
    ax.set_title("Si cluster energy density vs L (Tersoff)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_universal_density_si.png", dpi=180)
    plt.close(fig)

    summaries["meta"] = dict(
        L_values=L_VALUES, MC_SAMPLES=MC_SAMPLES, N_PER_L=N_PER_L,
        thresholds=P_C, A_SI=A_SI, R_VDW_SI=R_VDW_SI,
    )
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSummary  → {RESULTS_DIR / 'summary.json'}")
    print(f"Plots    → {RESULTS_DIR}/fig_*.png")


if __name__ == "__main__":
    main()
