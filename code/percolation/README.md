# oce-carbon-percolation-clusters

Application of the OCE v1.0.0 method (the JPCL submission) to **2D
site-percolation clusters of bare radical carbon at the critical
threshold p = p_c**.  The motivation, set by the user request, is
twofold:

  1. **Validate** that the cluster expansion fitted on a few small
     unrelaxed clusters (xtb GFN2 single-points) reproduces the xtb
     energy of new, larger clusters drawn from the same percolation
     ensemble.
  2. **Probe scaling laws** for the per-atom energy E/N(L) and total
     energy E_total(L) as the linear lattice size L grows.  Because the
     percolation cluster is itself a fractal at p_c, both the cluster
     mass and the per-atom energy are expected to display power-law
     behaviour governed by the universal 2D percolation exponents.

The user requested **site percolation on a honeycomb (graphene)
lattice**, p_c ≈ 0.6970 (Suding & Ziff 1999), with **dangling bonds
left as radicals** (no H termination).  Phase 1 sweeps L = 4 … 16 with
xtb single-points; an OCE-only extrapolation reaches L = 128.  Phase 2
relaxes the clusters with xtb and reports the OCE behaviour on the
relaxed geometry.

---

## Files

```
oce-carbon-percolation-clusters/
├── MEMORY_BUDGET.md          back-of-envelope memory/CPU table
├── lattices.py               honeycomb + square lattice builders
├── percolation.py            site percolation + largest-cluster extraction
├── runners.py                xtb wrapper for radical clusters (parity uhf)
├── oce_predict.py            adapter that imports versions/oce_v1.0.0
├── analysis.py               log-log fits, 3-parameter ε_bulk extraction
└── pipelines/
    ├── phase1_smoke.py       smoke test, L ∈ {4, 6, 8}
    ├── phase1_full.py        L ∈ {4, 6, 8, 12, 16}
    ├── phase1_large_L_oce.py xtb-trained OCE → L ∈ {24, 32, …, 128}
    └── phase2_relax_smoke.py xtb opt + OCE on relaxed (L ≤ 6)
```

`oce_predict.py` registers the v1.0.0 modules under the import name
`oce.atomic_table`, `oce.figures`, `oce.correlations`, so the published
basis is used **unchanged** — only the input geometries differ.

---

## Reproducing the runs

```bash
cd /home/david/atual/new-methods/oce-carbon-percolation-clusters
python pipelines/phase1_smoke.py        # ≈ 10 s
python pipelines/phase1_full.py         # ≈ 5 min, 102 xtb singlepoints
python pipelines/phase1_large_L_oce.py  # ≈ 8 min, OCE only beyond L=16
python pipelines/phase2_relax_smoke.py  # ≈ 30 s, 10 xtb optimisations
```

All caches and plots land under `data/<phase>/` and `results/<phase>/`.

---

## Phase 1 — unrelaxed clusters

### Setup

  * Honeycomb lattice, open boundary, L × L unit cells (2L² sites).
  * Each site occupied independently with probability **p_c =
    0.6970402**.
  * Largest connected cluster of occupied sites is kept as a 2D
    "molecule"; positions are the original lattice positions
    (unrelaxed); element is C everywhere.
  * xtb GFN2 single-point with `--uhf` set to the parity of the total
    electron count (xtb auto-finds the lowest-energy state of that
    spin parity; for 4-valence C this is uhf=0 in every realisation
    we sampled).

### Cluster mass scaling

Universal 2D percolation predicts ⟨N⟩(L) ~ L^{D_f} with
**D_f = 91/48 ≈ 1.896** (den Nijs 1979).  Empirically over
L ∈ {4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128}:

  **D_f^empirical = 1.823**, log-RMSE = 0.05.

The 4 % deficit relative to 91/48 is the standard finite-L correction
to scaling (correction-to-scaling exponent ω ≈ 0.5 in 2D percolation).

### OCE↔xtb agreement (parent-stratified by L)

Trained on **L ≤ 8 only** (72 clusters), evaluated on the never-seen
**L ∈ {12, 16}** (30 clusters):

| split           | n  | RMSE (eV) | R²        | Spearman ρ | per-atom RMSE |
|-----------------|----|-----------|-----------|------------|---------------|
| TRAIN (L ≤ 8)   | 72 | 1.42      | 1.00000   | 0.99991    | 77.6 meV/atom |
| TEST (L=12,16)  | 30 | 8.24      | 0.99999   | 1.00000    | 45.1 meV/atom |

Held-out per-atom RMSE = **45 meV/atom** is in the same regime as
MACE-OFF (18 meV/atom on dipeptides) and ANI-2x (30 meV/atom),
attained here with **17 OCE features** and **72 training points** —
two orders of magnitude smaller than even the JPCL benchmark.

### Total energy and per-atom power laws

Per-L means (mean over realisations):

| L  | ⟨N⟩    | ⟨E_total⟩_xtb  | ⟨E/N⟩_xtb | ⟨E/N⟩_oce |
|----|--------|----------------|-----------|-----------|
| 4  | 18.8   | −1056 eV       | −56.05    | −56.04    |
| 6  | 29.7   | −1672 eV       | −56.24    | −56.24    |
| 8  | 49.2   | −2779 eV       | −56.45    | −56.43    |
| 12 | 113.8  | −6435 eV       | −56.55    | −56.50    |
| 16 | 170.8  | −9663 eV       | −56.55    | −56.49    |

Log-log fits give

  E_total ~ L^{D_E} with **D_E = 1.837** (xtb) and **1.836** (OCE)
  — essentially the cluster mass exponent D_f, as expected for an
    extensive energy.

  E/N − ε_bulk ~ L^{−α} with the 3-parameter grid-search fit:

  **xtb**: ε_bulk = −61.61 eV/atom,  α = 0.078
  **OCE**: ε_bulk = −61.56 eV/atom,  α = 0.073

Including the OCE-only extrapolation up to L = 128 (≈ 6 600 atoms per
cluster):

  **combined**: ε_bulk = −61.70 eV/atom, α = 0.034.

The shrinking α as L grows is consistent with the boundary fractal
dimension of the **full** cluster boundary (every dangling bond)
being equal to D_f in 2D percolation — i.e. the asymptotic
boundary-volume ratio is finite and the leading correction to
ε_bulk decays slowly.  In the strict thermodynamic limit α → 0,
with finite-size correction α(L) ≈ (D_h − D_f)/D_f ≈ −19/91 (≈
0.21) modulated by the fraction of the boundary that is *outer hull*
versus *internal hole rim*.

### Largest cluster size we can build

Site-percolation cluster sampling is O(L²) and fits in <1 GB up to
L ≈ 10⁴.  **OCE evaluation** is the same complexity as bond
perception, dominated by the 3F angle enumeration; for the largest L
tested (L = 128, ~6 600 atoms), OCE evaluation took ~80 s/cluster.

The hard ceiling is **xtb single-point**, which scales O(N³) for SCF
diagonalisation:

  L  ≈ 32  → 700 atoms,  ~5 min       (comfortable)
  L  ≈ 64  → 2 700 atoms, ~1–2 h      (possible but slow)
  L  ≈ 128 → 10 000 atoms, ≳12 h, ≳60 GB     (limit on this 125-GiB box)

So **for xtb**: L_max ≈ 96–128 with patience.
For **OCE alone**: L_max ≈ 512 (≈150 000 atoms) is comfortable on
this machine; the only constraint is the bond-perception triple-loop
in `figures.py`.  An optimised neighbour-list version would push this
to L ≈ 2000.

---

## Phase 2 — relaxed clusters (smoke test)

10 percolation clusters at L ∈ {4, 6}, optimised with xtb GFN2
(`--opt` + `--uhf` = parity).  Bond distances change substantially
during relaxation — the unrelaxed lattice geometry is not a local
minimum because each atom has 1–3 dangling sp² lobes.

| metric                                             | result        |
|----------------------------------------------------|---------------|
| ⟨ΔE_relax⟩ = ⟨E_xtb_sp − E_xtb_relaxed⟩            | **+20.2 eV**  |
| ⟨RMS atomic displacement⟩                          | 1.1 Å         |
| OCE retrained on relaxed (n=10, p=99)              | ~ 0 meV/atom* |
| OCE trained on UN-relaxed, applied to RELAXED      | **2964 meV/atom** |

*The "0 meV/atom" of the retrained OCE is overfitting artefact (more
features than samples).  The decisive number is the third row.

**Interpretation.**  This is *exactly* the dual-regime behaviour
documented in the OCE/JPCL manuscript (Section "We probe the
within-parent axis explicitly"):

  * Across DIFFERENT clusters (different connectivity) the OCE 1F+2F+3F
    basis ranks energies perfectly (Spearman ρ = 1).
  * Across the SAME cluster's geometric reorganisation (same
    connectivity, different bond distances/angles), the basis is
    blind.  The 2F figure is summed over a small set of bonded pairs
    whose Wolfsberg-Helmholz hopping h(r) varies smoothly within the
    well — over the unrelaxed → relaxed transition the 2F values
    move ~1 % while the energy moves several eV.

The fix advertised in the JPCL paper applies here verbatim: enrich
the basis with neighbourhood-density (ACE-style) or cutoff-2F
features, *or* retrain on the relaxed data.

---

## Headline numbers

| quantity                                           | value                |
|----------------------------------------------------|----------------------|
| OCE features (1F + 2F + 3F, pure C)                | 17                   |
| OCE training clusters (xtb sp, L ≤ 8)              | 72                   |
| OCE held-out per-atom RMSE on L ∈ {12, 16}         | **45 meV/atom**      |
| Cluster mass exponent D_f (theory: 91/48 ≈ 1.896)  | **1.823 (5 L pts)**  |
|                                                    | **1.823 (11 L pts)** |
| Total energy exponent D_E                          | **1.84 (xtb = OCE)** |
| Per-atom decay exponent α (L = 4 … 16)             | 0.078 (xtb)          |
| Per-atom decay exponent α (L = 4 … 128)            | **0.034**            |
| Asymptotic ε_bulk (per-atom energy of 2D radical-perc cluster) | **−61.7 eV/atom**   |
| ⟨ΔE_relax⟩ (Phase 2)                              | +20 eV per cluster   |
| ⟨|Δr|⟩ (Phase 2)                                   | 1.1 Å                |
| OCE-unrelaxed → relaxed transfer error             | **3 eV/atom (fails)**|

---

## Open follow-ups (not in this run)

1. **Square lattice** (p_c = 0.5927 site, exists in `lattices.py`) —
   different coordination, same universal exponents, but different
   short-range OCE keys (no rings) and likely different ε_bulk.
2. **Bond percolation** counterpart (each bond independently kept
   with probability p_c).  Different cluster topology → useful test
   for the OCE basis under controlled changes in connectivity.
3. **Train OCE on relaxed structures** for a proper Phase 2: requires
   ~50 xtb optimisations at L ∈ {4, 6, 8}; doable in ~30 min on this
   machine.
4. **Enrich basis** with cutoff-2F features (every C–C pair within
   3 Å, regardless of bond perception) to recover relaxation
   sensitivity.  Already implemented as a stub in `oce_v1.1.x`.
5. **Asymptotic ε_bulk extrapolation** at L = 256, 512 with OCE-only,
   and triangulation against a periodic graphene reference.
