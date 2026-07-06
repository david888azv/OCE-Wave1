# Memory budget for site-percolation clusters on honeycomb at p_c

Universal 2D percolation fractal dimension: **D_f = 91/48 ≈ 1.896**.
Site-percolation threshold for honeycomb: **p_c ≈ 0.6970402** (Suding & Ziff 1999).

Largest connected cluster size at p_c on an L×L honeycomb (2L² sites total):
N_atoms ~ L^{D_f} = L^{91/48}.

| L  | 2L² (sites) | L^{1.896} (largest cluster, atoms) | xtb sp wall-time | OCE eval |
|----|-------------|------------------------------------|------------------|----------|
| 4  | 32          | ≈ 14                               | <1 s             | <1 ms    |
| 6  | 72          | ≈ 30                               | ~1 s             | <1 ms    |
| 8  | 128         | ≈ 51                               | ~2 s             | ~1 ms    |
| 12 | 288         | ≈ 109                              | ~10 s            | ~5 ms    |
| 16 | 512         | ≈ 188                              | ~30 s            | ~10 ms   |
| 24 | 1 152       | ≈ 411                              | ~2 min           | ~30 ms   |
| 32 | 2 048       | ≈ 705                              | ~5–10 min        | ~50 ms   |
| 48 | 4 608       | ≈ 1 542                            | ~30 min          | ~100 ms  |
| 64 | 8 192       | ≈ 2 685                            | ~1–2 h           | ~200 ms  |
| 96 | 18 432      | ≈ 5 762                            | ~6–10 h, ~30 GB  | ~500 ms  |
|128 | 32 768      | ≈10 005                            | ≳12 h, ≳60 GB    | ~1 s     |
|192 | 73 728      | ≈ 22 000                           | impractical xtb  | ~2 s     |
|256 | 131 072     | ≈ 38 000                           | impractical xtb  | ~3 s     |

System budget (this machine): **125 GiB RAM, 32 cores, xtb at /home/xtb/bin/xtb**.

xtb GFN2 memory scales O(N²) for the Fock matrix and O(N³) for diagonalisation;
empirical: ≈ 1 GB per 500 atoms. So ~30 GB at N=10⁴, 100+ GB at N=2×10⁴.

OCE evaluation memory is essentially negligible: design matrix is N_clusters × p_features
(p_features < 10² for pure C). The bottleneck is *xtb*, not OCE.

## Recommended L ranges

- **Phase 1 (small, xtb + OCE both):** L ∈ {4, 6, 8, 12, 16, 24, 32}.
- **Phase 1 large (OCE only, validated against xtb at L≤32):** L ∈ {48, 64, 96, 128}.
- **Asymptotic limit (OCE only, no xtb cross-check possible):** L ∈ {192, 256, 384, 512}.

For L ≥ 192 the largest cluster has ≳ 2×10⁴ atoms, and xtb would need
multi-day CPU time but **OCE evaluation finishes in seconds**.  This is the
regime where the cluster expansion shines and where any genuine power-law
behaviour `E/N ~ L^{-α}` can be sampled cleanly.

## Power-law expectations

Energy of a 2D fractal cluster with surface = boundary perimeter:

  E_total(L)   = ε_bulk · N(L) + ε_surf · S(L)
              = ε_bulk · L^{D_f} + ε_surf · L^{D_s}

where D_s is the *boundary fractal dimension*. For 2D incipient percolation
clusters the cluster *hull* (external perimeter) has D_h = 4/3 (Saleur–Duplantier)
and the full boundary (with internal holes) has D_e = 7/4.

Per atom:

  E/N(L) = ε_bulk + ε_surf · L^{D_s − D_f}
         ≈ ε_bulk + (const) · L^{D_h − D_f}     where D_h − D_f = 4/3 − 91/48 = -19/48 ≈ -0.396
         ≈ ε_bulk + (const) · L^{D_e − D_f}     where D_e − D_f = 7/4 − 91/48 = -7/48 ≈ -0.146

So we expect:

  **E/N(L) − ε_bulk ~ L^{-α}** with α between ≈ 0.15 and 0.40,
  depending on whether the perimeter is hull-only (radical-counting
  type) or full-boundary (every dangling bond).
