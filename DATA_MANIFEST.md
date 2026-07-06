# Data manifest & provenance — OCE-Wave1 v1.1.0

All paths are relative to the deposit root. "Generated in this work" = produced by
the author for this article; "Derived" = OCE feature matrices computed here from a
third-party dataset; "External" = third-party, not redistributed (cite the source).

## code/

| Path | Contents | Provenance |
|---|---|---|
| `code/oce/` | OCE core package: feature construction, `fit`, `selection`, `forces`, `calculator`, `slater_koster`, tests | Generated in this work (MIT) |
| `code/scripts/` | `build_features.py`, `fit_subset.py`, `gpu_benchmark.py`, `wave1_benchmark.py` | Generated in this work (MIT) |
| `code/perovskites/` | Perovskite dataset build, active-learning iterations, SIESTA drivers, Δ-learning, Wave-1.5 fit/test | Generated in this work (MIT) |
| `code/percolation/` | Carbon percolation-cluster library: `lattices.py`, `percolation.py`, `volume.py`, LAMMPS/SIESTA runners, `pipelines/`, `tests/` | Generated in this work (MIT) |

## data/perovskites/ — SIESTA-PBE halide-perovskite dataset (generated in this work)

Supports article §3.2 (accuracy + active-learning protocol), §3.5 (Δ-learning /
strain extrapolation) and Table 5 / Figure 5 (cross-functional HSE06 check).

| Path | Contents | Size |
|---|---|---|
| `structures.json`, `siesta_*_summary.json`, `validation_100.json` | perovskite structures + SIESTA-PBE energies | — |
| `features.npz`, `features_ct2f.npz`, `features_wave15.npz` + `feature_index*.json` | OCE feature matrices (Base / CT2F / Wave-1.5) | ~19 MB |
| `active_learning_iter{1,2,3}*.json`, `learning_curve_summary.json` | active-learning selections + curve (§3.2) | — |
| `delta_learning/` | Δ-learning against xtb-GFN2 baseline (§3.5) | 96 KB |
| `batio3_casestudy/` | BaTiO₃ polymorph substitution case study (SIESTA + OCE features) | 25 MB |
| `siesta_binaries_outputs/` | raw SIESTA runs for the 15 binary reference states (CsX/KX/BX₂) used in the decomposition energies | 34 MB |
| `pseudos/*.psml` | ONCVPSP scalar-relativistic pseudopotentials (PSML) — **PseudoDojo**, redistributable | 6.1 MB |
| `wave15_comparison.json`, `ct2f_comparison.json`, `hse_endmember_*` , `mannodi_aligned.json` | summary tables reproduced in the article | — |

## data/percolation_clusters/ — carbon percolation clusters (generated in this work)

Supports article §3.3 (non-OCE use of the OCE geometry library). 53 JSON datasets
across phases 1–17 (2D/3D lattices, Si/Ge, coordination, bond, p-sweep, DFT reps) +
analysis results.

| Path | Contents |
|---|---|
| `data/phase1_*` … `data/phase17_*` | cluster geometries + energies per phase |
| `results/` | cross-element analysis, aggregated metrics (`cross_element_analysis.json`, …) |

## data/feature_caches/ — OCE features derived from external datasets

Only the **derived feature matrices** are shipped; the raw third-party datasets are
NOT redistributed (obtain from the original source and cite).

| Path | Derived from | External source (cite) | Size |
|---|---|---|---|
| `feature_caches/qmof/features*.npz` (+ index) | QMOF MOFs | Rosen et al., QMOF database | 104 MB |
| `feature_caches/carbon_2d/features*.npz` | 2D carbon allotropes (DFTB+) | generated in this work (features + labels) | 176 KB |
| `feature_caches/spice_dipeptides/features*.npz` | SPICE 2.0 dipeptide subset | Eastman et al., SPICE 2.0 | 1.6 MB |

## External datasets referenced but NOT included

| Dataset | Reason | Where to get it |
|---|---|---|
| SPICE 2.0 (~36 GB) | too large / third-party | original SPICE deposition |
| QMOF (full) | third-party | Rosen et al. deposition |
| PEPCONF | third-party | Prasad et al., *Sci. Data* 2019 (DOI 10.1038/sdata.2018.310) |
| Mannodi-Kanakkithodi ABX₃ | third-party | Yang et al., *Digital Discovery* 2023 (DOI 10.1039/D3DD00015J) |

## Notes / caveats

- **Pseudopotential redistribution**: the bundled `*.psml` are PseudoDojo/ONCVPSP,
  which permit redistribution. If your Zenodo license selection conflicts with the
  PseudoDojo terms, remove `data/perovskites/pseudos/` and point users to PseudoDojo.
- **Package version string**: `oce/__init__.py` carries an internal dev string
  (`__version__ = "0.1.0"`); the *deposited release* is **v1.1.0** (Wave-1.5),
  matching the article's Data & Code Availability statement.
