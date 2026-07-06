# OCE-Wave1 — v1.1.0

Reproducibility package accompanying the *J. Chem. Theory Comput.* (JCTC) article:

> **Geometric Cutoff Features and Active Learning Extend Orbital Cluster
> Expansions Across Halide Perovskites, Metal–Organic Frameworks,
> Two-Dimensional Carbon Allotropes, and Biomolecular Conformers**
> D. L. Azevedo, Institute of Physics, University of Brasília (UnB).
> ORCID 0000-0002-3456-554X.

> **This GitHub repository holds the code only.** The datasets generated in this
> work (SIESTA-PBE halide-perovskite energies, carbon percolation clusters) and the
> pre-computed feature caches are large (200+ MB, one file >100 MB) and are archived
> together with this code on **Zenodo: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)**
> (v1.1.0). Download the Zenodo bundle to reproduce every table and figure.

This repository contains the **code** (OCE feature-construction library and the
training / evaluation / active-learning scripts). The **datasets generated in this
work** (SIESTA-PBE halide-perovskite energies and carbon percolation clusters) and
the **pre-computed feature caches** are archived on Zenodo (see banner above).

It builds on the OCE method introduced in the foundational letter
(D. L. Azevedo, *J. Phys. Chem. Lett.* **2026**, DOI
[10.1021/acs.jpclett.6c01491](https://doi.org/10.1021/acs.jpclett.6c01491);
code archived separately as `OCE-SPICE`,
[10.5281/zenodo.20068241](https://doi.org/10.5281/zenodo.20068241)).

## Layout

```
code/
  oce/                    OCE core package (feature construction, fit, selection, forces, …)
  scripts/                top-level build/fit/benchmark drivers (build_features.py, fit_subset.py, …)
  perovskites/            halide-perovskite pipeline (dataset build, active learning, SIESTA drivers, Δ-learning)
  percolation/            carbon percolation-cluster library (lattices, percolation, LAMMPS/SIESTA runners, tests)
data/
  perovskites/            SIESTA-PBE perovskite dataset generated in this work
                          (structures, energies, features, Δ-learning, BaTiO3 case study,
                           reference-binary SIESTA outputs, pseudopotentials)
  percolation_clusters/   carbon percolation-cluster datasets + analysis results (phases 1–17)
  feature_caches/         pre-computed OCE feature matrices (.npz) for the externally
                          sourced benchmarks (QMOF, 2D-carbon, SPICE dipeptides)
DATA_MANIFEST.md          file-by-file provenance and which article section each item supports
CHANGELOG.md              version history
LICENSE                   MIT (code); see DATA_MANIFEST.md for data/pseudo provenance
CITATION.cff              how to cite this deposit and the article
```

## Datasets NOT redistributed here (external, cite the original source)

- **SPICE 2.0** biomolecular DFT dataset (Eastman et al.) — ~36 GB, obtain from the
  original deposition; only the OCE feature caches derived from its dipeptide subset
  are included under `data/feature_caches/spice_dipeptides/`.
- **QMOF** database (Rosen et al.) — obtain from the original deposition; only the
  derived OCE feature caches are included under `data/feature_caches/qmof/`.
- **PEPCONF** peptide-conformer set (Prasad et al.) and the **Mannodi-Kanakkithodi
  ABX₃** halide-perovskite alloy database (Yang et al.) — cited in the article; not
  redistributed.

## Pseudopotentials

`data/perovskites/pseudos/*.psml` are ONCVPSP scalar-relativistic pseudopotentials
in PSML format, from the **PseudoDojo** project (openly redistributable). They are
bundled for turnkey reproduction of the SIESTA reference calculations.

## Reproducing the key numbers

The article's RMSE / ranking numbers are backed by the JSON summaries shipped here,
e.g. `data/perovskites/ct2f_comparison.json`,
`data/perovskites/learning_curve_summary.json`,
`data/perovskites/wave15_comparison.json`, and the percolation
`data/percolation_clusters/results/`. See `DATA_MANIFEST.md` for the full map.

## License

Code: MIT (see `LICENSE`). Data generated in this work: CC-BY-4.0.
Bundled PseudoDojo pseudopotentials retain their original PseudoDojo license.
