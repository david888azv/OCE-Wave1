# Changelog — `versions/`

One line per snapshot release. Newest at the top.

## 2026-05-08 (evening)

- `oce_v1.1.1`            — Patch. Fixes critical bug in
                            OCECalculator: ``self.results['forces']``
                            was pre-seeded with zeros, causing
                            ``get_forces()`` after a previous
                            ``get_potential_energy()`` to return
                            cached zeros. Forces now always computed.
                            Adds regression test
                            ``test_calculator_forces_nonzero_when_perturbed``
                            (10/10 PASS). Adds new experiment
                            ``oce/optimization_demo.py`` running the
                            production ridge model under ASE BFGS;
                            output in ``results/optimization_demo.log``.
                            Empirical finding: 1F+2F basis fits xtb
                            energies to ~25 meV but |F|_max ~ 15 eV/Å
                            at xtb equilibria — the basis lacks
                            short-range repulsion, so BFGS collapses
                            bonds. Documents structural motivation
                            for Phase 2 (Slater-Koster + repulsion).

## 2026-05-08 (afternoon)

- `oce_v1.1.0`            — Phase 1 of JCTC roadmap. Analytic forces
                            for 2F (Wolfsberg-Helmholz chain rule)
                            and 3F (cos θ Bekker formulae). New
                            modules forces.py + calculator.py +
                            test_forces.py (9/9 PASS, ≤1e-3 eV/Å
                            agreement vs central-diff on H2O/CH4/
                            C2H4 at eq + perturbed geometries).
                            Enables ASE BFGS/FIRE/MD without
                            finite-difference cost. 4F dihedral
                            gradient still mock — deferred to v1.1.2.
                            **Note: shipped with calculator caching
                            bug; superseded by v1.1.1 same day.**

## 2026-05-08 (morning)

- `oce_v1.0.0`            — JPCL submission baseline. 1F/2F/3F/4F
                            basis with Wolfsberg-Helmholz 2F and
                            Madelung column. 4/4 isomers correct.
- `oce_protein_v1.0.0`    — SPICE driver. Dipeptides 27 meV/atom,
                            DES370K 588 → 488 meV/atom (Madelung
                            ablation, J_M = 0.84).
- `oce_carbon_v1.0.0`     — Day-0 functional pipeline for 2D
                            carbon allotropes (1115 CIFs) on
                            DFTB+ engine.

## Planned

- `oce_v1.1.2`            — 4F dihedral gradient (∂cos 2φ/∂r) with
                            sign convention cross-validated against
                            figures.enumerate_figures.
- `oce_v1.2.0`            — Slater–Koster directional 2-centre
                            integrals (ssσ, spσ, ppσ, ppπ);
                            smooth Madelung damping; MBIS charges
                            promoted to package level.
- `oce_protein_v1.1.0`    — picks up forces; adds NVE MD validation
                            on dipeptides + pentapeptide.
- `oce_protein_v1.2.0`    — hierarchical transferability test
                            (di → tri → tetra) and active-learning
                            sampling.
