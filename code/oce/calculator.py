"""ASE Calculator wrapping a fitted OCE model.

Exposes ``energy`` and ``forces`` so an OCE model can drive
``ase.optimize`` (geometry relaxation) and ``ase.md`` (molecular
dynamics) without changes to the wider ASE ecosystem.

Usage
-----
>>> from oce.atomic_table import load_table
>>> from oce.fit import build_design_matrix, fit_model
>>> from oce.calculator import OCECalculator
>>> table = load_table("data/atoms/atomic_table.json")
>>> X, y, keys, _ = build_design_matrix(train_entries, table)
>>> model = fit_model(X, y, keys, alpha=1e-2)
>>> atoms.calc = OCECalculator(model=model, atomic_table=table)
>>> atoms.get_potential_energy()
>>> atoms.get_forces()

Topology handling
-----------------
By default the calculator re-perceives bonds at every ``calculate``
call (matches the existing :func:`oce.figures.enumerate_figures`
contract). For a force-stable optimisation/MD it is preferable to
hold the topology fixed at the initial geometry and recompute only
the geometric values inside each figure; that mode is selected with
``freeze_topology=True`` and is the recommended setting once the
analytic gradients in :mod:`oce.forces` are filled in (v1.1.0+).
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from oce.correlations import correlations_for_molecule
from oce.forces import FigureSet, compute_forces, freeze_topology


class OCECalculator(Calculator):
    """ASE Calculator returning OCE-predicted energies and forces."""

    implemented_properties = ['energy', 'forces']

    default_parameters = {
        'include_angles': True,
        'include_dihedrals': False,
        'freeze_topology': False,   # set True once analytic grads land
    }

    def __init__(self, model, atomic_table, **kwargs):
        """
        Parameters
        ----------
        model : oce.fit.OCEModel
            Fitted model with ``feature_keys`` and ``J``.
        atomic_table : dict
            Orbital ε table.
        include_angles, include_dihedrals : bool
            Which figure classes to enumerate.
        freeze_topology : bool
            If True, the bond/angle/dihedral connectivity is locked
            at the first ``calculate`` call. If False (default), it is
            re-perceived every call.
        """
        super().__init__(**kwargs)
        self.model = model
        self.atomic_table = atomic_table
        self._frozen: FigureSet | None = None

    # ------------------------------------------------------------ #
    # ASE Calculator interface                                      #
    # ------------------------------------------------------------ #

    def calculate(self, atoms=None, properties=('energy',),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        include_angles = self.parameters['include_angles']
        include_dihedrals = self.parameters['include_dihedrals']
        freeze = self.parameters['freeze_topology']

        # Resolve topology to use this step.
        if freeze:
            if self._frozen is None:
                self._frozen = freeze_topology(
                    atoms, self.atomic_table,
                    include_angles=include_angles,
                    include_dihedrals=include_dihedrals,
                )
            topology = self._frozen
        else:
            topology = freeze_topology(
                atoms, self.atomic_table,
                include_angles=include_angles,
                include_dihedrals=include_dihedrals,
            )

        # Energy: rebuild Π at the current geometry. We always re-enumerate
        # here because the geometric values stored in figure dataclasses
        # are only valid at the geometry at which they were perceived.
        topo_now = freeze_topology(
            atoms, self.atomic_table,
            include_angles=include_angles,
            include_dihedrals=include_dihedrals,
        )
        cv = correlations_for_molecule(
            topo_now.one, topo_now.two, self.atomic_table,
            three_figs=topo_now.three if include_angles else None,
            four_figs=topo_now.four if include_dihedrals else None,
        )
        self.results['energy'] = float(self.model.predict(cv))

        # Forces are cheap once the figures are enumerated, so compute
        # them unconditionally — this also avoids the ASE result-cache
        # trap where a previous calculate(['energy']) call would have
        # written zeros to self.results['forces'] and a subsequent
        # get_forces() would return the cached zeros without
        # triggering recomputation.
        self.results['forces'] = compute_forces(
            atoms, self.atomic_table, self.model,
            topology=topology,
        )

    # ------------------------------------------------------------ #
    # Convenience                                                   #
    # ------------------------------------------------------------ #

    def reset_topology(self) -> None:
        """Discard any frozen reference topology so the next
        ``calculate`` call re-perceives bonds. Only meaningful when
        the calculator was constructed with ``freeze_topology=True``.
        """
        self._frozen = None
