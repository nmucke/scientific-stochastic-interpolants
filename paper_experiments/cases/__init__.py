"""Case driver packages: analytical, navier_stokes, urban.

Each subpackage exposes a ``driver.py`` with an :class:`ExperimentRunner`
subclass that produces tidy :class:`ResultRecord`s for its case. The scientific
internals (samplers, metrics) are pulled from ``src/scisi`` and are mid-rebuild;
the drivers mark every such seam with a ``TODO`` referencing the GAP item.
"""

from __future__ import annotations
