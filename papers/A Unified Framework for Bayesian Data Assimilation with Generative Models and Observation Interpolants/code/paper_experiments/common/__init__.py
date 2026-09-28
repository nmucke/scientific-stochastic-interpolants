"""Shared experiment infrastructure for the paper experiments.

Contents
--------
* :mod:`seeding`     -- the fixed seed list + per-(case, scenario, test) seeding.
* :mod:`aggregation` -- mean +/- std over seeds (reproducibility Section 9).
* :mod:`runner`      -- the ``ExperimentRunner`` base class every case driver
  implements; turns a config + seed list into tidy :class:`ResultRecord`s.
"""

from __future__ import annotations
