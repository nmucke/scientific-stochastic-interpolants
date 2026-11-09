"""Shared bookkeeping for (possibly nonlinear) observation scenarios (NS + urban).

Pure stdlib + duck typing (no torch import), so the table pipeline can import it.

* :func:`scenario_key` -- the scenario label/key used for seeds
  (``seed_scenario``) or hyperparameter lookup (``hparam_scenario``). A linear
  scenario sets neither key, so it gets its own label back unchanged.
* :func:`states_tag` -- the filename tag of a saved-states ``.npz`` (``__c<c>``
  for a strength-carrying nonlinear operator, ``""`` otherwise). Used by the
  WRITER (``_save_states``) and the READER (the EnKF KL-reference glob), so the
  two can never drift apart.
* :func:`states_npz_fields` -- the observation-operator metadata stored in the
  ``.npz``.
* :func:`strength_results_path` -- ``results_file`` with a ``__c<c>`` suffix when
  ``+nonlinearity_strength`` is set, so the rows of a c-sweep do not mix.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_WARNED: set[tuple[str, str]] = set()


def scenario_key(
    scenario_cfg: Any, field: str, default: str, *, warn: bool = False
) -> str:
    """``scenario_cfg[field]`` if set, else ``default``.

    ``field`` is ``"seed_scenario"`` (label the mask / obs-noise seeds are derived
    from, so a nonlinear scenario can reuse its linear parent's draws verbatim)
    or ``"hparam_scenario"`` (key into the method YAMLs' ``[case][scenario][M]``
    tables). With ``warn=True`` a reuse of another scenario's key is logged once.
    """
    key = scenario_cfg.get(field, None) if scenario_cfg is not None else None
    if not key or str(key) == str(default):
        return str(default)
    if warn and (str(default), str(key)) not in _WARNED:
        _WARNED.add((str(default), str(key)))
        logger.warning(
            "Scenario %r reuses the %r hyperparameters (%s) -- tuned for a "
            "different observation operator; re-tune before quoting numbers.",
            default, str(key), field,
        )
    return str(key)


def _strength(obs_operator: Any) -> Optional[float]:
    """Strength ``c`` of a nonlinear operator that has one (pointwise ``g``)."""
    if getattr(obs_operator, "is_linear", True):
        return None
    c = getattr(obs_operator, "strength", None)
    return None if c is None else float(c)


def states_tag(obs_operator: Any) -> str:
    """Saved-states filename tag: ``__c<c>`` for a strength-carrying nonlinear
    operator, ``""`` for every linear operator (filenames unchanged)."""
    c = _strength(obs_operator)
    return "" if c is None else f"__c{c:g}"


def states_npz_fields(obs_operator: Any, hparam_scenario: str) -> dict[str, Any]:
    """Observation-operator metadata stored in a saved-states ``.npz``."""
    c = _strength(obs_operator)
    return {
        "obs_linear": bool(getattr(obs_operator, "is_linear", True)),
        "obs_nonlinearity": str(getattr(obs_operator, "nonlinearity", "") or ""),
        "obs_strength": float("nan") if c is None else c,
        "hparam_scenario": str(hparam_scenario),
    }


def strength_results_path(path: str | Path, strength: Any) -> Path:
    """``results_file`` with ``__c<c>`` before the suffix when ``strength`` is set.

    ``ns_arctan.csv`` + ``+nonlinearity_strength=2`` -> ``ns_arctan__c2.csv``.
    Unchanged when no strength override is given (every linear run) or when the
    name already ends in that tag. Only ``results_file`` is suffixed: give each c
    its own ``per_step_file`` if per-step curves are saved.
    """
    path = Path(path)
    if strength is None:
        return path
    tag = f"__c{float(strength):g}"
    if path.stem.endswith(tag):
        return path
    return path.with_name(f"{path.stem}{tag}{path.suffix}")


# ``<stem>__c<c>.<ext>`` as written by :func:`strength_results_path`.
STRENGTH_FILE_RE = re.compile(r"__c(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\.\w+$")


def strength_from_filename(name: str) -> Optional[str]:
    """The ``c`` tag of a strength-suffixed results file (``None`` otherwise)."""
    m = STRENGTH_FILE_RE.search(name)
    return m.group(1) if m else None


__all__ = [
    "scenario_key",
    "states_tag",
    "states_npz_fields",
    "strength_results_path",
    "strength_from_filename",
]
