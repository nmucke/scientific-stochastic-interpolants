"""Per-(assimilation-)step metric persistence (long format).

The scalar tidy file (``results_schema``) holds ONE value per (case, method,
scenario, variant, metric, E, M, seed) cell -- the mean over assimilation steps.
For figures and step-resolved tables we also want the *curve*: the metric at each
assimilation step. This module writes that curve as a compact long-format CSV,
one row per (cell, metric, step), so trajectories 2..N can keep their full metric
history WITHOUT paying the cost of saving the raw ensemble states (which is done
only for the first trajectory).

Columns::

    case, method, scenario, variant, metric, step, value, E, M, seed,
    test_index, NFE, seconds

* ``step`` is the assimilation-step index (0-based; step 0 is the first scored
  physical step after the seeded history prefix).
* ``value`` is the metric at that step.
* ``NFE`` / ``seconds`` are the per-step cost of the whole run (constant across
  the rows of one cell) so every row is self-describing about timing -- the user
  asked for timings on every run.

Append-only, CSV only, mirrors ``results_schema.ResultsWriter`` conventions so a
single run invocation streams its rows into one ``per_step_file`` without a global
rewrite.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# On-disk column order for the per-step file.
PER_STEP_FIELDNAMES: tuple[str, ...] = (
    "case",
    "method",
    "scenario",
    "variant",
    "metric",
    "step",
    "value",
    "E",
    "M",
    "seed",
    "test_index",
    "NFE",
    "seconds",
)


def per_step_rows(
    *,
    case: str,
    method: str,
    scenario: str,
    variant: str | None,
    E: int | None,
    M: int | None,
    seed: int,
    test_index: int | None,
    nfe: float | None,
    seconds: float | None,
    per_step: Mapping[str, Sequence[float]],
) -> list[dict[str, object]]:
    """Flatten a ``compute_metrics``-style ``per_step`` dict into tidy rows.

    ``per_step`` maps a metric name to its list of per-step values (as produced by
    the case pipelines' ``compute_metrics``). Metrics with an empty list (e.g.
    ``crps_unobserved`` under a full-observation super-res operator) are skipped.
    """
    rows: list[dict[str, object]] = []
    for metric, values in per_step.items():
        for step, value in enumerate(values):
            rows.append(
                {
                    "case": case,
                    "method": method,
                    "scenario": scenario,
                    "variant": variant,
                    "metric": metric,
                    "step": step,
                    "value": float(value),
                    "E": E,
                    "M": M,
                    "seed": seed,
                    "test_index": test_index,
                    "NFE": nfe,
                    "seconds": seconds,
                }
            )
    return rows


class PerStepWriter:
    """Append-only CSV writer for per-step metric rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None
        self._writer: csv.DictWriter | None = None

    def __enter__(self) -> "PerStepWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(PER_STEP_FIELDNAMES))
        if new_file:
            self._writer.writeheader()
        return self

    def extend(self, rows: Iterable[dict[str, object]]) -> None:
        assert self._writer is not None
        for row in rows:
            self._writer.writerow(
                {k: ("" if row.get(k) is None else row.get(k)) for k in PER_STEP_FIELDNAMES}
            )

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def append_per_step(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    """Convenience: append ``rows`` to the per-step CSV at ``path``."""
    path = Path(path)
    with PerStepWriter(path) as w:
        w.extend(rows)
    return path


def load_per_step(path: str | Path) -> list[dict[str, object]]:
    """Load every per-step row from a CSV (cells kept as strings/typed where easy)."""
    path = Path(path)
    out: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(dict(row))
    return out


__all__ = [
    "PER_STEP_FIELDNAMES",
    "per_step_rows",
    "PerStepWriter",
    "append_per_step",
    "load_per_step",
]
