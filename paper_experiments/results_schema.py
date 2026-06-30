"""Canonical tidy results schema for the paper experiments.

This module is the single source of truth for the format every case driver must
emit. It defines:

* :class:`ResultRecord` -- one tidy row, matching the spec columns exactly
  (Section 8 of ``paper_new/EXPERIMENTS_IMPLEMENTATION_SPEC.md``):
  ``case, method, scenario, metric, value, std, E, M, seed, NFE, seconds``.
* Canonical method / scenario / metric / case name constants so every case
  labels its rows identically. ``make_tables.py`` keys off these exact strings.
* An append-only writer and a loader (CSV and JSON-lines), proven to round-trip.

The tidy file is intentionally long-format (one metric value per row). One row
holds the mean (``value``) and the across-seed standard deviation (``std``); the
``seed`` column records which seed produced an *un-aggregated* row, or is set to
``SEED_AGGREGATED`` (-1) for an already-aggregated mean+/-std row. See
``common/aggregation.py`` for the mean+/-std-over-seeds reduction (reproducibility
Section 9).

Nothing here imports torch or ``scisi`` -- it is pure stdlib so the table
pipeline runs anywhere, including before the sampler rebuild lands.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

# --------------------------------------------------------------------------- #
# Canonical case names
# --------------------------------------------------------------------------- #


class Case(str, Enum):
    """The three test cases. Value is the tidy-file ``case`` string."""

    ANALYTICAL = "analytical"
    NAVIER_STOKES = "navier_stokes"
    URBAN = "urban"


# --------------------------------------------------------------------------- #
# Canonical method names
# --------------------------------------------------------------------------- #


class Method(str, Enum):
    """Canonical method labels. Value is the tidy-file ``method`` string.

    These map 1:1 to the row labels in ``sections/results.tex`` (see
    ``make_tables.METHOD_ROW_ORDER`` and ``METHOD_LATEX_LABEL``).
    """

    # ---- FINAL PAPER METHOD LINEUP -------------------------------------- #
    # Our three samplers (unified family; differ only in g_tau / w_tau / source).
    OURS_SI_SDE = "Ours (SI-SDE)"
    OURS_FM_ODE = "Ours (FM-ODE)"
    OURS_FM_SDE = "Ours (FM-SDE)"  # FM-SDE (a diffusion-model sampler); shown as "FM-SDE (DM)"

    # SI prior + SDE.
    FLOWDAS = "FlowDAS"

    # Flow-matching prior + ODE.
    GUIDED_FM_FIG = "Guided FM (FIG)"  # FIG measurement-interpolant guidance (yan_fig_2025)
    GUIDED_FM_OTODE = "Guided FM (OT-ODE)"  # OT-ODE guided flow (pokle_training-free_2024)
    D_FLOW_SGLD = "D-Flow SGLD"  # D-Flow (ben-hamu_d-flow_2024) optimised with SGLD -- TODO implement

    # Diffusion-model prior + SDE.
    SDA = "SDA"  # score-based DA (rozet_score-based_2023), single-window, on the DM prior
    SURGE = "SURGE"  # TODO implement

    # Classical data assimilation (ground-truth EnKF + conventional baselines).
    ENKF = "EnKF"
    LETKF = "LETKF"
    PARTICLE_FILTER = "Particle filter"
    ENSEMBLE_SCORE_FILTER = "Ensemble score filter"

    # ---- DEPRECATED (kept for back-compat with old result files; NOT in the
    # final paper lineup, removed from the run registry in driver.NS_METHODS). #
    GUIDED_FM = "Guided FM"  # legacy simple guided FM (yan_fig_2024)
    GUIDED_DIFFUSION = "Guided diffusion"  # DPS (chung_diffusion_2023)


METHODS: tuple[Method, ...] = tuple(Method)
"""All canonical methods, in declaration order."""


# --------------------------------------------------------------------------- #
# Canonical scenario names
# --------------------------------------------------------------------------- #


class Scenario(str, Enum):
    """Canonical observation-scenario labels. Value is the tidy-file string."""

    SUPERRES_32 = "32^2->128^2"
    SUPERRES_16 = "16^2->128^2"
    SPARSE_5 = "sparse 5%"
    SPARSE_1p5 = "sparse 1.5625%"
    ANALYTICAL = "analytical"  # Case 1 has a single (joint) scenario.


SCENARIOS: tuple[Scenario, ...] = tuple(Scenario)
"""All canonical scenarios, in declaration order."""


# --------------------------------------------------------------------------- #
# Canonical metric names
# --------------------------------------------------------------------------- #


class Metric(str, Enum):
    """Canonical metric keys. Value is the tidy-file ``metric`` string.

    Definitions live in Section 3 of the spec. ``make_tables.py`` maps a
    (method, scenario, metric) triple to a specific LaTeX cell.
    """

    # (a) Point accuracy.
    RMSE = "rmse"  # ensemble-mean RMSE (NS vorticity / generic)
    RMSE_VELOCITY = "rmse_velocity"  # urban, per-variable
    RMSE_TEMPERATURE = "rmse_temperature"  # urban, per-variable
    ENERGY_SPEC_RMSE = "energy_spec_rmse"  # log-spectrum RMSE

    # (b) Probabilistic calibration.
    CRPS = "crps"  # CRPS over the whole field (all grid points)
    CRPS_OBSERVED = "crps_observed"  # CRPS at observed grid points only
    CRPS_UNOBSERVED = "crps_unobserved"  # CRPS at unobserved grid points only
    SPREAD_SKILL = "spread_skill"  # report |1 - spread/skill| (0 = calibrated)

    # (c) Distributional fidelity.
    KL_POINTS = "kl_points"  # KL at points (fields + analytical)
    SLICED_W2 = "sliced_w2"  # Case 1 full joint

    # (d) Cost.
    NFE = "nfe"  # network evals per assimilation step
    SECONDS = "seconds"  # wall-clock s/step at matched E


METRICS: tuple[Metric, ...] = tuple(Metric)
"""All canonical metrics, in declaration order."""


# Sentinel seed for a row that already holds a mean +/- std reduced over seeds.
SEED_AGGREGATED: int = -1


# --------------------------------------------------------------------------- #
# The tidy record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ResultRecord:
    """One tidy results row.

    Columns are exactly the spec set (Section 8):
    ``case, method, scenario, metric, value, std, E, M, seed, NFE, seconds``.

    Notes
    -----
    * ``value`` is the metric value (a mean over seeds if ``seed ==
      SEED_AGGREGATED``).
    * ``std`` is the across-seed standard deviation; ``None`` for a single-seed
      (un-aggregated) row.
    * ``E`` (ensemble size) and ``M`` (pseudo-time steps) are sampler settings.
    * ``nfe`` and ``seconds`` are the cost columns; they are *also* expressible
      as their own metric rows (:attr:`Metric.NFE`, :attr:`Metric.SECONDS`) so
      that cost can flow through the same table machinery. Carrying them on
      every row as well keeps each accuracy row self-describing about cost.
    """

    case: str
    method: str
    scenario: str
    metric: str
    value: float
    std: float | None = None
    E: int | None = None
    M: int | None = None
    seed: int = SEED_AGGREGATED
    nfe: float | None = None
    seconds: float | None = None

    # Column order on disk. ``NFE``/``seconds`` keep their spec capitalisation in
    # the header while the dataclass field is lowercase (``nfe``) for Python style.
    # ClassVar so it is not treated as a dataclass field.
    FIELDNAMES: ClassVar[tuple[str, ...]] = (
        "case",
        "method",
        "scenario",
        "metric",
        "value",
        "std",
        "E",
        "M",
        "seed",
        "NFE",
        "seconds",
    )

    def to_row(self) -> dict[str, object]:
        """Return a dict keyed by the on-disk column names."""
        return {
            "case": self.case,
            "method": self.method,
            "scenario": self.scenario,
            "metric": self.metric,
            "value": self.value,
            "std": self.std,
            "E": self.E,
            "M": self.M,
            "seed": self.seed,
            "NFE": self.nfe,
            "seconds": self.seconds,
        }

    @classmethod
    def from_row(cls, row: dict[str, object]) -> "ResultRecord":
        """Inverse of :meth:`to_row`; tolerant of string-typed CSV cells."""
        return cls(
            case=str(row["case"]),
            method=str(row["method"]),
            scenario=str(row["scenario"]),
            metric=str(row["metric"]),
            value=_as_float(row["value"]),
            std=_as_opt_float(row.get("std")),
            E=_as_opt_int(row.get("E")),
            M=_as_opt_int(row.get("M")),
            seed=_as_opt_int(row.get("seed")) or SEED_AGGREGATED,
            nfe=_as_opt_float(row.get("NFE")),
            seconds=_as_opt_float(row.get("seconds")),
        )


# --------------------------------------------------------------------------- #
# Parsing helpers (CSV cells are strings; JSON cells may be typed or null)
# --------------------------------------------------------------------------- #


def _is_empty(v: object) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _as_float(v: object) -> float:
    if _is_empty(v):
        raise ValueError("required numeric field is empty")
    return float(v)  # type: ignore[arg-type]


def _as_opt_float(v: object) -> float | None:
    return None if _is_empty(v) else float(v)  # type: ignore[arg-type]


def _as_opt_int(v: object) -> int | None:
    return None if _is_empty(v) else int(float(v))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Append-only writer + loader
# --------------------------------------------------------------------------- #


class ResultsWriter:
    """Append-only writer for tidy results.

    Supports CSV (``.csv``) and JSON-lines (``.jsonl``), chosen by the path
    suffix. Append-only so concurrent case runs can each add their rows without
    a global rewrite; the header is written once when the file is created.

    Example
    -------
    >>> with ResultsWriter("results.csv") as w:
    ...     w.append(ResultRecord("analytical", "Ours (SI-SDE)",
    ...                           "analytical", "kl_points", 0.01))
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fmt = _infer_format(self.path)
        self._fh = None  # type: ignore[assignment]
        self._csv_writer: csv.DictWriter | None = None

    def __enter__(self) -> "ResultsWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="", encoding="utf-8")
        if self._fmt == "csv":
            self._csv_writer = csv.DictWriter(
                self._fh, fieldnames=list(ResultRecord.FIELDNAMES)
            )
            if new_file:
                self._csv_writer.writeheader()
        return self

    def append(self, record: ResultRecord) -> None:
        if self._fh is None:
            raise RuntimeError("ResultsWriter must be used as a context manager")
        if self._fmt == "csv":
            assert self._csv_writer is not None
            self._csv_writer.writerow(_csv_cells(record.to_row()))
        else:  # jsonl
            self._fh.write(json.dumps(record.to_row()) + "\n")

    def extend(self, records: Iterable[ResultRecord]) -> None:
        for r in records:
            self.append(r)

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def write_records(path: str | Path, records: Iterable[ResultRecord]) -> Path:
    """Convenience: append all ``records`` to ``path`` and return the path."""
    path = Path(path)
    with ResultsWriter(path) as w:
        w.extend(records)
    return path


def load_records(path: str | Path) -> list[ResultRecord]:
    """Load every tidy row from a ``.csv`` or ``.jsonl`` file."""
    path = Path(path)
    fmt = _infer_format(path)
    records: list[ResultRecord] = []
    with open(path, encoding="utf-8") as fh:
        if fmt == "csv":
            for row in csv.DictReader(fh):
                records.append(ResultRecord.from_row(row))
        else:  # jsonl
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ResultRecord.from_row(json.loads(line)))
    return records


def iter_records(path: str | Path) -> Iterator[ResultRecord]:
    """Stream tidy rows lazily (useful for large files)."""
    path = Path(path)
    fmt = _infer_format(path)
    with open(path, encoding="utf-8") as fh:
        if fmt == "csv":
            for row in csv.DictReader(fh):
                yield ResultRecord.from_row(row)
        else:
            for line in fh:
                line = line.strip()
                if line:
                    yield ResultRecord.from_row(json.loads(line))


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    raise ValueError(
        f"Unsupported results extension '{suffix}' for {path}; "
        "use .csv or .jsonl"
    )


def _csv_cells(row: dict[str, object]) -> dict[str, object]:
    """Render ``None`` as an empty cell for CSV (round-trips to ``None``)."""
    return {k: ("" if v is None else v) for k, v in row.items()}


__all__ = [
    "Case",
    "Method",
    "METHODS",
    "Scenario",
    "SCENARIOS",
    "Metric",
    "METRICS",
    "SEED_AGGREGATED",
    "ResultRecord",
    "ResultsWriter",
    "write_records",
    "load_records",
    "iter_records",
]


if dataclasses.is_dataclass(ResultRecord):  # pragma: no cover - import-time guard
    # Sanity: the dataclass column order must match the on-disk header (minus the
    # nfe/NFE rename). Keeps the schema and the writer from silently drifting.
    _expected = (
        "case",
        "method",
        "scenario",
        "metric",
        "value",
        "std",
        "E",
        "M",
        "seed",
        "nfe",
        "seconds",
    )
    _actual = tuple(f.name for f in dataclasses.fields(ResultRecord))
    assert _actual == _expected, (_actual, _expected)
