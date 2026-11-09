"""The Lorenz 96 lineup as the aggregation, figure and table scripts see it.

Two sweeps share every script: ``linear`` (h = every fourth site) and ``arctan``
(h = arctan of the same sites). Each has its own result root
``results/lorenz96/<obs>/`` and its own tidy ``scenario`` string, so their files
never mix (:func:`check_scenario`).

``ROWS`` is the one place that names the tidy (method, variant) of every cell, in
table order and with its table group. The cell names are those of
``cases/lorenz96/driver.py`` and ``run_lorenz96_grid.sh``; a renamed cell only needs
its row edited here. Pure stdlib (no torch), like ``results_schema``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

CASE = "lorenz96"
OBS_CHOICES: tuple[str, ...] = ("linear", "arctan")
# Tidy ``scenario`` per sweep (results_schema.Scenario values).
SCENARIO_OF: dict[str, str] = {"linear": "lorenz96", "arctan": "lorenz96_arctan"}

# Per-step and per-sequence metrics written by the SMC cells only (the five SURGE
# cells and pf_matched); every other row leaves these table columns blank.
SMC_ESS_FRACTION = "smc_ess_fraction"
SMC_ESS_MIN_FRACTION = "smc_ess_min_fraction"
SMC_RESAMPLES = "smc_resamples"
SMC_METRICS: tuple[str, ...] = (SMC_ESS_FRACTION, SMC_ESS_MIN_FRACTION, SMC_RESAMPLES)
# Distances of the ensemble to the PF reference posterior, per step (every cell).
KL_KNN = "kl_knn"  # k-NN KL estimate (Perez-Cruz)
ENERGY_DISTANCE = "energy_distance"

# The step counts of the M sweep; generative cells run at each, the M-independent
# cells once. HEADLINE_M is the main table's and the per-step figures' M (and the
# only M tuning uses).
M_VALUES: tuple[int, ...] = (25, 50, 100, 250)
HEADLINE_M = 50


@dataclass(frozen=True)
class Row:
    cell: str  # driver.py cell name
    method: str  # tidy method
    variant: str  # tidy variant, "" for none
    group: str  # key of GROUPS
    label: str  # short label of the variant column
    colour: str = "#000000"
    style: str = "-"
    smc: bool = False  # writes the smc_* metrics

    @property
    def generative(self) -> bool:
        """Runs at every M of the sweep (files carry ``__M<M>__``)."""
        return self.cell not in M_INDEPENDENT


# (key, title) in table order.
GROUPS: tuple[tuple[str, str], ...] = (
    ("reference", "PF reference floor"),
    ("classical", "Classical DA"),
    ("l96", "Lorenz 96 controls"),
    ("ours", "Ours"),
    ("baselines", "Generative baselines"),
)

SI, FM, DM = "Ours (SI-SDE)", "Ours (FM-ODE)", "Ours (DM-SDE)"
FC = "forecast_correction"
ROWS: tuple[Row, ...] = (
    Row("pf_reference", "Particle filter", "reference", "reference", "reference", "#666666", "-"),
    Row("pf_subsample", "Particle filter", "reference_subsample", "reference", "reference, E draws", "#666666", "--"),
    Row("enkf", "EnKF", "matched", "classical", "matched E", "#000000", ":"),
    Row("enkf_large", "EnKF", "large", "classical", "large E", "#000000", "-."),
    Row("pf_matched", "Particle filter", "matched", "classical", "matched E", "#E69F00", ":", smc=True),
    Row("oracle", "One-step oracle", "", "l96", "", "#D55E00", "--"),
    Row("oracle_fc", "One-step oracle", FC, "l96", "+ correction", "#D55E00", "-"),
    Row("prior_enkf", "EnKF", "trained_prior_si", "l96", "SI prior", "#56B4E9", ":"),
    Row("prior_enkf_fm", "EnKF", "trained_prior_fm", "l96", "FM prior", "#56B4E9", "-."),
    Row("si_sde", SI, "", "ours", "", "#0072B2", "-"),
    Row("fm_ode", FM, "", "ours", "", "#CC79A7", "-"),
    Row("dm_sde", DM, "", "ours", "", "#009E73", "-"),
    Row("si_sde_fc", SI, FC, "ours", "+ correction", "#0072B2", "--"),
    Row("fm_ode_fc", FM, FC, "ours", "+ correction", "#CC79A7", "--"),
    Row("dm_sde_fc", DM, FC, "ours", "+ correction", "#009E73", "--"),
    Row("surge_si_sde", SI, "surge", "ours", "+ SURGE", "#0072B2", ":", smc=True),
    Row("surge_dm_sde", DM, "surge", "ours", "+ SURGE", "#009E73", ":", smc=True),
    # The unguided SURGE control; its tidy label is an assumption (see README).
    Row("surge_unguided", "SURGE", "unguided", "ours", "unguided control", "#999999", ":", smc=True),
    Row("flowdas", "FlowDAS", "", "baselines", "", "#882255", "-"),
    Row("surge_flowdas", "SURGE (FlowDAS)", "", "baselines", "", "#882255", ":", smc=True),
    Row("sda", "SDA", "", "baselines", "", "#44AA99", "-"),
    Row("surge_sda", "SURGE (SDA)", "", "baselines", "", "#44AA99", ":", smc=True),
    Row("dflow", "D-Flow SGLD", "", "baselines", "", "#999933", "-"),
    Row("fig", "Guided FM (FIG)", "", "baselines", "", "#AA4499", "-"),
)
ROW_OF: dict[tuple[str, str], Row] = {(r.method, r.variant): r for r in ROWS}
# Cells without a step count: run once, not per M (run_lorenz96_grid.sh agrees).
M_INDEPENDENT: tuple[str, ...] = (
    "enkf", "enkf_large", "pf_matched", "pf_subsample", "pf_reference", "oracle", "oracle_fc",
)
# Horizontal reference lines of the metric-vs-M figures.
M_REFERENCE_LINES: tuple[str, ...] = ("pf_subsample", "enkf", "oracle")


def as_m(value) -> int | None:
    """The tidy ``M`` cell as an int, None when empty."""
    text = "" if value is None else str(value).strip()
    if text in ("", "None", "nan"):
        return None
    return int(float(text))


def pick_m(by_m: dict, row: Row, m: int = HEADLINE_M):
    """The entry of ``row`` at step count ``m`` from ``{M: entry}``.

    A generative row needs exactly ``m``; an M-independent row has one entry,
    whatever M it recorded. None when missing.
    """
    if row.generative:
        return by_m.get(m)
    if len(by_m) > 1:
        print(f"[{CASE}] warning: {row.cell} has entries at several M {sorted(by_m, key=str)}")
    return next(iter(by_m.values()), None)


def sweep_root(results: Path, obs: str) -> Path:
    if obs not in OBS_CHOICES:
        raise ValueError(f"Unknown observation sweep {obs!r}; expected one of {OBS_CHOICES}.")
    return results / CASE / obs


def norm_variant(value) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in ("", "None") else text


def check_scenario(paths, obs: str) -> None:
    """Refuse a tidy file of another sweep inside ``obs``'s root.

    Every row's ``scenario`` must be ``SCENARIO_OF[obs]``: a file copied into the
    wrong root would otherwise be averaged into this sweep's rows.
    """
    expected = SCENARIO_OF[obs]
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            found = {row.get("scenario", "") for row in csv.DictReader(handle)}
        if found - {expected}:
            raise ValueError(
                f"{path} has scenario {sorted(found - {expected})}, but it is in the "
                f"{obs!r} sweep root, whose scenario is {expected!r}: the two sweeps "
                "must not mix. Move the file to its own sweep root."
            )
