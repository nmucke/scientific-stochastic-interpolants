"""The Lorenz 96 lineup: the one place that names every cell and the shared constants.

Two sweeps share every script: ``linear`` (h = every fourth site) and ``arctan``
(h = arctan of the same sites). Each has its own result root
``results/lorenz96/<obs>/``, its own tidy ``scenario`` string and its own settings
block of ``configs/case/lorenz96.yaml`` (:data:`CONFIG_BLOCK`).

:data:`ROWS` gives, per ``driver.py`` cell and in table order, the tidy (method,
variant), the table group, the variant and legend labels, the line style and
whether the cell writes the SMC metrics. ``driver.CELLS`` is derived from it, and
the run script takes its cell lists from it::

    python lorenz96_lineup.py cells generative   # run at every M of M_VALUES
    python lorenz96_lineup.py cells fixed        # once per seed (M-independent)
    python lorenz96_lineup.py m-values           # M_VALUES

Pure stdlib (no torch), like ``results_schema``.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from results_schema import Method, Scenario  # noqa: E402

CASE = "lorenz96"
OBS_CHOICES: tuple[str, ...] = ("linear", "arctan")
# Tidy ``scenario`` per sweep (results_schema.Scenario values).
SCENARIO_OF: dict[str, str] = {
    "linear": Scenario.LORENZ96.value,
    "arctan": Scenario.LORENZ96_ARCTAN.value,
}
# Settings block of each sweep in configs/case/lorenz96.yaml.
CONFIG_BLOCK: dict[str, str] = {"linear": "methods", "arctan": "methods_arctan"}

# Per-step and per-sequence metrics written by the SMC cells only (the six SURGE
# cells and the particle filters); every other row leaves these table columns blank.
SMC_ESS_FRACTION = "smc_ess_fraction"
SMC_ESS_MIN_FRACTION = "smc_ess_min_fraction"
SMC_RESAMPLES = "smc_resamples"
SMC_METRICS: tuple[str, ...] = (SMC_ESS_FRACTION, SMC_ESS_MIN_FRACTION, SMC_RESAMPLES)
# Distances of the ensemble to the PF reference posterior, per step (every cell).
KL_KNN = "kl_knn"  # k-NN KL estimate (Perez-Cruz)
ENERGY_DISTANCE = "energy_distance"

# The step counts of the M sweep; generative cells run at each, the M-independent
# cells once. HEADLINE_M is the main table's and the per-step figures' M, and the
# only M tuning uses (the picks are reused at every M).
M_VALUES: tuple[int, ...] = (25, 50, 100, 250)
HEADLINE_M = 50


@dataclass(frozen=True)
class Row:
    cell: str  # driver.py cell name
    method: str  # tidy method (a results_schema.Method value)
    variant: str  # tidy variant, "" for none
    group: str  # key of GROUPS
    label: str  # short label of the variant column
    colour: str = "#000000"
    style: str = "-"
    smc: bool = False  # writes the smc_* metrics
    legend: str = ""  # figure legend; "" means the method

    @property
    def generative(self) -> bool:
        """Runs at every M of the sweep (files carry ``__M<M>__``)."""
        return self.cell not in M_INDEPENDENT

    @property
    def legend_label(self) -> str:
        return self.legend or self.method


# (key, title) in table order.
GROUPS: tuple[tuple[str, str], ...] = (
    ("reference", "PF reference floor"),
    ("classical", "Classical DA"),
    ("l96", "Lorenz 96 controls"),
    ("ours", "Ours"),
    ("baselines", "Generative baselines"),
)

PF, ENKF, ORACLE = Method.PARTICLE_FILTER.value, Method.ENKF.value, Method.ONE_STEP_ORACLE.value
SI, FM, DM = Method.OURS_SI_SDE.value, Method.OURS_FM_ODE.value, Method.OURS_DM_SDE.value
FC = "forecast_correction"
ROWS: tuple[Row, ...] = (
    Row("pf_reference", PF, "reference", "reference", "reference", "#666666", "-",
        smc=True, legend="PF reference"),
    Row("pf_subsample", PF, "reference_subsample", "reference", "reference, E draws",
        "#666666", "--", legend="PF reference, E draws"),
    Row("enkf", ENKF, "matched", "classical", "matched E", "#000000", ":",
        legend="EnKF (matched E)"),
    Row("enkf_large", ENKF, "large", "classical", "large E", "#000000", "-.",
        legend="EnKF (large E)"),
    Row("pf_matched", PF, "matched", "classical", "matched E", "#E69F00", ":", smc=True,
        legend="PF (matched E)"),
    Row("oracle", ORACLE, "", "l96", "", "#D55E00", "--", legend="One-step oracle"),
    Row("oracle_fc", ORACLE, FC, "l96", "+ correction", "#D55E00", "-",
        legend="Oracle + correction"),
    Row("prior_enkf", ENKF, "trained_prior_si", "l96", "SI prior", "#56B4E9", ":",
        legend="SI prior + EnKF"),
    Row("prior_enkf_fm", ENKF, "trained_prior_fm", "l96", "FM prior", "#56B4E9", "-.",
        legend="FM prior + EnKF"),
    Row("si_sde", SI, "", "ours", "", "#0072B2", "-", legend="Ours SI-SDE"),
    Row("fm_ode", FM, "", "ours", "", "#CC79A7", "-", legend="Ours FM-ODE"),
    Row("dm_sde", DM, "", "ours", "", "#009E73", "-", legend="Ours DM-SDE"),
    Row("si_sde_fc", SI, FC, "ours", "+ correction", "#0072B2", "--",
        legend="Ours SI-SDE + correction"),
    Row("fm_ode_fc", FM, FC, "ours", "+ correction", "#CC79A7", "--",
        legend="Ours FM-ODE + correction"),
    Row("dm_sde_fc", DM, FC, "ours", "+ correction", "#009E73", "--",
        legend="Ours DM-SDE + correction"),
    Row("surge_si_sde", SI, "surge", "ours", "+ SURGE", "#0072B2", ":", smc=True,
        legend="Ours SI-SDE + SURGE"),
    Row("surge_dm_sde", DM, "surge", "ours", "+ SURGE", "#009E73", ":", smc=True,
        legend="Ours DM-SDE + SURGE"),
    # The same SURGE SMC proposing from the unguided prior, one per wrapped prior
    # (the multistep convention): the ablation of the wrapped sampler.
    Row("surge_unguided_si", SI, "surge_unguided", "ours", "SURGE, unguided", "#0072B2",
        "-.", smc=True),
    Row("surge_unguided_dm", DM, "surge_unguided", "ours", "SURGE, unguided", "#009E73",
        "-.", smc=True),
    Row("flowdas", Method.FLOWDAS.value, "", "baselines", "", "#882255", "-"),
    Row("surge_flowdas", Method.SURGE_FLOWDAS.value, "", "baselines", "", "#882255", ":",
        smc=True),
    Row("sda", Method.SDA.value, "", "baselines", "", "#44AA99", "-"),
    Row("surge_sda", Method.SURGE_SDA.value, "", "baselines", "", "#44AA99", ":", smc=True),
    Row("dflow", Method.D_FLOW_SGLD.value, "", "baselines", "", "#999933", "-"),
    Row("fig", Method.GUIDED_FM_FIG.value, "", "baselines", "", "#AA4499", "-"),
)
ROW_OF: dict[tuple[str, str], Row] = {(r.method, r.variant): r for r in ROWS}
ROW_BY_CELL: dict[str, Row] = {r.cell: r for r in ROWS}
# Cells without a step count: run once, not per M (their config block has no
# num_steps; a test checks the two agree).
M_INDEPENDENT: tuple[str, ...] = (
    "enkf", "enkf_large", "pf_matched", "pf_subsample", "pf_reference", "oracle", "oracle_fc",
)
# Run once in all (not per seed): the reference scored against its own replicate.
SEED_INDEPENDENT: tuple[str, ...] = ("pf_reference",)
# Horizontal reference lines of the metric-vs-M figures.
M_REFERENCE_LINES: tuple[str, ...] = ("pf_subsample", "enkf", "oracle")


def run_cells(kind: str) -> list[str]:
    """``generative``: cells run at every M; ``fixed``: M-independent cells run
    once per seed (``pf_reference``, run once in all, is in neither)."""
    if kind == "generative":
        return [r.cell for r in ROWS if r.generative]
    if kind == "fixed":
        return [r.cell for r in ROWS if not r.generative and r.cell not in SEED_INDEPENDENT]
    raise ValueError(f"kind must be 'generative' or 'fixed', got {kind!r}")


# --------------------------------------------------------------------------- #
# Settings: inheritance and dotted keys (``correction.gain_scale``)
# --------------------------------------------------------------------------- #

# Cell -> the cell whose settings block it starts from; its own block adds to (and
# overrides) those keys. A corrected cell runs its plain cell's sampler, a SURGE
# cell around Ours the plain cell's likelihood.
INHERITS: dict[str, str] = {
    "si_sde_fc": "si_sde",
    "fm_ode_fc": "fm_ode",
    "dm_sde_fc": "dm_sde",
    "surge_si_sde": "si_sde",
    "surge_dm_sde": "dm_sde",
}


def settings_of(methods: dict, name: str) -> dict:
    """The settings of cell ``name`` from a plain ``methods:`` block (a dict)."""
    own = dict(methods.get(name) or {})
    if name in INHERITS:
        return {**settings_of(methods, INHERITS[name]), **own}
    return own


def set_dotted(settings: dict, flat: dict) -> dict:
    """A copy of ``settings`` with each dotted key of ``flat`` set in place:
    ``{"correction.gain_scale": 0.5}`` sets ``settings["correction"]["gain_scale"]``."""
    out = json.loads(json.dumps(settings))
    for key, value in flat.items():
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def parse_assignments(assignments) -> dict:
    """``["a.b=1", "mode=dps_jacobian_free"]`` -> ``{"a.b": 1, "mode": "dps_..."}``:
    JSON values, anything that is not JSON kept as a string (``--set``)."""
    flat = {}
    for assignment in assignments or ():
        key, sep, raw = assignment.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value, got {assignment!r}")
        try:
            flat[key] = json.loads(raw)
        except json.JSONDecodeError:
            flat[key] = raw
    return flat


# --------------------------------------------------------------------------- #
# Tidy files
# --------------------------------------------------------------------------- #


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


Entry = tuple[float, float, int]  # mean, std over seeds, seeds


def read_aggregated(path: Path) -> dict[tuple[str, str], dict[str, dict[int | None, Entry]]]:
    """``{(method, variant): {metric: {M: (mean, std, n)}}}`` of an ``all.csv``.

    The one reader of ``aggregated/all.csv`` (tables and figures). Its inputs were
    checked by ``aggregate_lorenz96`` (:func:`check_scenario`), so the scenario is
    not checked again. A (row, metric, M) recorded at several E keeps the entry
    with the most seeds, with a warning. Empty when the file does not exist.
    """
    data: dict = defaultdict(lambda: defaultdict(dict))
    if not path.exists():
        return data
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], norm_variant(row.get("variant")))
            entry = (float(row["value"]), float(row["std"] or "nan"), int(row["n_traj"]))
            m = as_m(row.get("M"))
            old = data[key][row["metric"]].get(m)
            if old is not None:
                print(
                    f"[{CASE}] warning: {key} {row['metric']} M={m} has rows at several E; "
                    "keeping the one with most seeds"
                )
                if old[2] >= entry[2]:
                    continue
            data[key][row["metric"]][m] = entry
    return data


def main(argv: list[str]) -> None:
    if argv == ["m-values"]:
        print(" ".join(str(m) for m in M_VALUES))
        return
    if len(argv) != 2 or argv[0] != "cells":
        raise SystemExit("usage: lorenz96_lineup.py cells generative|fixed | m-values")
    print(" ".join(run_cells(argv[1])))


if __name__ == "__main__":
    main(sys.argv[1:])
