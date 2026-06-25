"""Case 3 driver -- urban airflow CFD over a building array (uDALES).

Applied, multi-variable realism case (spec Section 6): coupled velocity and
temperature fields over bluff bodies, with solid cells, wakes, and anisotropic
statistics. Same observation scenarios as NS; sensors at physically plausible
locations (street level, facades).

Produces:
* ``tab:urban_accuracy``          -- velocity RMSE, temperature RMSE, KL-at-points
                                     x {32^2->128^2, 5%}.
* ``tab:urban_calibration_cost``  -- CRPS, |1-spread/skill|, NFE, s/step.
* ``fig:urban_fields``            -- geometry + truth/prior/posterior (figure TODO).

Author decision (GAP Section 6): the uDALES data will be author-provided
(``.nc`` + ``mask.npz``); no CFD generator is needed in-repo. Scope here is the
loader, solid-cell masking, channel-count fix, and metrics around the supplied
data.

Integration seams: same as the NS driver, plus solid-cell masking that must be
applied consistently in the model, the obs operator, and every metric (solid
cells excluded from RMSE/CRPS/KL -- reproducibility Section 9 and GAP E2).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from common.runner import ExperimentRunner, RunContext
from results_schema import Case, Method, Metric, ResultRecord, Scenario

# Same method list as NS.
URBAN_METHODS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_FM_SDE,
    Method.OURS_FM_ODE,
    Method.FLOWDAS,
    Method.GUIDED_FM,
    Method.GUIDED_DIFFUSION,
    Method.SDA,
    Method.ENSEMBLE_SCORE_FILTER,
    Method.ENKF,
    Method.PARTICLE_FILTER,
)

URBAN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario.SUPERRES_32,
    Scenario.SPARSE_5,
)

# Per-variable RMSE + shared distributional / calibration metrics.
URBAN_METRICS: tuple[Metric, ...] = (
    Metric.RMSE_VELOCITY,
    Metric.RMSE_TEMPERATURE,
    Metric.KL_POINTS,
    Metric.CRPS,
    Metric.SPREAD_SKILL,
)


class UrbanRunner(ExperimentRunner):
    """Runner for the urban airflow case."""

    case = Case.URBAN

    def methods(self) -> Sequence[Method]:
        return URBAN_METHODS

    def scenarios(self) -> Sequence[Scenario]:
        return URBAN_SCENARIOS

    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        # ------------------------------------------------------------------ #
        # TODO(E2): load author-provided uDALES data (.nc) + solid-cell mask
        #   (mask.npz). Fix the channel count (velocity components + temperature).
        # TODO(E2): apply the solid-cell mask consistently in the model input, the
        #   observation operator, and ALL metrics -- exclude solid cells from
        #   RMSE/CRPS/KL (reproducibility Section 9).
        # TODO(E1): obs operator per ctx.scenario; for sparse, place sensors at
        #   physically plausible locations (street level + facades), seeded mask.
        # TODO(E4/P1-P3): multi-channel unified sampler for ctx.method.
        # TODO: report KL at UNOBSERVED points in wake regions specifically
        #   (spec Section 6); per-variable RMSE (velocity, temperature).
        # ------------------------------------------------------------------ #
        raise NotImplementedError(
            "UrbanRunner.evaluate: pending author-provided uDALES data (GAP E2) "
            "and the unified-sampler rebuild (Phases 1-3). Emit per-seed "
            f"ResultRecord rows for metrics {[m.value for m in URBAN_METRICS]} "
            "plus NFE/seconds."
        )
