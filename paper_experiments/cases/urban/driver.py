"""Case 3 driver -- urban airflow CFD over a building array (uDALES).

Applied, multi-variable realism case (spec Section 6): coupled velocity and
temperature fields over bluff bodies, with solid cells, wakes, and anisotropic
statistics. Same observation scenarios as NS; sensors at physically plausible
(fluid) locations.

Produces:
* ``tab:urban_accuracy``          -- velocity RMSE, temperature RMSE
                                     x {32^2->128^2, 5%} (NO KL: no GT posterior).
* ``tab:urban_calibration_cost``  -- CRPS, |1-spread/skill|, NFE, s/step.
* ``fig:urban_fields``            -- geometry + truth/prior/posterior (figure TODO).

Author decision (PROJECT_HANDOFF B.4): the uDALES data is author-provided
(``data/udales/*.nc`` + ``data/udales/mask.npz``); no CFD generator is needed
in-repo. The trained SI / FM priors live under ``checkpoints/udales/``. This
driver is a close mirror of ``NavierStokesRunner`` -- it delegates the heavy
lifting to ``_urban_pipeline`` (which itself reuses the NS loader / sampler /
assimilation seams) and differs only where the urban case is GENERATIVE-ONLY and
MULTI-CHANNEL with solid cells:

* GENERATIVE-ONLY: there is no in-repo CFD solver, so the conventional /
  true-solver baselines (EnKF, LETKF, bootstrap PF, Ensemble Score Filter) CANNOT
  be run -- they propagate the ensemble with the genuine solver. ``URBAN_METHODS``
  omits them; only the learned-prior samplers run.
* MULTI-CHANNEL: the state is ``(u, v, w, thl)`` (4 channels); accuracy is a
  per-variable RMSE (velocity over ``u, v, w``; temperature over ``thl``).
* SOLID-CELL MASKING: the ``mask.npz`` solid-cell mask is excluded from every
  metric and from the sparse sensor pool (``_urban_pipeline`` applies it in the
  obs operator and the metrics; the model already receives it as ``field_cond``).
* NO KL-at-points: urban has only a ground-truth STATE, not a ground-truth
  posterior, so KL (which needs a reference posterior) is not computed and no
  large-E reference ensemble is drawn. Calibration = spread--skill + split CRPS.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

from common.runner import ExperimentRunner, RunContext
from common.seeding import mask_seed, obs_seed
from results_schema import Case, Method, Metric, ResultRecord, Scenario

from cases.urban import _urban_pipeline

logger = logging.getLogger(__name__)

# Urban (uDALES) has NO conventional/true-solver baseline available: the classical
# filters (EnKF, LETKF, bootstrap PF) -- and now the Ensemble Score Filter, which is
# also a true-solver method -- require the genuine forward solver to propagate the
# ensemble, which we do not have for the urban CFD here. The urban comparison is
# therefore GENERATIVE-ONLY -- our samplers vs the other deep generative / score-based
# methods, all sharing the trained prior. (Do NOT add EnKF / LETKF / particle filter /
# ensemble score filter here.)
URBAN_METHODS: tuple[Method, ...] = (
    # Ours (unified family).
    Method.OURS_SI_SDE,
    Method.OURS_FM_ODE,
    Method.OURS_FM_SDE,  # FM-SDE (DM)
    # SI + SDE.
    Method.FLOWDAS,
    # Flow matching + ODE.
    Method.GUIDED_FM_FIG,
    Method.GUIDED_FM_OTODE,
    Method.D_FLOW_SGLD,
    # Diffusion model + SDE.
    Method.SDA,
    Method.SURGE,
)

# Random-observation scenarios only (per user): the two sparse sensor densities.
# NO super-resolution for urban.
URBAN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario.SPARSE_5,
    Scenario.SPARSE_1p5,
)

# Per-variable RMSE + shared distributional / calibration metrics.
# Urban has only a ground-truth STATE, not a ground-truth posterior, so KL-at-points
# (which needs a reference posterior) is NOT computed. Calibration is assessed with the
# spread--skill ratio and CRPS (both scored against the ground-truth state), split into
# observed/unobserved grid points as for NS.
URBAN_METRICS: tuple[Metric, ...] = (
    Metric.RMSE_VELOCITY,
    Metric.RMSE_TEMPERATURE,
    Metric.CRPS,
    Metric.CRPS_OBSERVED,
    Metric.CRPS_UNOBSERVED,
    Metric.SPREAD_SKILL,
)

# Maps a wired Method to the YAML method-config name under configs/method/ (same
# wiring as NS; the urban case reuses every method config verbatim).
METHOD_CONFIG_NAME: dict[Method, str] = {
    Method.OURS_SI_SDE: "si_sde",
    Method.OURS_FM_SDE: "fm_sde",
    Method.OURS_FM_ODE: "fm_ode",
    Method.FLOWDAS: "flowdas",
    Method.GUIDED_FM_FIG: "guided_fm_fig",
    Method.GUIDED_FM_OTODE: "guided_fm_otode",
    Method.D_FLOW_SGLD: "dflow_sgld",
    Method.SDA: "sda",
    Method.SURGE: "surge",
}

# Maps a Scenario to its scenario-config name under configs/scenario/.
SCENARIO_CONFIG_NAME: dict[Scenario, str] = {
    Scenario.SPARSE_5: "sparse_5",
    Scenario.SPARSE_1p5: "sparse_1p5",
}


class UrbanRunner(ExperimentRunner):
    """Runner for the urban airflow case."""

    case = Case.URBAN

    def __init__(self, config, *, seeds=None):  # type: ignore[no-untyped-def]
        if seeds is None:
            from common.seeding import SEED_LIST

            seeds = SEED_LIST
        super().__init__(config, seeds=seeds)
        self._prior = None  # lazily loaded shared prior (load once)
        case_device = None
        try:
            case_device = config.case.get("device", None)
        except Exception:
            case_device = None
        self._device = str(self._cfg_get("device", case_device or "cpu"))

    # -- config helpers ---------------------------------------------------- #

    def _case_cfg(self):  # type: ignore[no-untyped-def]
        return self.config.case

    def _scenario_cfgs(self) -> dict:
        """Load the scenario configs referenced by this run (by name)."""
        from pathlib import Path

        from omegaconf import OmegaConf

        cfgs: dict[str, object] = {}
        root = Path(__file__).resolve().parents[2] / "configs" / "scenario"
        for scen in self.scenarios():
            name = SCENARIO_CONFIG_NAME[scen]
            cfgs[scen.value] = OmegaConf.load(root / f"{name}.yaml")
        return cfgs

    def _method_cfgs(self) -> dict:
        from pathlib import Path

        from omegaconf import OmegaConf

        cfgs: dict[Method, object] = {}
        root = Path(__file__).resolve().parents[2] / "configs" / "method"
        for m, name in METHOD_CONFIG_NAME.items():
            cfgs[m] = OmegaConf.load(root / f"{name}.yaml")
        return cfgs

    def _ensure_prior(self):  # type: ignore[no-untyped-def]
        if self._prior is None:
            self._prior = _urban_pipeline.load_prior(self._case_cfg(), self._device)
        return self._prior

    # -- subclass hooks ---------------------------------------------------- #

    def methods(self) -> Sequence[Method]:
        names = self._cfg_get("urban_methods", None)
        if names:
            return tuple(Method(n) for n in names)
        return URBAN_METHODS

    def scenarios(self) -> Sequence[Scenario]:
        names = self._cfg_get("urban_scenarios", None)
        if names:
            return tuple(Scenario(n) for n in names)
        return URBAN_SCENARIOS

    def make_context(self, method, scenario, seed) -> RunContext:  # type: ignore[no-untyped-def]
        case_cfg = self._case_cfg()
        return RunContext(
            case=self.case,
            method=method,
            scenario=scenario,
            seed=seed,
            ensemble_size=int(self._cfg_get("ensemble_size", case_cfg.ensemble_size)),
            num_steps=int(self._cfg_get("num_steps", case_cfg.num_steps)),
            extra={
                "num_physical_steps": int(case_cfg.num_physical_steps),
                "variance": self._resolve_variance(case_cfg),
                "test_index": int(
                    self._cfg_get("test_index", case_cfg.test_sample_indices[0])
                ),
                "likelihood_ensemble_size": int(case_cfg.likelihood_ensemble_size),
            },
        )

    @staticmethod
    def _resolve_variance(case_cfg) -> float:  # type: ignore[no-untyped-def]
        """Resolve the (scalar) observation-noise variance from the case config.

        The urban config carries a per-variable ``variance`` block (velocity /
        temperature) because the channels live on different scales; the
        observation operator + likelihood, however, take a single scalar ``R =
        sigma^2 I`` in the NORMALISED space (where every channel has unit std), so
        a single scalar variance applies uniformly. We read ``variance.normalised``
        when present, else fall back to a plain scalar ``variance``.

        TODO(spec Section 6): confirm the per-variable noise levels with the user
        and, if a genuinely per-channel R is wanted, generalise the obs operator /
        likelihood to a diagonal R (currently scalar). The normalised-space scalar
        is the faithful analogue of the NS ``variance: 0.0025`` (sigma=0.05).
        """
        var = case_cfg.get("variance", None)
        if var is None:
            return 0.0025
        # Scalar variance (already in normalised space).
        try:
            return float(var)
        except (TypeError, ValueError):
            pass
        # Mapping: prefer an explicit normalised-space scalar.
        normalised = var.get("normalised", None)
        if normalised is not None:
            return float(normalised)
        # Otherwise default to the NS-equivalent sigma=0.05 in normalised space.
        return 0.0025

    # -- per-(method, scenario, seed) evaluation --------------------------- #

    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        if ctx.method not in URBAN_METHODS:
            yield from self._todo_rows(ctx)
            return

        prior = self._ensure_prior()
        scen_cfg = self._scenario_cfgs()[ctx.scenario.value]
        method_cfg = self._method_cfgs()[ctx.method]
        device = self._device
        extra = ctx.extra

        # The STATE channel count comes from the loaded test trajectory (4 =
        # u, v, w, thl), NOT test_dataset.num_channels (hardcoded 5; only 4 are
        # stacked). Drives the obs operator's data_size and the fluid keep-mask.
        sample = prior.test_dataset[extra["test_index"]]
        num_channels = int(sample["x"].shape[0])
        data_size = (num_channels, prior.test_dataset.height, prior.test_dataset.width)

        fluid_mask = _urban_pipeline.fluid_keep_mask(prior, num_channels, device)

        obs_operator = _urban_pipeline.build_obs_operator(
            scenario_cfg=scen_cfg,
            data_size=data_size,
            mask_seed=mask_seed(self.case.value, ctx.scenario.value),
            fluid_mask=fluid_mask,
        )

        truth_obs = _urban_pipeline.prepare_truth_and_obs(
            prior=prior,
            test_index=extra["test_index"],
            obs_operator=obs_operator,
            variance=extra["variance"],
            num_physical_steps=extra["num_physical_steps"],
            obs_noise_seed=obs_seed(
                self.case.value, ctx.scenario.value, extra["test_index"], ctx.seed
            ),
            device=device,
        )

        model, posterior, stepper = _urban_pipeline.build_posterior(
            method_name=ctx.method.value,
            method_cfg=method_cfg,
            prior=prior,
            obs_operator=obs_operator,
            variance=extra["variance"],
            likelihood_ensemble_size=extra["likelihood_ensemble_size"],
            likelihood_mode=self._cfg_get("likelihood_mode", None),
        )

        nfe = _urban_pipeline.attach_nfe_counter(model)

        result = _urban_pipeline.run_assimilation(
            posterior=posterior,
            model=model,
            truth_obs=truth_obs,
            ensemble_size=ctx.ensemble_size,
            num_steps=ctx.num_steps,
            num_physical_steps=extra["num_physical_steps"],
            stepper=stepper,
            nfe_counter=nfe,
            # Anchor a0 = 0 (FM-path and DM-path: FM-SDE/FM-ODE, FIG, OT-ODE,
            # D-Flow, SDA, SURGE) inits from N(0, I); only the SI-path methods
            # (SI-SDE, FlowDAS) use the x0 point-mass init.
            gaussian_base=ctx.method not in (Method.OURS_SI_SDE, Method.FLOWDAS),
        )

        metrics = _urban_pipeline.compute_metrics(
            result=result,
            obs_operator=obs_operator,
            fluid_mask=fluid_mask,
            len_field_history=prior.len_field_history,
        )

        logger.info(
            "[URBAN] %s | %s | seed=%d | E=%d M=%d | %s",
            ctx.method.value,
            ctx.scenario.value,
            ctx.seed,
            ctx.ensemble_size,
            ctx.num_steps,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        )

        yield from self._metric_rows(ctx, metrics)

    # -- record emitters --------------------------------------------------- #

    def _metric_rows(self, ctx, metrics):  # type: ignore[no-untyped-def]
        for metric in URBAN_METRICS:
            yield ResultRecord(
                case=self.case.value,
                method=ctx.method.value,
                scenario=ctx.scenario.value,
                metric=metric.value,
                value=metrics[metric.value],
                E=ctx.ensemble_size,
                M=ctx.num_steps,
                seed=ctx.seed,
                nfe=metrics["nfe"],
                seconds=metrics["seconds"],
            )
        # Cost rows (so the calibration/cost table can read NFE / seconds).
        for metric, key in ((Metric.NFE, "nfe"), (Metric.SECONDS, "seconds")):
            yield ResultRecord(
                case=self.case.value,
                method=ctx.method.value,
                scenario=ctx.scenario.value,
                metric=metric.value,
                value=metrics[key],
                E=ctx.ensemble_size,
                M=ctx.num_steps,
                seed=ctx.seed,
                nfe=metrics["nfe"],
                seconds=metrics["seconds"],
            )

    def _todo_rows(self, ctx):  # type: ignore[no-untyped-def]
        """Emit NaN placeholder rows for any non-wired method (defensive)."""
        for metric in URBAN_METRICS + (Metric.NFE, Metric.SECONDS):
            yield ResultRecord(
                case=self.case.value,
                method=ctx.method.value,
                scenario=ctx.scenario.value,
                metric=metric.value,
                value=float("nan"),
                E=ctx.ensemble_size,
                M=ctx.num_steps,
                seed=ctx.seed,
            )


__all__ = ["UrbanRunner", "URBAN_METHODS", "URBAN_SCENARIOS", "URBAN_METRICS"]
