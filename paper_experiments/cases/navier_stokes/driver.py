"""Case 2 driver -- stochastic incompressible Navier--Stokes.

Main benchmark: learned SI/FM priors, high-dimensional chaotic field, realistic
observation operators (spec Section 5). Autoregressive assimilation of y^1..y^N.

Produces:
* ``tab:ns_accuracy``           -- vorticity RMSE, energy-spec RMSE, KL-at-points
                                   x {32^2->128^2, 5%}.
* ``tab:ns_calibration_cost``   -- CRPS, |1-spread/skill|, NFE, s/step x same.
* ``tab:ablation``              -- gain / g_tau / M / E ablations (Section 7).
* ``fig:ns_trajectories``, ``fig:ns_diagnostics`` -- via ``_ns_figures``.

The 16^2->128^2 and 1.5625% columns go to an appendix table in the same format.

Wiring (now landed; see ``_ns_pipeline.py`` for the heavy lifting):
* prior models  -- the trained SI prior (``FollmerStochasticInterpolant``) is
  loaded from ``checkpoints/<project>/<name>/``; the FM prior reuses the same
  UNet drift architecture paired with a rectified-flow interpolation so
  ``FlowMatchingModel.score`` is defined (GAP L1). When no ``model.pth`` is
  present the models run with random weights (smoke only).
* unified samplers -- ``scisi.posterior_models`` SI-SDE / FM-SDE / FM-ODE (E4).
* obs operators -- ``scisi.likelihood_models.observation_operators``: block-
  average super-res (E1) and seeded sparse masks (Section 9).
* metrics -- ensemble-mean RMSE (m1), log-spectrum energy RMSE (E11), unbiased
  CRPS + spread-skill (E7), KL-at-points (E9), NFE + wall-clock (E10).

Baselines DPS / EnKF / PF / SDA / ensemble score filter are emitted as TODO
rows (value=NaN) -- Phase 4, not implemented here. FlowDAS is wired.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence

from common.runner import ExperimentRunner, RunContext
from common.seeding import mask_seed, obs_seed
from results_schema import Case, Method, Metric, ResultRecord, Scenario

from cases.navier_stokes import _ns_figures, _ns_pipeline

logger = logging.getLogger(__name__)

# Full method list for the NS tables (results.tex row order).
NS_METHODS: tuple[Method, ...] = (
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

# Methods wired in this phase; the rest are emitted as TODO (NaN) rows.
WIRED_METHODS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_FM_SDE,
    Method.OURS_FM_ODE,
    Method.FLOWDAS,
)

# Maps a wired Method to the YAML method-config name under configs/method/.
METHOD_CONFIG_NAME: dict[Method, str] = {
    Method.OURS_SI_SDE: "si_sde",
    Method.OURS_FM_SDE: "fm_sde",
    Method.OURS_FM_ODE: "fm_ode",
    Method.FLOWDAS: "flowdas",
}

# Main-table scenarios; the other two go to the appendix table.
NS_MAIN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario.SUPERRES_32,
    Scenario.SPARSE_5,
)
NS_APPENDIX_SCENARIOS: tuple[Scenario, ...] = (
    Scenario.SUPERRES_16,
    Scenario.SPARSE_1p5,
)

# Maps a Scenario to its scenario-config name under configs/scenario/.
SCENARIO_CONFIG_NAME: dict[Scenario, str] = {
    Scenario.SUPERRES_32: "superres_32",
    Scenario.SUPERRES_16: "superres_16",
    Scenario.SPARSE_5: "sparse_5",
    Scenario.SPARSE_1p5: "sparse_1p5",
}

# Accuracy + calibration metrics produced per (method, scenario).
NS_FIELD_METRICS: tuple[Metric, ...] = (
    Metric.RMSE,
    Metric.ENERGY_SPEC_RMSE,
    Metric.KL_POINTS,
    Metric.CRPS,
    Metric.SPREAD_SKILL,
)

# Ablation tags consumed by make_tables.render_ablation_body (scenario column).
ABLATION_TAGS: tuple[str, ...] = (
    "ablation:gain_full",
    "ablation:gain_jacfree",
    "ablation:gain_none",
    "ablation:gdiff_sweep",
    "ablation:steps_sweep",
    "ablation:ensemble_sweep",
)


class NavierStokesRunner(ExperimentRunner):
    """Runner for the Navier--Stokes benchmark."""

    case = Case.NAVIER_STOKES

    def __init__(self, config, *, seeds=None):  # type: ignore[no-untyped-def]
        if seeds is None:
            from common.seeding import SEED_LIST

            seeds = SEED_LIST
        super().__init__(config, seeds=seeds)
        self._prior = None  # lazily loaded shared prior (load once)
        # device: top-level override wins, else case.device, else cpu.
        case_device = None
        try:
            case_device = config.case.get("device", None)
        except Exception:
            case_device = None
        self._device = str(self._cfg_get("device", case_device or "cpu"))
        # Cache of the large-E reference ensemble per (scenario, seed) so the
        # headline reference sampler runs once, not once per method (Blocker 1).
        self._ref_cache: dict[tuple[str, int], object] = {}

    # -- config helpers ---------------------------------------------------- #

    def _case_cfg(self):  # type: ignore[no-untyped-def]
        return self.config.case

    def _scenario_cfgs(self) -> dict:
        """Load the scenario configs referenced by this run (by name)."""
        from omegaconf import OmegaConf
        from pathlib import Path

        cfgs: dict[str, object] = {}
        root = Path(__file__).resolve().parents[2] / "configs" / "scenario"
        for scen in self.scenarios():
            name = SCENARIO_CONFIG_NAME[scen]
            cfgs[scen.value] = OmegaConf.load(root / f"{name}.yaml")
        return cfgs

    def _method_cfgs(self) -> dict:
        from omegaconf import OmegaConf
        from pathlib import Path

        cfgs: dict[Method, object] = {}
        root = Path(__file__).resolve().parents[2] / "configs" / "method"
        for m, name in METHOD_CONFIG_NAME.items():
            cfgs[m] = OmegaConf.load(root / f"{name}.yaml")
        return cfgs

    def _ensure_prior(self):  # type: ignore[no-untyped-def]
        if self._prior is None:
            self._prior = _ns_pipeline.load_prior(self._case_cfg(), self._device)
        return self._prior

    def _reference_trajectory(self, ctx, prior, scen_cfg, obs_operator, truth_obs):  # type: ignore[no-untyped-def]
        """Return (cached) large-E reference ensemble for KL-at-points.

        Drawn once per (scenario, seed) by the headline SI-SDE sampler on the
        same truth+obs+mask (spec Section 9). ``reference_ensemble_size`` comes
        from the case config (kept small for the smoke run).
        """
        key = (ctx.scenario.value, ctx.seed)
        if key in self._ref_cache:
            return self._ref_cache[key]
        case_cfg = self._case_cfg()
        ref_size = int(
            self._cfg_get(
                "reference_ensemble_size",
                case_cfg.get("reference_ensemble_size", 64),
            )
        )
        ref = _ns_pipeline.build_reference_trajectory(
            prior=prior,
            method_cfg=self._method_cfgs()[Method.OURS_SI_SDE],
            truth_obs=truth_obs,
            obs_operator=obs_operator,
            variance=ctx.extra["variance"],
            likelihood_ensemble_size=ctx.extra["likelihood_ensemble_size"],
            reference_ensemble_size=ref_size,
            num_steps=ctx.num_steps,
            num_physical_steps=ctx.extra["num_physical_steps"],
            # Match the run's mode so the reference draw is not silently the slow
            # full-Sigma_s (inflated) path when the run is jacobian-free.
            likelihood_mode=self._cfg_get("likelihood_mode", None),
        )
        self._ref_cache[key] = ref
        return ref

    # -- subclass hooks ---------------------------------------------------- #

    def methods(self) -> Sequence[Method]:
        names = self._cfg_get("ns_methods", None)
        if names:
            return tuple(Method(n) for n in names)
        return NS_METHODS

    def scenarios(self) -> Sequence[Scenario]:
        names = self._cfg_get("ns_scenarios", None)
        if names:
            return tuple(Scenario(n) for n in names)
        return NS_MAIN_SCENARIOS + NS_APPENDIX_SCENARIOS

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
                "variance": float(case_cfg.variance),
                "test_index": int(self._cfg_get("test_index", case_cfg.test_sample_indices[0])),
                "likelihood_ensemble_size": int(case_cfg.likelihood_ensemble_size),
            },
        )

    # -- per-(method, scenario, seed) evaluation --------------------------- #

    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        if ctx.method not in WIRED_METHODS:
            yield from self._todo_rows(ctx)
            return

        prior = self._ensure_prior()
        scen_cfg = self._scenario_cfgs()[ctx.scenario.value]
        method_cfg = self._method_cfgs()[ctx.method]
        device = self._device
        extra = ctx.extra

        data_size = (
            prior.test_dataset.num_channels,
            prior.test_dataset.height,
            prior.test_dataset.width,
        )

        obs_operator = _ns_pipeline.build_obs_operator(
            scenario_cfg=scen_cfg,
            data_size=data_size,
            mask_seed=mask_seed(self.case.value, ctx.scenario.value),
        )

        truth_obs = _ns_pipeline.prepare_truth_and_obs(
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

        model, posterior, stepper = _ns_pipeline.build_posterior(
            method_name=ctx.method.value,
            method_cfg=method_cfg,
            prior=prior,
            obs_operator=obs_operator,
            variance=extra["variance"],
            likelihood_ensemble_size=extra["likelihood_ensemble_size"],
            likelihood_mode=self._cfg_get("likelihood_mode", None),
        )

        nfe = _ns_pipeline.attach_nfe_counter(model)

        result = _ns_pipeline.run_assimilation(
            posterior=posterior,
            model=model,
            truth_obs=truth_obs,
            ensemble_size=ctx.ensemble_size,
            num_steps=ctx.num_steps,
            num_physical_steps=extra["num_physical_steps"],
            stepper=stepper,
            nfe_counter=nfe,
            gaussian_base=(ctx.method in (Method.OURS_FM_SDE, Method.OURS_FM_ODE)),
        )

        # Large-E reference ensemble for KL-at-points (cached per scenario+seed).
        reference = self._reference_trajectory(
            ctx, prior, scen_cfg, obs_operator, truth_obs
        )

        metrics = _ns_pipeline.compute_metrics(
            result=result,
            obs_operator=obs_operator,
            len_field_history=prior.len_field_history,
            reference_trajectory=reference,
        )

        logger.info(
            "[NS] %s | %s | seed=%d | E=%d M=%d | %s",
            ctx.method.value,
            ctx.scenario.value,
            ctx.seed,
            ctx.ensemble_size,
            ctx.num_steps,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        )

        # Optionally emit the figures (first wired method, main scenario, seed 0).
        if self._cfg_get("save_figures", False) and ctx.method == WIRED_METHODS[0] and ctx.seed == self.seeds[0]:
            self._save_figures(ctx, result, obs_operator, prior.len_field_history)

        yield from self._metric_rows(ctx, metrics)

    # -- ablation entrypoint (spec Section 7) ------------------------------ #

    def run_ablation(self, *, aggregate: bool = True) -> list[ResultRecord]:
        """Drive the ablation sweep and collect tidy ``tab:ablation`` rows.

        ``run()`` only invokes :meth:`evaluate`; this is the dedicated entrypoint
        for :meth:`evaluate_ablation` (the ablation table is otherwise never
        produced). Runs the FM-SDE sweep on a single scenario (the first of
        ``self.scenarios()``; override with ``ablation_scenario`` in config) over
        the seed list, then aggregates per (case, method, scenario-tag, metric,
        E, M) so distinct sweep points survive.
        """
        from common.seeding import seed_everything
        from common.aggregation import aggregate_over_seeds

        ablation_scenario = self._cfg_get("ablation_scenario", None)
        if ablation_scenario:
            scen = Scenario(ablation_scenario)
        else:
            scen = self.scenarios()[0]

        raw: list[ResultRecord] = []
        for seed in self.seeds:
            seed_everything(seed)
            ctx = self.make_context(Method.OURS_FM_SDE, scen, seed)
            raw.extend(self.evaluate_ablation(ctx))
        return aggregate_over_seeds(raw) if aggregate else raw

    def evaluate_ablation(self, ctx: RunContext) -> Iterable[ResultRecord]:
        """Produce tab:ablation rows (FM-SDE on one scenario, spec Section 7).

        Each axis emits RMSE/CRPS/spread-skill rows. To keep distinct sweep
        points from collapsing under across-seed aggregation (which groups by
        ``(case, method, scenario, metric, E, M)``), every point is emitted with:
          * a per-point tag ``ablation:<axis>:<value>`` carrying the ACTUAL swept
            E / M, so steps (M=10/50/100) and ensemble (E=16/64/256) points stay
            distinct in the tidy file; PLUS
          * a canonical-axis tag (``ablation:steps_sweep`` etc.) for ONE
            representative point per axis, which is what ``make_tables``'
            ``render_ablation_body`` keys off (one row per axis).

        Axes:
        * gain: ``dps_full`` (full G_tau) / ``dps_jacobian_free`` (Jacobian-free)
          / ``inflated`` (no correction, G=I) -> gain_full / gain_jacfree /
          gain_none.
        * g_tau: includes g=0 (== FM-ODE) -> gdiff_sweep.
        * M and E sweeps -> steps_sweep / ensemble_sweep.
        For the smoke run a cheaper subset (2 points/axis, isotropic gain) runs.
        """
        prior = self._ensure_prior()
        scen = ctx.scenario
        scen_cfg = self._scenario_cfgs()[scen.value]
        method_cfg = self._method_cfgs()[Method.OURS_FM_SDE]
        device = self._device
        extra = ctx.extra

        data_size = (
            prior.test_dataset.num_channels,
            prior.test_dataset.height,
            prior.test_dataset.width,
        )
        obs_operator = _ns_pipeline.build_obs_operator(
            scenario_cfg=scen_cfg,
            data_size=data_size,
            mask_seed=mask_seed(self.case.value, scen.value),
        )
        truth_obs = _ns_pipeline.prepare_truth_and_obs(
            prior=prior,
            test_index=extra["test_index"],
            obs_operator=obs_operator,
            variance=extra["variance"],
            num_physical_steps=extra["num_physical_steps"],
            obs_noise_seed=obs_seed(
                self.case.value, scen.value, extra["test_index"], ctx.seed
            ),
            device=device,
        )

        smoke = bool(self._cfg_get("ablation_smoke", True))
        base_M = ctx.num_steps
        base_E = ctx.ensemble_size

        # Axis points: (canonical_axis_tag, mode, g_mult, num_steps, E). The first
        # entry of each axis is the representative that also fills the canonical
        # make_tables row.
        if smoke:
            # Cheap: isotropic gain modes + small E/M so it completes on CPU.
            gain_points = [
                ("gain_jacfree", "dps_jacobian_free", 1.0, base_M, base_E),
                ("gain_none", "dps_jacobian_free", 1.0, base_M, base_E),
            ]
            gdiff_points = [
                ("gdiff_sweep", "dps_jacobian_free", 0.0, base_M, base_E),  # g=0 (FM-ODE)
                ("gdiff_sweep", "dps_jacobian_free", 1.0, base_M, base_E),
            ]
            steps_points = [
                ("steps_sweep", "dps_jacobian_free", 1.0, 5, base_E),
                ("steps_sweep", "dps_jacobian_free", 1.0, 10, base_E),
            ]
            ens_points = [
                ("ensemble_sweep", "dps_jacobian_free", 1.0, base_M, 2),
                ("ensemble_sweep", "dps_jacobian_free", 1.0, base_M, 4),
            ]
        else:
            # Full-scale (GPU): paper's three gain modes + 10/50/100, 16/64/256.
            gain_points = [
                ("gain_full", "dps_full", 1.0, base_M, base_E),
                ("gain_jacfree", "dps_jacobian_free", 1.0, base_M, base_E),
                ("gain_none", "inflated", 1.0, base_M, base_E),  # G=I (no correction)
            ]
            gdiff_points = [
                ("gdiff_sweep", "inflated", 0.0, base_M, base_E),  # g=0 (FM-ODE)
                ("gdiff_sweep", "inflated", 0.5, base_M, base_E),
                ("gdiff_sweep", "inflated", 1.0, base_M, base_E),
            ]
            steps_points = [
                ("steps_sweep", "inflated", 1.0, 10, base_E),
                ("steps_sweep", "inflated", 1.0, 50, base_E),
                ("steps_sweep", "inflated", 1.0, 100, base_E),
            ]
            ens_points = [
                ("ensemble_sweep", "inflated", 1.0, base_M, 16),
                ("ensemble_sweep", "inflated", 1.0, base_M, 64),
                ("ensemble_sweep", "inflated", 1.0, base_M, 256),
            ]

        def _run(mode, g_mult, num_steps, E):  # type: ignore[no-untyped-def]
            model, posterior, stepper = _ns_pipeline.build_posterior(
                method_name=Method.OURS_FM_SDE.value,
                method_cfg=method_cfg,
                prior=prior,
                obs_operator=obs_operator,
                variance=extra["variance"],
                likelihood_ensemble_size=extra["likelihood_ensemble_size"],
                likelihood_mode=mode,
                g_multiplier=g_mult,
            )
            nfe = _ns_pipeline.attach_nfe_counter(model)
            result = _ns_pipeline.run_assimilation(
                posterior=posterior,
                model=model,
                truth_obs=truth_obs,
                ensemble_size=E,
                num_steps=num_steps,
                num_physical_steps=extra["num_physical_steps"],
                stepper=stepper,
                nfe_counter=nfe,
                gaussian_base=True,
            )
            # KL is not emitted for ablation rows (only rmse/crps/spread_skill),
            # so a reference ensemble is unnecessary here.
            return _ns_pipeline.compute_metrics(
                result, obs_operator, prior.len_field_history
            )

        for axis_points in (gain_points, gdiff_points, steps_points, ens_points):
            # The gain axis has THREE distinct make_tables rows (gain_full /
            # gain_jacfree / gain_none -- one canonical tag each), so every point
            # emits its own canonical tag. The sweep axes (gdiff/steps/ensemble)
            # are a SINGLE make_tables row each, so only the representative (first)
            # point fills the canonical tag; all points still get a distinct
            # per-point tag so aggregation never collapses the sweep.
            axis_labels = [p[0] for p in axis_points]
            per_point_canonical = len(set(axis_labels)) == len(axis_labels)
            for i, (axis, mode, g, M, E) in enumerate(axis_points):
                metrics = _run(mode, g, M, E)
                if axis == "steps_sweep":
                    point_val = f"M{M}"
                elif axis == "ensemble_sweep":
                    point_val = f"E{E}"
                elif axis == "gdiff_sweep":
                    point_val = f"g{g}"
                else:  # gain axis: encode the mode
                    point_val = mode
                yield from self._ablation_rows(
                    ctx, f"ablation:{axis}:{point_val}", metrics, E=E, M=M
                )
                if per_point_canonical or i == 0:
                    yield from self._ablation_rows(
                        ctx, f"ablation:{axis}", metrics, E=E, M=M
                    )

    # -- record emitters --------------------------------------------------- #

    def _metric_rows(self, ctx, metrics):  # type: ignore[no-untyped-def]
        for metric in NS_FIELD_METRICS:
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

    def _ablation_rows(self, ctx, tag, metrics, E=None, M=None):  # type: ignore[no-untyped-def]
        # E / M carry the ACTUAL swept values (not the base ctx values) so the
        # across-seed aggregation keeps distinct sweep points separate.
        E = ctx.ensemble_size if E is None else E
        M = ctx.num_steps if M is None else M
        for metric in (Metric.RMSE, Metric.CRPS, Metric.SPREAD_SKILL):
            yield ResultRecord(
                case=self.case.value,
                method=Method.OURS_FM_SDE.value,
                scenario=tag,
                metric=metric.value,
                value=metrics[metric.value],
                E=E,
                M=M,
                seed=ctx.seed,
                nfe=metrics["nfe"],
                seconds=metrics["seconds"],
            )

    def _todo_rows(self, ctx):  # type: ignore[no-untyped-def]
        """Emit NaN placeholder rows for unimplemented baselines (Phase 4)."""
        for metric in NS_FIELD_METRICS + (Metric.NFE, Metric.SECONDS):
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

    # -- figures ----------------------------------------------------------- #

    def _save_figures(self, ctx, result, obs_operator, len_field_history):  # type: ignore[no-untyped-def]
        from pathlib import Path

        fig_root = Path(
            str(self._cfg_get("figures_root", "paper_new/figures/results"))
        )
        traj_path = _ns_figures.save_ns_trajectories(
            result, obs_operator, fig_root / "ns_trajectories.png"
        )
        diag_path = _ns_figures.save_ns_diagnostics(
            result, len_field_history, fig_root / "ns_diagnostics.png"
        )
        logger.info("[NS] wrote figures: %s, %s", traj_path, diag_path)


__all__ = ["NavierStokesRunner", "NS_METHODS", "NS_MAIN_SCENARIOS"]
