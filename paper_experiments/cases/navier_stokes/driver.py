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
* unified samplers -- ``scisi.posterior_models`` SI-SDE / DM-SDE / FM-ODE (E4).
* obs operators -- ``scisi.likelihood_models.observation_operators``: block-
  average super-res (E1) and seeded sparse masks (Section 9).
* metrics -- ensemble-mean RMSE (m1), log-spectrum energy RMSE (E11), unbiased
  CRPS + spread-skill (E7), KL-at-points (E9), NFE + wall-clock (E10).

Classical true-solver baselines EnKF / LETKF / PF / EnSF are wired through
``_evaluate_classical`` (jax-cfd solver forecast; EnSF uses a score-based
analysis). SDA + the generative DPS/FIG baselines are wired too. FlowDAS is wired.
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

# REDUCED paper method lineup (2026-07-01), grouped by (prior, sampler). The two
# "Ours" likelihood-covariance modes (jacfree / shared) are NOT separate methods
# here -- they are the same three samplers run under two ``likelihood_mode``
# settings, tagged apart by the tidy ``variant`` column (see ``_variant``). Dropped
# from the earlier lineup: Guided FM (FIG), Guided FM (OT-ODE), standalone SURGE,
# LETKF, Ensemble score filter.
NS_METHODS: tuple[Method, ...] = (
    # Ours (unified family) -- run twice (jacfree + shared) by the master script.
    Method.OURS_SI_SDE,
    Method.OURS_FM_ODE,
    Method.OURS_DM_SDE,  # DM-SDE
    # SI + SDE.
    Method.FLOWDAS,
    Method.SURGE_FLOWDAS,  # FlowDAS + SURGE
    # Diffusion model + SDE.
    Method.SDA,
    Method.SURGE_SDA,  # SDA + SURGE
    # Flow matching + ODE.
    Method.D_FLOW_SGLD,
    Method.GUIDED_FM_FIG,  # FIG measurement-interpolant corrector
    # Classical (ground-truth EnKF + conventional baseline).
    Method.ENKF,
    Method.PARTICLE_FILTER,
)

# Generative methods with a working build_posterior branch. D_FLOW_SGLD and SURGE
# are in NS_METHODS but NOT here yet (they fall through to TODO rows) until their
# samplers are implemented.
WIRED_METHODS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_DM_SDE,
    Method.OURS_FM_ODE,
    Method.FLOWDAS,
    Method.GUIDED_FM_FIG,
    Method.D_FLOW_SGLD,
    Method.SDA,
    Method.SURGE_SDA,
    Method.SURGE_FLOWDAS,
)

# The three "Ours" samplers -- the only methods that carry a ``variant`` tag
# (their likelihood-covariance mode). Every other method runs in a single mode.
OURS_METHODS: frozenset[Method] = frozenset(
    (Method.OURS_SI_SDE, Method.OURS_DM_SDE, Method.OURS_FM_ODE)
)

# likelihood_mode -> the tidy ``variant`` tag written for the Ours rows.
VARIANT_FROM_MODE: dict[str, str] = {
    "dps_jacobian_free": "jacfree",
    "inflated_shared": "shared",
}

# Classical / non-posterior DA baselines routed through ``_evaluate_classical``.
# ALL of these are TRUE-SOLVER (jax-cfd, no learned prior): ENKF, LETKF and
# PARTICLE_FILTER propagate the ensemble with the real solver and use a classical
# (Kalman / transform / weight) analysis; ENSEMBLE_SCORE_FILTER (EnSF) likewise
# forecasts with the true solver but uses a SCORE-based analysis update. jax is
# imported lazily (only when one of these methods runs).
CLASSICAL_METHODS: tuple[Method, ...] = (
    Method.ENKF,
    Method.LETKF,
    Method.PARTICLE_FILTER,
    Method.ENSEMBLE_SCORE_FILTER,
)

# Maps a Method to the YAML method-config name under configs/method/. Every entry
# here is loaded eagerly by ``_method_cfgs`` so each must have a config file (the
# two not-yet-wired methods carry placeholder configs).
METHOD_CONFIG_NAME: dict[Method, str] = {
    Method.OURS_SI_SDE: "si_sde",
    Method.OURS_DM_SDE: "dm_sde",
    Method.OURS_FM_ODE: "fm_ode",
    Method.FLOWDAS: "flowdas",
    Method.GUIDED_FM_FIG: "guided_fm_fig",
    Method.D_FLOW_SGLD: "dflow_sgld",
    Method.SDA: "sda",
    Method.SURGE_SDA: "surge_sda",
    Method.SURGE_FLOWDAS: "surge_flowdas",
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
    Metric.CRPS_OBSERVED,
    Metric.CRPS_UNOBSERVED,
    Metric.SPREAD_SKILL,
)

# Ablation tags consumed by make_tables.render_ablation_body (scenario column).
ABLATION_TAGS: tuple[str, ...] = (
    "ablation:cov_inflated",
    "ablation:cov_jacfree",
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

        # Reference-GENERATION runs (run_ns_reference.sh) set skip_kl_reference:
        # their output IS the KL reference, so drawing a separate SI-SDE reference
        # here is pointless -- and in the config-default ``inflated`` likelihood
        # mode it costs hours per cell (O(E_ref * N_y) JVPs/pseudo-step). KL is
        # then NaN in the reference run's own CSV, which is correct.
        if self._cfg_get("skip_kl_reference", False):
            self._ref_cache[key] = None
            return None

        # Ground-truth posterior reference: when ``kl_reference_states`` is set,
        # load the large-E true-solver EnKF ensemble (saved by a prior E=1000
        # non-localized EnKF run) for THIS (scenario, seed) and use it as the
        # KL-at-points reference, instead of the (self-biased) SI-SDE self-draw.
        # The same seeding makes the EnKF run's truth+obs identical to this cell's,
        # so its posterior ensemble is a valid reference for the same observations.
        # Returns None (-> KL NaN) when no matching reference file exists.
        ref_dir = self._cfg_get("kl_reference_states", None)
        if ref_dir:
            import glob
            import re
            from pathlib import Path

            import numpy as np
            import torch

            slug = re.sub(r"[^A-Za-z0-9]+", "_", str(ctx.scenario.value)).strip("_")
            pattern = str(
                Path(ref_dir) / f"{self.case.value}__EnKF__{slug}__seed{ctx.seed}__*.npz"
            )
            matches = sorted(glob.glob(pattern))
            ref = None
            if matches:
                data = np.load(matches[0], allow_pickle=True)
                ref = torch.from_numpy(data["posterior_trajectory"]).float()
                logger.info(
                    "[NS] KL reference = EnKF E=%d states %s",
                    ref.shape[0], Path(matches[0]).name,
                )
            else:
                logger.warning(
                    "[NS] no EnKF KL reference for %s seed=%d (KL -> NaN)",
                    ctx.scenario.value, ctx.seed,
                )
            self._ref_cache[key] = ref
            return ref

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
        if ctx.method in CLASSICAL_METHODS:
            yield from self._evaluate_classical(ctx)
            return
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
            # Shared-Jacobian refresh cadence (Ours + inflated_shared only);
            # override with jacobian_refresh_every=k, default 1 (every step).
            jacobian_refresh_every=self._cfg_get("jacobian_refresh_every", None),
            # Inflated-mode stabilisation knobs (Ours + inflated / inflated_shared
            # only); defaults reproduce current behaviour.
            jacobian_damping=self._cfg_get("jacobian_damping", None),
            sigma_bar_eig_floor=self._cfg_get("sigma_bar_eig_floor", None),
            isotropic_front_factor=self._cfg_get("isotropic_front_factor", None),
            residual_step_cap=self._cfg_get("residual_step_cap", None),
            # FM/DM drift-evaluation time offset (0=left default, 1=right endpoint).
            drift_time_shift=self._cfg_get("drift_time_shift", None),
            # Select the per-cell guidance scale (e.g. FlowDAS zeta) from the
            # method config's [case][scenario][M] table.
            case_key=self.case.value,
            scenario_key=SCENARIO_CONFIG_NAME[ctx.scenario],
            num_steps=ctx.num_steps,
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
            # Anchor a0 = 0 (FM-path and DM-path: DM-SDE/FM-ODE, FIG, OT-ODE,
            # D-Flow, SDA, SURGE, SURGE+SDA) inits from the N(0, I) latent; only
            # the SI-path methods (SI-SDE, FlowDAS, SURGE+FlowDAS) use the x0
            # point-mass init.
            gaussian_base=ctx.method
            not in (Method.OURS_SI_SDE, Method.FLOWDAS, Method.SURGE_FLOWDAS),
            # Early-abort divergence guard (opt-in; None = off, unchanged runs).
            divergence_rmse_threshold=self._cfg_get(
                "divergence_rmse_threshold", None
            ),
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
            {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if k != "per_step"},
        )

        # Optionally emit the figures (first wired method, main scenario, seed 0).
        if self._cfg_get("save_figures", False) and ctx.method == WIRED_METHODS[0] and ctx.seed == self.seeds[0]:
            self._save_figures(ctx, result, obs_operator, prior.len_field_history)

        # Optionally persist the raw posterior + truth states (traj1 only, driven
        # by the master script's save_states flag).
        if self._cfg_get("save_states", False):
            self._save_states(ctx, result, truth_obs, obs_operator, metrics=metrics)
        # Optionally persist the per-step metric curves (all trajectories).
        if self._cfg_get("save_per_step", False):
            self._save_per_step(ctx, metrics)

        yield from self._metric_rows(ctx, metrics)

    # -- classical true-solver baselines (EnKF / PF) ----------------------- #

    def _evaluate_classical(self, ctx: RunContext) -> Iterable[ResultRecord]:
        """Run a classical / score-filter baseline routed off the posterior path.

        Shares the SAME truth + observations + sensor mask + obs noise R as the
        generative methods: the prior (for its dataset + preprocesser + obs
        construction) and the obs operator are built exactly as in ``evaluate``,
        and ``truth_obs.observations`` is assimilated verbatim.

        * ENKF / LETKF / PARTICLE_FILTER propagate ensemble members with the real
          jax-cfd stochastic NS solver at 256^2 (no learned prior) and apply a
          classical (Kalman / local-transform / weight-resample) analysis.
        * ENSEMBLE_SCORE_FILTER (EnSF) ALSO forecasts with the true jax-cfd solver
          (no learned prior), but uses a SCORE-based analysis update instead of the
          Kalman gain.

        ``enkf_baseline`` is imported lazily so the jax import cost is only paid
        when one of these methods runs.
        """
        prior = self._ensure_prior()
        scen_cfg = self._scenario_cfgs()[ctx.scenario.value]
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

        from cases.navier_stokes import enkf_baseline  # lazy: pulls in jax (CPU)

        if ctx.method == Method.ENKF:
            # Localization geometry is only coherent for selection (sparse) masks
            # whose obs points are co-located on the lifted 256^2 grid; the
            # block-average super-res operator has no single obs-point geometry.
            # Default to the non-localized EnKF; set ``enkf_localization_radius``
            # in config to enable Gaspari-Cohn localization for sparse scenarios.
            localization_radius = self._cfg_get("enkf_localization_radius", None)
            # OPT-IN correlation-based localization (Vossepoel et al. 2025);
            # default "distance" keeps the Gaspari-Cohn behaviour. Request with
            # ``+enkf_localization_type=correlation``.
            localization_type = self._cfg_get("enkf_localization_type", "distance")
            corr_threshold = self._cfg_get("enkf_corr_threshold", None)
            result = enkf_baseline.run_enkf_baseline(
                truth_obs=truth_obs,
                obs_operator=obs_operator,
                ensemble_size=ctx.ensemble_size,
                variance=extra["variance"],
                num_physical_steps=extra["num_physical_steps"],
                len_field_history=prior.len_field_history,
                seed=ctx.seed,
                localization_radius=localization_radius,
                inflation=float(self._cfg_get("enkf_inflation", 1.0)),
                localization_type=str(localization_type),
                corr_threshold=(
                    float(corr_threshold) if corr_threshold is not None else None
                ),
                corr_inflation_max=float(self._cfg_get("enkf_corr_inflation_max", 4.0)),
                corr_inflation_beta=float(
                    self._cfg_get("enkf_corr_inflation_beta", 0.5)
                ),
            )
        elif ctx.method == Method.LETKF:
            # LETKF is inherently localized: it reads the same ``enkf_localization_radius``
            # knob, but defaults to a sensible radius (not a non-localized variant)
            # when the config leaves it None.
            localization_radius = self._cfg_get("enkf_localization_radius", None)
            # OPT-IN correlation-based localization (shares the same knobs as the
            # EnKF path); default "distance" keeps Gaspari-Cohn.
            localization_type = self._cfg_get("enkf_localization_type", "distance")
            corr_threshold = self._cfg_get("enkf_corr_threshold", None)
            result = enkf_baseline.run_letkf_baseline(
                truth_obs=truth_obs,
                obs_operator=obs_operator,
                ensemble_size=ctx.ensemble_size,
                variance=extra["variance"],
                num_physical_steps=extra["num_physical_steps"],
                len_field_history=prior.len_field_history,
                seed=ctx.seed,
                localization_radius=localization_radius,
                inflation=float(self._cfg_get("enkf_inflation", 1.0)),
                localization_type=str(localization_type),
                corr_threshold=(
                    float(corr_threshold) if corr_threshold is not None else None
                ),
                corr_inflation_max=float(self._cfg_get("enkf_corr_inflation_max", 4.0)),
                corr_inflation_beta=float(
                    self._cfg_get("enkf_corr_inflation_beta", 0.5)
                ),
            )
        elif ctx.method == Method.ENSEMBLE_SCORE_FILTER:
            # EnSF: TRUE-SOLVER forecast + score-based analysis update.
            result = enkf_baseline.run_ensf_baseline(
                truth_obs=truth_obs,
                obs_operator=obs_operator,
                ensemble_size=ctx.ensemble_size,
                variance=extra["variance"],
                num_physical_steps=extra["num_physical_steps"],
                len_field_history=prior.len_field_history,
                seed=ctx.seed,
                analysis_steps=int(self._cfg_get("ensf_analysis_steps", 20)),
            )
        else:  # Method.PARTICLE_FILTER
            result = enkf_baseline.run_particle_filter_baseline(
                truth_obs=truth_obs,
                obs_operator=obs_operator,
                ensemble_size=ctx.ensemble_size,
                variance=extra["variance"],
                num_physical_steps=extra["num_physical_steps"],
                len_field_history=prior.len_field_history,
                seed=ctx.seed,
            )

        # KL-at-points needs the shared large-E reference ensemble (same as the
        # generative methods); reuse the cached headline reference.
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
            "[NS] %s | %s | seed=%d | E=%d | %s",
            ctx.method.value,
            ctx.scenario.value,
            ctx.seed,
            ctx.ensemble_size,
            {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items() if k != "per_step"},
        )

        # Optionally persist the raw posterior + truth states (traj1 only).
        if self._cfg_get("save_states", False):
            self._save_states(ctx, result, truth_obs, obs_operator, metrics=metrics)
        # Optionally persist the per-step metric curves (all trajectories).
        if self._cfg_get("save_per_step", False):
            self._save_per_step(ctx, metrics)

        yield from self._metric_rows(ctx, metrics)

    # -- ablation entrypoint (spec Section 7) ------------------------------ #

    def run_ablation(self, *, aggregate: bool = True) -> list[ResultRecord]:
        """Drive the ablation sweep and collect tidy ``tab:ablation`` rows.

        ``run()`` only invokes :meth:`evaluate`; this is the dedicated entrypoint
        for :meth:`evaluate_ablation` (the ablation table is otherwise never
        produced). Runs the DM-SDE sweep on a single scenario (the first of
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
            ctx = self.make_context(Method.OURS_DM_SDE, scen, seed)
            raw.extend(self.evaluate_ablation(ctx))
        return aggregate_over_seeds(raw) if aggregate else raw

    def evaluate_ablation(self, ctx: RunContext) -> Iterable[ResultRecord]:
        """Produce tab:ablation rows (DM-SDE on one scenario, spec Section 7).

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
        * covariance: ``inflated`` / ``inflated_shared`` (full Sigma_s) vs
          ``dps_jacobian_free`` (isotropic).
        * g_tau: includes g=0 (== FM-ODE) -> gdiff_sweep.
        * M and E sweeps -> steps_sweep / ensemble_sweep.
        For the smoke run a cheaper subset (2 points/axis, isotropic gain) runs.
        """
        prior = self._ensure_prior()
        scen = ctx.scenario
        scen_cfg = self._scenario_cfgs()[scen.value]
        method_cfg = self._method_cfgs()[Method.OURS_DM_SDE]
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
            # Cheap: isotropic covariance + small E/M so it completes on CPU.
            cov_points = [
                ("cov_inflated", "dps_jacobian_free", 1.0, base_M, base_E),
                ("cov_jacfree", "dps_jacobian_free", 1.0, base_M, base_E),
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
            # Full-scale (GPU): the COVARIANCE comparison (inflated vs Jacobian-free)
            # + 10/50/100 steps, 16/64/256 ensemble. The inflated covariance uses
            # `inflated_shared` (ensemble-shared Jacobian; exact `inflated` is
            # O(B*N_y) net-Jacobians/step, days/cell at NS scale). The multiplicative
            # gain (former `dps_full` mode) was removed from the code: it did not
            # improve accuracy (inflated covariance ~0.16 vs gain/cheap on sparse;
            # analytical KL 0.001 inflated vs 0.174 gain), so it was dropped from the
            # paper and the runs. All points use the inflated covariance unless noted.
            cov_points = [
                ("cov_inflated", "inflated_shared", 1.0, base_M, base_E),
                ("cov_jacfree", "dps_jacobian_free", 1.0, base_M, base_E),
            ]
            gdiff_points = [
                ("gdiff_sweep", "inflated_shared", 0.0, base_M, base_E),  # g=0 (FM-ODE)
                ("gdiff_sweep", "inflated_shared", 0.5, base_M, base_E),
                ("gdiff_sweep", "inflated_shared", 1.0, base_M, base_E),
            ]
            steps_points = [
                ("steps_sweep", "inflated_shared", 1.0, 10, base_E),
                ("steps_sweep", "inflated_shared", 1.0, 50, base_E),
                ("steps_sweep", "inflated_shared", 1.0, 100, base_E),
            ]
            ens_points = [
                ("ensemble_sweep", "inflated_shared", 1.0, base_M, 16),
                ("ensemble_sweep", "inflated_shared", 1.0, base_M, 64),
                ("ensemble_sweep", "inflated_shared", 1.0, base_M, 256),
            ]

        def _run(mode, g_mult, num_steps, E):  # type: ignore[no-untyped-def]
            model, posterior, stepper = _ns_pipeline.build_posterior(
                method_name=Method.OURS_DM_SDE.value,
                method_cfg=method_cfg,
                prior=prior,
                obs_operator=obs_operator,
                variance=extra["variance"],
                likelihood_ensemble_size=extra["likelihood_ensemble_size"],
                likelihood_mode=mode,
                g_multiplier=g_mult,
                case_key=self.case.value,
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

        for axis_points in (cov_points, gdiff_points, steps_points, ens_points):
            # The covariance axis has TWO distinct make_tables rows (cov_inflated /
            # cov_jacfree -- one canonical tag each), so every point emits its own
            # canonical tag. The sweep axes (gdiff/steps/ensemble) are a SINGLE
            # make_tables row each, so only the representative (first) point fills
            # the canonical tag; all points still get a distinct per-point tag so
            # aggregation never collapses the sweep.
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
                else:  # covariance axis: encode the mode
                    point_val = mode
                yield from self._ablation_rows(
                    ctx, f"ablation:{axis}:{point_val}", metrics, E=E, M=M
                )
                if per_point_canonical or i == 0:
                    yield from self._ablation_rows(
                        ctx, f"ablation:{axis}", metrics, E=E, M=M
                    )

    # -- record emitters --------------------------------------------------- #

    def _variant(self, method) -> str | None:  # type: ignore[no-untyped-def]
        """Tidy ``variant`` tag for a cell: the Ours likelihood-covariance mode.

        Only the three Ours samplers carry a variant (``jacfree`` /``shared``,
        from ``likelihood_mode``); every baseline/classical method runs in a
        single mode and gets ``None`` (empty cell). Lets both Ours modes coexist
        as distinct rows under the same canonical method label.
        """
        if method not in OURS_METHODS:
            return None
        # Explicit override (e.g. distinguish an ablation like the cached-Jacobian
        # shared mode from the every-step shared mode, which share a likelihood_mode).
        override = self._cfg_get("variant_override", None)
        if override:
            return str(override)
        mode = self._cfg_get("likelihood_mode", None)
        return VARIANT_FROM_MODE.get(str(mode)) if mode is not None else None

    def _metric_rows(self, ctx, metrics):  # type: ignore[no-untyped-def]
        variant = self._variant(ctx.method)
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
                variant=variant,
            )
        # Cost rows (so the calibration/cost table can read NFE / seconds). Timing
        # is on EVERY row too (nfe/seconds columns); these are the explicit metric
        # rows so a table can read cost through the same machinery.
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
                variant=variant,
            )

    def _save_per_step(self, ctx, metrics):  # type: ignore[no-untyped-def]
        """Append this cell's per-step metric curves to ``per_step_file``.

        Enabled with ``save_per_step=true`` (path from ``per_step_file``). Lets
        trajectories 2..N keep their full metric-vs-step history WITHOUT saving the
        raw ensemble states (which is done only for the first trajectory). Carries
        the run timing (nfe/seconds per step) on every row.
        """
        from common.per_step_io import append_per_step, per_step_rows

        per_step = (metrics or {}).get("per_step", {})
        if not per_step:
            return
        path = self._cfg_get("per_step_file", None)
        if not path:
            logger.warning("[NS] save_per_step set but no per_step_file; skipping")
            return
        rows = per_step_rows(
            case=self.case.value,
            method=ctx.method.value,
            scenario=ctx.scenario.value,
            variant=self._variant(ctx.method),
            E=ctx.ensemble_size,
            M=ctx.num_steps,
            seed=ctx.seed,
            test_index=int(ctx.extra.get("test_index", -1)),
            nfe=metrics.get("nfe"),
            seconds=metrics.get("seconds"),
            per_step=per_step,
        )
        append_per_step(path, rows)
        logger.info("[NS] per-step curves -> %s (%d rows)", path, len(rows))

    def _ablation_rows(self, ctx, tag, metrics, E=None, M=None):  # type: ignore[no-untyped-def]
        # E / M carry the ACTUAL swept values (not the base ctx values) so the
        # across-seed aggregation keeps distinct sweep points separate.
        E = ctx.ensemble_size if E is None else E
        M = ctx.num_steps if M is None else M
        for metric in (Metric.RMSE, Metric.CRPS, Metric.SPREAD_SKILL):
            yield ResultRecord(
                case=self.case.value,
                method=Method.OURS_DM_SDE.value,
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
        variant = self._variant(ctx.method)
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
                variant=variant,
            )

    # -- raw state persistence (off by default) ---------------------------- #

    def _save_states(self, ctx, result, truth_obs, obs_operator, metrics=None):  # type: ignore[no-untyped-def]
        """Persist one cell's raw posterior + truth (+ observations) to ``.npz``.

        When ``metrics`` (the ``compute_metrics`` dict) is passed, the per-step
        metric curves (``metrics['per_step']``) are saved as ``per_step_*``
        arrays so the metrics-vs-time plots need no recomputation.

        Off by default; enable with ``save_states=true`` (``states_root`` sets the
        directory, default ``paper_experiments/results/states``). Lets figure
        generation and post-hoc analysis avoid recomputing the (sometimes
        expensive) samplers. Self-contained: stores the posterior ensemble
        trajectory ``[E, C, H, W, T]``, the truth ``[1, C, H, W, T]``, the
        assimilated observations, the sensor indices (sparse operators), the
        per-step cost, and the run metadata. Everything is reproducible from the
        seed regardless, so this is purely a convenience cache.
        """
        import re
        from pathlib import Path

        import numpy as np

        root = Path(str(self._cfg_get("states_root", "paper_experiments/results/states")))
        root.mkdir(parents=True, exist_ok=True)

        def _slug(s: object) -> str:
            return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

        # Variant (Ours jacfree/shared) goes in the filename so both modes' states
        # coexist for the same (method, scenario, seed) instead of clobbering.
        variant = self._variant(ctx.method)
        var_tag = f"__{variant}" if variant else ""
        path = root / (
            f"{self.case.value}__{_slug(ctx.method.value)}__{_slug(ctx.scenario.value)}"
            f"{var_tag}__seed{ctx.seed}__E{ctx.ensemble_size}_M{ctx.num_steps}.npz"
        )

        def _np(x):  # type: ignore[no-untyped-def]
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        # Per-(assimilation-)step metric curves, so the metrics-vs-time plots can
        # be made without recomputing. Saved as ``per_step_<metric>`` arrays
        # (length = number of scored steps). Empty when metrics aren't passed.
        per_step = (metrics or {}).get("per_step", {}) if metrics else {}
        per_step_arrays = {
            f"per_step_{k}": np.asarray(v, dtype=np.float64)
            for k, v in per_step.items()
        }

        np.savez_compressed(
            path,
            posterior_trajectory=_np(result.posterior_trajectory),  # [E, C, H, W, T]
            true_trajectory=_np(result.true_trajectory),            # [1, C, H, W, T]
            observations=_np(truth_obs.observations),               # [1, N_y, T]
            obs_indices=_np(getattr(obs_operator, "obs_indices", np.empty(0))),
            nfe_per_step=float(result.nfe_per_step),
            seconds_per_step=float(result.seconds_per_step),
            # Explicit per-run timing (s/step is the manuscript cost metric; the
            # total is s/step x assimilated steps for that run).
            seconds_total=float(result.seconds_per_step)
            * float(_np(result.posterior_trajectory).shape[-1]),
            method=ctx.method.value,
            scenario=ctx.scenario.value,
            variant=variant if variant else "",
            test_index=int(ctx.extra.get("test_index", -1)),
            seed=int(ctx.seed),
            E=int(ctx.ensemble_size),
            M=int(ctx.num_steps),
            **per_step_arrays,
        )
        logger.info("[NS] saved states -> %s", path)

    # -- figures ----------------------------------------------------------- #

    def _save_figures(self, ctx, result, obs_operator, len_field_history):  # type: ignore[no-untyped-def]
        from pathlib import Path

        fig_root = Path(
            str(self._cfg_get("figures_root", "manuscript/figures/results"))
        )
        traj_path = _ns_figures.save_ns_trajectories(
            result, obs_operator, fig_root / "ns_trajectories.png"
        )
        diag_path = _ns_figures.save_ns_diagnostics(
            result, len_field_history, fig_root / "ns_diagnostics.png"
        )
        logger.info("[NS] wrote figures: %s, %s", traj_path, diag_path)


__all__ = ["NavierStokesRunner", "NS_METHODS", "NS_MAIN_SCENARIOS"]
