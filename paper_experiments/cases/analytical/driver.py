"""Case 1 driver -- analytical linear--Gaussian (spec Section 4, GAP E4).

Closed-form posterior, no network trained. The correctness probe: every sampler
must reproduce the exact mean/cov as ``M`` grows in the ``inflated`` likelihood
mode, and the multiplicative-correction ablation (``inflated`` vs ``dps_full`` vs
``dps_jacobian_free``) must show the inflated mode is exact while the DPS
surrogates plateau away from it.

Produces:
* Table ``tab:analytical_results`` -- KL and sliced-W2 to the exact posterior, for
  SI-SDE / FM-SDE / FM-ODE / FlowDAS / Guided FM / Guided diffusion / EnKF /
  particle filter, at matched E, M. Emitted via the tidy schema.
* Figure ``fig:analytical_panels`` -- see ``figures.py``.

The samplers (``samplers.py``) are compact closed-form vector SDE/ODE integrators
on the analytic Gaussian prior; the KL / sliced-W2 estimators are reused from
``paper/scripts/analytical_utils/kl_divergence.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch

from common.runner import ExperimentRunner, RunContext
from common.seeding import derive_seed
from results_schema import Case, Method, Metric, ResultRecord, Scenario

import torch.nn as nn

from scisi.likelihood_models.dflow import DFlowSGLDLikelihood
from scisi.likelihood_models.gaussian_likelihood import (
    FlowdasGaussianLikelihood,
    InterpolantGaussianLikelihood,
)
from scisi.likelihood_models.guidance import GuidanceGaussianLikelihood
from scisi.likelihood_models.sda import SDALikelihood
from scisi.posterior_models.diffusion_posterior import DiffusionPosterior
from scisi.posterior_models.dflow_posterior import DFlowPosterior
from scisi.posterior_models.flow_matching_posterior import (
    FlowMatchingPosterior,
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.posterior_models.surge_posterior import SurgePosterior

from .prior_models import (
    build_dm_model,
    build_fm_model,
    build_identity_obs_operator,
    build_si_model,
)
from .samplers import (
    GaussianSystem,
    enkf_posterior,
    particle_filter_posterior,
    sample_posterior,
)

# Reuse the validated analytic KL / sliced-W2 estimators.
_PAPER_SCRIPTS = Path(__file__).resolve().parents[3] / "paper" / "scripts"
if str(_PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PAPER_SCRIPTS))
from analytical_utils.kl_divergence import (  # noqa: E402
    gaussian_kl_divergence,
    wasserstein_distance,
)

# Methods present in tab:analytical_results (no SDA / ensemble score filter rows).
ANALYTICAL_METHODS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_FM_SDE,
    Method.OURS_FM_ODE,
    Method.FLOWDAS,
    Method.GUIDED_FM_OTODE,
    Method.D_FLOW_SGLD,
    Method.SDA,
    Method.SURGE,
    Method.ENKF,
    Method.PARTICLE_FILTER,
)

# Map each method to (sampler, likelihood_mode). Our three samplers run the
# exact (inflated) mode. The new posterior-sampling baselines (FIG, OT-ODE,
# D-Flow SGLD, SDA, SURGE) are dispatched to their own closed-form samplers in
# ``_sample_and_count`` rather than via this table.
_METHOD_SPEC: dict[Method, tuple[str, str]] = {
    Method.OURS_SI_SDE: ("si_sde", "inflated"),
    Method.OURS_FM_SDE: ("fm_sde", "inflated"),
    Method.OURS_FM_ODE: ("fm_ode", "inflated"),
}

# Fixed analytic system parameters (spec Section 4).
_OBS_VAR = 1.0  # R = I
_PRIOR_VAR = 1.0  # Cov(x1 | x0) = I
_G0 = 1.0  # diffusion-strength base (gamma_tau = g0 (1 - tau))
_FLOWDAS_J = 16  # Monte-Carlo samples for the FlowDAS likelihood (reported)
_N_EVAL = 4096  # samples drawn for the metric estimates

# Locked hyperparameters for the new baselines (configs/method/*.yaml).
# OT-ODE preconditioner noise: the NS-locked sigma_y^2 = 0 assumes noiseless
# observations and collapses the spread under this case's H = I, R = 1 (full,
# noisy observation). Use the regime-appropriate sigma_y^2 = R and gamma_t = 1
# (gamma_t = 4 still over-contracts here); these recover the exact posterior.
_OTODE_OBS_VAR = _OBS_VAR  # OT-ODE sigma_y^2 = R in the preconditioner
_OTODE_GAMMA = 1.0  # OT-ODE adaptive weight gamma_t (regime-appropriate)
# D-Flow SGLD under-mixes at the NS-locked K = 20 (the chain starts at N(0,I)
# and needs more steps to equilibrate to the posterior); use K = 200 here.
_DFLOW_OPTIM_STEPS = 200  # D-Flow SGLD K (regime-appropriate)
_DFLOW_STEP_SIZE = 0.05  # D-Flow SGLD eta
_DFLOW_NOISE_SCALE = 1.0  # D-Flow SGLD Langevin noise multiplier
_SURGE_GUIDANCE = 1.0  # SURGE guidance strength (surge.yaml)
_SURGE_ESS = 0.5  # SURGE resample threshold ESS/N


class AnalyticalRunner(ExperimentRunner):
    """Runner for the analytical linear--Gaussian case."""

    case = Case.ANALYTICAL

    def methods(self) -> Sequence[Method]:
        return ANALYTICAL_METHODS

    def scenarios(self) -> Sequence[Scenario]:
        return (Scenario.ANALYTICAL,)

    # -- config helpers ---------------------------------------------------- #
    def _dim(self) -> int:
        """Dimension used for the table (the plot dimension, default 2)."""
        cfg = getattr(self.config, "case", None)
        if cfg is not None and hasattr(cfg, "plot_dimension"):
            return int(cfg.plot_dimension)
        return int(self._cfg_get("plot_dimension", 2))

    # -- the per-(method, scenario, seed) work ----------------------------- #
    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        d = self._dim()
        E, M = ctx.ensemble_size, ctx.num_steps
        sys_ = GaussianSystem(d=d, obs_var=_OBS_VAR, prior_var=_PRIOR_VAR)

        # Truth + observation are identical across methods for a given seed.
        sys_seed = derive_seed("analytical", "truth", ctx.seed)
        gx = torch.Generator().manual_seed(sys_seed)
        x0 = torch.randn(d, generator=gx)  # previous physical state x^0
        x1_true = x0 + torch.randn(d, generator=gx)  # x^1 ~ N(x0, I)
        y = x1_true + (_OBS_VAR**0.5) * torch.randn(d, generator=gx)  # H=I

        # Reference exact-posterior samples (for sliced-W2 and Gaussian KL).
        gr = torch.Generator().manual_seed(derive_seed("analytical", "ref", ctx.seed))
        ref = sys_.exact_posterior_samples(x0, y, _N_EVAL, gr).numpy()

        # Draw the method's posterior ensemble.
        gm = torch.Generator().manual_seed(
            derive_seed("analytical", ctx.method.value, ctx.seed)
        )
        samp, nfe = self._sample_and_count(ctx.method, sys_, x0, y, _N_EVAL, M, gm)

        samp_np = samp.numpy()
        kl = float(gaussian_kl_divergence(ref, samp_np))
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            w2 = float(
                wasserstein_distance(ref, samp_np, num_projections=128, seed=0)
            )

        yield ResultRecord(
            case=self.case.value, method=ctx.method.value,
            scenario=ctx.scenario.value, metric=Metric.KL_POINTS.value,
            value=kl, E=E, M=M, seed=ctx.seed, nfe=float(nfe),
        )
        yield ResultRecord(
            case=self.case.value, method=ctx.method.value,
            scenario=ctx.scenario.value, metric=Metric.SLICED_W2.value,
            value=w2, E=E, M=M, seed=ctx.seed, nfe=float(nfe),
        )

    def _sample_and_count(
        self,
        method: Method,
        sys_: GaussianSystem,
        x0: torch.Tensor,
        y: torch.Tensor,
        n: int,
        M: int,
        gen: torch.Generator,
    ) -> tuple[torch.Tensor, int]:
        """Run the sampler/baseline for ``method``; return (samples, NFE).

        The three "ours" methods and the baselines (FlowDAS, OT-ODE,
        D-Flow SGLD, SDA, SURGE) are routed through the src/scisi
        posterior samplers with closed-form analytical prior models.

        EnKF and particle filter stay with the hand-coded samplers in
        samplers.py (they do not use a generative prior model at all).
        """
        d = sys_.d

        # Pre-expand to ensemble size n so BasePosterior.sample sees
        # field_history.shape[0]==n from the start.  The base class captures
        # ensemble_size = field_history.shape[0] BEFORE calling _prepare_batch,
        # so passing shape-[1,...] tensors causes only base[0] to be updated in
        # the batch loop while the other n-1 particles stay at their initial values.
        x0_4d = x0.reshape(1, 1, 1, d)
        fh = x0_4d.unsqueeze(-1).repeat(n, 1, 1, 1, 1)  # [n, 1, 1, d, 1]
        base_si = x0_4d.repeat(n, 1, 1, 1)               # [n, 1, 1, d] for SI source
        y_obs = y.reshape(1, d)

        # ------------------------------------------------------------------
        # Our methods: SI-SDE, FM-SDE, FM-ODE (src/scisi posteriors)
        # ------------------------------------------------------------------
        if method == Method.OURS_SI_SDE:
            si_model = build_si_model(d=d, g0=_G0, prior_var=_PRIOR_VAR)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = InterpolantGaussianLikelihood(
                model=si_model, obs_operator=obs_op,
                variance=_OBS_VAR, model_class="si",
            )
            posterior = StochasticInterpolantPosterior(
                model=si_model, likelihood_model=likelihood,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=base_si, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        if method == Method.OURS_FM_SDE:
            fm_model = build_fm_model(d=d, prior_var=_PRIOR_VAR)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = InterpolantGaussianLikelihood(
                model=fm_model, obs_operator=obs_op,
                variance=_OBS_VAR, model_class="fm",
            )
            g_tau = endpoint_vanishing_diffusion(fm_model.interpolation, scale=_G0)
            posterior = FlowMatchingPosterior(
                model=fm_model, likelihood_model=likelihood, diffusion_term=g_tau,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        if method == Method.OURS_FM_ODE:
            fm_model = build_fm_model(d=d, prior_var=_PRIOR_VAR)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = InterpolantGaussianLikelihood(
                model=fm_model, obs_operator=obs_op,
                variance=_OBS_VAR, model_class="fm",
            )
            posterior = FlowMatchingPosterior(
                model=fm_model, likelihood_model=likelihood, diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        # ------------------------------------------------------------------
        # FlowDAS baseline (src/scisi FlowdasGaussianLikelihood)
        # ------------------------------------------------------------------
        if method == Method.FLOWDAS:
            si_model = build_si_model(d=d, g0=_G0, prior_var=_PRIOR_VAR)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = FlowdasGaussianLikelihood(
                model=si_model, obs_operator=obs_op,
                variance=_OBS_VAR, ensemble_size=_FLOWDAS_J,
            )
            posterior = StochasticInterpolantPosterior(
                model=si_model, likelihood_model=likelihood,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=base_si, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M * (1 + _FLOWDAS_J)

        # ------------------------------------------------------------------
        # Guided FM - OT-ODE (src/scisi GuidanceGaussianLikelihood)
        # ------------------------------------------------------------------
        if method == Method.GUIDED_FM_OTODE:
            fm_model = build_fm_model(d=d, prior_var=_PRIOR_VAR)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = GuidanceGaussianLikelihood(
                model=fm_model, obs_operator=obs_op,
                variance=_OBS_VAR,
                weighting="ot_ode",
                obs_variance=_OTODE_OBS_VAR,
                guidance_scale=_OTODE_GAMMA,
            )
            posterior = FlowMatchingPosterior(
                model=fm_model, likelihood_model=likelihood, diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        # ------------------------------------------------------------------
        # D-Flow SGLD (src/scisi DFlowPosterior)
        # ------------------------------------------------------------------
        if method == Method.D_FLOW_SGLD:
            fm_model = build_fm_model(d=d, prior_var=_PRIOR_VAR)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = DFlowSGLDLikelihood(
                model=fm_model, obs_operator=obs_op,
                variance=_OBS_VAR,
                num_optim_steps=_DFLOW_OPTIM_STEPS,
                step_size=_DFLOW_STEP_SIZE,
                noise_scale=_DFLOW_NOISE_SCALE,
            )
            posterior = DFlowPosterior(
                model=fm_model, likelihood_model=likelihood,
                num_optim_steps=_DFLOW_OPTIM_STEPS,
                step_size=_DFLOW_STEP_SIZE,
                noise_scale=_DFLOW_NOISE_SCALE,
                variance=_OBS_VAR,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), _DFLOW_OPTIM_STEPS * M

        # ------------------------------------------------------------------
        # SDA (src/scisi SDALikelihood + DiffusionPosterior)
        # ------------------------------------------------------------------
        if method == Method.SDA:
            dm_model = build_dm_model(d=d, prior_var=_PRIOR_VAR, g0=_G0)
            obs_op = build_identity_obs_operator(d=d)
            likelihood = SDALikelihood(
                model=dm_model, obs_operator=obs_op,
                variance=_OBS_VAR, model_class="fm",
            )
            posterior = DiffusionPosterior(
                model=dm_model, likelihood_model=likelihood,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        # ------------------------------------------------------------------
        # SURGE (src/scisi SurgePosterior)
        # ------------------------------------------------------------------
        if method == Method.SURGE:
            dm_model = build_dm_model(d=d, prior_var=_PRIOR_VAR, g0=_G0)
            obs_op = build_identity_obs_operator(d=d)
            posterior = SurgePosterior(
                model=dm_model,
                obs_operator=obs_op,
                variance=_OBS_VAR,
                guidance_scale=_SURGE_GUIDANCE,
                ess_threshold=_SURGE_ESS,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        # ------------------------------------------------------------------
        # Classical filters (use hand-coded samplers; no generative prior)
        # ------------------------------------------------------------------
        if method == Method.ENKF:
            return enkf_posterior(sys_, x0, y, ensemble_size=n, generator=gen), 1
        if method == Method.PARTICLE_FILTER:
            return (
                particle_filter_posterior(sys_, x0, y, ensemble_size=n, generator=gen),
                1,
            )
        raise ValueError(f"unhandled analytical method {method!r}")

    # -- covariance-mode ablation (appendix) ------------------------------- #
    # The three likelihood-covariance approximations, run for each of our three
    # samplers: individual (per-member) Jacobian, shared (ensemble-mean) Jacobian,
    # and the isotropic Jacobian-free corollary. In this linear-Gaussian case the
    # source covariance is state-independent, so individual == shared (exact); the
    # isotropic mode drops the curvature and is the only one that loses accuracy.
    _ABLATION_MODES: tuple[tuple[str, str], ...] = (
        ("cov_individual", "inflated"),
        ("cov_shared", "inflated_shared"),
        ("cov_jacfree", "dps_jacobian_free"),
    )
    _ABLATION_SAMPLERS: tuple[tuple[Method, str], ...] = (
        (Method.OURS_SI_SDE, "si_sde"),
        (Method.OURS_FM_SDE, "fm_sde"),
        (Method.OURS_FM_ODE, "fm_ode"),
    )

    def run_ablation(self, *, aggregate: bool = True) -> list[ResultRecord]:
        """Covariance-mode ablation over the fixed seed list (appendix table)."""
        from common.seeding import seed_everything
        from common.aggregation import aggregate_over_seeds

        raw: list[ResultRecord] = []
        for seed in self.seeds:
            seed_everything(seed)
            ctx = self.make_context(
                Method.OURS_SI_SDE, Scenario.ANALYTICAL, seed
            )
            raw.extend(self.evaluate_ablation(ctx))
        return aggregate_over_seeds(raw) if aggregate else raw

    def evaluate_ablation(self, ctx: RunContext) -> Iterable[ResultRecord]:
        """Run our 3 samplers x 3 covariance modes vs the exact posterior.

        Emits ``kl_points`` and ``sliced_w2`` rows tagged with the covariance
        mode in the ``scenario`` field (``ablation:cov_individual`` etc.), keyed
        by the sampler in ``method`` -- so the appendix table can show the mode
        effect for each sampler.
        """
        d = self._dim()
        E, M = ctx.ensemble_size, ctx.num_steps
        sys_ = GaussianSystem(d=d, obs_var=_OBS_VAR, prior_var=_PRIOR_VAR)

        # Truth + observation + reference are shared across modes for a given seed.
        gx = torch.Generator().manual_seed(derive_seed("analytical", "truth", ctx.seed))
        x0 = torch.randn(d, generator=gx)
        x1_true = x0 + torch.randn(d, generator=gx)
        y = x1_true + (_OBS_VAR**0.5) * torch.randn(d, generator=gx)
        gr = torch.Generator().manual_seed(derive_seed("analytical", "ref", ctx.seed))
        ref = sys_.exact_posterior_samples(x0, y, _N_EVAL, gr).numpy()

        for method, sampler in self._ABLATION_SAMPLERS:
            for tag, mode in self._ABLATION_MODES:
                gm = torch.Generator().manual_seed(
                    derive_seed("analytical_abl", f"{sampler}:{mode}", ctx.seed)
                )
                samp = sample_posterior(
                    sys_, x0, y, sampler=sampler, likelihood_mode=mode,
                    ensemble_size=_N_EVAL, num_steps=M, g0=_G0, generator=gm,
                ).numpy()
                kl = float(gaussian_kl_divergence(ref, samp))
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    w2 = float(
                        wasserstein_distance(ref, samp, num_projections=128, seed=0)
                    )
                scen = f"ablation:{tag}"
                yield ResultRecord(
                    case=self.case.value, method=method.value, scenario=scen,
                    metric=Metric.KL_POINTS.value, value=kl,
                    E=E, M=M, seed=ctx.seed,
                )
                yield ResultRecord(
                    case=self.case.value, method=method.value, scenario=scen,
                    metric=Metric.SLICED_W2.value, value=w2,
                    E=E, M=M, seed=ctx.seed,
                )


# --------------------------------------------------------------------------- #
# Convergence study (KL vs M and KL vs g_tau, per mode/dimension).
# --------------------------------------------------------------------------- #


def gaussian_kl_to_exact(
    sys_: GaussianSystem, x0: torch.Tensor, y: torch.Tensor, samples: torch.Tensor
) -> float:
    """Analytic Gaussian KL(sample || exact-posterior) from sample moments.

    Tractable in any dimension since everything is Gaussian (spec Section 4):
    compares the sample mean/cov to the exact posterior moments.
    """
    mean, cov_scale = sys_.exact_posterior_moments(x0, y)
    mu_q = mean.numpy()
    d = sys_.d
    sigma_q = float(cov_scale) * np.eye(d)
    x = samples.numpy()
    mu_p = x.mean(0)
    sigma_p = np.cov(x, rowvar=False)
    if d == 1:
        sigma_p = np.atleast_2d(sigma_p)
    sign_p, logdet_p = np.linalg.slogdet(sigma_p)
    _, logdet_q = np.linalg.slogdet(sigma_q)
    sigma_q_inv = np.linalg.inv(sigma_q)
    diff = mu_q - mu_p
    trace_term = np.trace(sigma_q_inv @ sigma_p)
    quad_term = diff @ sigma_q_inv @ diff
    return 0.5 * float(trace_term + quad_term - d + logdet_q - logdet_p)


__all__ = ["AnalyticalRunner", "ANALYTICAL_METHODS", "gaussian_kl_to_exact"]
