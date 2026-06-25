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

from .samplers import (
    GaussianSystem,
    enkf_posterior,
    flowdas_posterior,
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
    Method.GUIDED_FM,
    Method.GUIDED_DIFFUSION,
    Method.ENKF,
    Method.PARTICLE_FILTER,
)

# Map each method to (sampler, likelihood_mode). Our three samplers run the
# exact (inflated) mode; the generative baselines map onto the surrogate modes:
#   * Guided FM (FIG)        -> FM-ODE with the Jacobian-free corollary gain.
#   * Guided diffusion (DPS) -> FM-SDE with the uninflated DPS gain (dps_full).
_METHOD_SPEC: dict[Method, tuple[str, str]] = {
    Method.OURS_SI_SDE: ("si_sde", "inflated"),
    Method.OURS_FM_SDE: ("fm_sde", "inflated"),
    Method.OURS_FM_ODE: ("fm_ode", "inflated"),
    Method.GUIDED_FM: ("fm_ode", "dps_jacobian_free"),
    Method.GUIDED_DIFFUSION: ("fm_sde", "dps_full"),
}

# Fixed analytic system parameters (spec Section 4).
_OBS_VAR = 1.0  # R = I
_PRIOR_VAR = 1.0  # Cov(x1 | x0) = I
_G0 = 1.0  # diffusion-strength base (gamma_tau = g0 (1 - tau))
_FLOWDAS_J = 16  # Monte-Carlo samples for the FlowDAS likelihood (reported)
_N_EVAL = 4096  # samples drawn for the metric estimates


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
        """Run the sampler/baseline for ``method``; return (samples, NFE)."""
        if method in _METHOD_SPEC:
            sampler, mode = _METHOD_SPEC[method]
            samp = sample_posterior(
                sys_, x0, y, sampler=sampler, likelihood_mode=mode,
                ensemble_size=n, num_steps=M, g0=_G0, generator=gen,
            )
            return samp, M  # one drift eval per step
        if method == Method.FLOWDAS:
            samp = flowdas_posterior(
                sys_, x0, y, ensemble_size=n, num_steps=M, g0=_G0,
                n_mc=_FLOWDAS_J, generator=gen,
            )
            return samp, M * (1 + _FLOWDAS_J)  # MC likelihood evals
        if method == Method.ENKF:
            return enkf_posterior(sys_, x0, y, ensemble_size=n, generator=gen), 1
        if method == Method.PARTICLE_FILTER:
            return (
                particle_filter_posterior(sys_, x0, y, ensemble_size=n, generator=gen),
                1,
            )
        raise ValueError(f"unhandled analytical method {method!r}")


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
