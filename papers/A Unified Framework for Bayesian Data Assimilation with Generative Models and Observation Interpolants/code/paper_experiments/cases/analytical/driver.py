"""Case 1 driver -- analytical linear--Gaussian (spec Section 4, GAP E4).

Closed-form posterior, no network trained. The correctness probe: every sampler
must reproduce the exact mean/cov as ``M`` grows in the ``inflated`` likelihood
mode, and the covariance ablation (``inflated`` vs ``dps_jacobian_free``) must
show the inflated mode is exact while the Jacobian-free surrogate plateaus away
from it.

Produces:
* Table ``tab:analytical_results`` -- KL and sliced-W2 to the exact posterior, for
  SI-SDE / DM-SDE / FM-ODE / FlowDAS / Guided FM / Guided diffusion / EnKF /
  particle filter, at matched E, M. Emitted via the tidy schema.
* Figure ``fig:analytical_panels`` -- see ``figures.py``.

The samplers (``samplers.py``) are compact closed-form vector SDE/ODE integrators
on the analytic Gaussian prior; the KL / sliced-W2 estimators are reused from
``paper/scripts/analytical_utils/kl_divergence.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from common.runner import ExperimentRunner, RunContext
from common.paths import REPO_ROOT
from common.seeding import derive_seed
from results_schema import Case, Method, Metric, ResultRecord, Scenario

import torch.nn as nn

from scisi.likelihood_models.dflow import DFlowSGLDLikelihood
from scisi.likelihood_models.gaussian_likelihood import (
    FlowdasGaussianLikelihood,
    InterpolantGaussianLikelihood,
)
from scisi.likelihood_models.guidance import FIGGaussianLikelihood
from scisi.likelihood_models.sda import SDALikelihood
from scisi.likelihood_models.tmpd import TMPDLikelihood
from scisi.posterior_models.diffusion_posterior import DiffusionPosterior
from scisi.posterior_models.dflow_posterior import DFlowPosterior
from scisi.posterior_models.tmpd_posterior import TMPDPosterior
from scisi.posterior_models.flow_matching_posterior import (
    FlowMatchingPosterior,
    endpoint_vanishing_diffusion,
)
from scisi.posterior_models.stochastic_interpolant_posterior import (
    StochasticInterpolantPosterior,
)
from scisi.posterior_models.surge_posterior import (
    SurgeFlowMatchingPosterior,
    SurgeStochasticInterpolantPosterior,
)

from .prior_models import (
    build_dm_model,
    build_fm_model,
    build_identity_obs_operator,
    build_si_model,
)
from .classical_baselines import (
    GaussianSystem,
    enkf_posterior,
    particle_filter_posterior,
)

# Reuse the validated analytic KL / sliced-W2 estimators. `paper/` was untracked
# and moved to `archive/paper/` locally (commit ada9131), so accept either layout
# rather than hard-coding one and breaking the case for whoever has the other.
_REPO = REPO_ROOT
for _cand in (_REPO / "paper" / "scripts", _REPO / "archive" / "paper" / "scripts"):
    if (_cand / "analytical_utils").is_dir():
        _PAPER_SCRIPTS = _cand
        break
else:
    raise ModuleNotFoundError(
        "analytical_utils not found under paper/scripts or archive/paper/scripts "
        f"(searched from {_REPO}); the analytical case needs its KL/W2 estimators."
    )
if str(_PAPER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_PAPER_SCRIPTS))
from analytical_utils.kl_divergence import (  # noqa: E402
    gaussian_kl_divergence,
    wasserstein_distance,
)

# REDUCED paper lineup (2026-07-01), matched to the NS/urban lineup where it is
# meaningful for the closed-form linear-Gaussian case. The three "Ours" samplers
# each run under TWO likelihood-covariance modes (jacfree + shared), emitted as
# distinct ``variant`` rows (see ``_OURS_VARIANTS`` + ``evaluate``). Dropped from
# the earlier lineup: Guided FM (OT-ODE), standalone SURGE. Added: the two SURGE
# combos. EnKF + particle filter are the closed-form classical references.
ANALYTICAL_METHODS: tuple[Method, ...] = (
    Method.OURS_SI_SDE,
    Method.OURS_DM_SDE,
    Method.OURS_FM_ODE,
    Method.FLOWDAS,
    Method.SURGE_FLOWDAS,
    Method.SDA,
    Method.SURGE_SDA,
    Method.D_FLOW_SGLD,
    Method.GUIDED_FM_FIG,
    Method.ENKF,
    Method.PARTICLE_FILTER,
)
# Runnable by name (``+analytical_methods='["TMPD"]'``) but NOT on the default
# lineup, so the default grid and its result files are unchanged.
ANALYTICAL_OPT_IN_METHODS: tuple[Method, ...] = (Method.TMPD,)
# GUIDED_FM_FIG rejoined the lineup on 2026-07-28 (user request) so the
# analytical figures cover the same baselines as NS/urban. Its (k, c) cells in
# configs/method/guided_fm_fig.yaml are UNTUNED placeholders (k=1, c=1.0): it
# collapsed on earlier analytical probes (kl ~1e11), which the figure pipeline
# folds onto the "collapsed (off scale)" shelf. Sweep the analytical column of
# that YAML before reading its numbers as a fair comparison.

# The three "Ours" samplers dispatched through ``draw_interpolant_posterior``; each runs
# under BOTH likelihood-covariance modes below, emitted as distinct ``variant``
# rows. Mirrors the jacfree/shared split of the NS/urban master scripts (which set
# likelihood_mode on separate invocations).
_OURS_SAMPLER: dict[Method, str] = {
    Method.OURS_SI_SDE: "si_sde",
    Method.OURS_DM_SDE: "dm_sde",
    Method.OURS_FM_ODE: "fm_ode",
}
_OURS_VARIANTS: tuple[tuple[str, str], ...] = (
    ("jacfree", "dps_jacobian_free"),
    ("shared", "inflated_shared"),
)


def draw_interpolant_posterior(
    sys_: GaussianSystem,
    x0: torch.Tensor,
    y: torch.Tensor,
    *,
    sampler: str,  # "si_sde" | "dm_sde" | "fm_ode"
    likelihood_mode: str,  # "dps_jacobian_free" | "inflated_shared" | "inflated"
    ensemble_size: int,
    num_steps: int,
    g0: float = 1.0,
    seed: int = 0,
    drift_time_shift: float = 0.0,
    jacobian_damping: float = 1.0,
    sigma_bar_eig_floor: bool = False,
    isotropic_front_factor: bool = False,
    residual_step_cap: Optional[float] = None,
    obs_op: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Draw an Ours-sampler posterior ensemble via the canonical ``src/scisi`` models.

    Builds the closed-form analytic prior model (``prior_models``) + the identity
    observation operator, wraps them in ``InterpolantGaussianLikelihood`` (with the
    requested covariance ``likelihood_mode``) and the matching ``src/scisi``
    posterior (SI-SDE -> ``StochasticInterpolantPosterior``; DM-SDE / FM-ODE ->
    ``FlowMatchingPosterior`` with / without the endpoint-vanishing diffusion), and
    returns the posterior ensemble ``[ensemble_size, d]``. Reproducible via ``seed``
    (the ``src/scisi`` posteriors draw from the global RNG). This is the ONLY place
    the analytical case builds a generative posterior -- every Ours row AND the
    appendix figures go through it, so no sampler is hand-coded here.
    """
    d = sys_.d
    n = ensemble_size
    # ``obs_op`` lets the nonlinear harness (nonlinear_driver.py) pass a
    # PointwiseObservationOperator; None keeps the linear H = I path unchanged.
    obs_op = obs_op if obs_op is not None else build_identity_obs_operator(d=d)
    # Pre-expand to the ensemble so BasePosterior.sample captures E from the start.
    x0_4d = x0.reshape(1, 1, 1, d)
    fh = x0_4d.unsqueeze(-1).repeat(n, 1, 1, 1, 1)  # [n, 1, 1, d, 1]
    base_si = x0_4d.repeat(n, 1, 1, 1)              # [n, 1, 1, d] SI point-mass source
    y_obs = y.reshape(1, d)

    # Inflated-mode stabilisation knobs, resolved by the caller from the method
    # YAML. The analytical J is the EXACT (symmetric) Jacobian of a 2-D linear
    # model, so it carries none of the non-normal amplification that forces
    # lambda < 1 at NS scale -- the configs keep analytical at lambda=1.0 and the
    # rest off, i.e. these defaults reproduce the untouched theory.
    stabilisation = dict(
        jacobian_damping=jacobian_damping,
        sigma_bar_eig_floor=sigma_bar_eig_floor,
        isotropic_front_factor=isotropic_front_factor,
        residual_step_cap=residual_step_cap,
    )

    torch.manual_seed(int(seed))
    if sampler == "si_sde":
        model = build_si_model(d=d, g0=g0, prior_var=sys_.prior_var)
        likelihood = InterpolantGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=sys_.obs_var,
            model_class="si", likelihood_mode=likelihood_mode,
            **stabilisation,
        )
        posterior = StochasticInterpolantPosterior(
            model=model, likelihood_model=likelihood,
        )
        base = base_si
    elif sampler in ("dm_sde", "fm_ode"):
        model = build_fm_model(d=d, prior_var=sys_.prior_var)
        likelihood = InterpolantGaussianLikelihood(
            model=model, obs_operator=obs_op, variance=sys_.obs_var,
            model_class="fm", likelihood_mode=likelihood_mode,
            **stabilisation,
        )
        diffusion_term = (
            endpoint_vanishing_diffusion(model.interpolation, scale=g0)
            if sampler == "dm_sde" else None
        )
        posterior = FlowMatchingPosterior(
            model=model, likelihood_model=likelihood, diffusion_term=diffusion_term,
            drift_time_shift=drift_time_shift,
        )
        base = None
    else:
        raise ValueError(f"unknown Ours sampler {sampler!r}")

    with torch.no_grad():
        out = posterior.sample(
            base=base, batch_size=n, num_steps=num_steps,
            field_history=fh, observations=y_obs,
        )
    return out.reshape(n, d)

# Fixed analytic system parameters (spec Section 4).
_OBS_VAR = 1.0  # R = I
_PRIOR_VAR = 1.0  # Cov(x1 | x0) = I
_G0 = 1.0  # diffusion-strength base (gamma_tau = g0 (1 - tau))
_N_EVAL = 4096  # default samples drawn for the metric estimates (see _n_eval)


# Baseline hyperparameters come from each method's config (configs/method/*.yaml),
# the single source of truth shared with the NS/urban pipeline. The analytical case
# has ONE joint scenario ("analytical") but sweeps the SDE/ODE step count M, so its
# per-cell tables are the block ``analytical: { analytical: { <M>: v } }`` (the same
# case -> scenario -> M schema as NS, one scenario). ``_cfg_hparam`` resolves a knob
# for a given M via the shared resolver: a null/missing analytical cell (or an absent
# ``analytical:`` block) falls back to the table ``default:``; a plain scalar passes
# through. The shipped analytical matrix is all-null, so every value is the config
# ``default:`` until the cells are filled in (fill configs/method/*.yaml, no code).
_ANALYTICAL_CASE_KEY = "analytical"
_ANALYTICAL_SCENARIO_KEY = "analytical"  # single joint scenario (Scenario.ANALYTICAL)


def _cfg_hparam(block, key, num_steps, fallback):
    from cases.navier_stokes._ns_pipeline import resolve_scheduled_hparam

    val = block.get(key, fallback) if block is not None else fallback
    return resolve_scheduled_hparam(
        val, _ANALYTICAL_CASE_KEY, _ANALYTICAL_SCENARIO_KEY, int(num_steps),
        name=str(key),
    )


def _kl_w2_to_exact(ref: np.ndarray, samp_np: np.ndarray) -> tuple[float, float]:
    """KL + sliced-W2 of ``samp_np`` to the exact posterior ``ref``.

    A COLLAPSED sampler (e.g. FIG, whose corrector drives the covariance to zero
    under full noisy observation) yields a non-positive-definite sample covariance
    that ``gaussian_kl_divergence`` rejects. Rather than crash the whole run, such
    a degenerate draw reports NaN (the manuscript's "collapsed" outcome); the
    across-seed aggregation is NaN-safe.
    """
    try:
        kl = float(gaussian_kl_divergence(ref, samp_np))
    except ValueError:
        kl = float("nan")
    try:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            w2 = float(
                wasserstein_distance(ref, samp_np, num_projections=128, seed=0)
            )
    except ValueError:
        w2 = float("nan")
    return kl, w2


class AnalyticalRunner(ExperimentRunner):
    """Runner for the analytical linear--Gaussian case."""

    case = Case.ANALYTICAL

    def methods(self) -> Sequence[Method]:
        # Optional subset / opt-in selection, e.g. +analytical_methods='["TMPD","SDA"]'.
        names = self._cfg_get("analytical_methods", None)
        if names:
            allowed = set(ANALYTICAL_METHODS) | set(ANALYTICAL_OPT_IN_METHODS)
            methods = tuple(Method(n) for n in names)
            unknown = [m.value for m in methods if m not in allowed]
            if unknown:
                raise ValueError(f"Not an analytical method: {unknown}")
            return methods
        return ANALYTICAL_METHODS

    def scenarios(self) -> Sequence[Scenario]:
        return (Scenario.ANALYTICAL,)

    def run(self, *, aggregate: bool = True) -> list[ResultRecord]:
        """Linear case unchanged; ``case.nonlinear`` (analytical_nonlinear.yaml)
        delegates to the nonlinear-observation harness (``nonlinear_driver``)."""
        case_cfg = getattr(self.config, "case", None)
        if case_cfg is not None and case_cfg.get("nonlinear", None) is not None:
            from .nonlinear_driver import NonlinearAnalyticalHarness

            return NonlinearAnalyticalHarness.from_config(
                self.config, seeds=self.seeds
            ).run(aggregate=aggregate)
        return super().run(aggregate=aggregate)

    # -- config helpers ---------------------------------------------------- #

    # Baseline -> YAML config name under configs/method/. Loaded on demand to read
    # each method's ``default:`` hyperparameters (the analytical case tunes nothing).
    _METHOD_CONFIG_NAME: dict[Method, str] = {
        Method.FLOWDAS: "flowdas",
        Method.D_FLOW_SGLD: "dflow_sgld",
        Method.SDA: "sda",
        Method.TMPD: "tmpd",
        Method.SURGE_SDA: "surge_sda",
        Method.SURGE_FLOWDAS: "surge_flowdas",
        Method.GUIDED_FM_FIG: "guided_fm_fig",
        # Ours: same YAML source of truth as the baselines, so the inflated-mode
        # stabilisation knobs are read from configs/method/*.yaml here too rather
        # than being hard-coded (NS/urban read them via _ns_pipeline.build_posterior).
        Method.OURS_SI_SDE: "si_sde",
        Method.OURS_DM_SDE: "dm_sde",
        Method.OURS_FM_ODE: "fm_ode",
    }

    def _method_cfg(self, method: Method):
        """Load (and cache) a baseline's YAML config; source of its ``default:`` hparams."""
        from omegaconf import OmegaConf

        if not hasattr(self, "_mcfg_cache"):
            self._mcfg_cache: dict = {}
        if method not in self._mcfg_cache:
            root = Path(__file__).resolve().parents[2] / "configs" / "method"
            self._mcfg_cache[method] = OmegaConf.load(
                root / f"{self._METHOD_CONFIG_NAME[method]}.yaml"
            )
        return self._mcfg_cache[method]

    def _dim(self) -> int:
        """Dimension used for the table (the plot dimension, default 2)."""
        cfg = getattr(self.config, "case", None)
        if cfg is not None and hasattr(cfg, "plot_dimension"):
            return int(cfg.plot_dimension)
        return int(self._cfg_get("plot_dimension", 2))

    def _n_eval(self) -> int:
        """Samples drawn per (method, seed) for the KL / sliced-W2 estimates.

        Reads ``case.n_eval_samples`` (configs/case/analytical.yaml), with a
        top-level ``n_eval_samples=...`` CLI override winning, and ``_N_EVAL`` as
        the fallback. Larger values tighten the Monte-Carlo error on both metrics
        (the estimators are sample-moment based) at a linear cost in runtime.
        """
        cli = self._cfg_get("n_eval_samples", None)
        if cli is not None:
            return int(cli)
        cfg = getattr(self.config, "case", None)
        if cfg is not None and getattr(cfg, "n_eval_samples", None) is not None:
            return int(cfg.n_eval_samples)
        return _N_EVAL

    def _ours_hparams(self, method: Method, num_steps: int) -> dict:
        """Stabilisation kwargs of :func:`draw_interpolant_posterior` for an Ours row.

        Read from this sampler's own YAML (the same [case][scenario][M] tables the
        NS/urban pipeline resolves); a top-level CLI override still wins, so a
        sweep needs no edit to the tracked config. Shared by the linear
        ``evaluate`` and the nonlinear harness.
        """
        lik = self._method_cfg(method).get("likelihood_model", {})

        def hp(key, fallback):
            cli = self._cfg_get(key, None)
            return cli if cli is not None else _cfg_hparam(lik, key, num_steps, fallback)

        cap = hp("residual_step_cap", None)
        return dict(
            # FM/DM drift-evaluation time offset (P1 probe); 0 = current
            # left-endpoint Euler, 1 = bespoke right endpoint, 0.5 = midpoint.
            drift_time_shift=float(hp("drift_time_shift", 0.0)),
            jacobian_damping=float(hp("jacobian_damping", 1.0)),
            sigma_bar_eig_floor=bool(hp("sigma_bar_eig_floor", False)),
            isotropic_front_factor=bool(hp("isotropic_front_factor", False)),
            residual_step_cap=float(cap) if cap is not None else None,
        )

    # -- the per-(method, scenario, seed) work ----------------------------- #
    def evaluate(self, ctx: RunContext) -> Iterable[ResultRecord]:
        import time

        d = self._dim()
        n_eval = self._n_eval()
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
        ref = sys_.exact_posterior_samples(x0, y, n_eval, gr).numpy()

        def _emit(samp, nfe, seconds, variant):  # type: ignore[no-untyped-def]
            """KL + sliced-W2 rows for one draw, tagged with the variant + timing."""
            kl, w2 = _kl_w2_to_exact(ref, samp.numpy())
            for metric, val in (
                (Metric.KL_POINTS, kl),
                (Metric.SLICED_W2, w2),
            ):
                yield ResultRecord(
                    case=self.case.value, method=ctx.method.value,
                    scenario=ctx.scenario.value, metric=metric.value,
                    value=val, E=E, M=M, seed=ctx.seed,
                    nfe=float(nfe), seconds=float(seconds), variant=variant,
                )

        if ctx.method in _OURS_SAMPLER:
            # Ours: run BOTH likelihood-covariance modes -> two variant rows.
            sampler = _OURS_SAMPLER[ctx.method]
            hp = self._ours_hparams(ctx.method, M)
            for variant, mode in _OURS_VARIANTS:
                seed = derive_seed(
                    "analytical", f"{ctx.method.value}:{variant}", ctx.seed
                )
                t0 = time.perf_counter()
                samp = draw_interpolant_posterior(
                    sys_, x0, y, sampler=sampler, likelihood_mode=mode,
                    ensemble_size=n_eval, num_steps=M, seed=seed, **hp,
                )
                yield from _emit(samp, M, time.perf_counter() - t0, variant)
        else:
            gm = torch.Generator().manual_seed(
                derive_seed("analytical", ctx.method.value, ctx.seed)
            )
            t0 = time.perf_counter()
            self._last_vjp = None
            samp, nfe = self._sample_and_count(ctx.method, sys_, x0, y, n_eval, M, gm)
            seconds = time.perf_counter() - t0
            yield from _emit(samp, nfe, seconds, None)
            if self._last_vjp is not None:
                # TMPD: reverse-mode VJPs (backward passes over the batch) per
                # assimilation, reported beside -- not inside -- the NFE.
                yield ResultRecord(
                    case=self.case.value, method=ctx.method.value,
                    scenario=ctx.scenario.value, metric=Metric.VJP.value,
                    value=float(self._last_vjp), E=E, M=M, seed=ctx.seed,
                    nfe=float(nfe), seconds=float(seconds), variant=None,
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
        obs_op: Optional[nn.Module] = None,
        hparams: Optional[Mapping[str, object]] = None,
    ) -> tuple[torch.Tensor, int]:
        """Run a BASELINE / classical method for ``method``; return (samples, NFE).

        The generative baselines (FlowDAS, D-Flow SGLD, SDA, SURGE+SDA,
        SURGE+FlowDAS) are routed through the ``src/scisi`` posterior samplers with
        the closed-form analytical prior models (``prior_models``). The Ours
        samplers are NOT handled here -- they go through
        :func:`draw_interpolant_posterior` (both covariance variants).

        EnKF and particle filter are the only non-``src/scisi`` methods: they are
        the closed-form classical references in ``classical_baselines`` (no
        generative prior model at all).

        ``obs_op`` / ``hparams`` are the nonlinear-harness hooks: an explicit
        (possibly nonlinear) observation operator, and per-run hyperparameter
        overrides ``{key: value}`` that win over the method YAML (the R-dependent
        tuned values of ``configs/case/analytical_nonlinear.yaml``). Both ``None``
        (the linear case) leave this path exactly as before.
        """
        d = sys_.d
        # The system's own R / prior variance (the linear case: R = 1, prior 1).
        obs_var, prior_var = sys_.obs_var, sys_.prior_var
        given_op = obs_op  # None on the linear path (classical filters keep H = I code)
        obs_op = obs_op if obs_op is not None else build_identity_obs_operator(d=d)
        overrides = dict(hparams or {})

        # Read a baseline hyperparameter: an explicit override wins, else its
        # config resolved for THIS step count M against the analytical per-cell
        # matrix (null cell -> config `default:`). See _cfg_hparam.
        def _hp(block, key, fallback):
            if key in overrides:
                return overrides[key]
            return _cfg_hparam(block, key, M, fallback)

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
        # FlowDAS baseline (src/scisi FlowdasGaussianLikelihood)
        # ------------------------------------------------------------------
        if method == Method.FLOWDAS:
            lik = self._method_cfg(Method.FLOWDAS).get("likelihood_model", {})
            n_mc = int(_hp(lik, "num_mc_samples", 25))
            zeta = float(_hp(lik, "guidance_scale", 1.0))
            si_model = build_si_model(d=d, g0=_G0, prior_var=prior_var)
            likelihood = FlowdasGaussianLikelihood(
                model=si_model, obs_operator=obs_op,
                variance=obs_var, num_mc_samples=n_mc, guidance_scale=zeta,
            )
            posterior = StochasticInterpolantPosterior(
                model=si_model, likelihood_model=likelihood,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=base_si, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M * (1 + n_mc)

        # ------------------------------------------------------------------
        # D-Flow SGLD (src/scisi DFlowPosterior)
        # ------------------------------------------------------------------
        if method == Method.D_FLOW_SGLD:
            lik = self._method_cfg(Method.D_FLOW_SGLD).get("likelihood_model", {})
            step_size = float(_hp(lik, "step_size", 0.005))
            noise_scale = float(_hp(lik, "noise_scale", 1e-3))
            lambda_reg = float(_hp(lik, "lambda_reg", 1e-4))
            fm_model = build_fm_model(d=d, prior_var=prior_var)
            # PECULIAR TO D-FLOW: M drives the SGLD chain length num_optim_steps
            # (the FM-ODE rollout stays fixed at ode_steps=6), so the analytical
            # M-sweep {50,100,250,500} IS the Langevin-steps axis for D-Flow --
            # matching the KL-vs-steps figure caption. num_optim_steps is NOT a
            # config hyperparameter; step_size/noise_scale/lambda_reg come from the
            # dflow_sgld.yaml `default:`.
            dflow_K = M
            likelihood = DFlowSGLDLikelihood(
                model=fm_model, obs_operator=obs_op,
                variance=obs_var,
                num_optim_steps=dflow_K,
                step_size=step_size,
                noise_scale=noise_scale,
                lambda_reg=lambda_reg,
            )
            posterior = DFlowPosterior(
                model=fm_model, likelihood_model=likelihood,
                num_optim_steps=dflow_K,
                step_size=step_size,
                noise_scale=noise_scale,
                lambda_reg=lambda_reg,
                variance=obs_var,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), dflow_K * 6  # ~ Langevin steps x ode_steps(6)

        # ------------------------------------------------------------------
        # SDA (src/scisi SDALikelihood + DiffusionPosterior)
        # ------------------------------------------------------------------
        if method == Method.SDA:
            lik = self._method_cfg(Method.SDA).get("likelihood_model", {})
            gamma_sda = float(_hp(lik, "gamma_sda", 1e-2))
            dm_model = build_dm_model(d=d, prior_var=prior_var, g0=_G0)
            likelihood = SDALikelihood(
                model=dm_model, obs_operator=obs_op,
                variance=obs_var, model_class="fm", gamma_sda=gamma_sda,
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
        # TMPD (src/scisi TMPDLikelihood + TMPDPosterior): no tuned
        # hyperparameter, only the covariance choice.
        # Built like Ours (DM-SDE) -- the FM model with the endpoint-vanishing
        # g passed explicitly -- NOT like SDA's ``build_dm_model``: the stock
        # ``DenoiseDiffusionModel._compute_first_step`` drops ``v dt`` at
        # tau = 0, and TMPD is the head-to-head competitor of Ours (DM-SDE),
        # which takes the full first step. SDA keeps its own historical
        # construction (its rows are unchanged).
        # ------------------------------------------------------------------
        if method == Method.TMPD:
            cfg = self._method_cfg(Method.TMPD)
            lik = cfg.get("likelihood_model", {})
            fm_model = build_fm_model(d=d, prior_var=prior_var)
            likelihood = TMPDLikelihood(
                model=fm_model, obs_operator=obs_op, variance=obs_var,
                model_class="fm",
                covariance=str(_hp(lik, "covariance", "full")),
                vjp_chunk_size=int(_hp(lik, "vjp_chunk_size", 64)),
            )
            posterior = TMPDPosterior(
                model=fm_model, likelihood_model=likelihood,
                diffusion_term=endpoint_vanishing_diffusion(
                    fm_model.interpolation, scale=_G0
                ),
                guidance_weight=str(_hp(
                    cfg.get("posterior_model", {}), "guidance_weight",
                    "score_substitution",
                )),
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            # NFE = network forwards (one per step, as for SDA). The reverse-mode
            # VJPs TMPD adds on top are NOT in it; ``evaluate`` reports them as a
            # separate ``vjp`` row (``likelihood.vjp_count``).
            self._last_vjp = likelihood.vjp_count
            return out.reshape(n, d), M

        # ------------------------------------------------------------------
        # SURGE (SDA): SURGE SMC on the DM prior, SDA likelihood as guidance.
        # ------------------------------------------------------------------
        if method == Method.SURGE_SDA:
            cfg = self._method_cfg(Method.SURGE_SDA)
            gamma_sda = float(_hp(cfg.get("likelihood_model", {}), "gamma_sda", 1e-2))
            ess = float(_hp(cfg.get("posterior_model", {}), "ess_threshold", 0.5))
            dm_model = build_dm_model(d=d, prior_var=prior_var, g0=_G0)
            likelihood = SDALikelihood(
                model=dm_model, obs_operator=obs_op,
                variance=obs_var, model_class="fm", gamma_sda=gamma_sda,
            )
            posterior = SurgeFlowMatchingPosterior(
                model=dm_model, obs_operator=obs_op, variance=obs_var,
                likelihood_model=likelihood, ess_threshold=ess,
                diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M

        # ------------------------------------------------------------------
        # SURGE (FlowDAS): SURGE SMC on the SI prior, FlowDAS MC likelihood.
        # ------------------------------------------------------------------
        if method == Method.SURGE_FLOWDAS:
            cfg = self._method_cfg(Method.SURGE_FLOWDAS)
            lik = cfg.get("likelihood_model", {})
            n_mc = int(_hp(lik, "num_mc_samples", 25))
            zeta = float(_hp(lik, "guidance_scale", 1.0))
            ess = float(_hp(cfg.get("posterior_model", {}), "ess_threshold", 0.5))
            si_model = build_si_model(d=d, g0=_G0, prior_var=prior_var)
            likelihood = FlowdasGaussianLikelihood(
                model=si_model, obs_operator=obs_op,
                variance=obs_var, num_mc_samples=n_mc, guidance_scale=zeta,
            )
            posterior = SurgeStochasticInterpolantPosterior(
                model=si_model, obs_operator=obs_op, variance=obs_var,
                likelihood_model=likelihood, ess_threshold=ess,
                diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=base_si, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M * (1 + n_mc)

        # ------------------------------------------------------------------
        # Guided FM (FIG) baseline (src/scisi FIGGaussianLikelihood + FM-ODE)
        # ------------------------------------------------------------------
        if method == Method.GUIDED_FM_FIG:
            lik = self._method_cfg(Method.GUIDED_FM_FIG).get("likelihood_model", {})
            fig_k = int(_hp(lik, "guidance_steps", 1))
            fig_c = float(_hp(lik, "guidance_scale", 10.0))
            fig_w = float(_hp(lik, "interpolant_noise", 0.0))
            # Generic MC ensemble for the likelihood (FIG has no config field of its
            # own here); reuse FlowDAS's J so all analytical baselines share it.
            n_mc = int(_hp(
                self._method_cfg(Method.FLOWDAS).get("likelihood_model", {}),
                "num_mc_samples", 25))
            fm_model = build_fm_model(d=d, prior_var=prior_var)
            likelihood = FIGGaussianLikelihood(
                model=fm_model, obs_operator=obs_op, variance=obs_var,
                ensemble_size=n_mc,
                guidance_steps=fig_k, guidance_scale=fig_c,
                interpolant_noise=fig_w,
            )
            posterior = FlowMatchingPosterior(
                model=fm_model, likelihood_model=likelihood, diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), M * (1 + fig_k)

        # ------------------------------------------------------------------
        # Extra baselines used by the NONLINEAR harness only (not in the linear
        # ANALYTICAL_METHODS lineup, so the linear table is unaffected):
        # Guided FM (OT-ODE), DPS ("Guided diffusion"), standalone SURGE. The
        # analytical-regime settings follow the README (OT-ODE sigma_y^2 = R,
        # gamma = 1); none of these has a configs/method YAML.
        # ------------------------------------------------------------------
        if method == Method.GUIDED_FM_OTODE:
            from scisi.likelihood_models.guidance import GuidanceGaussianLikelihood

            fm_model = build_fm_model(d=d, prior_var=prior_var)
            likelihood = GuidanceGaussianLikelihood(
                model=fm_model, obs_operator=obs_op, variance=obs_var,
                weighting="ot_ode", obs_variance=obs_var,
                guidance_scale=float(_hp(None, "guidance_scale", 1.0)),
            )
            posterior = FlowMatchingPosterior(
                model=fm_model, likelihood_model=likelihood, diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), 2 * M

        if method == Method.GUIDED_DIFFUSION:
            from scisi.likelihood_models.guidance import DPSGaussianLikelihood

            dm_model = build_dm_model(d=d, prior_var=prior_var, g0=_G0)
            likelihood = DPSGaussianLikelihood(
                model=dm_model, obs_operator=obs_op, variance=obs_var,
                model_class="fm",
            )
            posterior = DiffusionPosterior(
                model=dm_model, likelihood_model=likelihood,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), 2 * M

        if method == Method.SURGE:
            from scisi.posterior_models.surge_posterior import SurgePosterior

            dm_model = build_dm_model(d=d, prior_var=prior_var, g0=_G0)
            posterior = SurgePosterior(
                model=dm_model, obs_operator=obs_op, variance=obs_var,
                guidance_scale=float(_hp(None, "guidance_scale", 1.0)),
                ess_threshold=0.5, diffusion_term=None,
            )
            with torch.no_grad():
                out = posterior.sample(
                    base=None, batch_size=n, num_steps=M,
                    field_history=fh, observations=y_obs,
                )
            return out.reshape(n, d), 2 * M

        # ------------------------------------------------------------------
        # Classical filters (use hand-coded samplers; no generative prior)
        # ------------------------------------------------------------------
        # Nonlinear h for the classical filters: row-wise forward of the given
        # operator on the [E, d] ensemble. None -> the original H = I code.
        obs_fn = None
        if given_op is not None:
            def obs_fn(xe: torch.Tensor) -> torch.Tensor:
                return given_op(xe.reshape(xe.shape[0], 1, 1, d))

        if method == Method.ENKF:
            return (
                enkf_posterior(
                    sys_, x0, y, ensemble_size=n, generator=gen, obs_fn=obs_fn
                ),
                1,
            )
        if method == Method.PARTICLE_FILTER:
            return (
                particle_filter_posterior(
                    sys_, x0, y, ensemble_size=n, generator=gen, obs_fn=obs_fn
                ),
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
        (Method.OURS_DM_SDE, "dm_sde"),
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
        n_eval = self._n_eval()
        E, M = ctx.ensemble_size, ctx.num_steps
        sys_ = GaussianSystem(d=d, obs_var=_OBS_VAR, prior_var=_PRIOR_VAR)

        # Truth + observation + reference are shared across modes for a given seed.
        gx = torch.Generator().manual_seed(derive_seed("analytical", "truth", ctx.seed))
        x0 = torch.randn(d, generator=gx)
        x1_true = x0 + torch.randn(d, generator=gx)
        y = x1_true + (_OBS_VAR**0.5) * torch.randn(d, generator=gx)
        gr = torch.Generator().manual_seed(derive_seed("analytical", "ref", ctx.seed))
        ref = sys_.exact_posterior_samples(x0, y, n_eval, gr).numpy()

        for method, sampler in self._ABLATION_SAMPLERS:
            for tag, mode in self._ABLATION_MODES:
                seed = derive_seed("analytical_abl", f"{sampler}:{mode}", ctx.seed)
                samp = draw_interpolant_posterior(
                    sys_, x0, y, sampler=sampler, likelihood_mode=mode,
                    ensemble_size=n_eval, num_steps=M, seed=seed,
                ).numpy()
                kl, w2 = _kl_w2_to_exact(ref, samp)
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
