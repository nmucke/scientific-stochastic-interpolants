"""SURGE posterior -- guided SDE + SMC particle filter (single window).

Single-window adaptation of SURGE (Wei, Ren, Shi & Lu, "SURGE: Approximation
and Training Free Particle Filter for Diffusion Surrogate", arXiv:2605.18745).

WHAT SURGE IS
-------------
SURGE is a *particle filter* (Sequential Monte Carlo) that runs on a generative
model of the dynamics. For each physical step it draws each particle through a
**guided reverse SDE** (observation guidance steers the proposal toward
observation-consistent states) and then *reweights* the particles along the
diffusion trajectory with a Girsanov-corrected SMC weight, resampling whenever
the effective sample size drops below a threshold. The Girsanov correction makes
the method **approximation-free**: an imperfect guidance ``G`` only affects
proposal efficiency, not the target -- the SMC weights debias it, so the sampler
converges to the true posterior regardless of the guidance used.

SURGE IS METHOD-AGNOSTIC (paper: combined with SDA, FlowDAS, DPS, ...)
---------------------------------------------------------------------
SURGE is the SMC layer; the **guidance** ``grad_x G`` is whatever observation-
score approximation you plug in. SDA and FlowDAS are *fully determined by their
likelihood approximations* (``scisi.likelihood_models.sda.SDALikelihood`` /
``scisi.likelihood_models.gaussian_likelihood.FlowdasGaussianLikelihood``), so
"SURGE + SDA" / "SURGE + FlowDAS" just means: take that method's likelihood
score as ``grad_x G`` in the guided proposal, and let the SMC layer debias it.
The SMC **reward** ``R = log p(y | x)`` is the *true* Gaussian observation
log-likelihood on the model-class Tweedie denoiser (computed here, NOT by the
plugged method), so the terminal potential is the exact likelihood and the
target is the true posterior -- independent of which guidance is plugged in.

    Propose (guided EM step), pseudo-time ``t : 0 -> 1``:
        x_{k+1} = x_k + [ b(x_k) + w(t) grad_x G(x_k | y) ] dt
                       + g(t) sqrt(dt) xi,        xi ~ N(0, I)

      where ``b`` is the prior reverse-SDE drift (FM: v + 1/2 g^2 s; SI: the
      trained SI drift b_theta), ``g(t)`` the diffusion coefficient, ``grad_x G``
      the plugged method's observation score, and ``w(t)`` that method's OWN SDE
      injection weight -- so ``w grad_x G`` is exactly the plugged method's guided
      drift and the combo proposal IS that method's guided SDE (paper's
      ``Sigma grad_x G``). ``w`` = FlowDAS's ``zeta`` (via ``sde_weight``), SDA's
      ``g^2``, ... ; the standalone-DPS ``SurgePosterior`` uses ``w = guidance_scale``.

    Reweight (incremental log-weight, Eq. 6 / Alg. 1), with u = (w / g) grad_x G
    (= Sigma^{-1/2} times the added drift w grad_x G):
        log w_{k+1} = log w_k
            + [ (t+dt) R(x_{k+1}) - t R(x_k) ]      # reward delta (true log p)
            - u(x_k | y) . ( sqrt(dt) xi )          # Girsanov martingale
            - 1/2 || u(x_k | y) ||^2 dt             # quadratic variation

    Resample (systematic) when ESS = 1 / sum_i w_i^2 < c * N.

CLASSES
-------
* :class:`SurgePosteriorBase` -- the SMC machinery (proposal, Girsanov weight,
  reward telescoping, resampling). Abstract in the denoiser (reward) only.
* :class:`SurgeFlowMatchingPosterior` -- FM prior (source ``N(0, I)``, anchor
  ``a0 = 0``); reward denoiser ``xhat_1 = (x + sigma^2 s) / beta``.
* :class:`SurgeStochasticInterpolantPosterior` -- SI prior (source is the point
  mass ``delta_{x0}``, anchor ``a0 = x0``); reward denoiser
  ``xhat_1 = (x + sigma^2 s - alpha x0) / beta`` with ``s`` recovered from the SI
  velocity. Both denoisers -> identity at ``t -> 1`` (``sigma -> 0``, ``beta ->
  1``), so the terminal reward is the exact observation log-likelihood.
* :class:`SurgePosterior` -- standalone SURGE: a :class:`SurgeFlowMatchingPosterior`
  whose guidance is the internal DPS gradient of the reward (no plugged
  likelihood). Kept for the "SURGE" method.

ADAPTATIONS (documented, like the SDA / FlowDAS baselines)
----------------------------------------------------------
* **Single-window.** Our harness is autoregressive (one physical step / window
  at a time), so SURGE's outer particle-filter loop over physical steps is the
  harness's autoregressive rollout; this module runs ONE window's guided-SDE +
  SMC pass. Each window's resampled ensemble is fed back as the next window's
  history (the base-posterior ``sample_trajectory`` threads this).
* **Guidance strength lives with the guidance method.** The plugged likelihood's
  ``.score()`` returns the raw observation score ``grad_x G``; SURGE multiplies it
  by that method's OWN SDE injection weight ``w(t)`` (:meth:`_injection_weight` --
  ``sde_weight`` for FlowDAS's ``zeta``, ``g^2`` for SDA) and forms the matching
  Girsanov weight ``(w / g) grad_x G`` from the SAME score, so the combo proposal
  is exactly that method's guided SDE and the SMC debiases it (Girsanov holds for
  any adapted drift). The tunable strength (e.g. FlowDAS's ``zeta``) therefore
  stays a **likelihood** hyperparameter, exactly as in ``flowdas.yaml`` -- NOT a
  SURGE posterior knob. Only the standalone :class:`SurgePosterior` (internal DPS,
  no plugged likelihood) owns its ``guidance_scale``.

The class subclasses :class:`BasePosterior` (NOT a likelihood plugged into a
``*Posterior``) because SURGE needs to (a) see the exact injected noise ``xi`` to
form the martingale weight and (b) resample ACROSS particles -- neither of which
the likelihood-only interface exposes.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.posterior_models.base_posterior import BasePosterior

logger = logging.getLogger(__name__)

MIN_TIME = 1e-4


class SurgePosteriorBase(BasePosterior):
    """SURGE guided-SDE + SMC machinery (single window); denoiser is abstract."""

    #: ``"fm"`` or ``"si"`` -- set by the concrete subclass (informational).
    model_class: str = "fm"

    def __init__(
        self,
        model: nn.Module,
        obs_operator: LinearObservationOperator,
        variance: float = 0.05,
        guidance_scale: float = 1.0,
        ess_threshold: float = 0.5,
        likelihood_model: Optional[nn.Module] = None,
        diffusion_term: Optional[Callable] = None,
        gaussian_base: bool = True,
    ) -> None:
        """Initialise the SURGE posterior.

        Args:
            model: The prior model (FM ``FlowMatchingModel`` / ``DenoiseDiffusionModel``
                or SI ``FollmerStochasticInterpolant``). Exposes ``.drift``,
                ``.score`` (FM) / ``.interpolation.score_from_velocity`` (SI) and
                ``.interpolation``.
            obs_operator: Linear observation operator ``H``.
            variance: Observation-noise variance ``sigma^2`` (``R = sigma^2 I``).
                Defines the SMC reward ``R = log p(y | x)``, so it MUST match the
                true observation noise.
            guidance_scale: Multiplier ``lambda`` on the plugged guidance score
                (SURGE's guidance strength). Tune per experiment.
            ess_threshold: Resample when ESS / N < ``ess_threshold`` (c in [0, 1]).
            likelihood_model: The guidance provider (``SDALikelihood`` /
                ``FlowdasGaussianLikelihood`` / ...) whose ``.score()`` returns
                ``grad_x G``. ``None`` -> the guidance must be supplied by an
                override (see :class:`SurgePosterior`, internal DPS).
            diffusion_term: Reverse-SDE coefficient ``g(t)``; ``None`` -> model
                default (SI: ``interpolation.gamma``; FM: ``model.diffusion_term``).
            gaussian_base: FM path (``N(0, I)`` init, anchor ``a0 = 0``) if True;
                SI path (point-mass ``delta_{x0}`` init) if False.
        """
        # ``likelihood_model`` may be None (internal-guidance subclass); pass a
        # trivial placeholder so BasePorterior.__init__ (and .device) is happy.
        super().__init__(
            model=model,
            likelihood_model=likelihood_model if likelihood_model is not None else nn.Identity(),
            diffusion_term=diffusion_term,
            gaussian_base=gaussian_base,
        )
        self.obs_operator = obs_operator
        self.variance = float(variance)
        self.guidance_scale = float(guidance_scale)
        self.ess_threshold = float(ess_threshold)
        self.interpolation = self.model.interpolation
        # Per-window SMC state (reset each window in ``_pre_step``).
        self._log_w: Optional[torch.Tensor] = None  # [E] cumulative log-weights
        self._incr_chunks: list[torch.Tensor] = []  # per-chunk incr this t-step
        self._is_first_window_step = False  # True on a window's first recorded step

    # ------------------------------------------------------------------ #
    # Reward R = log p(y | x): true Gaussian likelihood on the denoiser.
    # ------------------------------------------------------------------ #

    def _denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Tweedie denoiser ``xhat_1 = E[x_1 | x_t]`` (subclass: FM or SI)."""
        raise NotImplementedError

    def _reward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        observations: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """True observation log-likelihood ``R = -1/(2 sigma^2) ||y - H xhat_1||^2``."""
        with torch.no_grad():
            xhat1 = self._denoise(x, t, field_history, field_cond, pars_cond)
            residual = observations - self.obs_operator(xhat1)  # [E, N_y]
            return (-0.5 / self.variance) * (residual**2).sum(dim=1)  # [E]

    # ------------------------------------------------------------------ #
    # Guidance grad_x G (the plugged likelihood's observation score).
    # ------------------------------------------------------------------ #

    def _lik_kwargs(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
        prior_drift: torch.Tensor,
    ) -> dict:
        """Extra kwargs passed to ``likelihood_model.score`` (subclass-specific)."""
        return {}

    def _compute_guidance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        observations: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
        dt: torch.Tensor,
        prior_drift: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(grad_x G, reward_k_or_None)`` from the plugged likelihood.

        The default queries ``self.likelihood_model.score(...)`` (SDA / FlowDAS /
        ...), returning its raw guidance score and ``None`` for the reward (the
        reward is computed separately from the true likelihood via :meth:`_reward`).
        :class:`SurgePosterior` overrides this to compute the internal DPS
        gradient (and returns the reward alongside, to avoid a recompute).
        """
        out = self.likelihood_model.score(
            observations=observations,
            x=x,
            t=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            dt=dt,
            **self._lik_kwargs(x, t, field_history, field_cond, pars_cond, prior_drift),
        )
        guidance = out[0] if isinstance(out, tuple) else out
        return guidance.detach(), None

    def _prior_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Prior reverse-SDE drift ``b`` (FM: v + 1/2 g^2 s; SI: b_theta)."""
        return self.model.drift(x, t, field_history, field_cond, pars_cond).detach()

    def _injection_weight(self, t: torch.Tensor, g: torch.Tensor):
        """The coefficient the guidance score enters the SDE with -- the plugged
        method's OWN injection weight, so the combo proposal is that method's
        guided SDE (paper's ``Sigma grad_x G``).

        Mirrors the standalone posteriors: FlowDAS's tuned constant ``zeta`` via
        ``sde_weight`` (the strength lives in the likelihood), SDA's / any
        classifier-guidance footing ``g^2`` via ``guidance_weight == 'g_squared'``,
        else the SI/FM velocity--score weight ``w_tau = a_tau + 1/2 g^2``. Returned
        as a scalar (one window-time per step). :class:`SurgePosterior` overrides
        this for the internal-DPS standalone (no plugged likelihood).
        """
        lik = self.likelihood_model
        if hasattr(lik, "sde_weight"):
            w = lik.sde_weight(t, self.diffusion_term)  # FlowDAS zeta
        elif getattr(lik, "guidance_weight", None) == "g_squared":
            w = g ** 2  # SDA / classifier-guidance footing
        else:
            # SI/FM velocity--score weight w_tau = a_tau + 1/2 g^2.
            w = self.interpolation.velocity_score_coeff(t) + 0.5 * g ** 2
        return w.reshape(-1)[0] if torch.is_tensor(w) else float(w)

    # ------------------------------------------------------------------ #
    # Hooks.
    # ------------------------------------------------------------------ #

    def _pre_step(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Initialise weights at the first step; reset the per-step incr buffer."""
        E = base.shape[0]
        first_step = (
            self._log_w is None
            or self._log_w.shape[0] != E
            or float(t.reshape(-1)[0]) <= MIN_TIME + 1e-12
        )
        if first_step:
            # New window: uniform weights.
            self._log_w = torch.zeros(E, device=base.device)
        # Flag the first recorded step so _one_step banks the s_k R(x_k) baseline
        # skipped by model._compute_first_step (keeps the Eq. 6 telescoping exact).
        self._is_first_window_step = first_step
        self._incr_chunks = []
        return base

    def _one_step(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: torch.Tensor,
        pars_cond: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """One guided-SDE proposal + SMC log-weight increment for one chunk."""
        base = base.detach()
        t_next = torch.clamp(t + dt, max=1.0)

        # --- Prior reverse-SDE drift b(x_k) ----------------------------- #
        prior_drift = self._prior_drift(
            base, t, field_history, field_cond, pars_cond
        )

        # --- Guidance grad_x G and (maybe) reward R at x_k -------------- #
        grad_G, reward_k = self._compute_guidance(
            base, t, observations, field_history, field_cond, pars_cond, dt, prior_drift
        )
        if reward_k is None:
            reward_k = self._reward(
                base, t, observations, field_history, field_cond, pars_cond
            )

        # g(t) = Sigma^{1/2}(t), the reverse-SDE diffusion coefficient (scalar per
        # t-step). Floor away from 0 so the 1/g weight factor stays finite.
        g = self.diffusion_term(t).reshape(-1)[0].clamp(min=MIN_TIME)  # scalar
        # Injection weight w: the plugged method's OWN guidance strength (FlowDAS
        # zeta via sde_weight, SDA g^2, ...), so the proposal is that method's
        # guided SDE. Standalone SURGE overrides this to its DPS guidance_scale.
        w = self._injection_weight(t, g)

        # --- Guided Euler-Maruyama proposal (paper Eq. 4) --------------- #
        # x_{k+1} = x_k + [b + w grad_x G] dt + g sqrt(dt) xi (w grad_x G = Sigma grad_x G).
        noise = torch.randn_like(base)
        diffusion_incr = g * noise * dt.sqrt()
        base_next = base + (prior_drift + w * grad_G) * dt + diffusion_incr

        # --- SMC incremental log-weight (paper Alg. 1 / Eq. 6) ---------- #
        # Reward delta: (t+dt) R(x_{k+1}) - t R(x_k). R is the TRUE likelihood.
        reward_next = self._reward(
            base_next, t_next, observations, field_history, field_cond, pars_cond
        )

        t_s = t.reshape(-1)[0]
        t_s_next = t_next.reshape(-1)[0]
        reward_delta = t_s_next * reward_next - t_s * reward_k  # [E]
        if self._is_first_window_step:
            # The window's first diffusion step is taken by the UNGUIDED prior
            # (model._compute_first_step) and records no increment, so bank the
            # starting-state potential s_k R(x_k) here. Without it the telescoping
            # sum would carry a spurious -s_k R(x_k); with it the cumulative
            # log-weight terminates at exactly R(x_K) = log p(y | x_1) (Eq. 6).
            reward_delta = reward_delta + t_s * reward_k  # -> t_s_next * reward_next

        # Girsanov terms use u = Sigma^{1/2} grad_x G = (w / g) grad_x G -- the
        # SAME guidance that entered the proposal drift (w grad_x G) -- so the
        # weight debiases the proposal exactly (holds for any adapted drift).
        E = base.shape[0]
        u = ((w / g) * grad_G).reshape(E, -1)  # Sigma^{1/2} grad_x G
        flat_dW = (noise * dt.sqrt()).reshape(E, -1)  # sqrt(dt) xi
        martingale = -(u * flat_dW).sum(dim=1)  # [E]
        quad_var = -0.5 * (u ** 2).sum(dim=1) * dt  # [E]

        log_incr = (reward_delta + martingale + quad_var).detach()
        # Guard: a non-finite reward (e.g. a still-large denoiser) must not poison
        # the cumulative weights; clamp NaN/inf to a large finite penalty.
        log_incr = torch.nan_to_num(log_incr, nan=-1e30, neginf=-1e30, posinf=1e30)
        self._incr_chunks.append(log_incr)

        return base_next.detach()

    def _post_step(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        dt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate log-weights across all chunks, then ESS-resample."""
        if not self._incr_chunks:
            return base, field_history
        log_incr = torch.cat(self._incr_chunks, dim=0).to(base.device)  # [E]
        if self._log_w is None or self._log_w.shape[0] != base.shape[0]:
            self._log_w = torch.zeros(base.shape[0], device=base.device)
        self._log_w = self._log_w + log_incr

        base, field_history = self._maybe_resample(base, field_history)
        return base, field_history

    def _post_sample(
        self,
        base: torch.Tensor,
        observations: torch.Tensor,
        field_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Final resample so the returned ensemble is an equal-weight posterior."""
        base, field_history = self._maybe_resample(
            base, field_history, force=True
        )
        # Reset weights so the next window starts uniform.
        self._log_w = None
        return base, field_history

    # ------------------------------------------------------------------ #
    # SMC resampling.
    # ------------------------------------------------------------------ #

    def _normalized_weights(self) -> torch.Tensor:
        """Stable softmax of the cumulative log-weights -> [E] simplex."""
        lw = self._log_w
        lw = torch.nan_to_num(lw, nan=-1e30, neginf=-1e30, posinf=1e30)
        w = torch.softmax(lw - lw.max(), dim=0)
        return w

    def _maybe_resample(
        self,
        base: torch.Tensor,
        field_history: torch.Tensor,
        force: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Systematic resample when ESS / N < threshold (or ``force``)."""
        if self._log_w is None:
            return base, field_history
        E = base.shape[0]
        if E < 2:
            return base, field_history
        w = self._normalized_weights().to(base.device)
        ess = 1.0 / torch.clamp((w**2).sum(), min=1e-30)
        if not force and float(ess) >= self.ess_threshold * E:
            return base, field_history

        # Systematic resampling (low-variance).
        u0 = torch.rand(1, device=base.device) / E
        positions = u0 + torch.arange(E, device=base.device) / E
        cumsum = torch.cumsum(w, dim=0)
        cumsum[-1] = 1.0
        idx = torch.searchsorted(cumsum, positions).clamp(max=E - 1)

        base = base[idx].clone()
        field_history = field_history[idx].clone()
        # Reset to uniform weights after resampling.
        self._log_w = torch.zeros(E, device=base.device)
        return base, field_history


class SurgeFlowMatchingPosterior(SurgePosteriorBase):
    """SURGE on a flow-matching / diffusion prior (source ``N(0, I)``, ``a0 = 0``).

    Reward denoiser: FM Tweedie ``xhat_1 = (x + sigma^2 s) / beta`` with the score
    ``s = model.score(x_t)``. Plug any FM observation-score approximation
    (``SDALikelihood(model_class='fm')``, ``DPSGaussianLikelihood``, ...) as
    ``likelihood_model`` for the guidance.
    """

    model_class = "fm"

    def __init__(
        self,
        model: nn.Module,
        obs_operator: LinearObservationOperator,
        variance: float = 0.05,
        guidance_scale: float = 1.0,
        ess_threshold: float = 0.5,
        likelihood_model: Optional[nn.Module] = None,
        diffusion_term: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            guidance_scale=guidance_scale,
            ess_threshold=ess_threshold,
            likelihood_model=likelihood_model,
            diffusion_term=diffusion_term,
            gaussian_base=True,  # FM: N(0, I) init, anchor a0 = 0.
        )

    def _denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """FM Tweedie denoiser xhat_1 = (x + sigma^2 s) / beta (anchor a0 = 0)."""
        # Clamp t off the endpoints: beta = t -> 0 makes xhat_1 singular (the
        # Tweedie denoiser is undefined at the pure-noise end).
        t = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)
        t_grid = t.reshape(t.shape[0], *([1] * (x.dim() - 1))) if t.dim() < x.dim() else t
        beta = self.interpolation.beta(t_grid)
        sigma = self.interpolation.sigma(t_grid)
        score = self.model.score(x, t, field_history, field_cond, pars_cond)
        return (x + sigma**2 * score) / beta

    def _lik_kwargs(self, x, t, field_history, field_cond, pars_cond, prior_drift) -> dict:
        # Self-contained FM likelihoods (SDA-fm, DPS) recompute their own denoiser
        # from ``model.score``; pass the reverse-SDE drift for the odd consumer.
        return {"drift": prior_drift}


class SurgeStochasticInterpolantPosterior(SurgePosteriorBase):
    """SURGE on a stochastic-interpolant prior (source ``delta_{x0}``, ``a0 = x0``).

    Reward denoiser: SI Tweedie ``xhat_1 = (x + sigma^2 s - alpha x0) / beta`` with
    the score ``s`` recovered from the trained SI velocity ``b_theta`` via
    ``interpolation.score_from_velocity`` (anchor ``a0 = x0``). Plug any SI
    observation-score approximation (``FlowdasGaussianLikelihood``,
    ``SDALikelihood(model_class='si')``, ...) as ``likelihood_model``.
    """

    model_class = "si"

    def __init__(
        self,
        model: nn.Module,
        obs_operator: LinearObservationOperator,
        variance: float = 0.05,
        guidance_scale: float = 1.0,
        ess_threshold: float = 0.5,
        likelihood_model: Optional[nn.Module] = None,
        diffusion_term: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            model=model,
            obs_operator=obs_operator,
            variance=variance,
            guidance_scale=guidance_scale,
            ess_threshold=ess_threshold,
            likelihood_model=likelihood_model,
            diffusion_term=diffusion_term,
            gaussian_base=False,  # SI: point-mass delta_{x0} init.
        )

    def _denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """SI Tweedie denoiser xhat_1 = (x + sigma^2 s - alpha x0) / beta."""
        t = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)
        t_grid = t.reshape(t.shape[0], *([1] * (x.dim() - 1))) if t.dim() < x.dim() else t
        alpha = self.interpolation.alpha(t_grid)
        beta = self.interpolation.beta(t_grid)
        sigma = self.interpolation.sigma(t_grid)
        a0 = field_history[..., -1]  # SI anchor x0
        v = self.model.drift(x, t, field_history, field_cond, pars_cond)
        score = self.interpolation.score_from_velocity(x=x, v=v, t=t_grid, a0=a0)
        return (x + sigma**2 * score - alpha * a0) / beta

    def _lik_kwargs(self, x, t, field_history, field_cond, pars_cond, prior_drift) -> dict:
        # SI likelihoods (FlowDAS, SDA-si) take the SI drift b_theta; FlowDAS also
        # honours the diffusion schedule for its predictor spread.
        return {"drift": prior_drift, "diffusion_term": self.diffusion_term}


class SurgePosterior(SurgeFlowMatchingPosterior):
    """Standalone SURGE (internal DPS guidance) -- the ``SURGE`` method.

    A :class:`SurgeFlowMatchingPosterior` whose guidance is the DPS gradient of
    the reward (the true observation log-likelihood on the FM Tweedie denoiser),
    i.e. ``grad_x G = grad_x log p(y | x_t)`` with no plugged likelihood. The SMC
    layer then debiases this DPS proposal.
    """

    def _compute_guidance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        observations: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
        dt: torch.Tensor,
        prior_drift: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Internal DPS score grad_x log p(y | x_t) and the reward R (reused)."""
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            xhat1 = self._denoise(x_g, t, field_history, field_cond, pars_cond)
            residual = observations - self.obs_operator(xhat1)  # [E, N_y]
            reward = (-0.5 / self.variance) * (residual**2).sum(dim=1)  # [E]
            grad = torch.autograd.grad(reward.sum(), x_g)[0]  # [E, C, H, W]
        return grad.detach(), reward.detach()

    def _injection_weight(self, t: torch.Tensor, g: torch.Tensor):
        """Standalone DPS: scale the raw DPS score by ``guidance_scale`` directly
        (there is no plugged likelihood to supply an injection weight)."""
        return self.guidance_scale


__all__ = [
    "SurgePosteriorBase",
    "SurgeFlowMatchingPosterior",
    "SurgeStochasticInterpolantPosterior",
    "SurgePosterior",
]
