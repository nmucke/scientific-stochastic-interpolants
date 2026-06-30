"""SURGE posterior -- NEW BASELINE (diffusion model + guided SDE + SMC).

Single-window adaptation of SURGE (Wei, Ren, Shi & Lu, "SURGE: Approximation
and Training Free Particle Filter for Diffusion Surrogate", arXiv:2605.18745).

WHAT SURGE IS
-------------
SURGE is a *particle filter* (Sequential Monte Carlo) that uses a diffusion
model as the dynamics surrogate. For each physical step it draws each particle
through a **guided reverse-diffusion SDE** (DPS-style observation guidance steers
the proposal toward observation-consistent states) and then *reweights* the
particles along the diffusion trajectory with a Girsanov-corrected SMC weight,
resampling whenever the effective sample size drops below a threshold. The
Girsanov correction makes the method **approximation-free**: an imperfect
guidance ``G`` only affects proposal efficiency, not the target -- the SMC
weights debias it.

SURGE's per-diffusion-step update (paper Algorithm 1), in our forward pseudo-time
``t : 0 -> 1`` convention:

    Propose (guided EM step):
        x_{k+1} = x_k + [ v(x_k) + Sigma(t) grad_x G(x_k | y) ] dt
                       + Sigma^{1/2}(t) sqrt(dt) xi,        xi ~ N(0, I)

    Reweight (incremental log-weight, Eq. 6 / Alg. 1):
        log w_{k+1} = log w_k
            + [ (t+dt) R(x_{k+1}) - t R(x_k) ]                       # reward delta
            - Sigma^{1/2}(t) grad_x G(x_k | y) . ( sqrt(dt) xi )     # martingale
            - 1/2 Sigma(t) || grad_x G(x_k | y) ||^2 dt              # quad. variation

    Resample when ESS = 1 / sum_i w_i^2 < c * N.

ADAPTATIONS (documented, like the SDA / EnSF baselines)
-------------------------------------------------------
* **Single-window.** Our harness is autoregressive (one physical step / window
  at a time), so SURGE's outer particle-filter loop over physical steps is the
  harness's autoregressive rollout; this module runs ONE window's guided-SDE +
  SMC pass. Each window's resampled ensemble is fed back as the next window's
  history (the base-posterior ``sample_trajectory`` already threads this).
* **Guidance / reward = the DPS Gaussian observation log-likelihood** on the
  Tweedie denoiser ``xhat_1 = E[x_1 | x_t]`` (model_class='fm', anchor a0=0),
  i.e. ``log p(y | x_t) ~ -1/(2 sigma^2) || y - H xhat_1 ||^2``. ``grad_x G`` is
  the autograd gradient of this term (the DPS guidance score); the reward ``R``
  is the same log-likelihood. This is the standard DPS surrogate used by every
  diffusion-DA baseline here; SURGE's contribution over plain DPS is the SMC
  reweighting/resampling layer, which is implemented faithfully.
* ``Sigma(t)`` is the reverse-SDE diffusion coefficient ``g(t)^2`` and
  ``Sigma^{1/2}(t) = g(t)`` (``self.diffusion_term``), matching the
  :class:`DiffusionPosterior` prior drift / diffusion split.

The class subclasses :class:`BasePosterior` (NOT a likelihood plugged into
``DiffusionPosterior``) because SURGE needs to (a) see the exact injected noise
``xi`` to form the martingale weight term and (b) resample ACROSS particles --
neither of which the likelihood-only interface or ``DiffusionPosterior._one_step``
exposes.
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


class SurgePosterior(BasePosterior):
    """SURGE guided-SDE + SMC particle-filter posterior (single window)."""

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
        """Initialise the SURGE posterior.

        Args:
            model: The diffusion-model prior (DenoiseDiffusionModel; velocity /
                'fm' mode). Exposes ``.drift``, ``.score``,
                ``._get_velocity_from_score``, ``.interpolation``.
            obs_operator: Linear observation operator ``H``.
            variance: Observation-noise variance ``sigma^2`` (R = sigma^2 I).
            guidance_scale: Multiplier ``lambda`` on the DPS guidance gradient
                (paper's guidance strength).
            ess_threshold: Resample when ESS / N < ``ess_threshold`` (c in [0,1]).
            likelihood_model: Unused (SURGE owns its guidance); accepted for a
                uniform constructor signature. Defaults to None.
            diffusion_term: Reverse-SDE coefficient ``g(t)``; None -> model default.
        """
        # ``likelihood_model`` is required by BasePosterior.__init__ but SURGE
        # computes its own guidance, so pass a trivial placeholder.
        super().__init__(
            model=model,
            likelihood_model=likelihood_model or nn.Identity(),
            diffusion_term=diffusion_term,
            gaussian_base=True,  # DM path: init from N(0, I), anchor a0 = 0.
        )
        self.obs_operator = obs_operator
        self.variance = float(variance)
        self.guidance_scale = float(guidance_scale)
        self.ess_threshold = float(ess_threshold)
        self.interpolation = self.model.interpolation
        # Per-window state (reset each window in ``_pre_step`` on the first step).
        self._log_w: Optional[torch.Tensor] = None  # [E] cumulative log-weights
        self._incr_chunks: list[torch.Tensor] = []  # per-chunk incr this t-step

    # ------------------------------------------------------------------ #
    # Guidance / reward: DPS Gaussian log-likelihood on the Tweedie denoiser.
    # ------------------------------------------------------------------ #

    def _denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Tweedie denoiser xhat_1 = E[x_1 | x_t] (DM prior, anchor a0 = 0)."""
        # Clamp t off the endpoints: beta = t -> 0 makes xhat_1 = (x + sigma^2 s)/beta
        # blow up, which is the root of the gs>=1 divergence.
        t = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)
        t_grid = t.reshape(t.shape[0], *([1] * (x.dim() - 1))) if t.dim() < x.dim() else t
        alpha = self.interpolation.alpha(t_grid)
        beta = self.interpolation.beta(t_grid)
        sigma = self.interpolation.sigma(t_grid)
        score = self.model.score(x, t, field_history, field_cond, pars_cond)
        # a0 = 0 for the DM/FM path.
        xhat1 = (x + sigma**2 * score) / beta
        return xhat1

    def _guidance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        observations: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(grad_x G, R)``: DPS guidance drift and the reward (log p).

        ``R = -1/(2 sigma^2) || y - H xhat_1 ||^2`` (per member, [E]) is the TRUE
        SMC target potential (kept exact). The returned guidance ``grad_x G`` is
        the raw DPS gradient of ``R`` STEP-NORMALISED to unit L2 norm per member
        and scaled by ``guidance_scale`` -- a bounded, scale-free drift like the
        DPS / FlowDAS / FIG baselines. SURGE stays approximation-free: the
        Girsanov SMC weight uses this SAME ``grad_x G`` in both the martingale and
        quadratic-variation terms, so the proposal is exactly debiased regardless
        of how the drift is rescaled (Girsanov holds for any adapted drift). The
        raw 1/sigma^2 * denoiser-Jacobian gradient was unbounded and diverged to
        NaN for guidance_scale >= 1.
        """
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            xhat1 = self._denoise(x_g, t, field_history, field_cond, pars_cond)
            residual = observations - self.obs_operator(xhat1)  # [E, N_y]
            reward = -0.5 / self.variance * (residual**2).sum(dim=1)  # [E]
            grad = torch.autograd.grad(reward.sum(), x_g)[0]  # [E, C, H, W]

        # DPS step-normalisation: unit-L2 per member, then scale. The normalised
        # drift is still a valid adapted drift, so the Girsanov weight debiases it.
        E = grad.shape[0]
        gnorm = grad.reshape(E, -1).norm(dim=1).reshape(E, *([1] * (grad.dim() - 1)))
        grad_tilde = self.guidance_scale * grad / (gnorm + 1e-6)
        return grad_tilde.detach(), reward.detach()

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
            # New window: uniform weights and clean reward baseline.
            self._log_w = torch.zeros(E, device=base.device)
            self._prev_reward = torch.zeros(E, device=base.device)
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

        # --- Prior reverse-SDE drift (v + 1/2 g^2 s) --------------------- #
        drift = self.model.drift(
            base, t, field_history, field_cond, pars_cond
        ).detach()

        # --- DPS guidance grad_x G and reward R at x_k ------------------- #
        grad_G, reward_k = self._guidance(
            base, t, observations, field_history, field_cond, pars_cond
        )

        # Sigma^{1/2}(t) = g(t), Sigma(t) = g(t)^2. The diffusion term is a
        # scalar function of t here (one window-time per step); take the scalar
        # so the per-member weight terms stay [E] (not broadcast to [B, E]).
        g = self.diffusion_term(t).reshape(-1)[0]  # scalar
        sigma2 = g**2  # scalar

        # --- Guided Euler-Maruyama proposal ----------------------------- #
        # x_{k+1} = x_k + [v + Sigma grad_G] dt + Sigma^{1/2} sqrt(dt) xi
        noise = torch.randn_like(base)
        diffusion_incr = g * noise * dt.sqrt()
        base_next = base + (drift + sigma2 * grad_G) * dt + diffusion_incr

        # --- SMC incremental log-weight (paper Alg. 1 / Eq. 6) ---------- #
        # Reward delta: (t+dt) R(x_{k+1}) - t R(x_k). Recompute R at x_{k+1}.
        with torch.no_grad():
            xhat1_next = self._denoise(
                base_next, t_next, field_history, field_cond, pars_cond
            )
            resid_next = observations - self.obs_operator(xhat1_next)
            reward_next = -0.5 / self.variance * (resid_next**2).sum(dim=1)  # [E]

        t_s = t.reshape(-1)[0]
        t_s_next = t_next.reshape(-1)[0]
        reward_delta = t_s_next * reward_next - t_s * reward_k  # [E]

        # Flatten the per-element guidance / noise to [E, D] for the dot products.
        E = base.shape[0]
        flat_grad = (g * grad_G).reshape(E, -1)  # Sigma^{1/2} grad_G
        flat_dW = (noise * dt.sqrt()).reshape(E, -1)  # sqrt(dt) xi
        martingale = -(flat_grad * flat_dW).sum(dim=1)  # [E]
        quad_var = -0.5 * (sigma2 * (grad_G.reshape(E, -1) ** 2).sum(dim=1)) * dt  # [E]

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


__all__ = ["SurgePosterior"]
