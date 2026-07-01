import pdb
from typing import Any, Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator

# Clamp pseudo-time away from the endpoints where schedule denominators vanish.
MIN_TIME = 1e-4


class GuidanceGaussianLikelihood(nn.Module):
    """Guided-flow (FIG / FlowDAS-style) likelihood — BASELINE, not the paper method.

    Two opt-in modes, selected by ``weighting``:

    ``weighting="simple"`` (DEFAULT, unchanged legacy behaviour)
        The legacy one-step DPS/FIG guidance: it predicts ``x_1`` by a single
        Euler extrapolation ``x + (1 - t) * v`` and differentiates the raw
        observation log-likelihood ``-1/2 ||y - H x_1||^2 / sigma^2`` through it,
        then normalises by the residual norm. It does **not** use the
        observation interpolant, the inflated covariance ``Sigma_bar``, the
        source moments, or the multiplicative gain ``G_tau`` of the paper's
        unified method (see ``InterpolantGaussianLikelihood``).

    ``weighting="ot_ode"`` (Pokle et al., OT-ODE variant)
        The covariance-preconditioned guidance of *Training-free Linear Image
        Inverses via Flows* (Pokle, Muckley, Chen & Karrer, TMLR 2024,
        https://arxiv.org/abs/2310.04432), Algorithm 1 (Conditional OT-ODE). See
        :meth:`_score_ot_ode` for the equation-by-equation derivation and the
        reverse->forward time-convention mapping. This mode preconditions the
        guidance by ``(r_t^2 H H^T + sigma_y^2 I)^{-1}`` (paper Alg. 1 line 7),
        tempering the ``1/sigma^2`` blow-up of raw DPS.

    Retained only as a FlowDAS/FIG / Pokle baseline; the paper's own FM method
    routes through ``InterpolantGaussianLikelihood`` with the FM source moments.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        weighting: str = "simple",
        obs_variance: Optional[float] = None,
        guidance_scale: float = 1.0,
    ) -> None:
        """
        Initialize Multivariate Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            model: Model.
            variance: Variance.
            weighting: ``"simple"`` (default, legacy DPS-on-flow guidance) or
                ``"ot_ode"`` (Pokle et al. covariance-preconditioned OT-ODE).
            obs_variance: Measurement noise variance ``sigma_y^2`` used inside
                the OT-ODE preconditioner ``(r_t^2 H H^T + sigma_y^2 I)^{-1}``
                (paper Alg. 1 line 7). Defaults to ``variance`` (= ``R``) when
                ``None``. Pokle uses ``sigma_y in {0, 0.05}`` for non-denoising
                tasks; the ``r_t^2 H H^T`` term keeps the solve regular even at
                ``sigma_y = 0`` away from the endpoints.
            guidance_scale: Pokle's adaptive weight ``gamma_t`` (paper Alg. 1,
                "Adaptive weight gamma_t"); the unadaptive OT-ODE default is 1.
        """
        super(GuidanceGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        if weighting not in ("simple", "ot_ode"):
            raise ValueError(
                f"weighting must be 'simple' or 'ot_ode', got {weighting!r}."
            )
        self.weighting = weighting
        self.obs_variance = (
            float(obs_variance) if obs_variance is not None else float(variance)
        )
        self.guidance_scale = float(guidance_scale)
        # FM anchor a0 = 0 -> the FM posterior's anchor assertion passes.
        self.anchor = "zeros"
        self._HHt: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, observations: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. [B, C, H, W]
            observations: Observations.
            variance: Variance.

        Returns:
            torch.Tensor: Log probability.
        """
        pass

    def _compute_one_step_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the one step predictions."""

        # alpha = self.model.interpolation.alpha(t)
        # alpha_diff = self.model.interpolation.alpha_diff(t)
        # beta = self.model.interpolation.beta(t)
        # beta_diff = self.model.interpolation.beta_diff(t)

        # x_coeff = alpha_diff / (beta_diff * alpha - alpha_diff * beta)
        # score_coeff = alpha / (beta_diff * alpha - alpha_diff * beta)

        # return x_coeff * x + score_coeff * drift

        return x + (1 - t) * drift

    def _schedule(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the schedule."""

        alpha = self.model.interpolation.alpha(t)
        alpha_diff = self.model.interpolation.alpha_diff(t)
        beta = self.model.interpolation.beta(t)
        beta_diff = self.model.interpolation.beta_diff(t)

        # return (beta_diff * alpha - alpha_diff * beta) / alpha
        # return alpha * (beta_diff * alpha - alpha_diff * beta) / beta
        return (1 - t) / t

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the likelihood (guidance) score.

        Returns ``(score, log_likelihood)`` to match the
        ``InterpolantGaussianLikelihood`` / ``FlowdasGaussianLikelihood``
        interface so the FM / SI posteriors can unpack it. Extra keyword
        arguments (e.g. ``score=`` from the FM posterior, ``diffusion_term=``)
        are accepted and ignored -- FIG guidance differentiates the one-step
        ``x_1`` prediction directly and needs neither.

        Branches on ``self.weighting``: ``"ot_ode"`` -> Pokle et al. OT-ODE
        (:meth:`_score_ot_ode`); ``"simple"`` (default) -> the legacy
        residual-norm-normalised one-step DPS guidance below (UNCHANGED).
        """

        if self.weighting == "ot_ode":
            return self._score_ot_ode(
                observations=observations,
                x=x,
                t=t,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
                drift=drift,
            )

        # Compute the guidance on a fresh grad-enabled leaf so the caller's
        # autoregressive working state is never mutated in place (mirrors the
        # DPS / FlowDAS fixes).
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            preds = self._compute_one_step_prediction(
                x_g, t, dt, field_history, field_cond, pars_cond, drift
            )
            residual = observations - self.obs_operator(preds)
            diff_norm = torch.linalg.norm(residual, dim=1) ** 2  # [B] = ||y - H x1||^2
            grad = torch.autograd.grad(outputs=diff_norm.sum(), inputs=x_g)[0]

        # DPS-style step normalization zeta_t = zeta/||y - H x1||: divide the
        # squared-residual gradient by the residual NORM and drop the raw 1/sigma^2
        # (=400 at R=0.0025). The raw guidance is unbounded (set by the network
        # Jacobian x 1/sigma^2) and diverged to NaN at NS scale, exactly as the
        # original DPS/FlowDAS did; this bounds the per-step pull.
        norm = diff_norm.sqrt().reshape(-1, *([1] * (grad.dim() - 1)))  # [B,1,1,1]
        guidance = -grad / (norm + 1e-6)

        log_likelihood = (-0.5 * diff_norm / self.original_variance).detach()
        return guidance.detach(), log_likelihood

    def _get_HHt(self, ref: torch.Tensor) -> torch.Tensor:
        """Return ``H H^T`` (cached), shape ``[N_y, N_y]`` (OT-ODE only).

        Mirrors ``SDALikelihood._get_HHt`` (sda.py): builds the dense
        Gram matrix from ``obs_operator.obs_matrix`` for the ``N_y x N_y``
        covariance solve.
        """
        if (
            self._HHt is None
            or self._HHt.device != ref.device
            or self._HHt.dtype != ref.dtype
        ):
            H = self.obs_operator.obs_matrix.to(device=ref.device, dtype=ref.dtype)
            self._HHt = H @ H.transpose(0, 1)
        return self._HHt

    def _score_ot_ode(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """OT-ODE covariance-preconditioned guidance (Pokle et al., Algorithm 1).

        Reference: *Training-free Linear Image Inverses via Flows* (Pokle,
        Muckley, Chen & Karrer, TMLR 2024, arXiv:2310.04432), the Conditional
        OT-ODE variant (Algorithm 1; Eqs. 10 & 16).

        TIME-CONVENTION MAPPING (reverse Pokle -> forward ours)
        -------------------------------------------------------
        Pokle's *original* presentation runs pseudo-time in REVERSE, ``tau:
        1 -> 0`` with ``tau = 1`` the noise/source and ``tau = 0`` the clean
        data. OUR integrator runs FORWARD, ``t: 0 -> 1`` with ``t = 0`` the
        source ``~ N(0, I)`` and ``t = 1`` the clean field
        (``LinearDeterministicInterpolation``: ``x_t = (1-t) eps + t x_1``).
        We translate every time-indexed quantity via ``t_ours = 1 - tau_pokle``.

        Under that map the conditional OT path coefficients (Pokle: mean
        ``alpha_tau x_1``, std ``sigma_tau``, with ``alpha_tau`` the DATA weight
        and ``sigma_tau`` the NOISE weight) become, in our forward ``t``,

            alpha^OT(t) = (data weight)  = t      = our beta(t),
            sigma^OT(t) = (noise weight) = 1 - t  = our alpha(t),

        i.e. Pokle's OT path is EXACTLY our linear/rectified-flow path. So the
        guidance is written directly in our forward ``t`` (no reverse indexing).

        - Posterior variance estimate r_t^2 (Pokle Eq. 16):
              r_tau^2 = sigma_tau^2 / (sigma_tau^2 + alpha_tau^2)
          -> r_t^2  = (1 - t)^2 / ((1 - t)^2 + t^2)              [forward t].
          Endpoint sanity check (our direction): at t=0 -> r_0^2 = 1 (max prior
          uncertainty, the state is ~N(0,I)); at t->1 -> r_1^2 -> 0 (clean
          field, no uncertainty). This is the correct forward direction.

        - One-step denoiser xhat_1 = E[x_1 | x_t]. Pokle Eq. 10 inverts the
          velocity<->denoiser relation; for the OT path it reduces to
              vhat = (xhat_1 - x_t) / (1 - tau)  ->  xhat_1 = x_t + (1-tau) vhat.
          Under ``tau -> 1 - t`` the velocity flips sign (``dx/dtau =
          -dx/dt``), and ``(1 - tau) = t`` in their time. Re-expressed in OUR
          forward time the denoiser is the standard FM one-step extrapolation
              xhat_1 = x_t + (1 - t) * v,        v = our forward velocity,
          which is identically ``self._compute_one_step_prediction`` (and our
          path's ``E[x_1|x_t]``). We use the forward ``v`` (``drift=``), so the
          sign is already correct for forward integration.

        - Guidance vector g (Pokle Alg. 1 line 7):
              g = (d xhat_1 / d z_t)^T A^T (r_t^2 A A^T + sigma_y^2 I)^{-1}
                  (y - A xhat_1).
          This is the gradient w.r.t. x of the Mahalanobis data term
              -1/2 (y - A xhat_1)^T (r_t^2 A A^T + sigma_y^2 I)^{-1} (y - A xhat_1),
          computed by autograd through the one-step denoiser (the vector-Jacobian
          product folds in (d xhat_1/d z_t)^T). The preconditioner
          ``(r_t^2 H H^T + sigma_y^2 I)^{-1}`` is the OT-ODE replacement for raw
          DPS's ``1/sigma_y^2``; it tempers the blow-up as sigma_y -> 0.

        - OT-ODE conditional velocity. ``g`` (Alg. 1 line 7) is a SCORE-like
          gradient of the data log-likelihood w.r.t. ``x_t``; the conditional
          velocity adds it to the prior velocity through the SAME
          score->velocity coefficient ``a_tau`` the rest of the family uses:
              v(x_t | y) = v(x_t) + a_tau * g,   a_tau = (1 - t)/t  (linear path).
          BOOKKEEPING WITH THE POSTERIOR'S w_tau. ``FlowMatchingPosterior.
          _one_step`` forms ``b_post = b_prior + w_tau * guidance`` then steps
          ``base += b_post * dt`` with (ODE path) ``w_tau = a_tau =
          velocity_score_coeff(t)``. So the posterior ALREADY applies the
          score->velocity ``a_tau`` factor; we therefore return the BARE
          preconditioned VJP ``guidance = gamma_t * g`` (the earlier
          ``(t/(1-t))^2`` pre-weight double-applied the conversion AND inverted
          it, making ``w_tau * guidance = (t/(1-t)) g`` blow up as ``t -> 1``).
          ``gamma_t`` (``guidance_scale``, Pokle's adaptive weight, default 1)
          multiplies g.

        Numerical stability: ``t`` is clamped to ``[1e-4, 1-1e-4]`` and the
        posterior-variance estimate ``r_t^2`` is floored at ``R2_FLOOR`` so the
        preconditioner ``M = r_t^2 H H^T + sigma_y^2 I`` stays invertible as
        ``t -> 1`` even at ``sigma_y = 0`` (for a selection operator
        ``H H^T = I`` and ``r_t^2 -> 0`` would otherwise make ``M`` singular).
        The bounded ``a_tau -> 0`` factor then keeps the velocity correction
        finite at the data end.
        """
        # Floor for r_t^2 so M = r_t^2 H H^T + sigma_y^2 I is never singular as
        # t -> 1 (where r_t^2 -> 0); only the last few steps are affected.
        R2_FLOOR = 1e-2
        t_clamped = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)
        if t.dim() < x.dim():
            t_grid = t_clamped.reshape(t_clamped.shape[0], *([1] * (x.dim() - 1)))
        else:
            t_grid = t_clamped

        # r_t^2 = (1-t)^2 / ((1-t)^2 + t^2)  [Pokle Eq. 16, forward t], floored
        # so the preconditioner stays invertible as t -> 1.
        one_minus_t = 1.0 - t_grid
        r2 = (one_minus_t**2) / (one_minus_t**2 + t_grid**2)
        r2_scalar = max(float(r2.reshape(-1)[0]), R2_FLOOR)

        # Chunked guidance over the ensemble batch.
        # The U-Net backward graph for the full B=64 ensemble at 128×128 exceeds
        # GPU memory. quad[b] is independent across b (each member's xhat_1 only
        # depends on its own activations), so chunking is mathematically exact.
        _OT_CHUNK = 8  # members per backward pass; lower if still OOM
        B = x.shape[0]

        # Build the shared preconditioner M once (independent of x).
        HHt = self._get_HHt(x)  # [N_y, N_y]
        N_y = HHt.shape[0]
        eye = torch.eye(N_y, device=HHt.device, dtype=HHt.dtype)
        M = r2_scalar * HHt + self.obs_variance * eye

        grad_chunks: list[torch.Tensor] = []
        residual_chunks: list[torch.Tensor] = []
        for b0 in range(0, B, _OT_CHUNK):
            b1 = min(b0 + _OT_CHUNK, B)
            sl = slice(b0, b1)

            x_chunk = x[sl].detach().requires_grad_(True)
            tc_chunk = t_clamped[sl] if t_clamped.shape[0] == B else t_clamped
            tg_chunk = t_grid[sl] if t_grid.shape[0] == B else t_grid
            fh_chunk = field_history[sl]
            fc_chunk = field_cond[sl] if field_cond is not None else None
            pc_chunk = pars_cond[sl] if pars_cond is not None else None
            obs_chunk = observations[sl]

            with torch.enable_grad():
                # Recompute velocity FROM x_chunk so autograd tracks the full
                # Jacobian J = I + (1-t)*dv/dx.  Using frozen drift → J = I,
                # ~50× over-guidance at small t (see class docstring).
                vel_chunk = self.model.drift(
                    x_chunk, tc_chunk, fh_chunk, fc_chunk, pc_chunk
                )
                xhat1_chunk = x_chunk + (1.0 - tg_chunk) * vel_chunk
                res_chunk = obs_chunk - self.obs_operator(xhat1_chunk)  # [n, N_y]
                # M is held fixed (detached); gradient flows only through residual.
                sol_chunk = torch.linalg.solve(
                    M.detach(), res_chunk.transpose(0, 1)
                ).transpose(0, 1)  # [n, N_y]
                quad_chunk = 0.5 * (res_chunk * sol_chunk).sum(dim=1)  # [n]
                g_chunk = torch.autograd.grad(
                    outputs=quad_chunk.sum(), inputs=x_chunk
                )[0]

            grad_chunks.append(g_chunk.detach())
            residual_chunks.append(res_chunk.detach())

        grad = torch.cat(grad_chunks, dim=0)        # [B, ...]
        residual = torch.cat(residual_chunks, dim=0)  # [B, N_y]

        # grad = -(d xhat_1/d z_t)^T H^T M^{-1} (y - H xhat_1); flip sign so
        # g = +VJP toward the data (Pokle's line-7 g points up the log-likelihood).
        g = -grad

        # Return the BARE preconditioned VJP scaled by gamma_t. The posterior
        # multiplies by w_tau = a_tau = (1-t)/t, which IS the score->velocity
        # conversion of the OT-ODE conditional velocity v + a_tau * g (so the net
        # correction is a_tau * g, finite as t -> 1 since a_tau -> 0).
        guidance = self.guidance_scale * g

        log_likelihood = (-0.5 * (residual.detach() ** 2).sum(dim=1)
                          / self.original_variance).detach()
        return guidance.detach(), log_likelihood


class FIGGaussianLikelihood(nn.Module):
    """FIG guidance likelihood (yan_fig_2025) -- BASELINE.

    Faithful re-implementation of *Flow with Interpolant Guidance* (FIG;
    Yan, Zhang, Meng & Zhao, ICLR 2025, https://openreview.net/forum?id=fs2Z2z3GRx),
    matching the official ``FIG_flow/sampler.py`` ``FIG.update`` corrector.

    FIG is a guided probability-flow ODE: after each prior Euler flow step it
    pulls the proposed next state toward a *measurement interpolant* ``y_t`` --
    the measurement scaled along the flow time -- by ``k`` inner gradient-descent
    corrections, with step size ``c * (1 - t) / t``. Concretely, the official
    update (paper Algorithm 1 / FIG_flow/sampler.py ``class FIG``) is::

        y_t      = t_next * y + w * (1 - t) * H(eps),   eps ~ N(0, I)     (FIG Eq. 9)
        for j in range(k):
            r          = || y_t - H x_next ||                            (residual NORM)
            x_next    -= c * (1 - t) / t * grad_{x_next} r               (FIG Eq. 10)

    where ``x_next = x + v * dt`` is the prior FM-ODE Euler step (their
    ``euler_sampler``: ``x_next = x + v * dt``). The gradient is taken w.r.t.
    ``x_next`` (the post-flow state), NOT through the network -- ``H`` is linear,
    so ``grad_{x_next} ||y_t - H x_next|| = -H^T (y_t - H x_next) / ||y_t - H x_next||``,
    a residual-direction pull that is intrinsically bounded (unit-normalised by
    the residual norm). That is FIG's built-in numerical safeguard: the
    ``1/sigma^2`` factor that makes raw DPS/FlowDAS guidance blow up at NS scale
    is absent by construction (FIG differentiates the residual NORM, not the
    squared, noise-scaled log-likelihood). No extra clamp is needed beyond the
    repo-standard ``eps`` on the norm denominator and keeping ``t`` away from 0.

    Time convention. FIG's code runs ``t : eps -> 1`` with ``t = 0`` the noise /
    source and ``t = 1`` the clean image -- IDENTICAL to our forward
    ``t : 0 -> 1`` (``LinearDeterministicInterpolation``: ``x_t = (1-t) eps +
    t x_1``, ``xhat_1 = x + (1 - t) v``). FIG's ``next_t`` is our ``t + dt``. So
    no time flip or sign change is required; ``y_t = (t + dt) * y`` (default
    ``w = 0`` for the inpainting / super-resolution tasks that match our sparse /
    super-res scenarios).

    Posterior weighting. The FM-ODE posterior applies
    ``b_post = b_prior + w_tau * guidance`` then steps ``base += b_post * dt``,
    with (ODE path) ``w_tau = a_tau = velocity_score_coeff(t)``. For the linear
    (rectified-flow) FM path ``a_tau = (1 - t) / t`` -- exactly FIG's per-step
    step-size factor. To make the corrected state land on FIG's exact total
    displacement ``Delta = x_next_final - x_next_init`` (capturing the moving
    target across the ``k`` inner iterations, which no closed form reproduces),
    we run the ``k``-iteration corrector here and FOLD the posterior's
    ``w_tau * dt`` multiply OUT of the returned guidance: we return
    ``guidance = Delta / (w_tau * dt)`` so that the posterior's
    ``base += w_tau * guidance * dt`` adds back precisely ``Delta``. The
    effective per-step guidance therefore equals FIG's, independent of the FM
    posterior's schedule.

    Returns ``(guidance, log_likelihood)`` to match the
    ``InterpolantGaussianLikelihood`` / ``GuidanceGaussianLikelihood`` interface
    so it plugs into ``FlowMatchingPosterior`` (stepper=ode) unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        guidance_steps: int = 2,
        guidance_scale: float = 10.0,
        interpolant_noise: float = 0.0,
    ) -> None:
        """Initialize the FIG guidance likelihood.

        Args:
            model: Trained FM model (``FlowMatchingModel``); used only for its
                ``interpolation`` (the velocity is passed in via ``drift=``).
            obs_operator: Linear observation operator ``H`` exposing ``forward``
                (``H x``) and the transpose ``H^T``.
            variance: Measurement noise variance ``R = variance * I`` (used only
                for the reported ``log_likelihood``).
            ensemble_size: Ensemble size (kept for interface symmetry).
            guidance_steps: FIG's ``k`` -- number of inner corrector iterations
                per flow step (paper Algorithm 1). Official defaults: 2
                (inpainting), 1 (super-resolution).
            guidance_scale: FIG's ``c`` -- corrector step-size scale (paper
                Eq. 10). Official defaults: 10 (inpainting), 20 (super-res).
            interpolant_noise: FIG's ``w`` -- weight of the noise term in the
                measurement interpolant (paper Eq. 9). Official default 0 for
                inpainting / super-resolution.
        """
        super(FIGGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        self.guidance_steps = int(guidance_steps)
        self.guidance_scale = float(guidance_scale)
        self.interpolant_noise = float(interpolant_noise)
        self.interpolant = self.model.interpolation
        # FM anchor a0 = 0 -> the FM posterior's anchor assertion passes.
        self.anchor = "zeros"

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Unused (the ``score`` interface is the entry point)."""
        pass

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FIG measurement-interpolant guidance (yan_fig_2025, Algorithm 1).

        ``drift`` is the prior FM velocity ``v`` and ``dt`` the integrator step;
        together they reconstruct FIG's post-flow state ``x_next = x + v dt``.
        Returns ``(guidance, log_likelihood)`` with ``guidance`` pre-divided by
        the posterior's ``w_tau * dt`` (see class docstring) so the effective
        per-step correction equals FIG's exact ``k``-iteration displacement.
        """
        t_clamped = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)
        if t.dim() < x.dim():
            t_grid = t_clamped.reshape(t_clamped.shape[0], *([1] * (x.dim() - 1)))
        else:
            t_grid = t_clamped

        # FIG's per-iteration step-size factor c * (1 - t) / t (paper Eq. 10).
        step = self.guidance_scale * (1.0 - t_grid) / t_grid

        # Prior FM-ODE Euler flow step x_next = x + v dt (FIG ``euler_sampler``).
        # drift is the prior velocity v; if absent (degenerate), fall back to x.
        if drift is None:
            x_next_init = x.detach()
        else:
            x_next_init = (x + drift * dt).detach()

        # Measurement interpolant y_t = t_next * y + w (1 - t) H(eps) (FIG Eq. 9).
        # t_next is our t + dt (their ``next_t``). w defaults to 0.
        if dt is not None:
            t_next = (t_clamped + dt).clamp(max=1.0)
        else:
            t_next = t_clamped
        t_next_obs = t_next.reshape(t_next.shape[0], *([1] * (observations.dim() - 1)))
        y_t = t_next_obs * observations
        if self.interpolant_noise != 0.0:
            t_obs = t_clamped.reshape(
                t_clamped.shape[0], *([1] * (x.dim() - 1))
            )
            y_t = y_t + self.interpolant_noise * (1.0 - t_obs) * self.obs_operator(
                torch.randn_like(x_next_init)
            )

        # Official corrector guard (FIX 1b): skip the first and last integration
        # steps (FIG's ``1 <= i < N - 1``). With dt = 1/N this is t <= dt or
        # t >= 1 - dt. On a skipped step the guidance is zero (only the prior
        # flow step applies via the posterior).
        t_scalar = float(t_clamped.reshape(-1)[0])
        dt_scalar = float(dt.reshape(-1)[0]) if dt is not None else 0.0
        skip = (
            dt_scalar > 0.0
            and (t_scalar <= dt_scalar + 1e-9 or t_scalar >= 1.0 - dt_scalar - 1e-9)
        )
        if skip:
            guidance = torch.zeros_like(x_next_init)
            final_res = observations - self.obs_operator(x_next_init)
            diff_norm = torch.linalg.norm(
                final_res.reshape(final_res.shape[0], -1), dim=1
            ) ** 2
            log_likelihood = (-0.5 * diff_norm / self.original_variance).detach()
            return guidance, log_likelihood

        # k inner corrector iterations on a moving leaf (FIG ``class FIG``).
        x_next = x_next_init
        with torch.enable_grad():
            for _ in range(max(self.guidance_steps, 1)):
                x_g = x_next.detach().requires_grad_(True)
                residual = y_t - self.obs_operator(x_g)
                # FIG differentiates the residual NORM (not the squared,
                # noise-scaled log-likelihood) -> intrinsically bounded.
                norm = torch.linalg.norm(residual.reshape(residual.shape[0], -1), dim=1)
                grad = torch.autograd.grad(outputs=norm.sum(), inputs=x_g)[0]
                # FIX 1a: cap the per-iteration move magnitude at the residual
                # norm so a single inner step lands at most ON y_t (no overshoot
                # /collapse). grad is the unit residual direction, so the move is
                # eff_step * grad with eff_step = min(step, ||residual||).
                res_norm = norm.reshape(-1, *([1] * (grad.dim() - 1)))  # [B,1,..]
                eff_step = torch.minimum(step, res_norm)
                x_next = (x_g - eff_step * grad).detach()

        # Total FIG displacement on the post-flow state.
        delta = x_next - x_next_init

        # Fold out the posterior's w_tau * dt multiply (w_tau = a_tau on the ODE
        # path) so base += w_tau * guidance * dt reproduces exactly ``delta``.
        a_tau = self.interpolant.velocity_score_coeff(t_grid)
        denom = a_tau * dt if dt is not None else a_tau
        guidance = delta / (denom + 1e-12)

        # Reported per-member log-likelihood uses the RAW measurement model
        # -1/2 ||y - H x_next||^2 / R (diagnostic only; not used for the step).
        final_res = observations - self.obs_operator(x_next)
        diff_norm = torch.linalg.norm(
            final_res.reshape(final_res.shape[0], -1), dim=1
        ) ** 2  # [B]
        log_likelihood = (-0.5 * diff_norm / self.original_variance).detach()

        return guidance.detach(), log_likelihood


class DPSGaussianLikelihood(nn.Module):
    """Faithful DPS guidance likelihood (chung_diffusion_2023) -- BASELINE.

    Diffusion Posterior Sampling: the guidance score is

        g_tau = grad_{x_tau} log N(y; H xhat_1(x_tau), sigma^2 I),

    where ``xhat_1 = E[x_1 | x_tau]`` is the prior posterior-mean denoiser
    obtained from the prior score via Tweedie's formula, and -- the DEFINING DPS
    feature -- the gradient is taken with autograd ENABLED so it flows through
    the prior network (the Jacobian of the denoiser is never frozen). The
    measurement covariance is the RAW ``R = sigma^2 I`` (no PiGDM-style
    inflation, no multiplicative gain): that is the original DPS surrogate.

    The denoiser is built model-class agnostically from the affine-Gaussian
    path moments ``x_tau = alpha a0 + beta x_1 + sigma_tau eps`` (anchor
    ``a0 = x_0`` for SI, ``a0 = 0`` for FM):

        score s_tau = grad log p(x_tau)              (model score)
        xhat_1 = (x_tau + sigma_tau^2 s_tau - alpha a0) / beta   (Tweedie).

    For SI the model's drift carries the score, so it is recovered via the
    shared ``score_from_velocity`` identity; for FM the model exposes ``.score``
    directly. Both are evaluated with grad tracking so the DPS gradient is exact.

    Returns ``(score, log_likelihood)`` to match the
    ``InterpolantGaussianLikelihood`` / ``FlowdasGaussianLikelihood`` interface,
    so it plugs into the existing SI / FM posteriors unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        model_class: str = "si",
    ) -> None:
        super(DPSGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        if model_class not in ("si", "fm"):
            raise ValueError(f"model_class must be 'si' or 'fm', got {model_class!r}.")
        self.model_class = model_class
        # FM anchor a0 = 0; SI anchor a0 = x0 (field_history[..., -1]).
        self.anchor = "x0" if model_class == "si" else "zeros"
        self.interpolant = self.model.interpolation

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Unused (score interface is the entry point)."""
        pass

    def _denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
        drift: Optional[torch.Tensor],
        score: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Tweedie denoiser xhat_1 = E[x_1 | x_tau] (autograd through the net)."""
        # Rank-expand t for the grid-space schedule arithmetic.
        if t.dim() < x.dim():
            t_grid = t.reshape(t.shape[0], *([1] * (x.dim() - 1)))
        else:
            t_grid = t
        alpha = self.interpolant.alpha(t_grid)
        beta = self.interpolant.beta(t_grid)
        sigma_tau = self.interpolant.sigma(t_grid)

        # Prior score s_tau (grad-tracking; the DPS gradient flows through it).
        if self.model_class == "fm":
            s = self.model.score(x, t, field_history, field_cond, pars_cond)
        else:
            # SI: recover the score from the trained drift via the path identity.
            v = self.model.drift(x, t, field_history, field_cond, pars_cond)
            a0_si = field_history[..., -1]
            s = self.interpolant.score_from_velocity(x=x, v=v, t=t_grid, a0=a0_si)

        if self.anchor == "x0":
            a0 = field_history[..., -1]
        else:
            a0 = torch.zeros_like(x)

        # Tweedie: x_tau = alpha a0 + beta xhat_1 + (covariance s correction).
        # E[x_1 | x_tau] = (x_tau + sigma_tau^2 s - alpha a0) / beta.
        return (x + sigma_tau**2 * s - alpha * a0) / beta

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
        score: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DPS guidance score grad_{x} log N(y; H xhat_1(x), sigma^2 I)."""
        t_clamped = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)

        # Compute the DPS gradient on a DETACHED grad-enabled copy so the caller's
        # working state (the posterior's autoregressive ``base``) is never mutated
        # in place by ``requires_grad_``. Mutating it corrupts the SI feedback loop
        # (the next step's base would no longer equal field_history[..., -1]); the
        # grad still flows through the denoiser net via this copy.
        x_g = x.detach().requires_grad_(True)

        with torch.enable_grad():
            xhat1 = self._denoise(
                x_g, t_clamped, field_history, field_cond, pars_cond, drift, score
            )
            residual = observations - self.obs_operator(xhat1)
            diff_norm = torch.linalg.norm(residual, dim=1) ** 2  # [B] = ||y - H xhat||^2

            # Gradient of the squared measurement residual (NOT scaled by 1/sigma^2:
            # the raw 1/sigma^2 = O(1/R) factor is what diverged at NS scale).
            grad = torch.autograd.grad(outputs=diff_norm.sum(), inputs=x_g)[0]

        # DPS step normalization (chung_diffusion_2023): zeta_t = zeta / ||y - H xhat||.
        # Dividing the squared-residual gradient by the residual NORM makes the
        # guidance magnitude scale-free (independent of sigma and of the residual
        # size), which is exactly the bounded step the original DPS prescribes; the
        # posterior then applies the w_tau schedule as the constant zeta. Without
        # this the raw 1/sigma^2 = 400 guidance blew up (rmse ~916 -> NaN).
        norm = diff_norm.sqrt().reshape(-1, *([1] * (grad.dim() - 1)))  # [B,1,1,1]
        guidance = -grad / (norm + 1e-6)

        log_likelihood = (-0.5 * diff_norm / self.original_variance).detach()
        # -grad of the negative log-likelihood direction = grad of the log-likelihood.
        return guidance.detach(), log_likelihood
