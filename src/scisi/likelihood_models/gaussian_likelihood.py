from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


@contextmanager
def _math_sdpa():
    """Force the math scaled-dot-product-attention backend.

    ``torch.autograd.functional.jvp`` uses the double-backward trick, but on
    CUDA the flash / mem-efficient SDPA kernels have no double-backward
    (``derivative for aten::_scaled_dot_product_efficient_attention_backward is
    not implemented``). The math backend supports double-backward (it is what the
    CPU path already used), so the full-Sigma_s JVP through a UNet with attention
    must run under it. Numerically identical to the other backends.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # pragma: no cover - older torch without the public API
        yield
        return
    with sdpa_kernel(SDPBackend.MATH):
        yield

# Clamp pseudo-time away from the endpoints {0, 1} when evaluating
# schedule-derived quantities whose denominators vanish there. The guidance
# correction is only ever applied for tau >= dtau (see the posterior loops),
# so this is a numerical guard rather than a model assumption.
MIN_TIME = 1e-4


class InterpolantGaussianLikelihood(nn.Module):
    """Observation-interpolant Gaussian likelihood (paper Section
    ``obs_interpolation`` / ``multiplicative_correction``).

    Implements the canonical, model-agnostic interpolant likelihood shared by
    all three posterior samplers (SI-SDE, FM-SDE, FM-ODE). For a state
    ``x = x_tau`` it returns the interpolant likelihood score (the posterior
    multiplies it by the weight ``w_tau``). With (Lemma "Interpolated
    observation likelihood", Theorem "Multiplicative correction"), and
    ``sigma_tau^2`` the source variance, ``ybar = alpha H a0 + beta y``,
    ``mu_bar = H x - H mu_s``, ``R = sigma^2 I``:

        Sigma_bar = beta**2 R + H Sigma_s H^T
        Sbar      = (Sigma_s / sigma_tau^2) H^T Sigma_bar^{-1} (ybar - mu_bar)
        G_tau     = I + (1/beta**2) Sigma_s H^T R^{-1} H,

    and three internally-consistent modes select which ``Sigma_s`` is used (the
    SAME ``Sigma_s`` enters the covariance solve, the mean-Jacobian front factor
    ``Sigma_s/sigma_tau^2``, and the gain ``G_tau`` -- no mixing):

    - ``inflated`` (PiGDM-style; exact for the Gaussian case; default): full
      ``Sigma_s`` in ``Sigma_bar`` and the front factor, and **no gain**
      (``G_tau = I``).
    - ``dps_full`` (faithful to Theorem "Multiplicative correction", uninflated
      DPS surrogate): full ``Sigma_s`` everywhere AND the multiplicative gain
      ``G_tau`` with full ``Sigma_s``.
    - ``dps_jacobian_free`` (faithful to Corollary "Jacobian-free posterior
      drift"): isotropic ``Sigma_s = rho_tau I`` everywhere -> front factor
      ``~= H``, ``Sigma_bar = beta^2 R + rho HH^T``, ``G_tau = I + (rho/beta^2)
      H^T R^{-1} H``.

    Covariances are held fixed w.r.t. ``x`` (no autograd through the network);
    the full ``Sigma_s`` Jacobian term is applied via a Jacobian-vector product.

    The observation operator ``H`` is treated as a general linear operator via
    its forward (``H @ x``), adjoint (``transpose`` = ``H^T @ .``) and dense
    ``obs_matrix``; both selection (sparse / strided) and block-average
    super-resolution operators (``H^T H`` non-diagonal) are handled correctly.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        interpolant: Optional[nn.Module] = None,
        likelihood_mode: str = "inflated",
        model_class: str = "si",
        source_type: Optional[str] = None,
        anchor: Optional[str] = None,
        gain: Optional[str] = None,
        correct_likelihood_score: Optional[bool] = None,
    ) -> None:
        """Initialize the interpolant Gaussian likelihood.

        Args:
            model: Trained prior model (SI ``FollmerStochasticInterpolant`` or
                ``FlowMatchingModel``). Supplies the interpolation schedules and
                the score recovery.
            obs_operator: Linear observation operator exposing the forward
                ``H @ x``, the adjoint ``transpose`` (``H^T @ .``) and the dense
                ``obs_matrix``.
            variance: Scalar observation-noise variance ``sigma**2`` (``R =
                sigma**2 I``).
            ensemble_size: Ensemble size (kept for interface compatibility).
            interpolant: Optional interpolation override; defaults to the
                model's interpolation.
            likelihood_mode: One of ``"inflated"`` (PiGDM-style, full Sigma_s in
                Sigma_bar, no gain; exact for the Gaussian case -- default),
                ``"dps_full"`` (uninflated DPS surrogate: full Sigma_s plus the
                multiplicative gain G_tau; faithful to Theorem
                multiplicative_correction), or ``"dps_jacobian_free"`` (isotropic
                Sigma_s = rho I throughout; faithful to Corollary cheap_drift).
                The same Sigma_s is used in the covariance solve, the front
                factor and the gain within each mode.
            model_class: ``"si"`` or ``"fm"``. Selects the source moments
                (Wiener vs Tweedie) and the observation-interpolant anchor
                (``a0 = x_0`` for SI, ``a0 = 0`` for FM). ``source_type`` and
                ``anchor`` override it if given explicitly.
            source_type: Optional explicit ``"si"``/``"fm"`` source moments
                (defaults from ``model_class``).
            anchor: Optional explicit ``"x0"``/``"zeros"`` anchor (defaults from
                ``model_class``).
            gain: **Deprecated** legacy key, mapped onto ``likelihood_mode``
                (``"full"`` -> ``"dps_full"``, ``"jacobian_free"`` ->
                ``"dps_jacobian_free"``). Prefer ``likelihood_mode``.
            correct_likelihood_score: **Deprecated** legacy key. ``False`` forces
                the uncorrected DPS score (``dps_jacobian_free``); ``True`` is a
                no-op. Prefer ``likelihood_mode``.
        """
        super(InterpolantGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size

        # Backward-compatible mapping of the deprecated keys onto likelihood_mode
        # so existing YAML configs keep constructing. The new `likelihood_mode`
        # key takes precedence if it is set to a non-default value.
        if likelihood_mode == "inflated" and gain is not None:
            if gain == "full":
                likelihood_mode = "dps_full"
            elif gain == "jacobian_free":
                likelihood_mode = "dps_jacobian_free"
            else:
                raise ValueError(
                    f"legacy gain must be 'full'|'jacobian_free', got {gain!r}."
                )
        if likelihood_mode == "inflated" and correct_likelihood_score is False:
            likelihood_mode = "dps_jacobian_free"

        if likelihood_mode not in (
            "inflated",
            "inflated_shared",
            "dps_full",
            "dps_jacobian_free",
        ):
            raise ValueError(
                "likelihood_mode must be 'inflated', 'inflated_shared', "
                f"'dps_full' or 'dps_jacobian_free', got {likelihood_mode!r}."
            )
        self.likelihood_mode = likelihood_mode
        # Whether this mode uses the full (Jacobian) Sigma_s or the isotropic one.
        self.use_full_sigma_s = likelihood_mode in (
            "inflated",
            "inflated_shared",
            "dps_full",
        )
        # Whether this mode multiplies by the multiplicative gain G_tau. RETAINED
        # AS AN OPTION (``dps_full``) but OFF by default: the gain did not improve
        # accuracy in the experiments (analytical KL 0.001 inflated vs 0.174 gain;
        # NS sparse rmse ~0.16 inflated vs ~0.72 cheap), so it was dropped from the
        # paper and the runs. The accuracy comes from inflating the covariance
        # (G_tau = I), not from the gain. Every default mode sets G_tau = I.
        self.apply_gain = likelihood_mode == "dps_full"
        # ``inflated_shared``: tractable approximation of ``inflated`` for the
        # full-scale (UNet-prior) runs. The full source-covariance Jacobian is
        # evaluated ONCE at the ensemble-mean state (shared Sigma_s) instead of
        # per member, so Sigma_bar = beta^2 R + H Sigma_s H^T is a single
        # N_y x N_y matrix (factorised once, reused for every member's RHS) and
        # ``H Sigma_s H^T`` is built by a chunked column-batched JVP. This drops
        # the per-member factor (the exact ``inflated`` does B x N_y network
        # Jacobian evaluations per pseudo-step, which is intractable at NS scale).
        # It collapses EXACTLY to ``inflated`` when the ensemble members coincide.
        self.share_sigma_s = likelihood_mode == "inflated_shared"
        # Column chunk for the shared H Sigma_s H^T build (memory vs launches).
        self.shared_sigma_chunk = 64

        if model_class not in ("si", "fm"):
            raise ValueError(
                f"model_class must be 'si' or 'fm', got {model_class!r}."
            )
        self.model_class = model_class

        # Source moments and anchor default from the model class; explicit
        # overrides take precedence.
        self.source_type = source_type if source_type is not None else model_class
        if self.source_type not in ("si", "fm"):
            raise ValueError(
                f"source_type must be 'si' or 'fm', got {self.source_type!r}."
            )

        self.anchor = anchor if anchor is not None else (
            "x0" if model_class == "si" else "zeros"
        )
        if self.anchor not in ("x0", "zeros"):
            raise ValueError(f"anchor must be 'x0' or 'zeros', got {self.anchor!r}.")

        if interpolant is not None:
            self.interpolant = interpolant
        else:
            self.interpolant = self.model.interpolation

        # Cached H^T R^{-1} H (precomputed once; H, R time-independent).
        self._HtRinvH: Optional[torch.Tensor] = None
        self._obs_matrix: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, observations: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass (unused; the score interface is the entry point)."""
        pass

    # ------------------------------------------------------------------
    # Operator caches
    # ------------------------------------------------------------------
    def _get_obs_matrix(self, ref: torch.Tensor) -> torch.Tensor:
        """Return the dense observation matrix on ``ref``'s device/dtype."""
        H = self.obs_operator.obs_matrix.to(device=ref.device, dtype=ref.dtype)
        return H

    def _get_HtRinvH(self, ref: torch.Tensor) -> torch.Tensor:
        """Return ``H^T R^{-1} H`` (cached), with ``R = sigma**2 I``."""
        if (
            self._HtRinvH is None
            or self._HtRinvH.device != ref.device
            or self._HtRinvH.dtype != ref.dtype
        ):
            H = self._get_obs_matrix(ref)
            self._HtRinvH = (H.transpose(0, 1) @ H) / self.original_variance
        return self._HtRinvH

    # ------------------------------------------------------------------
    # Observation interpolant ybar_tau
    # ------------------------------------------------------------------
    def _interpolate_observations(
        self,
        observations: torch.Tensor,
        t: torch.Tensor,
        base_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Observation interpolant ``ybar_tau = alpha * H @ a0 + beta * y``.

        For SI (``anchor='x0'``) ``base_obs = H @ x_0`` so this is
        ``alpha * H @ x_0 + beta * y``; for FM (``anchor='zeros'``) the caller
        passes ``base_obs = 0`` so it reduces to ``beta * y``.
        """
        return self.interpolant.alpha(t) * base_obs + self.interpolant.beta(t) * observations

    # ------------------------------------------------------------------
    # Source conditional moments (Lemma "Source conditional moments")
    # ------------------------------------------------------------------
    def _source_mean_si(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        drift: torch.Tensor,
    ) -> torch.Tensor:
        """SI source mean mu_s = -gamma^2 t A_tau (beta b_theta - c_tau)."""
        gamma = self.interpolant.gamma(t)
        gamma_diff = self.interpolant.gamma_diff(t)
        beta = self.interpolant.beta(t)
        beta_diff = self.interpolant.beta_diff(t)
        alpha = self.interpolant.alpha(t)
        alpha_diff = self.interpolant.alpha_diff(t)

        A = t * gamma * (beta_diff * gamma - beta * gamma_diff)
        A = 1.0 / A

        c = beta_diff * x + (beta * alpha_diff - beta_diff * alpha) * field_history[..., -1]

        return -gamma**2 * t * A * (beta * drift - c)

    def _source_cov_diag_si(self, t: torch.Tensor) -> torch.Tensor:
        """Isotropic part of the SI source covariance: rho_tau = gamma^2 t."""
        return self.interpolant.gamma(t) ** 2 * t

    def _source_mean_fm(self, score: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """FM source mean mu_s = -alpha^2 s (Tweedie)."""
        return -self.interpolant.alpha(t) ** 2 * score

    def _source_cov_diag_fm(self, t: torch.Tensor) -> torch.Tensor:
        """Isotropic part of the FM source covariance: rho_tau = alpha^2."""
        return self.interpolant.alpha(t) ** 2

    # ------------------------------------------------------------------
    # Interpolant-likelihood score Sbar  (closed form, covariances detached)
    # ------------------------------------------------------------------
    def _build_HSHt(
        self, x: torch.Tensor, sigma_s_apply: Callable
    ) -> torch.Tensor:
        """Form ``H Sigma_s H^T`` per ensemble member, shape [B, N_y, N_y].

        Column ``j`` is ``H Sigma_s(x_b) (H^T e_j)``: the adjoint of the j-th
        observation basis vector is the same grid tensor for every member, but
        the full ``Sigma_s`` is state-dependent, so the Jacobian-vector product
        is evaluated at each member's ``x_b`` (one batched JVP per column).
        """
        H = self._get_obs_matrix(x)  # [N_y, N_u]
        N_y = H.shape[0]
        b = x.shape[0]
        eye_y = torch.eye(N_y, device=H.device, dtype=H.dtype)
        # H^T e_j on the grid, one per column.
        cols = self.obs_operator.transpose(eye_y)  # [N_y, C, H, W]

        HSHt = torch.empty(b, N_y, N_y, device=H.device, dtype=H.dtype)
        for j in range(N_y):
            v = cols[j : j + 1].expand(b, *cols.shape[1:])  # [B, C, H, W]
            HSHt[:, :, j] = self.obs_operator(sigma_s_apply(v))  # [B, N_y]
        return HSHt

    def _build_HSHt_shared(
        self, ref: torch.Tensor, sigma_s_apply: Callable
    ) -> torch.Tensor:
        """Form a single ``H Sigma_s H^T`` (shape ``[N_y, N_y]``).

        The ``inflated_shared`` approximation evaluates ``Sigma_s`` once at the
        ensemble-mean state, so a single matrix is built and reused for every
        member's RHS (vs the per-member ``[B, N_y, N_y]`` of :meth:`_build_HSHt`).
        Columns are processed in chunks: ``sigma_s_apply`` broadcasts the captured
        mean state to the chunk's batch, turning ``N_y`` single-column JVPs into
        ``ceil(N_y / chunk)`` batched ones. Column ``j`` is
        ``H Sigma_s (H^T e_j)``.
        """
        H = self._get_obs_matrix(ref)  # [N_y, N_u]
        N_y = H.shape[0]
        cols = self.obs_operator.transpose(
            torch.eye(N_y, device=H.device, dtype=H.dtype)
        )  # [N_y, C, H, W]

        HSHt = torch.empty(N_y, N_y, device=H.device, dtype=H.dtype)
        chunk = max(1, int(self.shared_sigma_chunk))
        for start in range(0, N_y, chunk):
            block = cols[start : start + chunk]  # [k, C, H, W]
            # obs_operator(Sigma_s block) is [k, N_y]; its row i is the
            # (start+i)-th COLUMN of H Sigma_s H^T.
            HSHt[:, start : start + block.shape[0]] = self.obs_operator(
                sigma_s_apply(block)
            ).transpose(0, 1)
        return HSHt

    def _interpolant_score(
        self,
        x: torch.Tensor,
        residual_obs: torch.Tensor,
        rho: torch.Tensor,
        sigma_tau_sq: torch.Tensor,
        beta: torch.Tensor,
        sigma_s_apply: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Closed-form Sbar = grad_xbar_mu^T Sigma_bar^{-1} (ybar - mu_bar).

        Interpolant-likelihood score (Lemma "Interpolated observation
        likelihood"), covariances held fixed w.r.t. ``x``. Within a mode the
        SAME ``Sigma_s`` is used in the covariance solve and the mean-Jacobian
        front factor:

            Sigma_bar    = beta^2 R + H Sigma_s H^T
            grad_xbar_mu = H Sigma_s / sigma_tau^2
            Sbar         = (Sigma_s / sigma_tau^2) H^T Sigma_bar^{-1} (ybar - mu_bar).

        ``sigma_s_apply`` (full ``Sigma_s``, applied via a batched
        Jacobian-vector product) is used by the ``inflated`` / ``dps_full``
        modes; ``None`` selects the isotropic ``Sigma_s = rho I`` of the
        ``dps_jacobian_free`` mode.

        Args:
            x: State tensor [B, C, H, W] (used for shape/device).
            residual_obs: ``ybar - mu_bar`` in observation space [B, N_y].
            rho: Isotropic source scale rho_tau (= sigma_tau^2).
            sigma_tau_sq: Source variance sigma_tau^2 (scalar tensor).
            beta: beta_tau (scalar tensor).
            sigma_s_apply: Operator ``v -> Sigma_s v`` on grid tensors (full
                modes); ``None`` -> isotropic ``Sigma_s = rho I``.

        Returns:
            Sbar as a full-grid tensor [B, C, H, W].
        """
        H = self._get_obs_matrix(x)  # [N_y, N_u]
        N_y = H.shape[0]

        beta_sq_R = float((beta.reshape(-1)[0] ** 2)) * self.original_variance
        rho_s = float(rho.reshape(-1)[0])
        sigma_tau_sq_s = float(sigma_tau_sq.reshape(-1)[0])
        eye = torch.eye(N_y, device=H.device, dtype=H.dtype)

        if sigma_s_apply is None:
            # Isotropic Sigma_s = rho I: Sigma_bar = beta^2 R + rho H H^T,
            # shared across the ensemble.
            Sigma_bar = beta_sq_R * eye + rho_s * (H @ H.transpose(0, 1))
            sol = torch.linalg.solve(
                Sigma_bar, residual_obs.transpose(0, 1)
            ).transpose(0, 1)  # [B, N_y]
            Ht_sol = self.obs_operator.transpose(sol)
            # grad_xbar_mu = H * (rho/sigma_tau^2); for rho = sigma_tau^2 this
            # is exactly H.
            return (rho_s / sigma_tau_sq_s) * Ht_sol

        if self.share_sigma_s:
            # Shared Sigma_s (inflated_shared): one Sigma_bar for the whole
            # ensemble, factorised once and solved for every member's RHS.
            HSHt = self._build_HSHt_shared(x, sigma_s_apply)  # [N_y, N_y]
            Sigma_bar = beta_sq_R * eye + HSHt  # [N_y, N_y]
            sol = torch.linalg.solve(
                Sigma_bar, residual_obs.transpose(0, 1)
            ).transpose(0, 1)  # [B, N_y]
            Ht_sol = self.obs_operator.transpose(sol)  # [B, C, H, W]
            # Front factor uses the SAME shared Sigma_s (broadcast over members).
            return sigma_s_apply(Ht_sol) / sigma_tau_sq_s

        # Full Sigma_s: Sigma_bar = beta^2 R + H Sigma_s H^T, per member.
        HSHt = self._build_HSHt(x, sigma_s_apply)  # [B, N_y, N_y]
        Sigma_bar = beta_sq_R * eye.unsqueeze(0) + HSHt  # [B, N_y, N_y]
        sol = torch.linalg.solve(
            Sigma_bar, residual_obs.unsqueeze(-1)
        ).squeeze(-1)  # [B, N_y]
        Ht_sol = self.obs_operator.transpose(sol)  # [B, C, H, W]
        # grad_xbar_mu front factor: (Sigma_s / sigma_tau^2) applied on grid.
        return sigma_s_apply(Ht_sol) / sigma_tau_sq_s

    def _apply_gain(
        self,
        sbar: torch.Tensor,
        rho: torch.Tensor,
        beta: torch.Tensor,
        sigma_s_apply: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Apply the multiplicative gain G_tau to Sbar (``dps_full`` mode).

        ``G_tau @ Sbar = Sbar + (1/beta^2) Sigma_s (H^T R^{-1} H) Sbar``, with
        the SAME ``Sigma_s`` as the covariance solve: full via ``sigma_s_apply``
        for ``dps_full``; the isotropic branch (``rho I``) is kept for symmetry
        but is unused since only ``dps_full`` applies the gain.
        """
        HtRinvH = self._get_HtRinvH(sbar)  # [N_u, N_u]
        b = sbar.shape[0]
        flat = sbar.reshape(b, -1)
        corr = (flat @ HtRinvH.transpose(0, 1)).reshape_as(sbar)  # (H^T R^{-1} H) Sbar

        inv_beta_sq = 1.0 / (beta**2)
        if sigma_s_apply is None:
            return sbar + inv_beta_sq * rho * corr
        return sbar + inv_beta_sq * sigma_s_apply(corr)

    # ------------------------------------------------------------------
    # Public score interface
    # ------------------------------------------------------------------
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
        diffusion_term: Optional[Callable] = None,
        score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the interpolant likelihood score for the configured mode.

        Returns the (possibly gain-multiplied) interpolant score ``Sbar`` /
        ``G_tau @ Sbar`` per ``likelihood_mode``; the posterior multiplies it by
        the weight ``w_tau``.

        Args:
            observations: Observation vector ``y`` [B, N_y].
            x: Current state ``x_tau`` [B, C, H, W].
            t: Pseudo-time [B, 1].
            field_history: Field history [B, C, H, W, L]; ``[..., -1]`` is
                ``x_0`` (the previous physical state / SI anchor).
            drift: SI drift ``b_theta`` at ``(x, t)`` (used for the SI source
                moments; ignored for FM).
            score: FM score ``s_tau`` at ``(x, t)`` (used for the FM source
                moments; ignored for SI).

        Returns:
            ``(score, log_likelihood)`` where ``score`` is ``Sbar`` (inflated /
            dps_jacobian_free) or ``G_tau @ Sbar`` (dps_full). The posterior
            multiplies it by the weight ``w_tau``.
        """

        t_clamped = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)

        # The source-moment and observation-interpolant terms multiply
        # schedule-derived coefficients (alpha, beta, gamma, ...) against the
        # 4D state ``x`` / grid tensors. The schedule functions preserve the
        # shape of their input, so ``t`` must carry the state rank for those
        # products to broadcast over the spatial dims; the drift net receives
        # the original ``[B, 1]`` time, but here we expand to ``x``'s rank.
        # Two shapes of the pseudo-time are needed: ``t_clamped`` stays ``[B, 1]``
        # for observation-space terms (the obs interpolant, the scalarised
        # covariance solve) and for the drift/score network; ``t_grid`` is rank-
        # expanded to ``x`` so the *grid-space* source moments ``mu_s`` (which
        # multiply schedule coefficients against the 4D state) broadcast over the
        # spatial dims. Mixing the two previously broadcast a ``[B,1,1,1]`` beta
        # against the ``[B, N_y]`` observations and corrupted the obs-space shape.
        t_net = t_clamped
        if t_clamped.dim() < x.dim():
            t_grid = t_clamped.reshape(
                t_clamped.shape[0], *([1] * (x.dim() - 1))
            )
        else:
            t_grid = t_clamped

        beta = self.interpolant.beta(t_clamped)

        # --- observation interpolant ybar_tau -------------------------------
        if self.anchor == "x0":
            base_obs = self.obs_operator(field_history[..., -1])
        else:  # FM: a0 = 0
            base_obs = torch.zeros_like(observations)
        interpolant_obs = self._interpolate_observations(
            observations, t_clamped, base_obs
        )

        # --- source conditional moments mu_s, rho_tau -----------------------
        # These are grid-space, so use the rank-expanded ``t_grid``.
        if self.source_type == "si":
            mu_s = self._source_mean_si(
                x=x, t=t_grid, field_history=field_history, drift=drift
            )
            rho = self._source_cov_diag_si(t_grid)
        else:  # fm
            mu_s = self._source_mean_fm(score=score, t=t_grid)
            rho = self._source_cov_diag_fm(t_grid)

        # --- likelihood mean mu_bar = H x - H mu_s --------------------------
        mu_bar = self.obs_operator(x) - self.obs_operator(mu_s)

        residual_obs = interpolant_obs - mu_bar  # ybar - mu_bar, [B, N_y]

        # --- full Sigma_s operator (inflated / dps_full modes) --------------
        # The same Sigma_s is used in the covariance solve, the front factor and
        # the gain; the isotropic mode (dps_jacobian_free) leaves it None.
        sigma_s_apply = None
        if self.use_full_sigma_s:
            if self.share_sigma_s:
                # inflated_shared: evaluate the Jacobian ONCE at the ensemble
                # mean (pseudo-time tau is identical across members, so the
                # schedule scalars are unchanged; only the state is averaged).
                def _mean0(z):
                    return None if z is None else z.mean(dim=0, keepdim=True)

                sigma_s_apply = self._build_full_sigma_s_apply(
                    x=_mean0(x),
                    t=t_grid[:1],
                    t_net=t_net[:1],
                    field_history=_mean0(field_history),
                    field_cond=_mean0(field_cond),
                    pars_cond=_mean0(pars_cond),
                    drift=None,
                    score=None,
                    rho=rho[:1] if rho.dim() > 0 else rho,
                )
            else:
                sigma_s_apply = self._build_full_sigma_s_apply(
                    x=x,
                    t=t_grid,
                    t_net=t_net,
                    field_history=field_history,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                    drift=drift,
                    score=score,
                    rho=rho,
                )

        # --- interpolant score Sbar (closed form) ---------------------------
        # The isotropic source scale rho_tau coincides with the source variance
        # sigma_tau^2 (= gamma^2 t for SI, alpha^2 for FM); the full Sigma_s
        # curvature enters via ``sigma_s_apply``.
        sigma_tau_sq = rho
        sbar = self._interpolant_score(
            x=x,
            residual_obs=residual_obs,
            rho=rho,
            sigma_tau_sq=sigma_tau_sq,
            beta=beta,
            sigma_s_apply=sigma_s_apply,
        )

        # --- multiplicative gain G_tau @ Sbar (dps_full only) ---------------
        # The inflated mode uses G_tau = I; only dps_full multiplies by the
        # gain, with the SAME (full) Sigma_s as the covariance solve.
        if self.apply_gain:
            corrected = self._apply_gain(
                sbar=sbar, rho=rho, beta=beta, sigma_s_apply=sigma_s_apply
            )
        else:
            corrected = sbar

        # Diagnostic log-likelihood (used for optional SMC resampling). The
        # tau >= dtau guard in the posterior loops keeps beta > 0, so no
        # additive denominator epsilon is needed (t is clamped to >= MIN_TIME).
        log_likelihood = -0.5 * (
            torch.linalg.norm(residual_obs, dim=1) ** 2
        ) / (beta.reshape(-1) ** 2 * self.original_variance)

        return corrected, log_likelihood

    # ------------------------------------------------------------------
    # Full source covariance (Jacobian) operator
    # ------------------------------------------------------------------
    def _build_full_sigma_s_apply(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
        drift: Optional[torch.Tensor],
        score: Optional[torch.Tensor],
        rho: torch.Tensor,
        t_net: Optional[torch.Tensor] = None,
    ) -> Callable:
        """Build an operator ``v -> Sigma_s v`` for the *full* source covariance.

        - SI: ``Sigma_s = gamma^2 t I + gamma^4 t^2 A_tau (beta J_b - bdot I)``,
          with ``J_b = grad_x b_theta`` applied as a Jacobian-vector product.
        - FM: ``Sigma_s = alpha^2 I + alpha^4 grad_x s``, with ``grad_x s``
          applied as a JVP.

        The Jacobian term is computed via a single ``torch.autograd`` JVP and
        detached from the outer graph, so no autograd flows through the network
        into the posterior drift.
        """
        # Schedule math uses the rank-expanded ``t``; the network is fed the
        # ``[B, 1]`` ``t_net`` (defaults to ``t`` when not supplied).
        if t_net is None:
            t_net = t
        beta = self.interpolant.beta(t)
        beta_diff = self.interpolant.beta_diff(t)

        def _bcast(tensor: Optional[torch.Tensor], k: int) -> Optional[torch.Tensor]:
            """Broadcast a captured ``[B0, ...]`` tensor to leading batch ``k``.

            The JVP is evaluated at the captured state ``x`` against a tangent
            ``v`` whose leading dim may differ: it equals ``B`` on the per-member
            path (no-op) but is the column-chunk / member count on the
            shared-state path, where ``x`` and the conditioning are captured at
            batch 1 (the ensemble mean). ``expand`` keeps it a view (no copy).
            """
            if tensor is None or tensor.shape[0] == k:
                return tensor
            if tensor.shape[0] == 1:
                return tensor.expand(k, *tensor.shape[1:])
            raise ValueError(
                f"cannot broadcast batch {tensor.shape[0]} to {k}."
            )

        if self.source_type == "si":
            gamma = self.interpolant.gamma(t)
            gamma_diff = self.interpolant.gamma_diff(t)
            A = t * gamma * (beta_diff * gamma - beta * gamma_diff)
            A = 1.0 / A
            jac_scale = gamma**4 * t**2 * A

            def jvp_fn(v: torch.Tensor) -> torch.Tensor:
                # J_b @ v, with b_theta = model.drift(x, ...). State and
                # conditioning are broadcast to v's batch (shared-state path).
                k = v.shape[0]
                xk = _bcast(x, k)
                tk, fhk = _bcast(t_net, k), _bcast(field_history, k)
                fck, pck = _bcast(field_cond, k), _bcast(pars_cond, k)

                def fn(inp: torch.Tensor) -> torch.Tensor:
                    return self.model.drift(inp, tk, fhk, fck, pck)

                with _math_sdpa():
                    _, jv = torch.autograd.functional.jvp(
                        fn, (xk,), (v,), create_graph=False
                    )
                return jv.detach()

            def sigma_s_apply(v: torch.Tensor) -> torch.Tensor:
                jac_term = jac_scale * (beta * jvp_fn(v) - beta_diff * v)
                return rho * v + jac_term

            return sigma_s_apply

        # FM
        alpha = self.interpolant.alpha(t)
        jac_scale = alpha**4

        def jvp_fn_fm(v: torch.Tensor) -> torch.Tensor:
            # grad_x s @ v with s = model.score(x, ...).
            k = v.shape[0]
            xk = _bcast(x, k)
            tk, fhk = _bcast(t_net, k), _bcast(field_history, k)
            fck, pck = _bcast(field_cond, k), _bcast(pars_cond, k)

            def fn(inp: torch.Tensor) -> torch.Tensor:
                return self.model.score(inp, tk, fhk, fck, pck)

            with _math_sdpa():
                _, jv = torch.autograd.functional.jvp(
                    fn, (xk,), (v,), create_graph=False
                )
            return jv.detach()

        def sigma_s_apply_fm(v: torch.Tensor) -> torch.Tensor:
            return rho * v + jac_scale * jvp_fn_fm(v)

        return sigma_s_apply_fm


class FlowdasGaussianLikelihood(nn.Module):
    """FlowDAS Monte-Carlo likelihood (baseline, ``chen_flowdas_2025``).

    NOT the paper's observation-interpolant method. Kept as a baseline: it
    draws one-step predictions of ``x_1`` and softmax-weights them by the raw
    observation likelihood ``N(y; H x_1, R)``.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        integration_order: int = 1,
    ) -> None:
        """
        Initialize Multivariate Gaussian likelihood.

        Either covariance_matrix or precision_matrix must be provided.

        Args:
            obs_operator: Observation operator.
            model: Model.
            variance: Variance.
        """
        super(FlowdasGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size
        self.dist = torch.distributions.MultivariateNormal
        self.integration_order = integration_order

        self.integral_variance = lambda t: 2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3

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

    def _denoiser_mean(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        drift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Deterministic one-step denoiser mean ``mu_x1 = E[x_1 | x_tau]``.

        A noise-free Milstein+RK (Heun) predictor of ``x_1`` from ``x_tau``.
        This is the FlowDAS predictor's *mean*; the Monte-Carlo spread is added
        separately in :meth:`score` so the guidance can be normalized.
        """
        drift_milstein = (
            drift
            if drift is not None
            else self.model.drift(x, t, field_history, field_cond, pars_cond)
        )
        pred = x + drift_milstein * (1.0 - t)
        drift_rk = self.model.drift(
            pred, torch.ones_like(t), field_history, field_cond, pars_cond
        )
        return x + 0.5 * (drift_milstein + drift_rk) * (1.0 - t)

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
        """FlowDAS guidance (chen_flowdas_2025, Algorithm 2 line 10).

        Faithful Monte-Carlo guidance: draw ``J = n_mc`` one-step predictions of
        ``x_1`` from ``x_tau`` as ``X1_hat^(j) = mu_x1 + sqrt(v1) eps_j``, with
        ``eps_j ~ N(0, I)`` DETACHED constants so the only ``x_tau``-dependence is
        through the denoiser mean ``mu_x1(x_tau)``. Compute DETACHED importance
        weights ``w_j = softmax_j(-||y - H X1_hat^(j)||^2 / (2 sigma^2))`` and then
        take the gradient of the weighted observation error THROUGH the predictor

            g = -grad_{x_tau} sum_j w_j ||y - H X1_hat^(j)||^2
              = -grad_{x_tau} ||y - H xhat1||^2   (weights detached),
              xhat1 = sum_j w_j X1_hat^(j).

        The crucial point (vs the old bounded surrogate ``(x1_hat - mu_x1)/(v1+R)``
        which vanishes as ``v1 -> 0``): the residual ``(y - H xhat1)`` is pulled
        THROUGH the denoiser Jacobian by autograd, so it points toward the
        observation and does NOT vanish at the data end.

        The raw ``1/sigma^2``-scaled gradient through the network was what diverged
        at NS scale (the step SIZE, not the algorithm), so we step-normalize like
        the other guided baselines (DPS-style): ``guidance = -grad / (||grad|| +
        eps)``. The posterior applies ``w_tau`` as the schedule. Returns
        ``(guidance, log_likelihood)``; computed on a detached grad-enabled leaf so
        the caller's autoregressive working state is never mutated in place.
        """
        n_mc = self.ensemble_size
        R = self.original_variance

        # Conditional spread (FlowDAS predictor std); state-independent scalar.
        s = self.integral_variance(t)

        # Detached grad-enabled leaf: the DPS gradient flows through mu_x1(x_g)
        # but the caller's working tensor is untouched.
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            mu_x1 = self._denoiser_mean(
                x_g, t, field_history, field_cond, pars_cond, drift
            )
            b, c, h, w = mu_x1.shape

            # J one-step predictions with DETACHED source noise; the x_g-dependence
            # is ONLY through mu_x1 (eps_j are constants).
            eps = torch.randn(
                (n_mc, *mu_x1.shape), device=mu_x1.device, dtype=mu_x1.dtype
            )
            preds = mu_x1.unsqueeze(0) + s * eps  # [J, B, C, H, W]

            Hpreds = self.obs_operator(preds.reshape(n_mc * b, c, h, w)).reshape(
                n_mc, b, -1
            )
            residual = observations.unsqueeze(0) - Hpreds  # [J, B, N_y]
            sq_err = (residual**2).sum(dim=-1)  # [J, B]

            # DETACHED importance weights (constants in the autograd graph).
            log_w = -0.5 * sq_err.detach() / R  # [J, B]
            weights = torch.softmax(log_w, dim=0)  # [J, B]

            # Weighted observation error, differentiated through the predictor.
            weighted_err = (weights * sq_err).sum(dim=0).sum()  # scalar
            grad = torch.autograd.grad(outputs=weighted_err, inputs=x_g)[0]

        # DPS-style step normalization (the step SIZE was the divergence, not the
        # algorithm): unit-L2 per member; descent on the error => -grad.
        gnorm = grad.reshape(b, -1).norm(dim=1).reshape(b, *([1] * (grad.dim() - 1)))
        guidance = -grad / (gnorm + 1e-6)

        # Per-member log marginal likelihood (for optional SMC reweighting).
        log_likelihood = (
            torch.logsumexp(log_w, dim=0)
            - torch.log(torch.tensor(float(n_mc), device=log_w.device))
        ).detach()

        return guidance.detach(), log_likelihood
