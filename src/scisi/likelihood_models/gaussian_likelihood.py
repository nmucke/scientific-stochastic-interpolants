from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


@contextmanager
def _math_sdpa():
    """Force the math scaled-dot-product-attention backend.

    The full-Sigma_s Jacobian-vector products differentiate through the UNet's
    attention, but on CUDA the flash / mem-efficient SDPA kernels implement
    neither forward-mode AD (``torch.func.jvp``) nor double-backward. The math
    backend supports both (it is what the CPU path already used), so every JVP
    through a UNet with attention must run under it. Numerically identical to
    the other backends.
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
    ``obs_interpolation``).

    Implements the canonical, model-agnostic interpolant likelihood shared by
    all three posterior samplers (SI-SDE, FM-SDE, FM-ODE). For a state
    ``x = x_tau`` it returns the interpolant likelihood score ``Sbar`` (the
    posterior multiplies it by the weight ``w_tau``). With (Lemma "Interpolated
    observation likelihood"), ``ybar = alpha H a0 + beta y``, ``mu_bar = H x -
    H mu_s``, ``R = sigma^2 I``, and covariances held fixed w.r.t. ``x``:

        Sigma_bar = beta**2 R + H Sigma_s H^T
        Sbar      = (Sigma_s / rho_tau) H^T Sigma_bar^{-1} (ybar - mu_bar)

    where ``rho_tau`` is the isotropic source variance (``gamma^2 tau`` for
    SI, ``alpha^2`` for FM). The ``likelihood_mode`` selects which ``Sigma_s``
    is used -- the SAME ``Sigma_s`` enters the covariance solve and the
    mean-Jacobian front factor, no mixing:

    - ``inflated`` (PiGDM-style; exact for the Gaussian case; default): full
      ``Sigma_s = c_iso I + c_jac J`` with the network Jacobian ``J`` applied
      per member via a forward-mode Jacobian-vector product.
    - ``inflated_shared``: tractable approximation of ``inflated`` for the
      full-scale (UNet-prior) runs. ``J`` is evaluated ONCE at the
      ensemble-mean state, so ``Sigma_bar`` is a single ``N_y x N_y`` matrix
      (factorised once, reused for every member's RHS) instead of the
      per-member ``B x N_y`` network Jacobian evaluations of ``inflated``,
      which are intractable at NS scale. Collapses EXACTLY to ``inflated``
      when the ensemble members coincide.
    - ``dps_jacobian_free`` (faithful to Corollary "Jacobian-free posterior
      drift"): isotropic ``Sigma_s = rho_tau I`` -> front factor ``= H``,
      ``Sigma_bar = beta^2 R + rho H H^T``.

    The multiplicative gain ``G_tau`` of Theorem "Multiplicative correction"
    (former ``dps_full`` mode) was removed: it did not improve accuracy
    (analytical KL 0.001 inflated vs 0.174 with gain; NS sparse rmse ~0.16 vs
    ~0.72) and was dropped from the paper and the runs. The accuracy comes
    from inflating the covariance, not from the gain.

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
        jacobian_refresh_every: int = 1,
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
            likelihood_mode: ``"inflated"`` (default), ``"inflated_shared"`` or
                ``"dps_jacobian_free"`` -- see the class docstring.
            model_class: ``"si"`` or ``"fm"``. Selects the source moments
                (Wiener vs Tweedie) and the observation-interpolant anchor
                (``a0 = x_0`` for SI, ``a0 = 0`` for FM).
            jacobian_refresh_every: ``inflated_shared`` only -- recompute the
                shared network Jacobian every k-th pseudo-time step (default 1 =
                every step, no approximation). Between refreshes the cached
                ``H J H^T`` and frozen-state JVP are reused; the schedule
                scalars (rho, beta, ...) stay EXACT at the current tau -- only
                the Jacobian factor J is lagged. The cache resets whenever
                pseudo-time restarts (a new SDE/ODE pass, i.e. the next
                assimilation step).
        """
        super(InterpolantGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        self.ensemble_size = ensemble_size

        if likelihood_mode not in (
            "inflated",
            "inflated_shared",
            "dps_jacobian_free",
        ):
            raise ValueError(
                "likelihood_mode must be 'inflated', 'inflated_shared' or "
                f"'dps_jacobian_free', got {likelihood_mode!r}."
            )
        self.likelihood_mode = likelihood_mode
        # Whether this mode uses the full (Jacobian) Sigma_s or the isotropic one.
        self.use_full_sigma_s = likelihood_mode in ("inflated", "inflated_shared")
        self.share_sigma_s = likelihood_mode == "inflated_shared"
        # Column chunk for the shared H J H^T build (memory vs launches).
        self.shared_sigma_chunk = 64
        self.jacobian_refresh_every = int(jacobian_refresh_every)
        if self.jacobian_refresh_every < 1:
            raise ValueError(
                f"jacobian_refresh_every must be >= 1, got {jacobian_refresh_every!r}."
            )
        # (t_last, steps_since, jvp_fn, HJHt) of the current shared Jacobian.
        # ``t_last`` tracks the last pseudo-time seen so repeated calls at the
        # same tau (extra ensemble batches) reuse the cache, and a tau DECREASE
        # (pseudo-time restarted -> next assimilation window) forces a refresh.
        self._shared_jac_cache: Dict[str, Any] = {
            "t_last": None, "steps_since": 0, "jvp_fn": None, "HJHt": None,
        }

        if model_class not in ("si", "fm"):
            raise ValueError(
                f"model_class must be 'si' or 'fm', got {model_class!r}."
            )
        self.model_class = model_class
        self.anchor = "x0" if model_class == "si" else "zeros"

        if interpolant is not None:
            self.interpolant = interpolant
        else:
            self.interpolant = self.model.interpolation

        # Cached H H^T and adjoint basis columns (H, R time-independent).
        self._HHt: Optional[torch.Tensor] = None
        self._cols: Optional[torch.Tensor] = None
        self._cols_key: Optional[tuple] = None

    # ------------------------------------------------------------------
    # Operator caches
    # ------------------------------------------------------------------
    def _get_obs_matrix(self, ref: torch.Tensor) -> torch.Tensor:
        """Return the dense observation matrix on ``ref``'s device/dtype."""
        H = self.obs_operator.obs_matrix.to(device=ref.device, dtype=ref.dtype)
        return H

    def _get_HHt(self, ref: torch.Tensor) -> torch.Tensor:
        """Return ``H H^T`` (cached), shape ``[N_y, N_y]``.

        Time-independent, so it is formed ONCE instead of re-running the
        ``[N_y, N_u] @ [N_u, N_y]`` matmul on every pseudo-time step.
        """
        if (
            self._HHt is None
            or self._HHt.device != ref.device
            or self._HHt.dtype != ref.dtype
        ):
            H = self._get_obs_matrix(ref)
            self._HHt = H @ H.transpose(0, 1)
        return self._HHt

    def _get_adjoint_basis(self, ref: torch.Tensor) -> torch.Tensor:
        """Return the grid tensors ``H^T e_j`` for all obs rows, ``[N_y, C, H, W]``.

        Time-independent (one grid tensor per observation basis vector), so it
        is built ONCE per device/dtype instead of allocating an ``N_y x N_u``
        block on every ``H Sigma_s H^T`` build. Keyed on the identity of
        ``obs_matrix`` so ``load_mask`` invalidates it.
        """
        H = self._get_obs_matrix(ref)
        key = (id(self.obs_operator.obs_matrix), H.device, H.dtype)
        if self._cols is None or self._cols_key != key:
            eye_y = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            self._cols = self.obs_operator.transpose(eye_y)
            self._cols_key = key
        return self._cols

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
    def _si_A(self, t: torch.Tensor) -> torch.Tensor:
        """SI schedule factor ``A_tau = 1 / (t gamma (beta' gamma - beta gamma'))``.

        Shared by the SI source mean and the SI ``Sigma_s`` coefficients.
        """
        gamma = self.interpolant.gamma(t)
        gamma_diff = self.interpolant.gamma_diff(t)
        beta = self.interpolant.beta(t)
        beta_diff = self.interpolant.beta_diff(t)
        A = t * gamma * (beta_diff * gamma - beta * gamma_diff)
        return 1.0 / A

    def _source_mean_si(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        drift: torch.Tensor,
    ) -> torch.Tensor:
        """SI source mean mu_s = -gamma^2 t A_tau (beta b_theta - c_tau)."""
        gamma = self.interpolant.gamma(t)
        beta = self.interpolant.beta(t)
        beta_diff = self.interpolant.beta_diff(t)
        alpha = self.interpolant.alpha(t)
        alpha_diff = self.interpolant.alpha_diff(t)

        A = self._si_A(t)
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
        # H^T e_j on the grid, one per column (cached; time-independent).
        cols = self._get_adjoint_basis(x)  # [N_y, C, H, W]

        HSHt = torch.empty(b, N_y, N_y, device=H.device, dtype=H.dtype)
        for j in range(N_y):
            v = cols[j : j + 1].expand(b, *cols.shape[1:])  # [B, C, H, W]
            HSHt[:, :, j] = self.obs_operator(sigma_s_apply(v))  # [B, N_y]
        return HSHt

    def _build_HJHt_shared(
        self, ref: torch.Tensor, jvp_fn: Callable
    ) -> torch.Tensor:
        """Form the Jacobian-only ``H J H^T`` (shape ``[N_y, N_y]``).

        The ``inflated_shared`` approximation evaluates the network Jacobian
        ``J`` once at the ensemble-mean state, so a single matrix is built and
        reused for every member's RHS (vs the per-member ``[B, N_y, N_y]`` of
        :meth:`_build_HSHt`). Columns are processed in chunks: ``jvp_fn``
        broadcasts the captured mean state to the chunk's batch, turning
        ``N_y`` single-column JVPs into ``ceil(N_y / chunk)`` batched ones.
        Column ``j`` is ``H J (H^T e_j)``.

        The schedule scalars are NOT baked in: the caller assembles
        ``H Sigma_s H^T = c_iso H H^T + c_jac H J H^T`` per pseudo-step, so
        this matrix can be cached across steps (``jacobian_refresh_every > 1``)
        with the scalars staying exact -- only ``J`` is lagged.
        """
        H = self._get_obs_matrix(ref)  # [N_y, N_u]
        N_y = H.shape[0]
        cols = self._get_adjoint_basis(ref)  # [N_y, C, H, W]

        HJHt = torch.empty(N_y, N_y, device=H.device, dtype=H.dtype)
        chunk = max(1, int(self.shared_sigma_chunk))
        for start in range(0, N_y, chunk):
            block = cols[start : start + chunk]  # [k, C, H, W]
            # obs_operator(J block) is [k, N_y]; its row i is the (start+i)-th
            # COLUMN of H J H^T.
            HJHt[:, start : start + block.shape[0]] = self.obs_operator(
                jvp_fn(block)
            ).transpose(0, 1)
        return HJHt

    def _interpolant_score(
        self,
        x: torch.Tensor,
        residual_obs: torch.Tensor,
        rho: torch.Tensor,
        beta: torch.Tensor,
        sigma_s_apply: Optional[Callable] = None,
        HSHt_shared: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Closed-form Sbar = grad_xbar_mu^T Sigma_bar^{-1} (ybar - mu_bar).

        Interpolant-likelihood score (Lemma "Interpolated observation
        likelihood"), covariances held fixed w.r.t. ``x``. Within a mode the
        SAME ``Sigma_s`` is used in the covariance solve and the mean-Jacobian
        front factor:

            Sigma_bar    = beta^2 R + H Sigma_s H^T
            Sbar         = (Sigma_s / rho_tau) H^T Sigma_bar^{-1} (ybar - mu_bar)

        (the isotropic source scale rho_tau coincides with the source variance
        sigma_tau^2, so ``Sigma_s = rho I`` makes the front factor exactly
        ``H``).

        Two layouts: a single shared ``[N_y, N_y]`` covariance, factorised once
        and solved for every member's RHS (``dps_jacobian_free``: ``rho H H^T``;
        ``inflated_shared``: the caller-assembled ``HSHt_shared``), or the
        per-member ``[B, N_y, N_y]`` batched solve of the exact ``inflated``
        mode. In all cases the ``beta^2 R`` term is added on the diagonal in
        place of forming ``beta^2 R * I`` explicitly (addition commutes,
        identical result).

        Args:
            x: State tensor [B, C, H, W] (used for shape/device).
            residual_obs: ``ybar - mu_bar`` in observation space [B, N_y].
            rho: Isotropic source scale rho_tau (= sigma_tau^2).
            beta: beta_tau (scalar tensor).
            sigma_s_apply: Operator ``v -> Sigma_s v`` on grid tensors (full
                modes); ``None`` -> isotropic ``Sigma_s = rho I``.
            HSHt_shared: ``inflated_shared`` only -- the assembled
                ``H Sigma_s H^T`` (``[N_y, N_y]``, fresh tensor) built by the
                caller from the cached ``H J H^T`` and the current schedule
                scalars.

        Returns:
            Sbar as a full-grid tensor [B, C, H, W].
        """
        beta_sq_R = float((beta.reshape(-1)[0] ** 2)) * self.original_variance
        rho_s = float(rho.reshape(-1)[0])

        if sigma_s_apply is not None and not self.share_sigma_s:
            # Exact inflated mode: Sigma_bar per member, batched solve.
            HSHt = self._build_HSHt(x, sigma_s_apply)  # [B, N_y, N_y]
            Sigma_bar = HSHt  # fresh tensor; add beta^2 R on the diagonals in place
            Sigma_bar.diagonal(dim1=-2, dim2=-1).add_(beta_sq_R)
            sol = torch.linalg.solve(
                Sigma_bar, residual_obs.unsqueeze(-1)
            ).squeeze(-1)  # [B, N_y]
            Ht_sol = self.obs_operator.transpose(sol)  # [B, C, H, W]
            # grad_xbar_mu front factor: (Sigma_s / rho_tau) applied on grid.
            return sigma_s_apply(Ht_sol) / rho_s

        # Shared covariance (dps_jacobian_free / inflated_shared): one
        # [N_y, N_y] Sigma_bar for the whole ensemble, factorised once and
        # solved for every member's RHS. Both inputs are fresh tensors, so
        # the diagonal add mutates neither cache.
        if HSHt_shared is not None:
            Sigma_bar = HSHt_shared
        else:
            Sigma_bar = rho_s * self._get_HHt(x)
        Sigma_bar.diagonal().add_(beta_sq_R)
        sol = torch.linalg.solve(
            Sigma_bar, residual_obs.transpose(0, 1)
        ).transpose(0, 1)  # [B, N_y]
        Ht_sol = self.obs_operator.transpose(sol)  # [B, C, H, W]
        if sigma_s_apply is None:
            # Isotropic front factor rho/rho_tau = 1: Sbar is exactly H^T sol.
            return Ht_sol
        # Shared Sigma_s front factor (broadcast over members).
        return sigma_s_apply(Ht_sol) / rho_s

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
            ``(score, log_likelihood)`` where ``score`` is the interpolant
            score ``Sbar``. The posterior multiplies it by the weight
            ``w_tau``.
        """

        t_clamped = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)

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
        if self.model_class == "si":
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

        # --- full Sigma_s operator (inflated modes) --------------------------
        # Both full modes use the unified split Sigma_s = c_iso I + c_jac J
        # with the network Jacobian J applied by a forward-mode JVP; the
        # isotropic mode (dps_jacobian_free) leaves it None.
        sigma_s_apply = None
        hsht_shared = None
        if self.use_full_sigma_s:
            if self.share_sigma_s:
                # inflated_shared: evaluate the Jacobian ONCE at the ensemble
                # mean (pseudo-time tau is identical across members, so the
                # schedule scalars are unchanged; only the state is averaged),
                # refreshed every ``jacobian_refresh_every``-th pseudo-step.
                c_iso, c_jac = self._sigma_s_coeffs(
                    t_grid[:1], rho[:1] if rho.dim() > 0 else rho
                )
                jvp_fn = self._refresh_shared_jacobian(
                    x=x,
                    t_clamped=t_clamped,
                    t_net=t_net,
                    field_history=field_history,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                )
                cache = self._shared_jac_cache
                if cache["HJHt"] is None:
                    cache["HJHt"] = self._build_HJHt_shared(x, jvp_fn)
                # Assemble H Sigma_s H^T with the CURRENT schedule scalars; the
                # cached H J H^T is the only lagged factor (and is exact for
                # jacobian_refresh_every = 1). Scalar multiplication creates a
                # fresh tensor, so neither cache is mutated downstream.
                hsht_shared = (
                    float(c_iso.reshape(-1)[0]) * self._get_HHt(x)
                    + float(c_jac.reshape(-1)[0]) * cache["HJHt"]
                )
            else:
                c_iso, c_jac = self._sigma_s_coeffs(t_grid, rho)
                jvp_fn = self._build_jvp_fn(
                    x=x,
                    t_net=t_net,
                    field_history=field_history,
                    field_cond=field_cond,
                    pars_cond=pars_cond,
                )

            def sigma_s_apply(
                v: torch.Tensor, _ci=c_iso, _cj=c_jac, _jvp=jvp_fn
            ) -> torch.Tensor:
                return _ci * v + _cj * _jvp(v)

        # --- interpolant score Sbar (closed form) ---------------------------
        sbar = self._interpolant_score(
            x=x,
            residual_obs=residual_obs,
            rho=rho,
            beta=beta,
            sigma_s_apply=sigma_s_apply,
            HSHt_shared=hsht_shared,
        )

        # Diagnostic log-likelihood (used for optional SMC resampling). The
        # tau >= dtau guard in the posterior loops keeps beta > 0, so no
        # additive denominator epsilon is needed (t is clamped to >= MIN_TIME).
        log_likelihood = -0.5 * (
            torch.linalg.norm(residual_obs, dim=1) ** 2
        ) / (beta.reshape(-1) ** 2 * self.original_variance)

        return sbar, log_likelihood

    # ------------------------------------------------------------------
    # Full source covariance (Jacobian) operator
    # ------------------------------------------------------------------
    def _sigma_s_coeffs(
        self, t: torch.Tensor, rho: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coefficients of the unified split ``Sigma_s = c_iso I + c_jac J``.

        - SI: ``Sigma_s = gamma^2 t I + gamma^4 t^2 A_tau (beta J_b - bdot I)``
          -> ``c_iso = rho - jac_scale * beta_diff``, ``c_jac = jac_scale *
          beta`` with ``jac_scale = gamma^4 t^2 A_tau``.
        - FM: ``Sigma_s = alpha^2 I + alpha^4 grad_x s`` -> ``c_iso = rho``,
          ``c_jac = alpha^4``.

        ``J`` is the network Jacobian (``grad_x b_theta`` for SI, ``grad_x s``
        for FM), applied via :meth:`_build_jvp_fn`. Splitting the scalars from
        ``J`` lets the shared mode cache ``H J H^T`` across pseudo-steps while
        keeping the schedule scalars exact at the current ``tau``.
        """
        if self.model_class == "si":
            beta = self.interpolant.beta(t)
            beta_diff = self.interpolant.beta_diff(t)
            gamma = self.interpolant.gamma(t)
            jac_scale = gamma**4 * t**2 * self._si_A(t)
            return rho - jac_scale * beta_diff, jac_scale * beta

        # FM
        alpha = self.interpolant.alpha(t)
        return rho, alpha**4

    def _build_jvp_fn(
        self,
        x: torch.Tensor,
        t_net: torch.Tensor,
        field_history: Optional[torch.Tensor],
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> Callable:
        """Build a forward-mode operator ``v -> J v`` at the captured state.

        ``J`` is the network Jacobian (``grad_x b_theta`` for SI, ``grad_x s``
        for FM), applied via ``torch.func.jvp`` (forward-mode AD): one dual
        forward per tangent batch, vs the ~7x slower double-backward
        construction this replaced (measured ~21-32 ms/tangent double-backward
        vs ~4 ms/tangent forward-mode on the NS UNet; identical values to
        5e-16 rel. in fp64). Run under ``no_grad`` it builds no backward graph,
        so nothing flows into the posterior drift.

        The captured state and conditioning broadcast to the tangent's batch:
        it equals ``B`` on the per-member path (no-op) but is the column chunk
        / member count on the shared path, where the state is captured at
        batch 1 (the ensemble mean). Broadcasts are materialised
        (``contiguous``) because forward-mode AD rejects writes into stride-0
        expanded views inside the network.
        """
        if self.model_class == "si":
            net_fn = self.model.drift  # J_b @ v, with b_theta = model.drift
        else:
            net_fn = self.model.score  # grad_x s @ v, with s = model.score

        x = x.detach()

        def _bcast(tensor: Optional[torch.Tensor], k: int) -> Optional[torch.Tensor]:
            if tensor is None or tensor.shape[0] == k:
                return tensor
            if tensor.shape[0] == 1:
                return tensor.expand(k, *tensor.shape[1:]).contiguous()
            raise ValueError(
                f"cannot broadcast batch {tensor.shape[0]} to {k}."
            )

        def jvp_fn(v: torch.Tensor) -> torch.Tensor:
            k = v.shape[0]
            xk = _bcast(x, k)
            tk, fhk = _bcast(t_net, k), _bcast(field_history, k)
            fck, pck = _bcast(field_cond, k), _bcast(pars_cond, k)
            # The tangent must be materialised too: the per-member H Sigma_s H^T
            # build hands over expanded basis columns (stride-0 batch).
            v = v.detach().contiguous()
            with torch.no_grad(), _math_sdpa():
                _, jv = torch.func.jvp(
                    lambda z: net_fn(z, tk, fhk, fck, pck), (xk,), (v,)
                )
            return jv

        return jvp_fn

    def _refresh_shared_jacobian(
        self,
        x: torch.Tensor,
        t_clamped: torch.Tensor,
        t_net: torch.Tensor,
        field_history: Optional[torch.Tensor],
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> Callable:
        """Return the shared-state JVP, refreshed every k-th pseudo-step.

        The shared Jacobian is evaluated at the ensemble-mean state and kept
        for ``jacobian_refresh_every`` distinct pseudo-time values (k = 1, the
        default, refreshes on every step -- no approximation). Bookkeeping is
        by the clamped pseudo-time: an UNCHANGED tau (another ensemble batch of
        the same step, or a solver re-evaluation) reuses the cache without
        counting; a DECREASED tau means the pseudo-time loop restarted (next
        assimilation window), which always forces a refresh. Refreshing also
        drops the cached ``H J H^T`` so it is lazily rebuilt from the new
        Jacobian.
        """
        cache = self._shared_jac_cache
        t_s = float(t_clamped.reshape(-1)[0])

        refresh = cache["jvp_fn"] is None
        if not refresh and t_s != cache["t_last"]:
            if t_s < cache["t_last"]:
                refresh = True  # pseudo-time restarted -> new SDE/ODE pass
            else:
                cache["steps_since"] += 1
                refresh = cache["steps_since"] >= self.jacobian_refresh_every
        cache["t_last"] = t_s

        if refresh:
            def _mean0(z: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                return None if z is None else z.mean(dim=0, keepdim=True)

            cache["jvp_fn"] = self._build_jvp_fn(
                x=_mean0(x),
                t_net=t_net[:1],
                field_history=_mean0(field_history),
                field_cond=_mean0(field_cond),
                pars_cond=_mean0(pars_cond),
            )
            cache["HJHt"] = None
            cache["steps_since"] = 0

        return cache["jvp_fn"]


class FlowdasGaussianLikelihood(nn.Module):
    """FlowDAS Monte-Carlo likelihood (baseline, ``chen_flowdas_2025``).

    NOT the paper's observation-interpolant method. Kept as a baseline: it
    draws ``J`` one-step predictions of ``x_1`` from ``x_tau``, softmax-weights
    them by the observation likelihood ``N(y; H x_1, R)``, and returns the RAW
    guidance score ``grad log p(y|x_tau)`` differentiated through the predictor
    (magnitude and ``1/sigma**2`` intact). The posterior multiplies it by the
    tuned constant step size ``zeta`` (:meth:`sde_weight`) -- FlowDAS's own
    guidance mechanism, rather than the SI-SDE velocity--score weight used by
    the interpolant-likelihood method.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        integration_order: int = 1,
        guidance_scale: float = 1.0,
        max_grad_norm: Optional[float] = None,
        num_mc_samples: Optional[int] = 25,
    ) -> None:
        """Initialize the FlowDAS Monte-Carlo likelihood.

        Args:
            obs_operator: Observation operator.
            model: Trained SI prior (supplies drift + interpolation schedule).
            variance: Observation-noise variance ``sigma**2`` (``R``).
            ensemble_size: The DA ensemble size ``E``. Kept for interface
                compatibility only -- it is **not** used for the Monte-Carlo
                estimate (that is ``num_mc_samples``).
            integration_order: Kept for interface compatibility.
            guidance_scale: FlowDAS guidance step size ``zeta`` (the paper's tuned
                constant). It is returned by :meth:`sde_weight` and multiplies the
                RAW likelihood score in the posterior drift, so it must be tuned
                per experiment (the raw ``1/sigma**2`` score is what diverged at
                NS scale before -- start small and/or set ``max_grad_norm``).
            max_grad_norm: Optional per-member L2 cap on the score. ``None`` leaves
                the raw magnitude untouched; a float clips blow-ups WITHOUT
                rescaling scores below the cap (unlike unit-normalisation, which
                destroyed the time schedule).
            num_mc_samples: Number of one-step ``x_1`` predictions ``J`` drawn
                PER ensemble member for the likelihood estimate (paper Eq. 10;
                recommended ``J=25`` irrespective of ``E``). This is decoupled
                from ``ensemble_size``. ``None`` falls back to the ``25`` default.
        """
        super(FlowdasGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        # DA ensemble size E (interface only; NOT the Monte-Carlo count).
        self.ensemble_size = ensemble_size
        # J = number of one-step x_1 predictions per member (paper Eq. 10).
        self.num_mc_samples = int(num_mc_samples) if num_mc_samples is not None else 25
        self.dist = torch.distributions.MultivariateNormal
        self.integration_order = integration_order
        self.guidance_scale = guidance_scale
        self.max_grad_norm = max_grad_norm

        # Gamma-multiplier of the trained interpolant: the whole stochastic part
        # of the interpolant (hence the predictive spread of x_1 | x_tau) scales
        # with it, but the hard-coded `integral_variance` below was derived for
        # gamma_multiplier = 1. Scale by it so the Monte-Carlo cloud matches the
        # ACTUAL trained noise level (NS prior uses gamma_multiplier = 0.1).
        self._gamma_mult = float(
            getattr(self.model.interpolation, "gamma_multiplier", 1.0)
        )

        # Predictive spread of x_1 given x_tau for gamma_multiplier = 1: large at
        # tau -> 0 (only x_0 known) and 0 at tau -> 1 (x_1 determined). NOTE: this
        # is used below as a STANDARD DEVIATION; whether the FlowDAS paper's
        # predictor uses this as a std or a variance should be confirmed against
        # the arXiv TeX (the `integral_variance` name suggests a variance).
        self.integral_variance = lambda t: 2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3

    def sde_weight(
        self, t: torch.Tensor, diffusion_term: Optional[Callable] = None
    ) -> torch.Tensor:
        """Coefficient multiplying the likelihood score in the posterior drift.

        FlowDAS applies a TUNED CONSTANT guidance step size ``zeta`` to the raw
        likelihood score (paper: "``zeta_n`` typically constant"), rather than
        the SI-SDE velocity--score weight ``a_tau + 1/2 g**2`` used by the
        interpolant-likelihood method. The posterior queries this so each
        likelihood owns how its score enters the SDE.
        """
        return self.guidance_scale * torch.ones_like(t)

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
        ``x_1`` from ``x_tau`` as ``X1_hat^(j) = mu_x1 + s eps_j``, with
        ``eps_j ~ N(0, I)`` DETACHED constants so the only ``x_tau``-dependence is
        through the denoiser mean ``mu_x1(x_tau)``. Compute DETACHED importance
        weights ``w_j = softmax_j(-||y - H X1_hat^(j)||^2 / (2 sigma^2))`` and
        return the RAW likelihood score, differentiated THROUGH the predictor

            grad log p(y|x_tau) = grad_{x_tau} [ -1/(2 sigma^2) sum_j w_j ||y - H X1_hat^(j)||^2 ]
                                                                 (weights detached).

        This is the paper's guidance term with its magnitude and ``1/sigma^2``
        INTACT (no unit-normalisation). The posterior multiplies it by the tuned
        constant step size ``zeta`` returned by :meth:`sde_weight` -- FlowDAS's
        actual mechanism -- and optional ``max_grad_norm`` only caps blow-ups
        (the NS-scale divergence) without rescaling smaller scores. Returns
        ``(score, log_likelihood)``; computed on a detached grad-enabled leaf so
        the caller's autoregressive working state is never mutated in place.
        """
        n_mc = self.num_mc_samples  # J: MC x_1 draws per member (decoupled from E)
        R = self.original_variance

        # Predictive spread of x_1 | x_tau, scaled by the interpolant's
        # gamma_multiplier so the Monte-Carlo cloud matches the trained noise
        # level (state-independent scalar).
        s = self._gamma_mult * self.integral_variance(t)

        # Detached grad-enabled leaf: the score flows through mu_x1(x_g) but the
        # caller's working tensor is untouched.
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

            # log p(y|x_tau) with the 1/(2 sigma^2) factor kept; differentiate
            # THROUGH the predictor (weights detached) to get the raw score. The
            # -1/(2R) sign makes `grad` already point UP the log-likelihood, so
            # the posterior ascends with a positive step size zeta.
            log_lik = -0.5 * (weights * sq_err).sum(dim=0).sum() / R  # scalar
            grad = torch.autograd.grad(outputs=log_lik, inputs=x_g)[0]

        score = grad
        # Optional safety cap: clip only members whose score exceeds max_grad_norm,
        # leaving the direction and the relative magnitude of smaller scores intact
        # (unlike unit-normalisation, which flattened the schedule).
        if self.max_grad_norm is not None:
            gn = score.reshape(b, -1).norm(dim=1).reshape(
                b, *([1] * (score.dim() - 1))
            )
            score = score * (self.max_grad_norm / (gn + 1e-12)).clamp(max=1.0)

        # Per-member log marginal likelihood (for optional SMC reweighting).
        log_likelihood = (
            torch.logsumexp(log_w, dim=0)
            - torch.log(torch.tensor(float(n_mc), device=log_w.device))
        ).detach()

        return score.detach(), log_likelihood
