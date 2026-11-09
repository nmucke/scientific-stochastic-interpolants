import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import (
    LinearObservationOperator,
    PointwiseObservationOperator,
)

# Gauss-Hermite nodes of the statistical linearization. Saturating g (arctan)
# under the wide small-tau Gaussians converges slowly: 20 nodes leave ~1e-1
# errors in E g at tau = 0.1, 64 ~1e-2. Elementwise [B, N_y, n] work.
GH_NODES = 64


class _MemberJacobian:
    """Per-member ``J_b = J_h(x_lin[b])``: the slice of the linear-operator
    interface the likelihood uses, via the operator's ``jvp`` / ``vjp`` /
    ``gram`` (no dense Jacobian)."""

    per_member = True

    def __init__(self, obs_operator: nn.Module, x_lin: torch.Tensor) -> None:
        self.obs = obs_operator
        self.x_lin = x_lin
        self.num_obs = int(obs_operator.num_obs)

    # Materialised: forward-mode AD (torch.func defaults) rejects stride-0 views.
    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self.obs.jvp(self.x_lin, v.contiguous())

    def transpose(self, u: torch.Tensor) -> torch.Tensor:
        return self.obs.vjp(self.x_lin, u.contiguous())

    def gram(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.obs.gram(self.x_lin)


def _gauss_hermite_moments(
    obs: PointwiseObservationOperator, m: torch.Tensor, s2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``E g(z)``, ``E g'(z)``, ``Var g(z)`` for ``z ~ N(m, s2)`` per sensor."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(GH_NODES)
    xi = torch.as_tensor(nodes, dtype=m.dtype, device=m.device)
    w = torch.as_tensor(weights / np.sqrt(2.0 * np.pi), dtype=m.dtype, device=m.device)
    Z = m.unsqueeze(-1) + s2.sqrt().unsqueeze(-1) * xi  # [B, N_y, n]
    gZ = obs.g(Z)
    mean = (gZ * w).sum(-1)
    dmean = (obs.dg(Z) * w).sum(-1)
    var = ((gZ - mean.unsqueeze(-1)) ** 2 * w).sum(-1)
    return mean, dmean, var


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

    Nonlinear ``h`` (``obs_operator.is_linear`` False): with ``ybar = alpha
    h(a0) + beta y`` the anchor cancels and the residual is the exact
    per-member ``beta (y - h(xhat1))``, ``xhat1 = (x_tau - alpha a0 - mu_s) /
    beta``; every ``H`` above becomes ``J = J_h(x_lin)``. ``inflated_shared``
    shares only the NETWORK Jacobian: for a ``PointwiseObservationOperator``
    ``h = g(L x)`` it linearizes ``h`` per member around the shared ``L
    Sigma_s L^T``; for other operators it linearizes once at the ensemble mean of ``xhat1``
    (dense ``J``, refreshed with the shared network Jacobian, so the cost per
    refresh is that of the linear ``H J H^T`` build), which is unsafe for
    non-injective ``h`` (warned at run time by ``_check_shared_descent``);
    ``inflated`` / ``dps_jacobian_free`` linearize
    per member via the operator's ``jvp`` / ``vjp`` / ``gram`` (``inflated``:
    ``N_y`` network JVPs as for linear ``H``; jacfree: one ``gram``). Linear
    operators take the unchanged linear path.
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
        jacobian_damping: float = 1.0,
        sigma_bar_eig_floor: bool = False,
        isotropic_front_factor: bool = False,
        residual_step_cap: Optional[float] = None,
        obs_linearization_method: str = "first_order",
    ) -> None:
        """Initialize the interpolant Gaussian likelihood.

        Args:
            model: Trained prior model (SI ``FollmerStochasticInterpolant`` or
                ``FlowMatchingModel``). Supplies the interpolation schedules and
                the score recovery.
            obs_operator: An ``ObservationOperator``. Linear: forward ``H @ x``,
                adjoint ``transpose`` and dense ``obs_matrix``; nonlinear: see
                the class docstring.
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
            jacobian_damping: Jacobian-term damping ``lambda in [0, 1]`` for the
                full modes (``inflated`` / ``inflated_shared``). Replaces
                ``Sigma_s = c_iso I + c_jac J`` with ``c_iso I + lambda c_jac J``
                in BOTH the covariance solve and the mean front factor, so the
                whole non-normal ``J`` contribution is scaled consistently.
                ``1.0`` (default) reproduces the current full ``Sigma_s``; ``0.0``
                collapses to the isotropic (jacfree) ``Sigma_s`` while still
                paying the JVP cost. Prior art: SDA's ``gamma_sda`` tames its
                ``H H^T`` term the same way. Inert for ``dps_jacobian_free``.
            sigma_bar_eig_floor: ``inflated_shared`` only. When ``True``, the
                assembled ``Sigma_bar = beta^2 R + c_iso H H^T + c_jac H J H^T``
                is symmetrised and its eigenvalues are floored at the
                jacfree-equivalent value ``beta^2 R + c_iso lambda_min(H H^T)``
                before the solve, so the non-symmetric / shrinking ``H J H^T``
                term can never make ``Sigma_bar^{-1}`` larger than the isotropic
                covariance would. ``False`` (default) solves the raw assembled
                ``Sigma_bar`` (current behaviour). ``N_y`` is small (<= 1024) so
                the ``eigh`` is negligible next to the JVPs.
            isotropic_front_factor: full modes. When ``True``, the inflated
                ``Sigma_s`` is used only inside the covariance solve; the mean
                front factor is the isotropic ``H`` (PiGDM-style inflation
                without a second application of ``J``), halving the ``J``
                amplification path. ``False`` (default) uses the matching full
                ``Sigma_s / rho`` front factor.
            residual_step_cap: full modes. Optional relative cap ``kappa`` on the
                guidance magnitude: each member's returned ``Sbar`` is clamped so
                ``||H Sbar|| <= kappa ||ybar - mu_bar||`` (schedule-consistent,
                inactive in the healthy regime). ``None`` (default) applies no
                cap.
            obs_linearization_method: nonlinear ``h`` only. ``"first_order"``
                (default; Taylor linearization, see the class docstring) or
                ``"statistical"`` (``PointwiseObservationOperator`` only, per
                member in every mode): per-sensor Gauss--Hermite moments of
                ``g(z_k)``, ``z_k ~ N((L xhat1)_k, (L C L^T)_kk)``, ``C =
                Sigma_s / beta^2``, giving the mean ``E g``, gain ``diag(E g')``
                and extra variance ``Var g - (E g')^2 (L C L^T)_kk``.
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
        # --- inflated-mode stabilisation knobs (defaults reproduce current
        # behaviour: lambda=1 full Sigma_s, no eig floor, full front factor,
        # no step cap). All are inert for dps_jacobian_free.
        self.jacobian_damping = float(jacobian_damping)
        if self.jacobian_damping < 0.0:
            raise ValueError(
                f"jacobian_damping must be >= 0, got {jacobian_damping!r}."
            )
        # lambda > 1 is AMPLIFICATION, not damping: it scales c_jac beyond the
        # value the theory prescribes. Allowed (and useful -- with a lagged
        # Jacobian the lambda=1 divergence is suppressed and the rmse curve is
        # still falling at 1.0, so the optimum can sit above it), but it is an
        # empirical over-inflation with no derivation behind it; the upper end is
        # bounded only by divergence, so sweep it, do not assume it.
        self.sigma_bar_eig_floor = bool(sigma_bar_eig_floor)
        self.isotropic_front_factor = bool(isotropic_front_factor)
        self.residual_step_cap = (
            float(residual_step_cap) if residual_step_cap is not None else None
        )
        if self.residual_step_cap is not None and self.residual_step_cap <= 0.0:
            raise ValueError(
                f"residual_step_cap must be > 0 or None, got {residual_step_cap!r}."
            )
        # (t_last, steps_since, jvp_fn, HJHt, lin) of the current shared Jacobian.
        # ``t_last`` tracks the last pseudo-time seen so repeated calls at the
        # same tau (extra ensemble batches) reuse the cache, and a tau DECREASE
        # (pseudo-time restarted -> next assimilation window) forces a refresh.
        # ``lin``: nonlinear-h ensemble-mean linearization, refreshed with it.
        self._shared_jac_cache: Dict[str, Any] = {
            "t_last": None, "steps_since": 0, "jvp_fn": None, "HJHt": None,
            "lin": None,
        }

        if obs_linearization_method not in ("first_order", "statistical"):
            raise ValueError(
                "obs_linearization_method must be 'first_order' or 'statistical', "
                f"got {obs_linearization_method!r}."
            )
        if (
            obs_linearization_method == "statistical"
            and not obs_operator.is_linear
            and not isinstance(obs_operator, PointwiseObservationOperator)
        ):
            raise ValueError(
                "obs_linearization_method='statistical' needs a "
                "PointwiseObservationOperator."
            )
        self.obs_linearization_method = obs_linearization_method

        # inflated_shared + nonlinear h: a pointwise h = g(L x) is linearized per
        # member around the shared L Sigma_s L^T (cheap, safe for non-injective
        # g); other operators are linearized once at the ensemble-mean xhat1.
        self._shared_member_linearization = isinstance(
            obs_operator, PointwiseObservationOperator
        )

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

        # Cached adjoint basis columns H^T e_j (H time-independent), keyed on
        # the identity of the source ``obs_matrix`` (a reference is held), so
        # ``load_mask`` / a refreshed linearization invalidates them. ``H H^T``
        # is cached by the operator (``obs_operator.gram``).
        self._cols: Optional[torch.Tensor] = None
        self._cols_src: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Operator caches
    # ------------------------------------------------------------------
    def _get_adjoint_basis(
        self, ref: torch.Tensor, op: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """Return the grid tensors ``H^T e_j`` for all obs rows, ``[N_y, C, H, W]``.

        Time-independent (one grid tensor per observation basis vector), so it
        is built ONCE per device/dtype instead of allocating an ``N_y x N_u``
        block on every ``H Sigma_s H^T`` build.
        """
        op = self.obs_operator if op is None else op
        src = op.obs_matrix
        if (
            self._cols is None
            or self._cols_src is not src
            or self._cols.device != ref.device
            or self._cols.dtype != ref.dtype
        ):
            eye_y = torch.eye(src.shape[0], device=ref.device, dtype=ref.dtype)
            self._cols = op.transpose(eye_y)
            self._cols_src = src
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
        self, x: torch.Tensor, sigma_s_apply: Callable, op: Any = None
    ) -> torch.Tensor:
        """Form ``H Sigma_s H^T`` per ensemble member, shape [B, N_y, N_y].

        Column ``j`` is ``H Sigma_s(x_b) (H^T e_j)``: the adjoint of the j-th
        observation basis vector is the same grid tensor for every member, but
        the full ``Sigma_s`` is state-dependent, so the Jacobian-vector product
        is evaluated at each member's ``x_b`` (one batched JVP per column).
        For a per-member ``op`` (:class:`_MemberJacobian`) the column
        ``J_b^T e_j`` differs per member and is one batched VJP.
        """
        op = self.obs_operator if op is None else op
        b = x.shape[0]
        if getattr(op, "per_member", False):
            N_y = op.num_obs
            eye = torch.eye(N_y, device=x.device, dtype=x.dtype)
            HSHt = torch.empty(b, N_y, N_y, device=x.device, dtype=x.dtype)
            for j in range(N_y):
                col = op.transpose(eye[j].expand(b, N_y))
                HSHt[:, :, j] = op(sigma_s_apply(col))
            return HSHt

        # H^T e_j on the grid, one per column (cached; time-independent).
        cols = self._get_adjoint_basis(x, op)  # [N_y, C, H, W]
        N_y = cols.shape[0]

        HSHt = torch.empty(b, N_y, N_y, device=x.device, dtype=x.dtype)
        for j in range(N_y):
            v = cols[j : j + 1].expand(b, *cols.shape[1:])  # [B, C, H, W]
            HSHt[:, :, j] = op(sigma_s_apply(v))  # [B, N_y]
        return HSHt

    def _build_HJHt_shared(
        self, ref: torch.Tensor, jvp_fn: Callable, op: Any = None
    ) -> torch.Tensor:
        """Form the Jacobian-only ``H J H^T`` (shape ``[N_y, N_y]``).

        The ``inflated_shared`` approximation evaluates the network Jacobian
        ONCE at the ensemble mean, so ``H J H^T`` is a single matrix. With the
        split ``Sigma_s = c_iso I + c_jac J`` the caller assembles
        ``H Sigma_s H^T = c_iso H H^T + c_jac H J H^T`` per pseudo-step, so
        this matrix can be cached across steps (``jacobian_refresh_every > 1``)
        with the scalars staying exact -- only ``J`` is lagged.
        """
        op = self.obs_operator if op is None else op
        cols = self._get_adjoint_basis(ref, op)  # [N_y, C, H, W]
        N_y = cols.shape[0]

        HJHt = torch.empty(N_y, N_y, device=ref.device, dtype=ref.dtype)
        chunk = max(1, int(self.shared_sigma_chunk))
        for start in range(0, N_y, chunk):
            block = cols[start : start + chunk]  # [k, C, H, W]
            # op(J block) is [k, N_y]; its row i is the (start+i)-th
            # COLUMN of H J H^T.
            HJHt[:, start : start + block.shape[0]] = op(
                jvp_fn(block)
            ).transpose(0, 1)
        return HJHt

    def _shared_HSHt(
        self, x: torch.Tensor, jvp_fn: Callable, c_iso: Any, c_jac: Any, op: Any = None
    ) -> torch.Tensor:
        """``inflated_shared``: ``c_iso H H^T + c_jac H J H^T`` (fresh tensor).

        Assembled with the CURRENT schedule scalars; the cached ``H J H^T`` is
        the only lagged factor (exact for ``jacobian_refresh_every = 1``).
        """
        op = self.obs_operator if op is None else op
        cache = self._shared_jac_cache
        if cache["HJHt"] is None:
            cache["HJHt"] = self._build_HJHt_shared(x, jvp_fn, op)
        return (
            float(c_iso.reshape(-1)[0]) * op.gram(x)
            + float(c_jac.reshape(-1)[0]) * cache["HJHt"]
        )

    @staticmethod
    def _solve(Sigma_bar: torch.Tensor, residual_obs: torch.Tensor) -> torch.Tensor:
        """``Sigma_bar^{-1} r`` per member: shared ``[N_y, N_y]`` (factorised
        once for every RHS) or batched ``[B, N_y, N_y]``."""
        if Sigma_bar.dim() == 2:
            return torch.linalg.solve(
                Sigma_bar, residual_obs.transpose(0, 1)
            ).transpose(0, 1)
        return torch.linalg.solve(Sigma_bar, residual_obs.unsqueeze(-1)).squeeze(-1)

    def _interpolant_score(
        self,
        x: torch.Tensor,
        residual_obs: torch.Tensor,
        rho: torch.Tensor,
        beta: torch.Tensor,
        sigma_s_apply: Optional[Callable] = None,
        HSHt_shared: Optional[torch.Tensor] = None,
        op: Any = None,
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
        mode (and of jacfree with a per-member ``op``). In all cases the
        ``beta^2 R`` term is added on the diagonal in place.

        Args:
            x: State tensor [B, C, H, W] (used for shape/device).
            residual_obs: ``ybar - mu_bar`` in observation space [B, N_y].
            rho: Isotropic source scale rho_tau (= sigma_tau^2).
            beta: beta_tau (scalar tensor).
            sigma_s_apply: Operator ``v -> Sigma_s v`` on grid tensors (full
                modes); ``None`` -> isotropic ``Sigma_s = rho I``.
            HSHt_shared: ``inflated_shared`` only -- the assembled
                ``H Sigma_s H^T`` (``[N_y, N_y]``, fresh tensor).
            op: Linear(ized) operator playing the role of ``H`` (default
                ``obs_operator``): a shared ``LinearizedObservationOperator`` or a
                per-member :class:`_MemberJacobian` (nonlinear ``h``).

        Returns:
            Sbar as a full-grid tensor [B, C, H, W].
        """
        op = self.obs_operator if op is None else op
        beta_sq_R = float((beta.reshape(-1)[0] ** 2)) * self.original_variance
        rho_s = float(rho.reshape(-1)[0])

        if sigma_s_apply is None:
            # Jacobian-free: Sigma_bar = beta^2 R + rho H H^T (fresh tensor).
            Sigma_bar = rho_s * op.gram(x)
            Sigma_bar.diagonal(dim1=-2, dim2=-1).add_(beta_sq_R)
        elif not self.share_sigma_s:
            # Exact inflated mode: Sigma_bar per member (fresh tensor).
            Sigma_bar = self._build_HSHt(x, sigma_s_apply, op)
            Sigma_bar.diagonal(dim1=-2, dim2=-1).add_(beta_sq_R)
        else:
            Sigma_bar = HSHt_shared  # fresh tensor
            Sigma_bar.diagonal().add_(beta_sq_R)
            if self.sigma_bar_eig_floor:
                # Symmetrise + floor eigenvalues at the jacfree-equivalent value
                # so the non-symmetric / shrinking H J H^T term can never make
                # Sigma_bar^{-1} larger than the isotropic covariance would.
                Sigma_bar = self._floor_sigma_bar(Sigma_bar, x, rho_s, beta_sq_R, op)
        Ht_sol = op.transpose(self._solve(Sigma_bar, residual_obs))  # [B, C, H, W]
        if sigma_s_apply is None:
            # Isotropic front factor rho/rho_tau = 1: Sbar is exactly H^T sol.
            return Ht_sol
        # grad_xbar_mu front factor: (Sigma_s / rho_tau) applied on grid, or
        # the isotropic H (Sbar = Ht_sol) when the front factor is damped.
        sbar = (
            Ht_sol
            if self.isotropic_front_factor
            else sigma_s_apply(Ht_sol) / rho_s
        )
        return self._cap_step(sbar, residual_obs, op)

    def _floor_sigma_bar(
        self,
        Sigma_bar: torch.Tensor,
        x: torch.Tensor,
        rho_s: float,
        beta_sq_R: float,
        op: Any = None,
        HHt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Symmetrise ``Sigma_bar`` and floor its eigenvalues (shared mode).

        The assembled ``Sigma_bar`` (already carrying ``beta^2 R`` on its
        diagonal) is replaced by ``V clamp(Lambda, floor) V^T`` where
        ``(Lambda, V) = eigh(1/2 (Sigma_bar + Sigma_bar^T))`` and
        ``floor = beta^2 R + rho_tau lambda_min(H H^T)`` -- the smallest
        eigenvalue the isotropic (jacfree) ``rho H H^T + beta^2 R`` would have.
        This both drops the non-normal part of the once-evaluated ``H J H^T``
        (theory has a symmetric ``Sigma_s``) and caps ``||Sigma_bar^{-1}||`` at
        the jacfree level, so a shrunk / indefinite Jacobian term can no longer
        blow the guidance up. ``N_y <= 1024`` -> the ``eigh`` is negligible.
        A batched ``HHt`` (``[B, N_y, N_y]``, statistical linearization) gives a
        per-member floor.
        """
        if HHt is None:
            HHt = (self.obs_operator if op is None else op).gram(x)
        sym = 0.5 * (Sigma_bar + Sigma_bar.transpose(-2, -1))
        evals, evecs = torch.linalg.eigh(sym)
        if HHt.dim() == 2:
            floor = beta_sq_R + rho_s * float(torch.linalg.eigvalsh(HHt)[0])
            evals = evals.clamp_min(floor)
        else:
            floor = beta_sq_R + rho_s * torch.linalg.eigvalsh(HHt)[..., :1]
            evals = torch.maximum(evals, floor)
        return (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-2, -1)

    def _cap_step(
        self, sbar: torch.Tensor, residual_obs: torch.Tensor, op: Any = None
    ) -> torch.Tensor:
        """Relative per-member cap on the guidance (full modes; no-op if unset).

        Clamps each member so ``||H Sbar|| <= kappa ||ybar - mu_bar||`` with
        ``kappa = self.residual_step_cap`` (``H`` -> the linearization ``op``
        for nonlinear ``h``): one guidance step can move the state
        at most ``kappa`` times the current observation-space residual. The cap
        is schedule-consistent (it scales with the residual, not a fixed
        constant) so it is inactive in the healthy regime and only bites the
        blow-ups. Directions and the relative magnitude of sub-cap members are
        untouched.
        """
        if self.residual_step_cap is None:
            return sbar
        op = self.obs_operator if op is None else op
        b = sbar.shape[0]
        hs_norm = op(sbar).reshape(b, -1).norm(dim=1)  # ||H Sbar||
        res_norm = residual_obs.reshape(b, -1).norm(dim=1)
        cap = self.residual_step_cap * res_norm
        scale = (cap / (hs_norm + 1e-12)).clamp(max=1.0)
        return sbar * scale.reshape(b, *([1] * (sbar.dim() - 1)))
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

        # --- full Sigma_s operator (inflated modes) --------------------------
        # Both full modes use the unified split Sigma_s = c_iso I + c_jac J
        # with the network Jacobian J applied by a forward-mode JVP; the
        # isotropic mode (dps_jacobian_free) leaves it None.
        sigma_s_apply, c_iso, c_jac, jvp_fn = self._sigma_s_operator(
            x=x,
            t_clamped=t_clamped,
            t_grid=t_grid,
            t_net=t_net,
            rho=rho,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        if self.obs_operator.is_linear:
            # --- observation interpolant ybar_tau ---------------------------
            if self.anchor == "x0":
                base_obs = self.obs_operator(field_history[..., -1])
            else:  # FM: a0 = 0
                base_obs = torch.zeros_like(observations)
            interpolant_obs = self._interpolate_observations(
                observations, t_clamped, base_obs
            )

            # --- likelihood mean mu_bar = H x - H mu_s ----------------------
            mu_bar = self.obs_operator(x) - self.obs_operator(mu_s)

            residual_obs = interpolant_obs - mu_bar  # ybar - mu_bar, [B, N_y]

            hsht_shared = (
                self._shared_HSHt(x, jvp_fn, c_iso, c_jac)
                if self.share_sigma_s
                else None
            )

            # --- interpolant score Sbar (closed form) -----------------------
            sbar = self._interpolant_score(
                x=x,
                residual_obs=residual_obs,
                rho=rho,
                beta=beta,
                sigma_s_apply=sigma_s_apply,
                HSHt_shared=hsht_shared,
            )
        else:
            sbar, residual_obs = self._score_nonlinear(
                x=x,
                observations=observations,
                t_grid=t_grid,
                beta=beta,
                mu_s=mu_s,
                rho=rho,
                field_history=field_history,
                sigma_s_apply=sigma_s_apply,
                c_iso=c_iso,
                c_jac=c_jac,
                jvp_fn=jvp_fn,
            )

        # Diagnostic log-likelihood (used for optional SMC resampling). The
        # tau >= dtau guard in the posterior loops keeps beta > 0, so no
        # additive denominator epsilon is needed (t is clamped to >= MIN_TIME).
        log_likelihood = -0.5 * (
            torch.linalg.norm(residual_obs, dim=1) ** 2
        ) / (beta.reshape(-1) ** 2 * self.original_variance)

        return sbar, log_likelihood

    # ------------------------------------------------------------------
    # Nonlinear observation operator h
    # ------------------------------------------------------------------
    def _score_nonlinear(
        self,
        x: torch.Tensor,
        observations: torch.Tensor,
        t_grid: torch.Tensor,
        beta: torch.Tensor,
        mu_s: torch.Tensor,
        rho: torch.Tensor,
        field_history: torch.Tensor,
        sigma_s_apply: Optional[Callable],
        c_iso: Optional[torch.Tensor],
        c_jac: Optional[torch.Tensor],
        jvp_fn: Optional[Callable],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Sbar, residual)`` for a nonlinear ``h`` (see the class docstring).

        Residual ``beta (y - h(xhat1))`` with the source-moment denoiser
        ``xhat1 = (x_tau - alpha a0 - mu_s) / beta`` (for linear ``H`` exactly
        ``ybar - mu_bar``); ``H -> J = J_h(x_lin)`` everywhere else, with
        ``x_lin`` the ensemble mean of ``xhat1`` (``inflated_shared``) or
        ``xhat1_b`` (other modes) -- never ``x_tau``.
        """
        alpha_g = self.interpolant.alpha(t_grid)
        beta_g = self.interpolant.beta(t_grid)
        if self.anchor == "x0":
            xhat1 = (x - alpha_g * field_history[..., -1] - mu_s) / beta_g
        else:  # FM: a0 = 0
            xhat1 = (x - mu_s) / beta_g
        xhat1 = xhat1.detach()

        if self.obs_linearization_method == "statistical":
            return self._score_statistical(
                x, xhat1, observations, beta, rho, sigma_s_apply, c_iso, c_jac, jvp_fn
            )
        if self.share_sigma_s and self._shared_member_linearization:
            # Shared network Jacobian, per-member pointwise J_h (see __init__).
            return self._score_statistical(
                x, xhat1, observations, beta, rho, sigma_s_apply, c_iso, c_jac,
                jvp_fn, first_order=True,
            )

        residual_obs = beta * (observations - self.obs_operator(xhat1))  # [B, N_y]
        hsht_shared = None
        if self.share_sigma_s:
            op: Any = self._shared_linearization(xhat1)
            hsht_shared = self._shared_HSHt(x, jvp_fn, c_iso, c_jac, op)
        else:
            op = _MemberJacobian(self.obs_operator, xhat1)
        sbar = self._interpolant_score(
            x=x,
            residual_obs=residual_obs,
            rho=rho,
            beta=beta,
            sigma_s_apply=sigma_s_apply,
            HSHt_shared=hsht_shared,
            op=op,
        )
        if self.share_sigma_s:
            self._check_shared_descent(xhat1, op, sbar)
        return sbar, residual_obs

    def _check_shared_descent(
        self, xhat1: torch.Tensor, lin: Any, sbar: torch.Tensor
    ) -> None:
        """Warn when the ensemble-mean linearization gives ascent steps.

        The shared step ``Sbar_b`` is built so that the shared model predicts
        the observation change ``J_bar Sbar_b`` (aligned with the residual);
        the member's actual first-order change is ``J_b Sbar_b``. If
        ``<J_b Sbar_b, J_bar Sbar_b> < 0`` the two point in opposite
        directions, i.e. the step moves the member AWAY from its observation
        (``J_b`` and ``J_bar`` disagree in sign: the member sits on another
        branch of a non-injective ``h``). Uncorrected this is the Riccati
        runaway (Riccati blow-up ``dx ~ -x^2``) of a member on the other branch. Two ``h``-JVPs,
        no network calls.
        """
        b = xhat1.shape[0]
        own = self.obs_operator.jvp(xhat1, sbar).reshape(b, -1)
        shared = lin(sbar).reshape(b, -1)
        bad = (own * shared).sum(dim=1) < 0
        if bool(bad.any()):
            warnings.warn(
                f"inflated_shared: the ensemble-mean linearization of the "
                f"nonlinear observation operator gives {int(bad.sum())}/{b} "
                "members an ASCENT step (their local Jacobian disagrees in sign "
                "with the shared one, e.g. a non-injective h with members on "
                "different branches). Such members can diverge; use "
                "likelihood_mode='inflated' for this operator.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _shared_linearization(self, xhat1: torch.Tensor) -> nn.Module:
        """``J_h`` at the ensemble mean of ``xhat1`` (dense), cached with and
        refreshed on the cadence of the shared network Jacobian."""
        cache = self._shared_jac_cache
        if cache["lin"] is None:
            cache["lin"] = self.obs_operator.linearize(xhat1.mean(dim=0, keepdim=True))
        return cache["lin"]

    def _score_statistical(
        self,
        x: torch.Tensor,
        xhat1: torch.Tensor,
        observations: torch.Tensor,
        beta: torch.Tensor,
        rho: torch.Tensor,
        sigma_s_apply: Optional[Callable],
        c_iso: Optional[torch.Tensor],
        c_jac: Optional[torch.Tensor],
        jvp_fn: Optional[Callable],
        first_order: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Statistical linearization of ``h = g(L x)`` (per member, any mode).

        With ``M = L Sigma_s L^T`` (built as the linear code builds ``H Sigma_s
        H^T`` with ``H := L``; same network-JVP cost), per sensor ``z_k ~
        N((L xhat1)_k, s_k^2)``, ``s_k^2 = M_kk / beta^2``, ``m = E g(z)``,
        ``D = diag(E g'(z))``:

            Sigma_bar = beta^2 R + D M D + beta^2 diag(Var g - D^2 s^2),
            Sbar      = (Sigma_s / rho) L^T D Sigma_bar^{-1} beta (y - m),

        the first-order formula with ``g, g'`` replaced by their Gaussian
        expectations plus the linearization-error variance. Identity ``g``
        gives the linear result.

        ``first_order=True``: the per-member Taylor linearization through the
        same code, ``m = g(L xhat1)``, ``D = diag(g'(L xhat1))``, no extra
        variance -- the ``inflated_shared`` route for a pointwise ``h`` (shared ``M``,
        per-member ``D``).
        """
        obs = self.obs_operator
        L = obs.inner
        beta_s = float(beta.reshape(-1)[0])
        beta_sq_R = beta_s**2 * self.original_variance
        rho_s = float(rho.reshape(-1)[0])

        if sigma_s_apply is None:
            M = rho_s * L.gram(x)
        elif self.share_sigma_s:
            M = self._shared_HSHt(x, jvp_fn, c_iso, c_jac, L)
        else:
            M = self._build_HSHt(x, sigma_s_apply, L)

        if first_order:
            z = L(xhat1)
            mh, D = obs.g(z), obs.dg(z)
            s2 = var = torch.zeros_like(z)
        else:
            s2 = torch.diagonal(M, dim1=-2, dim2=-1).clamp_min(0.0) / beta_s**2
            mh, D, var = _gauss_hermite_moments(obs, L(xhat1), s2)
        residual_obs = beta * (observations - mh)  # [B, N_y]

        Sigma_bar = D.unsqueeze(-1) * M * D.unsqueeze(-2)  # fresh [B, N_y, N_y]
        diag = Sigma_bar.diagonal(dim1=-2, dim2=-1)
        diag.add_(beta_sq_R)
        diag.add_(beta_s**2 * (var - D**2 * s2).clamp_min(0.0))
        if self.share_sigma_s and self.sigma_bar_eig_floor:
            JJt = D.unsqueeze(-1) * L.gram(x) * D.unsqueeze(-2)
            Sigma_bar = self._floor_sigma_bar(Sigma_bar, x, rho_s, beta_sq_R, HHt=JJt)

        Ht_sol = L.transpose(D * self._solve(Sigma_bar, residual_obs))  # J^T sol
        if sigma_s_apply is None:
            return Ht_sol, residual_obs
        sbar = (
            Ht_sol
            if self.isotropic_front_factor
            else sigma_s_apply(Ht_sol) / rho_s
        )
        return self._cap_step(sbar, residual_obs, lambda v: D * L(v)), residual_obs
    # ------------------------------------------------------------------
    # Full source covariance (Jacobian) operator
    # ------------------------------------------------------------------
    def _sigma_s_operator(
        self,
        x: torch.Tensor,
        t_clamped: torch.Tensor,
        t_grid: torch.Tensor,
        t_net: torch.Tensor,
        rho: torch.Tensor,
        field_history: Optional[torch.Tensor],
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> tuple[Optional[Callable], Any, Any, Optional[Callable]]:
        """``(sigma_s_apply, c_iso, c_jac, jvp_fn)`` for the full modes.

        ``sigma_s_apply(v) = c_iso v + c_jac J v`` (``None`` everywhere for
        ``dps_jacobian_free``). ``inflated_shared`` evaluates the Jacobian ONCE
        at the ensemble mean (pseudo-time tau is identical across members, so
        the schedule scalars are unchanged; only the state is averaged),
        refreshed every ``jacobian_refresh_every``-th pseudo-step.
        """
        if not self.use_full_sigma_s:
            return None, None, None, None
        if self.share_sigma_s:
            c_iso, c_jac = self._sigma_s_coeffs(
                t_grid[:1], rho[:1] if rho.dim() > 0 else rho
            )
            c_jac = c_jac * self.jacobian_damping  # lambda-damp J (solve + front)
            jvp_fn = self._refresh_shared_jacobian(
                x=x,
                t_clamped=t_clamped,
                t_net=t_net,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )
        else:
            c_iso, c_jac = self._sigma_s_coeffs(t_grid, rho)
            c_jac = c_jac * self.jacobian_damping  # lambda-damp J (solve + front)
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

        return sigma_s_apply, c_iso, c_jac, jvp_fn

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
            cache["lin"] = None  # nonlinear h: re-linearize lazily
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
        mc_chunk: Optional[int] = None,
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
            mc_chunk: How many of the ``J`` Monte-Carlo members are materialised
                (and pushed through ``obs_operator``) AT ONCE. ``None`` (default)
                means "all J", i.e. exactly the unchunked code path. A finite
                value caps the transient at ``[chunk, B, C, H, W]`` instead of
                ``[J, B, C, H, W]`` and the observation-operator batch at
                ``chunk * B`` instead of ``J * B`` -- required at urban scale
                (J=25, B=64, C=4, 128x128 is a 400 MiB block per tensor and
                1600 samples through the operator, which OOMs a 24 GB card).
                The estimator is unchanged: ``eps`` is still drawn in a single
                ``randn`` call over the full ``[J, B, C, H, W]`` block (identical
                random stream) and the importance-weight softmax stays GLOBAL
                over all ``J``; only the observation-operator evaluation and the
                gradient accumulation are split. Costs one extra chunked forward
                pass through ``obs_operator`` (the weights must be known before
                the differentiated pass). ``mc_chunk >= J`` (and ``None``) is
                BITWISE the old result; a genuine split is bitwise-exact in the
                weights but not in the score, because summing the per-member
                gradients chunk-by-chunk regroups a float sum -- measured at
                <= 1.4 ULP of ``max|score|`` (rel. 3e-16 in f64, 1.6e-7 in f32),
                i.e. pure associativity noise, not a different estimator.
        """
        super(FlowdasGaussianLikelihood, self).__init__()
        self.obs_operator = obs_operator
        self.model = model
        self.original_variance = variance
        # DA ensemble size E (interface only; NOT the Monte-Carlo count).
        self.ensemble_size = ensemble_size
        # J = number of one-step x_1 predictions per member (paper Eq. 10).
        self.num_mc_samples = int(num_mc_samples) if num_mc_samples is not None else 25
        # Monte-Carlo chunk over J (memory vs one extra obs-operator pass).
        # None -> no chunking, i.e. the historical single-shot code path.
        self.mc_chunk = int(mc_chunk) if mc_chunk is not None else None
        if self.mc_chunk is not None and self.mc_chunk < 1:
            raise ValueError(f"mc_chunk must be >= 1 or None, got {mc_chunk!r}.")
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

    def _sq_obs_err(
        self,
        mu_x1: torch.Tensor,
        eps: torch.Tensor,
        s: torch.Tensor,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Squared observation residuals ``||y - H(mu_x1 + s eps_j)||^2``.

        Evaluated for the ``k`` Monte-Carlo members carried by ``eps``
        (``[k, B, C, H, W]``); returns ``[k, B]``. Every operation here is
        elementwise or a reduction over the OBSERVATION axis only, so slicing
        ``eps`` along its leading (member) axis and concatenating the results is
        bitwise the same as one call with all ``J`` members -- which is what
        makes the chunked path in :meth:`score` an exact refactor rather than an
        approximation. This is the only place the ``[k, B, C, H, W]`` block and
        the ``k * B``-sample operator batch are materialised, hence the only
        place the memory peak has to be controlled.
        """
        k = eps.shape[0]
        b, c, h, w = mu_x1.shape
        preds = mu_x1.unsqueeze(0) + s * eps  # [k, B, C, H, W]
        Hpreds = self.obs_operator(preds.reshape(k * b, c, h, w)).reshape(k, b, -1)
        residual = observations.unsqueeze(0) - Hpreds  # [k, B, N_y]
        return (residual**2).sum(dim=-1)  # [k, B]

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

        # How many MC members to materialise at once (None -> all J at once,
        # the historical path). Clamped to J so `mc_chunk >= J` costs nothing.
        chunk = n_mc if self.mc_chunk is None else min(self.mc_chunk, n_mc)

        # Detached grad-enabled leaf: the score flows through mu_x1(x_g) but the
        # caller's working tensor is untouched.
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            mu_x1 = self._denoiser_mean(
                x_g, t, field_history, field_cond, pars_cond, drift
            )
            b = mu_x1.shape[0]  # mu_x1 is [B, C, H, W]; C/H/W only matter inside
            #                     _sq_obs_err, which re-reads them from mu_x1.

            # J one-step predictions with DETACHED source noise; the x_g-dependence
            # is ONLY through mu_x1 (eps_j are constants). Drawn in ONE randn call
            # over the full [J, B, C, H, W] block whether or not the evaluation is
            # chunked: eps is not what overflows the card (the obs_operator over
            # J*B samples is), and per-chunk draws would silently move the random
            # stream, so every chunked result would differ from every unchunked one.
            eps = torch.randn(
                (n_mc, *mu_x1.shape), device=mu_x1.device, dtype=mu_x1.dtype
            )

            if chunk >= n_mc:
                # --- unchunked: one shot, exactly as before.
                sq_err = self._sq_obs_err(mu_x1, eps, s, observations)  # [J, B]

                # DETACHED importance weights (constants in the autograd graph).
                log_w = -0.5 * sq_err.detach() / R  # [J, B]
                weights = torch.softmax(log_w, dim=0)  # [J, B]

                # log p(y|x_tau) with the 1/(2 sigma^2) factor kept; differentiate
                # THROUGH the predictor (weights detached) to get the raw score. The
                # -1/(2R) sign makes `grad` already point UP the log-likelihood, so
                # the posterior ascends with a positive step size zeta.
                log_lik = -0.5 * (weights * sq_err).sum(dim=0).sum() / R  # scalar
                grad = torch.autograd.grad(outputs=log_lik, inputs=x_g)[0]
            else:
                # --- chunked over J, in two passes. The weights are a softmax
                # over ALL J members, so they cannot be formed chunk-by-chunk:
                # pass 1 assembles the (cheap, [J, B]) squared errors WITHOUT a
                # graph, the softmax is then taken globally, and pass 2 rebuilds
                # the same quantities inside the graph one chunk at a time. The
                # extra forward pass buys the memory bound; the arithmetic is the
                # unchunked one element for element (see _sq_obs_err).
                with torch.no_grad():
                    sq_err = torch.cat(
                        [
                            self._sq_obs_err(
                                mu_x1, eps[j0 : j0 + chunk], s, observations
                            )
                            for j0 in range(0, n_mc, chunk)
                        ],
                        dim=0,
                    )  # [J, B]

                # GLOBAL softmax over all J (never per chunk). Already detached.
                log_w = -0.5 * sq_err / R  # [J, B]
                weights = torch.softmax(log_w, dim=0)  # [J, B]

                # log_lik is a plain SUM over j of per-member terms, so its
                # gradient is the sum of the per-chunk gradients. Accumulate that
                # sum at mu_x1 rather than at x_g: the denoiser then gets ONE
                # backward pass (two model evaluations), exactly as unchunked,
                # instead of one per chunk. Each chunk's graph is freed by its own
                # grad() call, which is what keeps the peak at chunk size; the
                # graph UPSTREAM of mu_x1 is never traversed here, so it survives
                # for the final backward. Regrouping that sum is the ONE thing
                # chunking cannot keep bitwise (float addition is not
                # associative); the measured drift is <= 1.4 ULP of max|score|.
                g_mu = torch.zeros_like(mu_x1)
                for j0 in range(0, n_mc, chunk):
                    sq_err_c = self._sq_obs_err(
                        mu_x1, eps[j0 : j0 + chunk], s, observations
                    )
                    log_lik_c = (
                        -0.5
                        * (weights[j0 : j0 + chunk] * sq_err_c).sum(dim=0).sum()
                        / R
                    )
                    g_mu = g_mu + torch.autograd.grad(log_lik_c, mu_x1)[0]
                grad = torch.autograd.grad(
                    outputs=mu_x1, inputs=x_g, grad_outputs=g_mu
                )[0]

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
