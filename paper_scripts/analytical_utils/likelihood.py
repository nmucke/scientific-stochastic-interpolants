
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm
from torch.func import jacrev, vmap


class InterpolantLikelihood(nn.Module):
    """Interpolant likelihood.

    Computes the score of the interpolated observation likelihood,
        nabla_x log p(ybar_tau | x_tau, x_0),
    using the mean correction d_tau and (optionally) the covariance
    correction C_tau from Lemma 4.2 and Lemma 4.3.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        ensemble_size: int = 100,
        use_covariance_correction: bool = True,
        perturbation: Optional[str] = None,
        target_variance: float = 1.0,
        num_quad: int = 200,
    ) -> None:
        """Initialize interpolant likelihood.

        Args:
            obs_matrix: Observation matrix H, shape (N_y, N_u).
            drift_model: Trained (or analytical) drift model.
            original_variance: Scalar observation noise variance (R = original_variance * I).
            ensemble_size: Not used currently, kept for interface compatibility.
            use_covariance_correction: If True, include the C_tau correction
                from the conditional covariance of W_tau.  Requires computing
                the Jacobian of the drift, which costs O(N_u) backward passes.
            perturbation: Which LG likelihood correction to apply. One of
                ``None`` (no correction), ``"true"`` (exact analytical LG
                correction, requires forward integration of Phi and knowing
                the prior variance), ``"tangent"`` (cheap tangent-linear
                surrogate for Phi using ensemble variance -- Heuristic 1 of
                ``appendix_cheap_corrections.tex``), or ``"ensemble"``
                (ensemble-calibrated innovation variance, rescales the
                interpolant score magnitude -- Heuristic 3).
        """
        super(InterpolantLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size
        self.use_covariance_correction = use_covariance_correction
        if perturbation not in (
            None, "true", "tangent", "ensemble", "residual", "deint"
        ):
            raise ValueError(
                f"perturbation must be None, 'true', 'tangent', 'ensemble', "
                f"'residual', or 'deint', got {perturbation!r}"
            )
        self.perturbation = perturbation
        self.target_variance = target_variance
        self.num_quad = num_quad

    def forward(self) -> None:
        """Forward pass."""
        pass

    def _compute_W_covariance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the  Wcovariance

        Returns:
            W_cov: shape (batch, N_y, N_y).
        """
        gamma = self.drift_model.interpolation.gamma(t)[0,0].item()
        beta = self.drift_model.interpolation.beta(t)[0,0].item()
        beta_diff = self.drift_model.interpolation.beta_diff(t)[0,0].item()
        gamma_diff = self.drift_model.interpolation.gamma_diff(t)[0,0].item()

        A_tau = 1.0 / (
            t[0, 0] * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-8
        ).item()

        # --- Compute J_b via autograd on a detached copy of x ---

        x = x.unsqueeze(0)
        x0 = x0.unsqueeze(0)

        J_b = jacrev(
            lambda x: self.drift_model._compute_drift(x, t[0:1], x0).sum(dim=0),
        )(x)
        J_b = J_b.transpose(0, 1).squeeze(0)

        I = torch.eye(J_b.shape[-1], device=J_b.device, dtype=J_b.dtype)
        return t[0,0] * I + gamma**2 * t[0,0]**2 * A_tau * (beta * J_b - beta_diff * I)

    def mu_fn(self, x, x0, t, observations):
        """Compute the mean."""
        gamma = self.drift_model.interpolation.gamma(t)[0,0]
        beta = self.drift_model.interpolation.beta(t)[0,0]
        alpha = self.drift_model.interpolation.alpha(t)[0,0]

        i_obs = alpha * torch.matmul(self.obs_matrix, x0) + beta * observations

        x_obs = torch.matmul(self.obs_matrix, x)

        drift = self.drift_model._compute_drift(x.unsqueeze(0), t, x0.unsqueeze(0))
        model_score = self.drift_model._compute_score_from_drift(
            x.unsqueeze(0), t[0:1], x0.unsqueeze(0), drift
        ).squeeze(0)

        E_W = -gamma * t[0,0] * model_score  # E[W_tau | x_tau, x_0]
        d = -gamma * torch.matmul(self.obs_matrix, E_W)

        return i_obs - x_obs - d
    
    def sigma_fn(self, x, x0, t):
        """Compute the covariance."""

        gamma = self.drift_model.interpolation.gamma(t)[0,0].item()
        beta = self.drift_model.interpolation.beta(t)[0,0].item()

        R = beta ** 2 * self.original_variance

        cov_W = self._compute_W_covariance(x, t, x0)

        cov_W = self.obs_matrix@ cov_W @ self.obs_matrix.T
        cov_W = gamma**2 * cov_W

        sigma = R * torch.eye(
            self.obs_matrix.shape[0], device=x.device
        ) + cov_W

        return sigma

    def _phi_and_vcond(self, tau: float) -> tuple[float, float]:
        """Compute Phi_{1,tau} and V_cond(tau) by quadrature (LG case)."""
        interp = self.drift_model.interpolation
        c = self.target_variance

        s = torch.linspace(tau, 1.0, self.num_quad).view(-1, 1)
        beta = interp.beta(s).view(-1)
        beta_diff = interp.beta_diff(s).view(-1)
        gamma = interp.gamma(s).view(-1)
        gamma_diff = interp.gamma_diff(s).view(-1)
        s_flat = s.view(-1)

        num = beta * beta_diff * c + s_flat * gamma * gamma_diff
        den = beta ** 2 * c + s_flat * gamma ** 2 + 1e-12
        a_vals = num / den

        log_phi = torch.trapz(a_vals, s_flat)
        phi = torch.exp(log_phi).item()

        t_tensor = torch.tensor([[tau]])
        beta_t = interp.beta(t_tensor)[0, 0].item()
        gamma_t = interp.gamma(t_tensor)[0, 0].item()
        V_tau = beta_t ** 2 * c + tau * gamma_t ** 2
        V_cond = max(c - phi ** 2 * V_tau, 0.0)
        return phi, V_cond

    def likelihood_perturbation(self, x, x0, t, observations):
        """Analytical LG correction to the interpolant likelihood score.

        Adds the term that turns the interpolant likelihood score into the
        exact posterior score for the linear-Gaussian test case of
        ``appendix_simple_test_case.tex``:

            perturbation = A_tau H^T (y - H x_0)  -  B_tau H^T H (x_tau - x_0)

        with

            A_tau = Phi / v_S  -  beta^3 c / (v_I V_tau)
            B_tau = Phi^2 / v_S  -  beta^4 c^2 / (v_I V_tau^2)

        and

            V_tau  = beta^2 c + tau gamma^2                       (SI marginal var)
            v_S    = V_cond + sigma^2,   V_cond = c - Phi^2 V_tau (SDE conditional)
            v_I    = beta^2 sigma^2 + gamma^2 tau
                     + gamma^4 tau^2 A_code (beta a - beta_diff)  (matches sigma_fn)

        The J_b-correction piece (last term in v_I) is essential: the score
        function in this class uses the full ``sigma_fn`` covariance, not the
        J_b-free surrogate -- so the analytical correction must be computed
        against the same v_I that the Gaussian log-likelihood uses, otherwise
        there is a residual bias and the perturbation fails to cancel.

        Assumptions: isotropic Gaussian prior N(x_0, c I), linear analytical
        drift (so J_b = a(tau) * I is a scalar), and observation operator H
        such that H^T H is isotropic (the test case uses H = I). Outside this
        setting the correction is only locally valid.
        """
        tau = t[0, 0].item()
        phi, V_cond = self._phi_and_vcond(tau)

        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        beta_diff = interp.beta_diff(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        gamma_diff = interp.gamma_diff(t)[0, 0].item()

        c = self.target_variance
        sigma2 = self.original_variance

        V_tau = beta ** 2 * c + tau * gamma ** 2
        v_S = V_cond + sigma2

        # Full scalar v_I matching sigma_fn: analytical drift Jacobian
        # J_b = a(tau) I enters the covariance correction as
        #   gamma^4 tau^2 A_code (beta a - beta_diff).
        a_tau = (beta * beta_diff * c + tau * gamma * gamma_diff) / V_tau
        A_code = 1.0 / (
            tau * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-12
        )
        jac_correction = gamma ** 4 * tau ** 2 * A_code * (beta * a_tau - beta_diff)
        v_I = beta ** 2 * sigma2 + gamma ** 2 * tau + jac_correction

        A_tau = phi / v_S - (beta ** 3 * c) / (v_I * V_tau)
        B_tau = phi ** 2 / v_S - (beta ** 4 * c ** 2) / (v_I * V_tau ** 2)

        H = self.obs_matrix
        y = observations[0]                               # (N_y,)
        innovation = y.unsqueeze(0) - x0 @ H.T            # (B, N_y)
        displacement = (x - x0) @ (H.T @ H)               # (B, N_u)

        return A_tau * (innovation @ H) - B_tau * displacement

    def _perturbation_tangent(self, x, x0, t, observations):
        """Heuristic 1: tangent-linear surrogate for Phi using ensemble variance.

        Keeps the additive LG form of ``likelihood_perturbation`` but replaces
        the quadrature-based ``Phi_{1,tau}`` by a one-step Taylor expansion
        ``Phi ~ 1 + (1-tau) * a_hat(tau)`` and replaces the analytical marginal
        variance ``V_tau`` by the mean per-coordinate ensemble variance. No
        forward integration, no extra sampling.
        """
        tau = t[0, 0].item()
        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        beta_diff = interp.beta_diff(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        gamma_diff = interp.gamma_diff(t)[0, 0].item()

        c = self.target_variance
        sigma2 = self.original_variance

        x_centered = x - x.mean(dim=0, keepdim=True)
        V_tau_hat = (x_centered ** 2).sum(dim=-1).mean().item() / x.shape[-1]
        V_tau_hat = max(V_tau_hat, 1e-6)

        a_tau = (beta * beta_diff * c + tau * gamma * gamma_diff) / V_tau_hat
        phi = 1.0 + (1.0 - tau) * a_tau
        V_cond = max(c - phi ** 2 * V_tau_hat, 0.0)
        v_S = V_cond + sigma2

        A_code = 1.0 / (
            tau * gamma * (beta_diff * gamma - beta * gamma_diff) + 1e-12
        )
        jac_correction = gamma ** 4 * tau ** 2 * A_code * (beta * a_tau - beta_diff)
        v_I = beta ** 2 * sigma2 + gamma ** 2 * tau + jac_correction

        A_tau = phi / v_S - (beta ** 3 * c) / (v_I * V_tau_hat)
        B_tau = phi ** 2 / v_S - (beta ** 4 * c ** 2) / (v_I * V_tau_hat ** 2)

        H = self.obs_matrix
        y = observations[0]
        innovation = y.unsqueeze(0) - x0 @ H.T
        displacement = (x - x0) @ (H.T @ H)

        return A_tau * (innovation @ H) - B_tau * displacement

    def _perturbation_ensemble_factor(self, x, t) -> float:
        """Heuristic 3: scalar rescaling by ensemble-calibrated v_S.

        Returns ``v_I / hat_v_S`` with
            hat_v_S = sigma^2 + (1/N_y) tr(H Cov_hat(x_tau) H^T),
        estimated from the current ensemble, and ``v_I = beta^2 sigma^2 +
        gamma^2 tau`` (the paper's J_b-free surrogate in scalar form). The
        score returned by ``score()`` is multiplied by this factor to calibrate
        its magnitude against the actual spread of the ensemble in observation
        space. This corrects only the scale of the interpolant score, not its
        slope, cf. ``appendix_cheap_corrections.tex``.
        """
        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        gamma = interp.gamma(t)[0, 0].item()
        tau = t[0, 0].item()
        sigma2 = self.original_variance

        v_I = beta ** 2 * sigma2 + gamma ** 2 * tau

        H = self.obs_matrix
        x_centered = x - x.mean(dim=0, keepdim=True)
        Hx = x_centered @ H.T
        v_S_hat = sigma2 + (Hx ** 2).sum(dim=-1).mean().item() / H.shape[0]

        return v_I / max(v_S_hat, 1e-8)

    def _perturbation_residual(self, x, x0, t, observations):
        """Heuristic 4: ensemble residual matching.

        Pairs a one-step Tweedie endpoint prediction with the H3 variance
        calibration, using the actual observation through the ensemble
        residual ``r^(i) = y - H (x_tau^(i) + (1-tau) drift_theta^(i))``. The
        correction is additive in the LG form of
        ``eq:lg_score_correction_final``:

            A_tau H^T r_bar  -  (beta^2 / v_S_hat) H^T H (x_tau - x_0),

        with ``A_tau = 1 / v_S_hat`` and
        ``v_S_hat = sigma^2 + (1/N_y) tr Cov_hat(H x1_hat)``. No auxiliary
        ensembles or Jacobians: the drift is already being evaluated at this
        step.
        """
        tau = t[0, 0].item()
        interp = self.drift_model.interpolation
        beta = interp.beta(t)[0, 0].item()
        sigma2 = self.original_variance

        drift = self.drift_model._compute_drift(x, t[0:1], x0)
        x1_hat = x + (1.0 - tau) * drift                       # (B, N_u)

        H = self.obs_matrix
        y = observations[0]
        Hx1 = x1_hat @ H.T                                     # (B, N_y)

        residuals = y.unsqueeze(0) - Hx1                       # (B, N_y)
        r_bar = residuals.mean(dim=0)                          # (N_y,)

        Hx1_centered = Hx1 - Hx1.mean(dim=0, keepdim=True)
        v_S_hat = sigma2 + (Hx1_centered ** 2).sum(dim=-1).mean().item() / H.shape[0]
        v_S_hat = max(v_S_hat, 1e-8)

        A_tau = 1.0 / v_S_hat
        B_tau = beta ** 2 / v_S_hat

        displacement = (x - x0) @ (H.T @ H)
        innovation_term = (r_bar @ H).unsqueeze(0).expand_as(displacement)

        return A_tau * innovation_term - B_tau * displacement

    def _perturbation_deinterpolation(self, x, x0, t, observations):
        """Heuristic 5: drift-free de-interpolation residual.

        Uses the interpolation identity
            x_tau = alpha * x_0 + beta * x_1 + gamma * W_tau
        to form a drift-free endpoint estimate
            x1_hat = (x_tau - alpha * x_0) / beta
        (dropping the zero-mean Wiener term), then feeds it through the
        same residual/variance calibration as ``_perturbation_residual``.
        No drift evaluations, no Jacobians -- purely algebraic.
        """
        tau = t[0, 0].item()
        interp = self.drift_model.interpolation
        alpha = interp.alpha(t)[0, 0].item()
        beta = interp.beta(t)[0, 0].item()
        sigma2 = self.original_variance

        beta_safe = beta if abs(beta) > 1e-3 else 1e-3
        x1_hat = (x - alpha * x0) / beta_safe                 # (B, N_u)

        H = self.obs_matrix
        y = observations[0]
        Hx1 = x1_hat @ H.T                                    # (B, N_y)

        residuals = y.unsqueeze(0) - Hx1
        r_bar = residuals.mean(dim=0)                         # (N_y,)

        Hx1_centered = Hx1 - Hx1.mean(dim=0, keepdim=True)
        v_S_hat = sigma2 + (Hx1_centered ** 2).sum(dim=-1).mean().item() / H.shape[0]
        v_S_hat = max(v_S_hat, 1e-8)

        A_tau = 1.0 / v_S_hat
        B_tau = beta ** 2 / v_S_hat

        displacement = (x - x0) @ (H.T @ H)
        innovation_term = (r_bar @ H).unsqueeze(0).expand_as(displacement)

        return A_tau * innovation_term - B_tau * displacement

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Compute the likelihood score nabla_x log p(ybar_tau | x_tau, x_0).

        The score is obtained by constructing the Gaussian log-likelihood
        with the corrected mean (d_tau) and optionally the corrected
        covariance (C_tau), then differentiating w.r.t. x via autograd.
        """

        diff_fun = lambda x, x0: self.mu_fn(x, x0, t, observations[0])
        diff_fun_vmap = lambda x, x0: torch.vmap(
            diff_fun, in_dims=0, out_dims=0
        )(x, x0)
        diff = diff_fun_vmap(x, x0)

        sigma_fn = lambda x, x0: self.sigma_fn(x, x0, t)
        Sigma = vmap(sigma_fn, in_dims=0, out_dims=0)(x, x0)        # (B, d_y, d_y)

        log_prb_fun = lambda diff, Sigma: (
            -0.5 * torch.dot(diff, torch.linalg.solve(Sigma, diff))
        )

        log_prb = torch.vmap(log_prb_fun, in_dims=0, out_dims=0)(diff, Sigma)

        score = torch.autograd.grad(log_prb.sum(), x)[0]

        if self.perturbation == "true":
            score = score + self.likelihood_perturbation(x, x0, t, observations)
        elif self.perturbation == "tangent":
            score = score + self._perturbation_tangent(x, x0, t, observations)
        elif self.perturbation == "ensemble":
            score = score * self._perturbation_ensemble_factor(x, t)
        elif self.perturbation == "residual":
            score = score + self._perturbation_residual(x, x0, t, observations)
        elif self.perturbation == "deint":
            score = score + self._perturbation_deinterpolation(
                x, x0, t, observations
            )

        return score


class SDEConditionalLikelihood(nn.Module):
    """Likelihood using the SDE conditional p_SDE(x_1 | x_tau, x_0).

    Unlike ``InterpolantLikelihood`` (which uses the interpolation joint
    from Lemma 4.2/4.3), this class computes the exact Doob h-transform
    of the prior SDE for the isotropic linear-Gaussian test case:

        p(x_0) = N(mu_0, c * I),  target_mean(x_0) = x_0.

    Under that assumption the prior drift is affine,
        b(x, tau, x_0) = a(tau) * (x - x_0),
    with
        a(tau) = [beta*beta' * c + tau*gamma*gamma'] / [beta^2 * c + tau*gamma^2].

    The SDE transition is then closed form and gives
        x_1 | x_tau, x_0 ~ N( x_0 + Phi * (x_tau - x_0),  V_cond * I ),
    where Phi = exp(int_tau^1 a(s) ds) and
          V_cond = c - Phi^2 * V_tau,  V_tau = beta^2 c + tau gamma^2.

    The resulting Gaussian likelihood score yields the exact LG posterior.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        target_variance: float = 1.0,
        num_quad: int = 200,
    ) -> None:
        super(SDEConditionalLikelihood, self).__init__()
        self.obs_matrix = obs_matrix
        self.drift_model = drift_model
        self.original_variance = original_variance
        self.target_variance = target_variance
        self.num_quad = num_quad

    def forward(self) -> None:
        """Forward pass."""
        pass

    def _phi_and_vcond(self, tau: float) -> tuple[float, float]:
        """Compute Phi_{1,tau} and V_cond(tau) by quadrature."""
        interp = self.drift_model.interpolation
        c = self.target_variance

        s = torch.linspace(tau, 1.0, self.num_quad).view(-1, 1)
        beta = interp.beta(s).view(-1)
        beta_diff = interp.beta_diff(s).view(-1)
        gamma = interp.gamma(s).view(-1)
        gamma_diff = interp.gamma_diff(s).view(-1)
        s_flat = s.view(-1)

        num = beta * beta_diff * c + s_flat * gamma * gamma_diff
        den = beta ** 2 * c + s_flat * gamma ** 2 + 1e-12
        a_vals = num / den

        log_phi = torch.trapz(a_vals, s_flat)
        phi = torch.exp(log_phi).item()

        t_tensor = torch.tensor([[tau]])
        beta_t = interp.beta(t_tensor)[0, 0].item()
        gamma_t = interp.gamma(t_tensor)[0, 0].item()
        V_tau = beta_t ** 2 * c + tau * gamma_t ** 2
        V_cond = max(c - phi ** 2 * V_tau, 0.0)
        return phi, V_cond

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        """Score nabla_x log p(y | x_tau, x_0) under the SDE conditional."""
        tau = t[0, 0].item()
        phi, V_cond = self._phi_and_vcond(tau)

        x1_mean = x0 + phi * (x - x0)                       # (B, N_u)
        y_mean = torch.matmul(x1_mean, self.obs_matrix.T)   # (B, N_y)

        H = self.obs_matrix
        HHt = H @ H.T
        I_y = torch.eye(H.shape[0], device=x.device, dtype=x.dtype)
        Sigma = V_cond * HHt + self.original_variance * I_y  # (N_y, N_y)

        diff = observations[0].unsqueeze(0) - y_mean         # (B, N_y)
        sol = torch.linalg.solve(Sigma, diff.T).T            # (B, N_y)
        log_prb = -0.5 * (diff * sol).sum(dim=-1)

        return torch.autograd.grad(log_prb.sum(), x)[0]


class LinearizedDriftLikelihood(nn.Module):
    """Option 1: single linearization of the learned drift.

    At the current state (x_tau, tau), compute b = b_theta(x_tau, x_0, tau)
    and J = d b_theta / d x |_{x_tau}, then treat the SDE as affine-linear
    from tau to 1 with these frozen coefficients.  The mean and covariance
    of x_1 | x_tau, x_0 are propagated by Euler integration of
        dm/ds = b + J (m - x_tau),            m(tau) = x_tau
        dP/ds = J P + P J^T + gamma_s^2 I,    P(tau) = 0
    and the resulting Gaussian is used as the SDE conditional.

    Generalizes ``SDEConditionalLikelihood`` to learned / non-isotropic
    drifts -- no need to know target_mean / target_cov.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        num_quad: int = 50,
    ) -> None:
        super(LinearizedDriftLikelihood, self).__init__()
        self.obs_matrix = obs_matrix
        self.drift_model = drift_model
        self.original_variance = original_variance
        self.num_quad = num_quad

    def forward(self) -> None:
        """Forward pass."""
        pass

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        tau = t[0, 0].item()
        B, d = x.shape
        t_b = t[0:1]

        b0 = self.drift_model._compute_drift(x, t_b, x0)  # (B, d)

        def single_drift(xi, x0i):
            return self.drift_model._compute_drift(
                xi.unsqueeze(0), t_b, x0i.unsqueeze(0)
            ).squeeze(0)

        J = vmap(jacrev(single_drift, argnums=0))(x, x0)  # (B, d, d)

        num_q = max(self.num_quad, 1)
        ds = (1.0 - tau) / num_q
        s_nodes = torch.linspace(tau, 1.0 - ds, num_q).view(-1, 1)
        gamma_vals = self.drift_model.interpolation.gamma(s_nodes).view(-1)

        m = x.clone()
        P = torch.zeros(B, d, d, device=x.device, dtype=x.dtype)
        I_d = torch.eye(d, device=x.device, dtype=x.dtype).expand(B, d, d)

        for k in range(num_q):
            g2 = gamma_vals[k] ** 2
            residual = (m - x).unsqueeze(-1)
            Jr = torch.bmm(J, residual).squeeze(-1)
            m = m + (b0 + Jr) * ds
            JP = torch.bmm(J, P)
            PJT = torch.bmm(P, J.transpose(-1, -2))
            P = P + (JP + PJT + g2 * I_d) * ds

        H = self.obs_matrix
        y_mean = m @ H.T
        HP = torch.einsum('ij,bjk->bik', H, P)
        HPH = torch.einsum('bik,lk->bil', HP, H)
        I_y = torch.eye(H.shape[0], device=x.device, dtype=x.dtype)
        Sigma = HPH + self.original_variance * I_y

        diff = observations[0].unsqueeze(0) - y_mean
        sol = torch.linalg.solve(Sigma, diff.unsqueeze(-1)).squeeze(-1)
        log_prb = -0.5 * (diff * sol).sum(dim=-1)

        return torch.autograd.grad(log_prb.sum(), x)[0]


class MultiStepLinearizedDriftLikelihood(nn.Module):
    """Option 3: K-step re-linearization of the learned drift.

    Splits [tau, 1] into ``num_substeps`` segments.  On each segment,
    the drift is linearized around the current mean at the start of
    the segment (re-computing b_theta and J = d b_theta / d x there),
    and then the linear SDE is integrated across the segment by
    ``num_euler_per_substep`` Euler steps.  More accurate than
    ``LinearizedDriftLikelihood`` when the drift is strongly nonlinear
    or when (1 - tau) is large.  Cost is ~K times higher.
    """

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        num_substeps: int = 5,
        num_euler_per_substep: int = 10,
    ) -> None:
        super(MultiStepLinearizedDriftLikelihood, self).__init__()
        self.obs_matrix = obs_matrix
        self.drift_model = drift_model
        self.original_variance = original_variance
        self.num_substeps = num_substeps
        self.num_euler_per_substep = num_euler_per_substep

    def forward(self) -> None:
        """Forward pass."""
        pass

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Callable,
    ) -> torch.Tensor:
        tau = t[0, 0].item()
        B, d = x.shape
        num_sub = max(self.num_substeps, 1)
        num_euler = max(self.num_euler_per_substep, 1)
        sub_ds = (1.0 - tau) / num_sub
        inner_ds = sub_ds / num_euler

        m = x.clone()
        P = torch.zeros(B, d, d, device=x.device, dtype=x.dtype)
        I_d = torch.eye(d, device=x.device, dtype=x.dtype).expand(B, d, d)

        for k in range(num_sub):
            tau_k = tau + k * sub_ds
            t_k = torch.tensor([[tau_k]], device=x.device, dtype=x.dtype)

            b_k = self.drift_model._compute_drift(m, t_k, x0)  # (B, d)

            def single_drift(mi, x0i):
                return self.drift_model._compute_drift(
                    mi.unsqueeze(0), t_k, x0i.unsqueeze(0)
                ).squeeze(0)

            J_k = vmap(jacrev(single_drift, argnums=0))(m, x0)  # (B, d, d)

            m_ref = m

            for e in range(num_euler):
                s_e = tau_k + e * inner_ds
                s_tensor = torch.tensor([[s_e]], device=x.device, dtype=x.dtype)
                g = self.drift_model.interpolation.gamma(s_tensor)[0, 0]
                g2 = g ** 2

                residual = (m - m_ref).unsqueeze(-1)
                Jr = torch.bmm(J_k, residual).squeeze(-1)
                m = m + (b_k + Jr) * inner_ds
                JP = torch.bmm(J_k, P)
                PJT = torch.bmm(P, J_k.transpose(-1, -2))
                P = P + (JP + PJT + g2 * I_d) * inner_ds

        H = self.obs_matrix
        y_mean = m @ H.T
        HP = torch.einsum('ij,bjk->bik', H, P)
        HPH = torch.einsum('bik,lk->bil', HP, H)
        I_y = torch.eye(H.shape[0], device=x.device, dtype=x.dtype)
        Sigma = HPH + self.original_variance * I_y

        diff = observations[0].unsqueeze(0) - y_mean
        sol = torch.linalg.solve(Sigma, diff.unsqueeze(-1)).squeeze(-1)
        log_prb = -0.5 * (diff * sol).sum(dim=-1)

        return torch.autograd.grad(log_prb.sum(), x)[0]


class FlowdasLikelihood(nn.Module):
    """Interpolant likelihood."""

    def __init__(
        self,
        obs_matrix: torch.Tensor,
        drift_model: nn.Module,
        original_variance: float,
        ensemble_size: int = 25,
        multiplier: float = 4.0,
    ) -> None:
        """Initialize interpolant likelihood."""
        super(FlowdasLikelihood, self).__init__()
        self.drift_model = drift_model
        self.obs_matrix = obs_matrix
        self.original_variance = original_variance
        self.ensemble_size = ensemble_size
        self.multiplier = multiplier

    def forward(
        self,
    ) -> None:
        """Forward pass."""
        pass

    def _compute_one_step_prediction(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the one step prediction."""

        drift_milstein = self.drift_model(x, t, x0)
        pred = x + drift_milstein * (1.0 - t)
        # Add noise = integral of the diffusion term from t to 1
        pred = pred + torch.randn_like(x) * (
            2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3
        )
        # RK step
        drift_rk = self.drift_model(pred, torch.ones_like(t), x0)
        pred = x + 0.5 * (drift_milstein + drift_rk) * (1 - t)

        # Expand the prediction to the ensemble size
        pred = pred.repeat(self.ensemble_size, 1, 1)

        # Add noise = integral of the diffusion term from t to 1
        return pred + torch.randn_like(pred) * (
            2 / 3 - t.sqrt() + (1 / 3) * (t.sqrt()) ** 3
        )

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        observations: torch.Tensor,
        dt: torch.Tensor,
        diffusion_term: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Score function."""
        preds = self._compute_one_step_prediction(x, t, dt, x0)

        pred_obs = torch.matmul(preds, self.obs_matrix.T)

        diff = pred_obs - observations.unsqueeze(0)
        diff_norm = torch.linalg.norm(diff, dim=-1)
        diff_norm = -diff_norm / (2 * self.original_variance)

        weights = torch.softmax(diff_norm.detach() ** 2, dim=0)

        return (
            torch.autograd.grad((diff_norm * weights).sum(), x)[0]
            * dt
            * self.multiplier
        )