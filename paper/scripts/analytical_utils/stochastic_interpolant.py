from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm


class AnalyticalDriftModel(nn.Module):
    """Analytical drift model."""

    def __init__(
        self,
        interpolation: nn.Module,
        target_mean: Callable,
        target_cov: Callable,
        diffusion_term: Callable,
    ) -> None:
        """Initialize analytical drift model."""
        super(AnalyticalDriftModel, self).__init__()
        self.interpolation = interpolation
        self.target_mean = target_mean
        self.target_cov = target_cov
        self.diffusion_term = diffusion_term

    def _get_coefs(self, t: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Get the coefficients."""
        alpha = self.interpolation.alpha(t)
        alpha_diff = self.interpolation.alpha_diff(t)
        beta = self.interpolation.beta(t)
        beta_diff = self.interpolation.beta_diff(t)
        gamma = self.interpolation.gamma(t)
        gamma_diff = self.interpolation.gamma_diff(t)
        return alpha, alpha_diff, beta, beta_diff, gamma, gamma_diff

    def _compute_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the drift."""

        I = torch.eye(x0.shape[1]).expand(x0.shape[0], x0.shape[1], x0.shape[1])

        alpha, alpha_diff, beta, beta_diff, gamma, gamma_diff = self._get_coefs(t)

        out = alpha_diff * x0 + beta_diff * self.target_mean(x0)

        if t.any() < 1e-8:
            return out

        mean_bar = alpha * x0 + beta * self.target_mean(x0)

        cov_bar = (
            beta.unsqueeze(1) ** 2 * self.target_cov(x0)
            + (t * gamma**2).unsqueeze(1) * I
        )
        cov_bar_inv = torch.linalg.inv(cov_bar)

        out_2 = beta.unsqueeze(1) * beta_diff[0, 0] * self.target_cov(x0)
        out_2 = out_2 + (t * gamma * gamma_diff).unsqueeze(1) * I
        out_2 = torch.bmm(out_2, cov_bar_inv)
        out_2 = torch.bmm(out_2, (x - mean_bar).unsqueeze(-1)).squeeze(-1)

        return out + out_2

    def _compute_score_from_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        drift: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the score from the drift."""

        alpha, alpha_diff, beta, beta_diff, gamma, gamma_diff = self._get_coefs(t)

        A = t * gamma * (beta_diff * gamma - beta * gamma_diff)
        A = 1 / (A + 1e-6)

        c = beta_diff * x + (beta * alpha_diff - beta_diff * alpha) * x0

        return A * (beta * drift - c)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""

        drift = self._compute_drift(x, t, x0)
        # score = self._compute_score_from_drift(x, t, x0, drift)

        return (
            drift
            # + 0.5
            # * (self.diffusion_term(t) ** 2 - self.interpolation.gamma(t) ** 2)
            # * score
        )


class AnalyticalStochasticInterpolant(nn.Module):
    """Analytical stochastic interpolant."""

    def __init__(
        self,
        interpolation: nn.Module,
        drift_model: nn.Module,
        diffusion_term: Callable,
    ) -> None:
        """Initialize analytical stochastic interpolant."""
        super(AnalyticalStochasticInterpolant, self).__init__()
        self.interpolation = interpolation
        self.drift_model = drift_model
        self.diffusion_term = diffusion_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass

    def sample(
        self,
        x0: torch.Tensor,
        num_steps: int,
        diffusion_term: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Sample from the analytical stochastic interpolant."""

        if diffusion_term is None:
            diffusion_term = self.diffusion_term

        t_vec = torch.linspace(0, 1, num_steps).unsqueeze(0).repeat(x0.shape[0], 1)
        dt = torch.tensor(1 / num_steps)

        x = x0.clone()

        pbar = tqdm.tqdm(range(num_steps))
        for i in pbar:
            t = t_vec[:, i : i + 1]
            x = x + self.drift_model(x, t, x0) * dt
            x = x + self.diffusion_term(t) * torch.randn_like(x) * torch.sqrt(dt)
        return x
