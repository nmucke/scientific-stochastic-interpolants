# import torch
# import torch.nn as nn
# import tqdm
# from typing import Optional
# from paper_scripts.analytical_utils.likelihood import InterpolantLikelihood

# class PosteriorModel(nn.Module):
#     """Posterior model."""

#     def __init__(
#         self,
#         drift_model: nn.Module,
#         likelihood_model: nn.Module,
#     ) -> None:
#         """Initialize posterior model."""
#         super(PosteriorModel, self).__init__()
#         self.drift_model = drift_model
#         self.likelihood_model = likelihood_model

#     def forward(
#         self,
#     ) -> None:
#         """Forward pass."""
#         pass

#     def sample(
#         self,
#         x0: torch.Tensor,
#         num_steps: int,
#         observations: torch.Tensor,
#         diffusion_term: Optional[nn.Module] = None,
#     ) -> torch.Tensor:
#         """Sample from the posterior model."""
#         if diffusion_term is None:
#             diffusion_term = self.drift_model.diffusion_term

#         t_vec = torch.linspace(0, 1, num_steps).unsqueeze(0).repeat(x0.shape[0], 1)
#         dt = torch.tensor(1 / num_steps)

#         x = x0.clone()

#         pbar = tqdm.tqdm(range(num_steps - 1))
#         for i in pbar:
#             t = t_vec[:, i : i + 1]
#             new = x + self.drift_model(x, t, x0) * dt
#             new = new + diffusion_term(t) * torch.randn_like(x) * torch.sqrt(dt)

#             if t.any() < 1e-6:
#                 x = new
#                 continue
#             x.requires_grad = True

#             score = self.likelihood_model.score(
#                 x, t, x0, observations, dt, diffusion_term
#             )

#             if isinstance(self.likelihood_model, InterpolantLikelihood):
#                 x = new + score * diffusion_term(t) ** 2 * dt 
#             else:
#                 x = new + score
#             x = x.detach()
#         return x

import torch
import torch.nn as nn
import tqdm
from typing import Optional
from paper_scripts.analytical_utils.likelihood import InterpolantLikelihood


class PosteriorModel(nn.Module):
    """Posterior model."""

    def __init__(
        self,
        drift_model: nn.Module,
        likelihood_model: nn.Module,
    ) -> None:
        """Initialize posterior model."""
        super(PosteriorModel, self).__init__()
        self.drift_model = drift_model
        self.likelihood_model = likelihood_model

    def forward(
        self,
    ) -> None:
        """Forward pass."""
        pass

    def sample(
        self,
        x0: torch.Tensor,
        num_steps: int,
        observations: torch.Tensor,
        diffusion_term: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Sample from the posterior model.

        Solves the posterior SDE (Theorem 4.1) via Euler-Maruyama:

            x_{i+1} = x_i + b_theta(x_i, x_0, t_i) * dt
                     + gamma_t^2 * score(x_i) * dt
                     + gamma_t * sqrt(dt) * z

        where gamma_t is the interpolant diffusion coefficient.

        If diffusion_term is provided, it overrides gamma_t for the noise
        and score scaling. This is correct only when the tunable diffusion
        drift correction is active in the drift model.
        """
        t_vec = torch.linspace(0, 1, num_steps).unsqueeze(0).repeat(x0.shape[0], 1)
        dt = torch.tensor(1.0 / num_steps)

        x = x0.clone()

        # Use interpolation gamma by default (consistent with Theorem 4.1).
        # Fall back to diffusion_term if explicitly provided (tunable diffusion).
        gamma_fn = self.drift_model.interpolation.gamma
        if diffusion_term is not None:
            gamma_fn = diffusion_term

        pbar = tqdm.tqdm(range(num_steps - 1))
        for i in pbar:
            t = t_vec[:, i : i + 1]

            gamma_t = gamma_fn(t)

            # Prior drift + noise
            new = x + self.drift_model(x, t, x0) * dt
            new = new + gamma_t * torch.randn_like(x) * torch.sqrt(dt)

            # Skip likelihood correction at tau ~ 0
            if (t < 1e-6).all():
                x = new
                continue

            x.requires_grad = True

            score = self.likelihood_model.score(
                x, t, x0, observations, dt, gamma_fn
            )

            if isinstance(self.likelihood_model, InterpolantLikelihood):
                x = new + score * gamma_t**2 * dt
            else:
                x = new + score
            x = x.detach()
        
        return x