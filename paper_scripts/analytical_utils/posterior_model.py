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
        """Sample from the posterior model."""
        if diffusion_term is None:
            diffusion_term = self.drift_model.diffusion_term

        t_vec = torch.linspace(0, 1, num_steps).unsqueeze(0).repeat(x0.shape[0], 1)
        dt = torch.tensor(1 / num_steps)

        x = x0.clone()

        pbar = tqdm.tqdm(range(num_steps - 1))
        for i in pbar:
            t = t_vec[:, i : i + 1]
            new = x + self.drift_model(x, t, x0) * dt
            new = new + diffusion_term(t) * torch.randn_like(x) * torch.sqrt(dt)

            if t.any() < 1e-6:
                x = new
                continue
            x.requires_grad = True

            score = self.likelihood_model.score(
                x, t, x0, observations, dt, diffusion_term
            )

            if isinstance(self.likelihood_model, InterpolantLikelihood):
                x = new + score * dt * diffusion_term(t) ** 2
            else:
                x = new + score
            x = x.detach()
        return x
