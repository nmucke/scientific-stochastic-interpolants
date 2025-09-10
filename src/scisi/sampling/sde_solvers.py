from typing import Callable

import torch
import torch.nn as nn


def euler_maruyama_step(
    drift_model: nn.Module,
    diffusion_model: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    field_cond: torch.Tensor | None = None,
    pars_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler-Maruyama step."""
    wiener_process = torch.randn_like(x, device=x.device) * torch.sqrt(dt)
    drift = drift_model(x, t, field_cond, pars_cond)
    diffusion = diffusion_model(t)
    return x + drift * dt + diffusion * wiener_process
