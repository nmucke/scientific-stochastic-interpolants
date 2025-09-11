from typing import Callable

import torch
import torch.nn as nn


def euler_maruyama_step(
    drift_model: nn.Module,
    diffusion_term: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    field_history: torch.Tensor | None = None,
    field_cond: torch.Tensor | None = None,
    pars_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler-Maruyama step."""
    wiener_process = torch.randn_like(x, device=x.device) * torch.sqrt(dt)
    drift = drift_model(x, t, field_history, field_cond, pars_cond)
    diffusion = diffusion_term(t)
    return x + drift * dt + diffusion * wiener_process


def heun_step(
    drift_model: nn.Module,
    diffusion_term: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    field_history: torch.Tensor | None = None,
    field_cond: torch.Tensor | None = None,
    pars_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Heun step."""
    wiener_process = torch.randn_like(x, device=x.device) * torch.sqrt(dt)

    predictor_diffusion = diffusion_term(t)
    predictor_drift = drift_model(x, t, field_history, field_cond, pars_cond)
    predcitor_step = x + predictor_drift * dt + predictor_diffusion * wiener_process

    corrector_drift = drift_model(
        predcitor_step, t + dt, field_history, field_cond, pars_cond
    )
    corrector_diffusion = diffusion_term(t + dt)

    final_drift = 0.5 * (predictor_drift + corrector_drift)
    final_diffusion = 0.5 * (predictor_diffusion + corrector_diffusion)
    return x + final_drift * dt + final_diffusion * wiener_process
