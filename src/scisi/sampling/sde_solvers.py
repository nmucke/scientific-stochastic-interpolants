import pdb
from typing import Callable

import torch
import torch.nn as nn


def _wiener_increment(x, dt, noise_covariance=None):
    """sqrt(dt) xi with white xi, coloured as C^{1/2} sqrt(dt) xi if C is given."""
    wiener_process = torch.randn_like(x, device=x.device) * torch.sqrt(dt)  # * mask
    if noise_covariance is not None:
        wiener_process = noise_covariance.sqrt(wiener_process)
    return wiener_process


def euler_maruyama_step(
    drift_model: nn.Module,
    diffusion_term: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    field_history: torch.Tensor | None = None,
    field_cond: torch.Tensor | None = None,
    pars_cond: torch.Tensor | None = None,
    mask: torch.Tensor = torch.tensor(1.0),
    noise_covariance=None,
) -> torch.Tensor:
    """Euler-Maruyama step; ``noise_covariance`` (C) colours the increment.

    x + b dt + g(t) C^{1/2} sqrt(dt) xi, for additive, state-independent noise.
    """
    wiener_process = _wiener_increment(x, dt, noise_covariance)
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
    mask: torch.Tensor = torch.tensor(1.0),
    noise_covariance=None,
) -> torch.Tensor:
    """Heun step; ``noise_covariance`` (C) colours the increment."""
    wiener_process = _wiener_increment(x, dt, noise_covariance)

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
