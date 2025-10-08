import torch
import torch.nn as nn


def euler_step(
    drift_model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    field_history: torch.Tensor | None = None,
    field_cond: torch.Tensor | None = None,
    pars_cond: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler step."""
    drift = drift_model(x, t, field_history, field_cond, pars_cond)
    return x + drift * dt
