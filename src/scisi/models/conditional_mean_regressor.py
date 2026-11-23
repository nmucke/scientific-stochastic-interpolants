"""Deterministic conditional-mean forecaster trained with MSE.

Its validation residuals ``y - m(h)`` estimate the forecast covariance that a
coloured interpolant driver should carry (WP1).
"""

import torch
from torch import nn


class ConditionalMeanRegressor(nn.Module):
    """m(h) = c + N(c, h): the MSE-optimal point forecast, c the base state.

    ``network`` is any backbone with the drift-model signature
    ``(x, cond, field_history, field_cond, pars_cond)``; its scalar time input
    is fed zeros. Trained by ``FlowTrainer``: ``forward`` returns
    (prediction, target) and ignores the sampled ``t`` and ``noise``.
    """

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def predict(self, base, field_history=None, field_cond=None, pars_cond=None):
        """Return the conditional-mean forecast m(h), shaped like ``base``."""
        cond = torch.zeros(base.shape[0], 1, device=base.device, dtype=base.dtype)
        return base + self.network(
            x=base,
            cond=cond,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

    def forward(
        self,
        base,
        target,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        **ignored,
    ):
        return self.predict(base, field_history, field_cond, pars_cond), target
