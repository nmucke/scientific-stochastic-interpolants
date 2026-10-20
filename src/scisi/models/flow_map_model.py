"""Conditional deterministic flow maps, including progressive map matching."""

from scisi.models.base_map_model import BaseMapModel


class FlowMapModel(BaseMapModel):
    """A two-time ODE map with Gaussian initialization for ensemble forecasts."""

    gaussian_base = True

    def training_pair(self, base, target, t, noise):
        return (
            self.interpolation(base=noise, target=target, t=t),
            self.interpolation.forward_diff(base=noise, target=target, t=t),
        )
