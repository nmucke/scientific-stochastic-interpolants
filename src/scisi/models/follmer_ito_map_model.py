"""Conditional Föllmer maps with shared additive-noise Brownian paths."""

import torch

from scisi.models.base_map_model import BaseMapModel, batch_time
from scisi.sampling.brownian_path import (
    KarhunenLoeveBrownianPath,
    WeightedBrownianPath,
)

BROWNIAN_FEATURES = ("local", "karhunen_loeve")


class FollmerItoMapModel(BaseMapModel):
    """Learn average SDE drift while retaining exact martingale increments.

    The current implementation uses the repository's affine diffusion schedule
    ``gamma(t)=gamma_multiplier*(1-t)``. Local path features approximate the
    drift's Brownian dependence; the explicit martingale increment is exact.

    ``brownian_features`` selects the path representation fed to the network:
    ``"local"`` (increments over equal subdivisions of [s,t]) or
    ``"karhunen_loeve"`` (the first ``num_path_features`` global KL
    coefficients of W on [0,1], the same for every interval of a path).
    """

    def __init__(
        self,
        interpolation,
        map_model,
        num_path_features=4,
        mask_path=None,
        brownian_features="local",
    ):
        super().__init__(interpolation, map_model, mask_path)
        if not isinstance(num_path_features, int) or num_path_features < 1:
            raise ValueError("num_path_features must be a positive integer.")
        if brownian_features not in BROWNIAN_FEATURES:
            raise ValueError(f"brownian_features must be one of {BROWNIAN_FEATURES}.")
        self.brownian_features = brownian_features
        if not torch.allclose(
            interpolation.transform_noise(torch.ones(1), torch.tensor([0.25])),
            torch.tensor([0.5]),
        ):
            raise ValueError("Föllmer Itô maps require wiener_process=True.")
        self.num_path_features = num_path_features
        self.gamma_multiplier = float(interpolation.gamma_multiplier)
        probe = torch.tensor([0.0, 0.25, 0.75, 1.0])
        if not torch.allclose(
            interpolation.gamma(probe), self.gamma_multiplier * (1 - probe)
        ):
            raise ValueError(
                "WeightedBrownianPath requires gamma(t)=gamma_multiplier*(1-t)."
            )

    def make_brownian_path(self, x, generator=None):
        if self.brownian_features == "karhunen_loeve":
            return KarhunenLoeveBrownianPath.sample_like(
                x,
                self.gamma_multiplier,
                generator=generator,
                num_modes=self.num_path_features,
            )
        return WeightedBrownianPath.sample_like(
            x, self.gamma_multiplier, generator=generator
        )

    def average_velocity(
        self,
        x,
        s,
        t,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        brownian_path=None,
    ):
        s, t = batch_time(s, x), batch_time(t, x)
        if torch.any(t < s):
            raise ValueError("An Itô map requires s <= t.")
        if brownian_path is None:
            if torch.any(t != s):
                raise ValueError(
                    "Off-diagonal Itô maps require a shared brownian_path."
                )
            features = x.new_zeros(
                x.shape[0], self.num_path_features * x.shape[1], *x.shape[2:]
            )
        else:
            features = brownian_path.features(s, t, self.num_path_features)
        return self.map_model(
            x=x,
            cond=torch.cat([s, t], -1),
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
            path_features=features,
        )

    def noise_increment(self, x, s, t, brownian_path=None):
        if brownian_path is None:
            if torch.any(t != s):
                raise ValueError(
                    "Off-diagonal Itô maps require a shared brownian_path."
                )
            return torch.zeros_like(x)
        return brownian_path.increment(s, t)

    def training_pair(self, base, target, t, noise):
        return (
            self.interpolation(base=base, target=target, t=t, noise=noise),
            self.interpolation.forward_diff(base=base, target=target, t=t, noise=noise),
        )
