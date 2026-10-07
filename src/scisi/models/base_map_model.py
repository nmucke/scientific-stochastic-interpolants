"""Common sampling and forecasting interface for learned two-time maps."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import nn


def batch_time(t, x: torch.Tensor) -> torch.Tensor:
    """Normalize scalar or batched pseudo-time to [B,1]."""
    value = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
        raise ValueError("Pseudo-time must be finite and lie in [0, 1].")
    if value.numel() == 1:
        return value.reshape(1, 1).expand(x.shape[0], 1)
    if value.shape not in ((x.shape[0],), (x.shape[0], 1)):
        raise ValueError("Time must be scalar or have one value per state.")
    return value.reshape(x.shape[0], 1)


def spatial_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))


class BaseMapModel(nn.Module, ABC):
    """Map counterpart of BaseModel, retaining its forecasting tensor contract."""

    gaussian_base = False

    def __init__(
        self,
        interpolation: nn.Module,
        map_model: nn.Module,
        mask_path: Optional[str] = None,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.map_model = map_model
        mask = torch.tensor(1.0)
        if mask_path is not None:
            mask = torch.as_tensor(np.load(mask_path)["mask"], dtype=torch.float32)[
                None, None
            ]
        self.register_buffer("mask", mask)

    @property
    def model(self):
        return self.map_model

    @property
    def device(self):
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else self.mask.device

    @property
    def dtype(self):
        parameter = next(self.parameters(), None)
        return parameter.dtype if parameter is not None else self.mask.dtype

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
        return self.map_model(
            x=x,
            cond=torch.cat([s, t], -1),
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

    def noise_increment(self, x, s, t, brownian_path=None):
        return torch.zeros_like(x)

    def map(
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
        velocity = self.average_velocity(
            x,
            s,
            t,
            field_history,
            field_cond,
            pars_cond,
            brownian_path,
        )
        increment = spatial_time(t - s, x) * velocity
        increment = increment + self.noise_increment(x, s, t, brownian_path)
        return x + increment * self.mask

    def forward(self, x, s, t, **kwargs):
        return self.map(x, s, t, **kwargs)

    @abstractmethod
    def training_pair(self, base, target, t, noise):
        """Return an interpolant state and its diagonal drift regression target."""
        raise NotImplementedError

    def make_brownian_path(self, x, generator=None):
        return None

    @staticmethod
    def _repeat_condition(value, batch_size):
        if value is None:
            return None
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] != 1:
            raise ValueError(
                "Condition batch must match the state batch or contain one item."
            )
        return value.expand(batch_size, *value.shape[1:])

    @torch.no_grad()
    def sample(
        self,
        field_history,
        batch_size=1,
        num_steps=1,
        base=None,
        field_cond=None,
        pars_cond=None,
        return_field_history=False,
        brownian_path=None,
        generator=None,
    ):
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_steps must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if field_history.ndim != 5 or field_history.shape[-1] < 1:
            raise ValueError("field_history must have shape [B,C,H,W,L] with L>=1.")
        n = max(
            batch_size, field_history.shape[0], base.shape[0] if base is not None else 1
        )
        field_history = self._repeat_condition(field_history, n).to(
            self.device, self.dtype
        )
        field_cond = self._repeat_condition(field_cond, n)
        pars_cond = self._repeat_condition(pars_cond, n)
        if field_cond is not None:
            field_cond = field_cond.to(self.device, self.dtype)
        if pars_cond is not None:
            pars_cond = pars_cond.to(self.device, self.dtype)
        if base is None:
            x = field_history[..., -1]
            if self.gaussian_base:
                x = torch.randn(
                    x.shape, device=x.device, dtype=x.dtype, generator=generator
                )
        else:
            x = self._repeat_condition(base, n).to(self.device, self.dtype)
        if x.shape != field_history.shape[:-1]:
            raise ValueError("base must match field_history's state shape.")
        if brownian_path is None:
            brownian_path = self.make_brownian_path(x, generator)
        times = torch.linspace(0, 1, num_steps + 1, device=x.device, dtype=x.dtype)
        for s, t in zip(times[:-1], times[1:]):
            x = self.map(x, s, t, field_history, field_cond, pars_cond, brownian_path)
        result = x.detach().cpu()
        if return_field_history:
            history = torch.cat(
                [field_history[..., 1:].cpu(), result.unsqueeze(-1)], -1
            )
            return result, history
        return result

    @torch.no_grad()
    def sample_trajectory(
        self,
        field_history,
        base=None,
        batch_size=1,
        num_steps=1,
        num_physical_steps=10,
        field_cond=None,
        pars_cond=None,
        generator=None,
    ):
        """Forecast a total trajectory length including the supplied history."""
        length = field_history.shape[-1]
        if num_physical_steps < length:
            raise ValueError(
                "num_physical_steps must include the complete input history."
            )
        n = max(
            batch_size, field_history.shape[0], base.shape[0] if base is not None else 1
        )
        history = self._repeat_condition(field_history, n)
        trajectory = [history[..., i].cpu() for i in range(length)]
        for i in range(num_physical_steps - length):
            fc = (
                field_cond[..., i]
                if field_cond is not None and field_cond.ndim == 5
                else field_cond
            )
            pc = pars_cond[:, i : i + 1] if pars_cond is not None else None
            state, history = self.sample(
                field_history=history,
                base=base if i == 0 else None,
                batch_size=n,
                num_steps=num_steps,
                field_cond=fc,
                pars_cond=pc,
                return_field_history=True,
                generator=generator,
            )
            trajectory.append(state)
        return torch.stack(trajectory, -1)
