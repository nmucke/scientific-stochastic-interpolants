"""Coherent Gaussian drivers for conditional Itô maps.

The repository's stochastic interpolants use diffusion ``a * (1 - t)``.
Its integrated noise is a Brownian motion in the variance clock
``q(t) = a**2 * (t - t**2 + t**3 / 3)``. Lazy Gaussian bridges give
exact joint laws at any finite set of queried times, without an SDE grid.
"""

from bisect import bisect_left
import math
from typing import Optional

import torch


class WeightedBrownianPath:
    """A batch of independent paths of ``M_t = integral a*(1-r) dW_r``.

    Each coordinate of the state has its own independent driver. Reusing this
    object in a composed map preserves the same driving realization. Times
    may differ across batch entries. Path queries are random sampling
    operations and are intentionally not differentiable in query times.

    Values are cached; querying additional times samples their conditional
    Gaussian bridge law. Seeds reproduce an identical *query sequence*.
    Different query orders produce the same distribution, not necessarily
    the same realization for a given seed.
    """

    def __init__(
        self,
        x: torch.Tensor,
        gamma_multiplier: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("x must have a nonempty batch and a channel dimension")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        amplitude = float(gamma_multiplier)
        if not math.isfinite(amplitude) or amplitude < 0:
            raise ValueError("gamma_multiplier must be finite and nonnegative")
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype
        self.gamma_multiplier = amplitude
        self.generator = generator
        zero = torch.zeros_like(x, requires_grad=False)
        end = self._randn(x.shape) * (amplitude / math.sqrt(3.0))
        self._times = [[0.0, 1.0] for _ in range(x.shape[0])]
        self._values = [[zero[b], end[b]] for b in range(x.shape[0])]

    @classmethod
    def sample_like(
        cls,
        x: torch.Tensor,
        gamma_multiplier: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> "WeightedBrownianPath":
        """Sample a driver with the batch, state shape, device and dtype of x."""
        return cls(x, gamma_multiplier=gamma_multiplier, generator=generator)

    def fork(
        self, generator: Optional[torch.Generator] = None
    ) -> "WeightedBrownianPath":
        """Copy the observed path values, then independently refine between them.

        Cached tensors are shared and treated as immutable; the time/value lists
        are copied so new queries cannot modify the parent or sibling caches.
        The provided generator is used only for new queries. With ``None``, use
        the global torch RNG, advancing it normally; the parent's generator
        state is never copied or reset. For independent reproducible forks,
        provide differently seeded generators or one generator advanced across
        calls. Identically reseeding each fork would duplicate its refinements.
        """
        child = object.__new__(type(self))
        child.shape = self.shape
        child.device = self.device
        child.dtype = self.dtype
        child.gamma_multiplier = self.gamma_multiplier
        child.generator = generator
        child._times = [times.copy() for times in self._times]
        child._values = [values.copy() for values in self._values]
        return child

    def _randn(self, shape) -> torch.Tensor:
        return torch.randn(
            shape, device=self.device, dtype=self.dtype, generator=self.generator
        )

    def _batch_times(self, t) -> list[float]:
        # Use double precision for the variance clock, also for float32 states.
        times = torch.as_tensor(t, dtype=torch.float64).detach().to(device="cpu")
        times = times.reshape(-1)
        if times.numel() == 1:
            times = times.expand(self.shape[0])
        if times.numel() != self.shape[0]:
            raise ValueError("time must be scalar or contain one value per batch entry")
        if not torch.isfinite(times).all() or ((times < 0) | (times > 1)).any():
            raise ValueError("time must lie in [0, 1] and be finite")
        return times.tolist()

    def variance(self, s, t) -> torch.Tensor:
        """Return exact increment variances, with shape [batch, 1]."""
        starts, ends = self._batch_times(s), self._batch_times(t)
        if any(end < start for start, end in zip(starts, ends)):
            raise ValueError("increment end must not precede its start")
        values = [self._variance_between(a, b) for a, b in zip(starts, ends)]
        return torch.tensor(values, device=self.device, dtype=self.dtype)[:, None]

    def _variance_between(self, s: float, t: float) -> float:
        # Factor the difference of cubes to remain accurate on tiny intervals.
        a, b = 1.0 - s, 1.0 - t
        return self.gamma_multiplier**2 * (t - s) * (a * a + a * b + b * b) / 3.0

    def value(self, t) -> torch.Tensor:
        """Query M_t, preserving the exact Gaussian joint law with cached values."""
        query = self._batch_times(t)
        result: list[Optional[torch.Tensor]] = [None] * self.shape[0]
        missing = []
        means = []
        variances = []
        for batch, time in enumerate(query):
            times = self._times[batch]
            values = self._values[batch]
            index = bisect_left(times, time)
            if index < len(times) and times[index] == time:
                result[batch] = values[index]
                continue
            left, right = times[index - 1], times[index]
            left_value, right_value = values[index - 1], values[index]
            left_variance = self._variance_between(left, time)
            right_variance = self._variance_between(time, right)
            total_variance = left_variance + right_variance
            weight = left_variance / total_variance if total_variance > 0 else 0.0
            means.append(left_value + weight * (right_value - left_value))
            variances.append(weight * right_variance)
            missing.append((batch, time, index))
        if missing:
            scale = torch.tensor(variances, device=self.device, dtype=self.dtype)
            scale = scale.sqrt().reshape(-1, *([1] * (len(self.shape) - 1)))
            draws = torch.stack(means) + scale * self._randn(
                (len(missing), *self.shape[1:])
            )
            for draw, (batch, time, index) in zip(draws, missing):
                self._times[batch].insert(index, time)
                self._values[batch].insert(index, draw)
                result[batch] = draw
        return torch.stack(result)

    def increment(self, s, t) -> torch.Tensor:
        """Return M_t - M_s from the shared realization; zero for s=t."""
        starts, ends = self._batch_times(s), self._batch_times(t)
        if any(end < start for start, end in zip(starts, ends)):
            raise ValueError("increment end must not precede its start")
        # Passing double tensors avoids rounding Python-float lists to float32.
        starts = torch.tensor(starts, dtype=torch.float64)
        ends = torch.tensor(ends, dtype=torch.float64)
        return self.value(ends) - self.value(starts)

    def features(self, s, t, num_features: int) -> torch.Tensor:
        """Return chronological local martingale increments as spatial channels.

        Equal subdivisions of [s,t] give K features per physical channel.
        Output shape is [B, K*C, ...], with feature-major channel ordering.
        They retain their physical noise scale (no variance normalization).
        Their sum is the full interval increment. Using finitely many such
        features compresses the path; this is distinct from the exact joint
        law of the represented Gaussian values themselves.
        """
        if isinstance(num_features, bool) or not isinstance(num_features, int):
            raise TypeError("num_features must be a positive integer")
        if num_features < 1:
            raise ValueError("num_features must be a positive integer")
        starts = torch.tensor(self._batch_times(s), dtype=torch.float64)
        ends = torch.tensor(self._batch_times(t), dtype=torch.float64)
        if (ends < starts).any():
            raise ValueError("feature interval end must not precede its start")
        values = [self.value(starts)]
        for index in range(1, num_features):
            time = starts + (ends - starts) * (index / num_features)
            values.append(self.value(time))
        values.append(self.value(ends))
        return torch.cat(
            [right - left for left, right in zip(values[:-1], values[1:])], dim=1
        )
