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


class _WeightedPathBase:
    """Shared time validation, variances and increments of ``M_t``.

    Subclasses cache M per batch entry and implement ``value``, ``features``
    and ``_copy_cache`` (used by ``fork``).
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

    def fork(self, generator: Optional[torch.Generator] = None):
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
        self._copy_cache(child)
        return child

    def _copy_cache(self, child) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    def increment(self, s, t) -> torch.Tensor:
        """Return M_t - M_s from the shared realization; zero for s=t."""
        starts, ends = self._batch_times(s), self._batch_times(t)
        if any(end < start for start, end in zip(starts, ends)):
            raise ValueError("increment end must not precede its start")
        # Passing double tensors avoids rounding Python-float lists to float32.
        starts = torch.tensor(starts, dtype=torch.float64)
        ends = torch.tensor(ends, dtype=torch.float64)
        return self.value(ends) - self.value(starts)

    def _feature_interval(self, s, t, num_features):
        """Validate the feature count and interval; return double [B] times."""
        if isinstance(num_features, bool) or not isinstance(num_features, int):
            raise TypeError("num_features must be a positive integer")
        if num_features < 1:
            raise ValueError("num_features must be a positive integer")
        starts = torch.tensor(self._batch_times(s), dtype=torch.float64)
        ends = torch.tensor(self._batch_times(t), dtype=torch.float64)
        if (ends < starts).any():
            raise ValueError("feature interval end must not precede its start")
        return starts, ends


class WeightedBrownianPath(_WeightedPathBase):
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
        super().__init__(x, gamma_multiplier, generator)
        zero = torch.zeros_like(x, requires_grad=False)
        end = self._randn(x.shape) * (self.gamma_multiplier / math.sqrt(3.0))
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

    def _copy_cache(self, child) -> None:
        child._times = [times.copy() for times in self._times]
        child._values = [values.copy() for values in self._values]

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

    def features(self, s, t, num_features: int) -> torch.Tensor:
        """Return chronological local martingale increments as spatial channels.

        Equal subdivisions of [s,t] give K features per physical channel.
        Output shape is [B, K*C, ...], with feature-major channel ordering.
        They retain their physical noise scale (no variance normalization).
        Their sum is the full interval increment. Using finitely many such
        features compresses the path; this is distinct from the exact joint
        law of the represented Gaussian values themselves.
        """
        starts, ends = self._feature_interval(s, t, num_features)
        values = [self.value(starts)]
        for index in range(1, num_features):
            time = starts + (ends - starts) * (index / num_features)
            values.append(self.value(time))
        values.append(self.value(ends))
        return torch.cat(
            [right - left for left, right in zip(values[:-1], values[1:])], dim=1
        )


class KarhunenLoeveBrownianPath(_WeightedPathBase):
    """``M_t`` driven by W whose first K Karhunen–Loève coefficients are explicit.

    On [0,1], ``W_t = sum_n xi_n phi_n(t)`` with ``xi_n = int e_n dW`` iid
    N(0,1), ``e_n(r) = sqrt(2) cos(w_n r)``, ``w_n = (n - 1/2) pi``. Then
    ``M_t = sum_{n<=K} xi_n m_n(t) + R_t`` with ``m_n(t) = E[M_t xi_n]`` and a
    centered Gaussian residual R independent of xi_1..xi_K, with covariance
    ``C_R(t,t') = q(min(t,t')) - sum_n m_n(t) m_n(t')``. R is not Markov, so a
    new time is drawn from R's exact Gaussian conditional given *all* cached
    times of its batch entry; the conditioning weights are scalars shared by
    all state coordinates. Features are xi_1..xi_K (Itô maps, arXiv
    2606.11156, Sec. 2.4), identical for every interval of one path.

    Caching, forking and seeding semantics match ``WeightedBrownianPath``.
    """

    # Relative eigenvalue cutoff of the pseudo-inverse and relative variance
    # below which a cached time carries no information about R (e.g. t=0).
    _RTOL = 1e-10

    def __init__(
        self,
        x: torch.Tensor,
        gamma_multiplier: float = 1.0,
        generator: Optional[torch.Generator] = None,
        num_modes: int = 4,
    ) -> None:
        super().__init__(x, gamma_multiplier, generator)
        if isinstance(num_modes, bool) or not isinstance(num_modes, int):
            raise TypeError("num_modes must be a positive integer")
        if num_modes < 1:
            raise ValueError("num_modes must be a positive integer")
        self.num_modes = num_modes
        self._xi = self._randn((x.shape[0], num_modes, *x.shape[1:]))
        zero = torch.zeros_like(x, requires_grad=False)
        self._times = [[0.0] for _ in range(x.shape[0])]
        self._values = [[zero[b]] for b in range(x.shape[0])]
        self._residuals = [[zero[b]] for b in range(x.shape[0])]

    @classmethod
    def sample_like(
        cls,
        x: torch.Tensor,
        gamma_multiplier: float = 1.0,
        generator: Optional[torch.Generator] = None,
        num_modes: int = 4,
    ) -> "KarhunenLoeveBrownianPath":
        """Sample a driver with the batch, state shape, device and dtype of x."""
        return cls(
            x,
            gamma_multiplier=gamma_multiplier,
            generator=generator,
            num_modes=num_modes,
        )

    def _copy_cache(self, child) -> None:
        child.num_modes = self.num_modes
        child._xi = self._xi
        child._times = [times.copy() for times in self._times]
        child._values = [values.copy() for values in self._values]
        child._residuals = [values.copy() for values in self._residuals]

    def _modes(self, times: torch.Tensor) -> torch.Tensor:
        """Unit-amplitude couplings m_n(t)/a, double [len(times), K]."""
        omega = torch.arange(1, self.num_modes + 1, dtype=torch.float64)
        omega = (omega - 0.5) * math.pi
        wt = times[:, None] * omega
        # 1 - cos(wt) = 2 sin^2(wt/2) avoids cancellation at small t.
        return math.sqrt(2.0) * (
            (1 - times[:, None]) * torch.sin(wt) / omega
            + 2 * torch.sin(wt / 2) ** 2 / omega**2
        )

    def _residual_kernel(self, times: torch.Tensor) -> torch.Tensor:
        """Unit-amplitude C_R/a^2 on the given double times."""
        low = torch.minimum(times[:, None], times[None, :])
        modes = self._modes(times)
        return low * (1 - low + low * low / 3) - modes @ modes.T

    def value(self, t) -> torch.Tensor:
        """Query M_t, preserving the exact Gaussian joint law with cached values."""
        query = self._batch_times(t)
        result: list[Optional[torch.Tensor]] = [None] * self.shape[0]
        missing = []
        means = []
        variances = []
        for batch, time in enumerate(query):
            times = self._times[batch]
            index = bisect_left(times, time)
            if index < len(times) and times[index] == time:
                result[batch] = self._values[batch][index]
                continue
            grid = torch.tensor(times + [time], dtype=torch.float64)
            kernel = self._residual_kernel(grid)
            prior = kernel[-1, -1].clamp_min(0.0).item()
            # Times with zero residual variance (t=0) carry no information; the
            # eigenvalue cutoff below handles near-degenerate cached times.
            keep = kernel.diagonal()[:-1] > 0.0
            keep_index = keep.nonzero().flatten()
            if prior <= 0.0 or keep_index.numel() == 0:
                weights = torch.zeros(0, dtype=torch.float64)
                variance = prior
            else:
                gram = kernel[keep_index][:, keep_index]
                cross = kernel[keep_index, -1]
                eigval, eigvec = torch.linalg.eigh(gram)
                cutoff = self._RTOL * eigval.abs().max()
                inverse = torch.where(
                    eigval > cutoff, 1 / eigval, torch.zeros_like(eigval)
                )
                weights = eigvec @ (inverse * (eigvec.T @ cross))
                variance = max(prior - float(cross @ weights), 0.0)
            residuals = self._residuals[batch]
            if weights.numel():
                stacked = torch.stack([residuals[i] for i in keep_index.tolist()])
                weights = weights.to(device=self.device, dtype=self.dtype)
                mean = torch.tensordot(weights, stacked, dims=1)
            else:
                mean = torch.zeros_like(residuals[0])
            means.append(mean)
            variances.append(self.gamma_multiplier**2 * variance)
            missing.append((batch, time, index))
        if missing:
            scale = torch.tensor(variances, device=self.device, dtype=self.dtype)
            scale = scale.sqrt().reshape(-1, *([1] * (len(self.shape) - 1)))
            draws = torch.stack(means) + scale * self._randn(
                (len(missing), *self.shape[1:])
            )
            rows = [batch for batch, _, _ in missing]
            modes = self._modes(
                torch.tensor([time for _, time, _ in missing], dtype=torch.float64)
            )
            modes = (self.gamma_multiplier * modes).to(self.device, self.dtype)
            coupled = torch.einsum("bk,bk...->b...", modes, self._xi[rows])
            for residual, path, (batch, time, index) in zip(
                draws, coupled + draws, missing
            ):
                self._times[batch].insert(index, time)
                self._residuals[batch].insert(index, residual)
                self._values[batch].insert(index, path)
                result[batch] = path
        return torch.stack(result)

    def features(self, s, t, num_features: int) -> torch.Tensor:
        """Return the KL coefficients xi_1..xi_K as spatial channels.

        Output shape is [B, K*C, ...], feature-major ([xi_1 over C channels,
        xi_2 over C channels, ...]); unit variance and independent of [s,t].
        ``num_features`` must equal ``num_modes``.
        """
        self._feature_interval(s, t, num_features)
        if num_features != self.num_modes:
            raise ValueError("num_features must equal the number of KL modes")
        return self._xi.reshape(self.shape[0], -1, *self.shape[2:]).clone()
