from typing import Any, Callable, Optional

import torch
import torch.nn as nn


def get_grid_observation_matrix(
    data_size: tuple[int, int, int], skip_grid: int
) -> torch.Tensor:
    """Get grid observation matrix."""

    C, H, W = data_size
    num_dofs = C * H * W

    # Calculate number of observed points based on grid spacing
    obs_h = H // skip_grid
    obs_w = W // skip_grid
    num_obs = C * obs_h * obs_w

    # Create observation matrix
    obs_matrix = torch.zeros(num_obs, num_dofs)

    # Fill in ones at grid points
    obs_idx = 0
    for c in range(C):
        for h in range(0, H, skip_grid):
            for w in range(0, W, skip_grid):
                flat_idx = c * (H * W) + h * W + w
                obs_matrix[obs_idx, flat_idx] = 1.0
                obs_idx += 1

    # Get indices where observations are made
    obs_indices = torch.nonzero(obs_matrix.sum(dim=0))

    return obs_matrix, num_obs, obs_indices


def get_random_observation_matrix(
    data_size: tuple[int, int, int],
    percent_obs: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Get random observation matrix.

    The selected indices form a fixed (seeded) sparse-sensor mask. Passing a
    ``seed`` makes the mask reproducible across calls and across methods, which
    the experiments require (a single shared mask per scenario). With
    ``seed=None`` the legacy (non-deterministic) behaviour is preserved.

    Args:
        data_size: Data size ``(C, H, W)``.
        percent_obs: Fraction of degrees of freedom to observe.
        seed: Optional RNG seed for a reproducible sensor mask.

    Returns:
        Tuple ``(obs_matrix, num_obs, obs_indices)``.
    """
    num_obs = int(data_size[0] * data_size[1] * data_size[2] * percent_obs)
    num_dofs = data_size[0] * data_size[1] * data_size[2]

    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(num_dofs, generator=generator)
    else:
        perm = torch.randperm(num_dofs)

    # Sort the selected indices so the mask is order-independent and identical
    # regardless of where it is consumed.
    obs_indices = torch.sort(perm[:num_obs]).values
    obs_matrix = torch.zeros(num_obs, num_dofs)
    for row in range(num_obs):
        obs_matrix[row, obs_indices[row]] = 1.0
    return obs_matrix, num_obs, obs_indices


def get_block_average_observation_matrix(
    data_size: tuple[int, int, int],
    low_res: int,
    high_res: Optional[int] = None,
) -> torch.Tensor:
    """Get a block-average super-resolution observation matrix.

    ``H`` performs block-average down-pooling from the model grid
    ``high_res x high_res`` to the low-resolution observation grid
    ``low_res x low_res`` by a factor ``factor = high_res // low_res``. This is
    equivalent to ``nn.functional.avg_pool2d`` with ``kernel = stride = factor``:
    every output cell is the mean of the ``factor x factor`` block of input
    cells beneath it. The operator is applied independently per channel/variable
    so the multi-channel layout matches the other operators.

    Because each block is averaged, every matrix entry is ``1 / factor**2``. The
    correct adjoint ``H^T`` therefore spreads each low-res cell value back over
    its block with the same ``1 / factor**2`` weighting, which guarantees the
    dot-product identity ``<H x, y> == <x, H^T y>``.

    Args:
        data_size: Data size ``(C, H, W)`` of the model grid. ``H`` and ``W``
            must equal ``high_res`` (when provided) and be divisible by
            ``factor``.
        low_res: Side length of the low-resolution observation grid.
        high_res: Side length of the model grid. Defaults to ``H`` from
            ``data_size``.

    Returns:
        Tuple ``(obs_matrix, num_obs, obs_indices)``. ``obs_indices`` holds the
        flat indices of every model-grid cell that contributes to an
        observation (all of them, since block-average touches the whole grid).
    """

    C, H, W = data_size

    if high_res is None:
        high_res = H

    if H != high_res or W != high_res:
        raise ValueError(
            f"Block-average operator expects a square {high_res}x{high_res} grid "
            f"matching data_size, got ({H}, {W})."
        )
    if high_res % low_res != 0:
        raise ValueError(
            f"high_res ({high_res}) must be divisible by low_res ({low_res})."
        )

    factor = high_res // low_res
    block_area = factor * factor

    num_dofs = C * H * W
    num_obs = C * low_res * low_res

    obs_matrix = torch.zeros(num_obs, num_dofs)

    weight = 1.0 / block_area
    obs_idx = 0
    for c in range(C):
        for oh in range(low_res):
            for ow in range(low_res):
                for bh in range(factor):
                    for bw in range(factor):
                        h = oh * factor + bh
                        w = ow * factor + bw
                        flat_idx = c * (H * W) + h * W + w
                        obs_matrix[obs_idx, flat_idx] = weight
                obs_idx += 1

    # Every grid cell contributes to exactly one observation.
    obs_indices = torch.nonzero(obs_matrix.sum(dim=0))

    return obs_matrix, num_obs, obs_indices


get_observation_matrix_factory = {
    "grid": get_grid_observation_matrix,
    "random": get_random_observation_matrix,
    "super_res": get_block_average_observation_matrix,
    "avg_pool": get_block_average_observation_matrix,
}


class ObservationOperator(nn.Module):
    """Base class for (possibly nonlinear) observation operators ``y = h(x)``.

    Shapes: ``forward(x)`` ``[B, C, H, W] -> [B, N_y]``; ``jvp(x, v)`` =
    ``J_h(x_b) v_b`` -> ``[B, N_y]``; ``vjp(x, u)`` = ``J_h(x_b)^T u_b`` ->
    ``[B, C, H, W]``; ``gram(x)`` = ``J_h(x_b) J_h(x_b)^T`` ``[B, N_y, N_y]``;
    ``jacobian(x_ref)`` dense ``[N_y, N_x]`` at ONE state; ``linearize(x_ref)``
    the linear map ``J_h(x_ref)`` (no offset). Derivatives are per batch member.
    The defaults use ``torch.func`` (any differentiable ``forward``);
    subclasses with an analytic Jacobian override them.

    ``is_linear`` gates the linear-only paths (closed-form ``H H^T``,
    ``obs_matrix`` / ``transpose``), which nonlinear operators refuse loudly.
    """

    is_linear: bool = False

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Per-member Jacobian-vector product ``J_h(x) v``."""
        _, out = torch.func.jvp(self.forward, (x,), (v,))
        return out

    def vjp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Per-member vector-Jacobian product ``J_h(x)^T u``."""
        _, pullback = torch.func.vjp(self.forward, x)
        return pullback(u)[0]

    def jacobian(self, x_ref: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """Dense Jacobian ``[N_y, N_x]`` at a single state (batched VJPs)."""
        x1 = x_ref.reshape(1, self.C, self.H, self.W)
        eye = torch.eye(self.num_obs, device=x1.device, dtype=x1.dtype)
        rows = []
        for start in range(0, self.num_obs, chunk):
            e = eye[start : start + chunk]
            xb = x1.expand(e.shape[0], -1, -1, -1).contiguous()
            rows.append(self.vjp(xb, e).reshape(e.shape[0], -1))
        return torch.cat(rows, dim=0)

    def gram(self, x: torch.Tensor, chunk_elems: int = 2**24) -> torch.Tensor:
        """``J_h(x_b) J_h(x_b)^T``, ``[B, N_y, N_y]`` (chunked VJP -> JVP).

        Each chunk of ``k`` basis vectors costs one batched VJP + JVP on
        ``k * B`` states, ``k * B * N_x <= chunk_elems``.
        """
        b, n_y = x.shape[0], self.num_obs
        k = max(1, chunk_elems // max(1, b * x[0].numel()))
        eye = torch.eye(n_y, device=x.device, dtype=x.dtype)
        out = x.new_empty(b, n_y, n_y)
        for s in range(0, n_y, k):
            e = eye[s : s + k]
            kk = e.shape[0]
            # Materialised: forward-mode AD rejects stride-0 (expanded) inputs.
            u = e.unsqueeze(1).expand(kk, b, n_y).reshape(kk * b, n_y).contiguous()
            xr = x.unsqueeze(0).expand(kk, *x.shape).reshape(kk * b, *x.shape[1:])
            xr = xr.contiguous()
            col = self.jvp(xr, self.vjp(xr, u).contiguous()).reshape(kk, b, n_y)
            out[:, :, s : s + kk] = col.permute(1, 2, 0)
        return out

    def linearize(self, x_ref: torch.Tensor) -> "LinearizedObservationOperator":
        """The linear map ``J_h(x_ref)`` (dense, no affine offset)."""
        return LinearizedObservationOperator(
            self.jacobian(x_ref), (self.C, self.H, self.W)
        )


def _cached_gram(op: nn.Module, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``H H^T`` of a dense ``src`` on ``ref``'s device/dtype, cached on ``op``.

    The cache holds a reference to ``src`` and compares with ``is``, so a
    reassigned matrix (``load_mask``) invalidates it and an id is never reused.
    """
    c = getattr(op, "_gram_cache", None)
    if c is None or c[0] is not src or c[1].device != ref.device or c[1].dtype != ref.dtype:
        H = src.to(device=ref.device, dtype=ref.dtype)
        c = (src, H @ H.transpose(0, 1))
        op._gram_cache = c
    return c[1]


class LinearObservationOperator(ObservationOperator):
    """Linear observation operator."""

    is_linear = True

    def __init__(
        self,
        type: str = "grid",
        data_size: tuple[int, int, int] = (1, 128, 128),
        skip_grid: Optional[int] = None,
        percent_obs: Optional[float] = None,
        low_res: Optional[int] = None,
        high_res: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize linear observation operator.

        Args:
            type: Type of observation operator. One of ``"grid"``, ``"random"``,
                ``"super_res"``/``"avg_pool"``.
            data_size: Data size ``(C, H, W)``.
            skip_grid: Grid spacing (``type="grid"``).
            percent_obs: Fraction of observed degrees of freedom
                (``type="random"``).
            low_res: Low-resolution side length (``type="super_res"``).
            high_res: High-resolution (model grid) side length
                (``type="super_res"``). Defaults to ``H`` from ``data_size``.
            seed: RNG seed for reproducible sparse-sensor masks
                (``type="random"``). Shared across methods so every method sees
                the same fixed mask.
        """
        super(LinearObservationOperator, self).__init__()

        self.C, self.H, self.W = data_size

        self.num_dofs = self.C * self.H * self.W

        self.type = type
        self.seed = seed

        if type == "grid":
            (
                self.obs_matrix,
                self.num_obs,
                self.obs_indices,
            ) = get_grid_observation_matrix(data_size, skip_grid)  # type: ignore[arg-type]
        elif type == "random":
            (
                self.obs_matrix,
                self.num_obs,
                self.obs_indices,
            ) = get_random_observation_matrix(
                data_size, percent_obs, seed=seed  # type: ignore[arg-type]
            )
        elif type in ("super_res", "avg_pool"):
            (
                self.obs_matrix,
                self.num_obs,
                self.obs_indices,
            ) = get_block_average_observation_matrix(
                data_size, low_res, high_res  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"Unknown observation operator type: {type}")

    @property
    def obs_indices_on_grid(self) -> torch.Tensor:
        """Get observation indices."""
        indices = torch.zeros(self.num_dofs)
        indices[self.obs_indices] = 1
        return indices.view(self.C, self.H, self.W)

    @property
    def obs_indices_c_h_w(self) -> torch.Tensor:
        """Get observation indices."""
        """Get x, y coordinates of observation points.

        Returns:
            Tensor of shape (num_obs, 3) containing [channel, height, width] indices of observations.
        """
        indices = self.obs_indices_on_grid
        obs_indices = torch.zeros((self.num_obs, 3), dtype=torch.long)
        idx = 0

        for c in range(self.C):
            y_coords, x_coords = torch.where(indices[c] == 1)
            num_coords = len(y_coords)
            obs_indices[idx : idx + num_coords, 0] = c
            obs_indices[idx : idx + num_coords, 1] = y_coords
            obs_indices[idx : idx + num_coords, 2] = x_coords
            idx += num_coords

        return obs_indices

    def save_mask(self, path: str) -> None:
        """Persist the observation index set so the mask can be reused.

        Args:
            path: Destination ``.pt`` file path.
        """
        torch.save(
            {
                "type": self.type,
                "data_size": (self.C, self.H, self.W),
                "num_obs": self.num_obs,
                "obs_indices": self.obs_indices.cpu(),
                "seed": self.seed,
            },
            path,
        )

    def load_mask(self, path: str) -> None:
        """Load a previously saved observation index set in place.

        This overwrites the current selection mask (``obs_indices`` and the
        corresponding selection ``obs_matrix``) with the persisted one, so a
        fixed mask can be shared verbatim across methods/runs. Only supported
        for selection operators (``grid``/``random``).

        Args:
            path: Source ``.pt`` file written by :meth:`save_mask`.
        """
        if self.type in ("super_res", "avg_pool"):
            raise ValueError(
                "load_mask is only supported for selection operators "
                "(grid/random), not block-average super-resolution."
            )

        checkpoint = torch.load(path)
        obs_indices = checkpoint["obs_indices"]
        num_obs = int(checkpoint["num_obs"])

        device = self.obs_matrix.device
        obs_matrix = torch.zeros(num_obs, self.num_dofs, device=device)
        for row in range(num_obs):
            obs_matrix[row, obs_indices[row]] = 1.0

        self.obs_indices = obs_indices.to(device)
        self.num_obs = num_obs
        self.obs_matrix = obs_matrix

    def _matrix_on(self, ref: torch.Tensor) -> torch.Tensor:
        """Return ``obs_matrix`` on ``ref``'s device/dtype, cached.

        ``obs_matrix`` is a plain tensor attribute (not a registered buffer), so
        ``.to(device)`` on the module never moves it and the hot path
        (``forward`` / ``transpose``, called once per JVP column / step) would
        otherwise re-copy a ``[N_y, N_u]`` CPU->GPU matrix every call -- which
        dominates wall-clock and pollutes seconds/step. We cache the moved matrix
        and re-copy only when the device/dtype changes or ``obs_matrix`` is
        reassigned (e.g. via ``load_mask``; cache is keyed on tensor identity).
        """
        cache = getattr(self, "_obs_matrix_cache", None)
        src = self.obs_matrix
        if (
            cache is None
            or cache["src"] is not src
            or cache["matrix"].device != ref.device
            or cache["matrix"].dtype != ref.dtype
        ):
            moved = src.to(device=ref.device, dtype=ref.dtype)
            cache = {"src": src, "matrix": moved}
            self._obs_matrix_cache = cache
        return cache["matrix"]

    def _selection_index_on(self, ref: torch.Tensor) -> Optional[torch.Tensor]:
        """Flat DOF index per observation row for pure-selection operators.

        For ``grid``/``random`` operators row ``j`` of ``obs_matrix`` is the
        single 1.0 at ``obs_indices[j]`` (rows are built in ascending-index
        order, and ``load_mask`` rebuilds them the same way), so ``H @ x`` is a
        gather and ``H^T @ y`` a scatter -- bitwise identical to the dense
        matmul (the dropped terms are exact ``0.0 * x`` products) at a tiny
        fraction of its cost. Returns ``None`` for the block-average operator,
        which keeps the dense path. Cached per device, keyed on the identity of
        ``obs_indices`` so ``load_mask`` invalidates it.
        """
        if self.type not in ("grid", "random"):
            return None
        cache = getattr(self, "_sel_idx_cache", None)
        src = self.obs_indices
        if (
            cache is None
            or cache["src"] is not src
            or cache["idx"].device != ref.device
        ):
            idx = src.reshape(-1).to(dtype=torch.long)
            # One-time sanity check of the row <-> index correspondence.
            assert idx.numel() == self.num_obs
            cache = {"src": src, "idx": idx.to(ref.device)}
            self._sel_idx_cache = cache
        return cache["idx"]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: apply ``H`` (``H @ x``).

        Selection operators (``grid``/``random``) use a gather instead of the
        dense matmul (identical result, see :meth:`_selection_index_on`).

        Args:
            x: Input tensor. [B, C, H, W]

        Returns:
            Output tensor. [B, num_obs]
        """

        b = x.shape[0]

        idx = self._selection_index_on(x)
        if idx is not None:
            return x.reshape(b, -1).index_select(1, idx)

        return x.reshape(b, -1) @ self._matrix_on(x).T

    def transpose(
        self,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adjoint pass: apply ``H^T`` (``H^T @ y``).

        For selection operators this scatters observed values back to their grid
        locations (done directly via ``index_copy``; identical to the dense
        matmul); for the block-average super-resolution operator it spreads
        each low-res cell value equally over its ``factor x factor`` block with
        the ``1 / factor**2`` weighting, so that ``<H x, y> == <x, H^T y>``.

        Args:
            y: Observation tensor. [B, num_obs]

        Returns:
            Full-grid tensor. [B, C, H, W]
        """

        b = y.shape[0]

        idx = self._selection_index_on(y)
        if idx is not None:
            out = y.new_zeros(b, self.num_dofs).index_copy(1, idx, y)
            return out.view(b, self.C, self.H, self.W)

        out = y @ self._matrix_on(y)

        return out.view(b, self.C, self.H, self.W)

    # ------------------------------------------------------------------
    # ObservationOperator interface (exact for a linear operator)
    # ------------------------------------------------------------------
    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """``J v = H v`` (independent of ``x``)."""
        return self.forward(v)

    def vjp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """``J^T u = H^T u`` (independent of ``x``)."""
        return self.transpose(u)

    def gram(self, x: Optional[torch.Tensor] = None, chunk_elems: int = 0) -> torch.Tensor:
        """``H H^T``, ``[N_y, N_y]`` (state independent; broadcasts over ``B``)."""
        return _cached_gram(self, self.obs_matrix, self.obs_matrix if x is None else x)

    def jacobian(self, x_ref: Optional[torch.Tensor] = None, chunk: int = 256) -> torch.Tensor:
        """Dense ``H``."""
        if x_ref is None:
            return self.obs_matrix
        return self._matrix_on(x_ref)

    def linearize(self, x_ref: Optional[torch.Tensor] = None) -> "LinearObservationOperator":
        """A linear operator is its own linearization."""
        return self


class LinearizedObservationOperator(ObservationOperator):
    """Dense linear map ``y = J x`` (e.g. ``J = J_h(x_ref)`` from ``linearize``).

    Only the linear part: no affine offset, so ``is_linear`` is truthful and
    every linear code path (``obs_matrix``, ``transpose``, ``gram``) applies.
    """

    is_linear = True

    def __init__(self, J: torch.Tensor, data_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.obs_matrix = J
        self.C, self.H, self.W = data_size
        self.num_obs, self.num_dofs = J.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        J = self.obs_matrix.to(device=x.device, dtype=x.dtype)
        return x.reshape(x.shape[0], -1) @ J.T

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        J = self.obs_matrix.to(device=y.device, dtype=y.dtype)
        return (y @ J).view(y.shape[0], self.C, self.H, self.W)

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.forward(v)

    def vjp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.transpose(u)

    def gram(self, x: Optional[torch.Tensor] = None, chunk_elems: int = 0) -> torch.Tensor:
        """``J J^T``, ``[N_y, N_y]`` (cached)."""
        return _cached_gram(self, self.obs_matrix, self.obs_matrix if x is None else x)

    def jacobian(self, x_ref: Optional[torch.Tensor] = None, chunk: int = 256) -> torch.Tensor:
        return self.obs_matrix

    def linearize(self, x_ref: Optional[torch.Tensor] = None) -> "LinearizedObservationOperator":
        return self


# ----------------------------------------------------------------------
# Elementwise nonlinearities: name -> (z, c, eps) -> (g(z), g'(z)). The
# strength c -> 0 recovers g(z) = z for arctan / tanh / saturating / cubic / exp.
# ----------------------------------------------------------------------
def _tanh(z, c, eps):
    th = torch.tanh(c * z)
    return th / c, 1.0 - th**2


def _saturating(z, c, eps):  # z / (1 + c|z|)  (SDA's Kolmogorov observation)
    d = 1.0 + c * z.abs()
    return z / d, 1.0 / d**2


def _abs(z, c, eps):  # smoothed |z|  (non-injective)
    r = torch.sqrt(z**2 + eps**2)
    return r, z / r


def _exp(z, c, eps):  # (exp(c z) - 1) / c
    e = torch.exp(c * z)
    return (e - 1.0) / c, e


NONLINEARITIES: dict[str, Callable] = {
    "identity": lambda z, c, eps: (z, torch.ones_like(z)),
    "arctan": lambda z, c, eps: (torch.atan(c * z) / c, 1.0 / (1.0 + (c * z) ** 2)),
    "tanh": _tanh,
    "saturating": _saturating,
    "cubic": lambda z, c, eps: (z + c * z**3, 1.0 + 3.0 * c * z**2),
    "square": lambda z, c, eps: (c * z**2, 2.0 * c * z),  # non-injective
    "abs": _abs,
    "exp": _exp,
}


def _g_and_dg(name: str, z: torch.Tensor, c: float, eps: float):
    """``(g(z), g'(z))`` of the named nonlinearity."""
    if name not in NONLINEARITIES:
        raise ValueError(f"Unknown nonlinearity: {name!r}")
    return NONLINEARITIES[name](z, c, eps)


class SensorComposedOperator(ObservationOperator):
    """Nonlinear operator built on an inner linear sensor map ``L`` (``.inner``).

    ``inner`` is a ``LinearObservationOperator`` or a dict of its constructor
    kwargs (merged with ``**inner_kwargs``, e.g. hydra's ``data_size``). Grid
    and mask metadata are delegated to ``inner``; ``obs_matrix`` /
    ``transpose`` raise so linear-only code fails loudly.
    """

    is_linear = False

    def __init__(self, inner: Any, **inner_kwargs: Any) -> None:
        super().__init__()
        if isinstance(inner, dict):
            inner = LinearObservationOperator(**{**inner, **inner_kwargs})
        self.inner = inner

    @property
    def C(self) -> int:  # type: ignore[override]
        return self.inner.C

    @property
    def H(self) -> int:  # type: ignore[override]
        return self.inner.H

    @property
    def W(self) -> int:  # type: ignore[override]
        return self.inner.W

    @property
    def num_dofs(self) -> int:
        return self.inner.num_dofs

    @property
    def num_obs(self) -> int:
        return self.inner.num_obs

    @property
    def type(self) -> str:
        return self.inner.type

    @property
    def seed(self) -> Optional[int]:
        return self.inner.seed

    @property
    def obs_indices(self) -> torch.Tensor:
        return self.inner.obs_indices

    @property
    def obs_indices_on_grid(self) -> torch.Tensor:
        return self.inner.obs_indices_on_grid

    @property
    def obs_indices_c_h_w(self) -> torch.Tensor:
        return self.inner.obs_indices_c_h_w

    def save_mask(self, path: str) -> None:
        self.inner.save_mask(path)

    def load_mask(self, path: str) -> None:
        self.inner.load_mask(path)

    @property
    def obs_matrix(self) -> torch.Tensor:
        raise AttributeError(
            f"{type(self).__name__} is nonlinear and has no obs_matrix; use "
            ".jacobian(x_ref) / .linearize(x_ref), or .inner.obs_matrix for the "
            "linear sensor map."
        )

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "A nonlinear observation operator has no transpose; use .vjp(x, u)."
        )


class PointwiseObservationOperator(SensorComposedOperator):
    """``h(x) = g(L x)``: linear sensor map ``L``, elementwise ``g``.

    ``g`` is one of :data:`NONLINEARITIES` (``arctan``: EnSF / FlowDAS;
    ``saturating`` ``z/(1+|z|)``: SDA's Kolmogorov flow; non-injective
    ``square`` / ``abs``). Analytic Jacobian ``J_h(x) = diag(g'(L x)) L``, so
    ``jvp`` / ``vjp`` / ``gram`` cost the same as ``L``.

    Args:
        inner: ``L`` (operator or dict of its kwargs).
        nonlinearity: Name of ``g``.
        strength: Nonlinearity strength ``c``.
        eps: Smoothing for ``abs``.
    """

    def __init__(
        self,
        inner: Any,
        nonlinearity: str = "arctan",
        strength: float = 1.0,
        eps: float = 1e-3,
        **inner_kwargs: Any,
    ) -> None:
        super().__init__(inner, **inner_kwargs)
        if nonlinearity not in NONLINEARITIES:
            raise ValueError(
                f"nonlinearity must be one of {tuple(NONLINEARITIES)}, got {nonlinearity!r}"
            )
        if strength <= 0 and nonlinearity not in ("identity", "abs"):
            raise ValueError(f"strength must be > 0, got {strength!r}")
        self.nonlinearity = nonlinearity
        self.strength = float(strength)
        self.eps = float(eps)

    def g(self, z: torch.Tensor) -> torch.Tensor:
        return _g_and_dg(self.nonlinearity, z, self.strength, self.eps)[0]

    def dg(self, z: torch.Tensor) -> torch.Tensor:
        return _g_and_dg(self.nonlinearity, z, self.strength, self.eps)[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(self.inner(x))

    def jvp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.dg(self.inner(x)) * self.inner(v)

    def vjp(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.inner.transpose(self.dg(self.inner(x)) * u)

    def gram(self, x: torch.Tensor, chunk_elems: int = 0) -> torch.Tensor:
        """``D (L L^T) D``, ``D = diag(g'(L x_b))``, ``[B, N_y, N_y]``."""
        d = self.dg(self.inner(x))
        return d.unsqueeze(-1) * self.inner.gram(x) * d.unsqueeze(-2)

    def jacobian(self, x_ref: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        x1 = x_ref.reshape(1, self.C, self.H, self.W)
        return self.dg(self.inner(x1)).reshape(-1, 1) * self.inner._matrix_on(x1)
