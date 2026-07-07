import pdb
from typing import Optional

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


class LinearObservationOperator(nn.Module):
    """Linear observation operator."""

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
            or cache["src_id"] != id(src)
            or cache["matrix"].device != ref.device
            or cache["matrix"].dtype != ref.dtype
        ):
            moved = src.to(device=ref.device, dtype=ref.dtype)
            cache = {"src_id": id(src), "matrix": moved}
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
            or cache["src_id"] != id(src)
            or cache["idx"].device != ref.device
        ):
            idx = src.reshape(-1).to(dtype=torch.long)
            # One-time sanity check of the row <-> index correspondence.
            assert idx.numel() == self.num_obs
            cache = {"src_id": id(src), "idx": idx.to(ref.device)}
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
