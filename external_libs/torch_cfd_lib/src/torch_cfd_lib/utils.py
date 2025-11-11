from typing import Tuple

import numpy as np
import torch
import xarray
from torch_cfd.grids import GridVariable


def get_tensor_from_grid_variables(
    v: Tuple[GridVariable, GridVariable],
) -> torch.Tensor:
    """Get a tensor from a grid variable."""

    return torch.stack(
        [v[0].data.detach().cpu().clone(), v[1].data.detach().cpu().clone()],
        dim=0,
    ).squeeze(1)


def get_vorticity_from_grid_variables(
    v: Tuple[GridVariable, GridVariable]
) -> torch.Tensor:
    """Compute vorticity from velocity field."""

    nx = v[0].data.shape[1]
    ny = v[0].data.shape[2]
    batch_size = v[0].data.shape[0]

    def vorticity_fn(ds: xarray.Dataset) -> xarray.DataArray:
        """Compute vorticity from velocity field."""
        return (ds.v.differentiate("x") - ds.u.differentiate("y")).rename("vorticity")

    coords = {
        "batch": np.linspace(0, batch_size - 1, batch_size, dtype=np.int64),
        "x": np.linspace(0, 2, nx, dtype=np.float64),
        "y": np.linspace(0, 1, ny, dtype=np.float64),
    }

    u_data = xarray.DataArray(
        v[0].data.detach().cpu(), dims=["batch", "x", "y"], coords=coords
    ).to_dataset(name="u")

    v_data = xarray.DataArray(
        v[1].data.detach().cpu(), dims=["batch", "x", "y"], coords=coords
    ).to_dataset(name="v")

    data = xarray.merge([u_data, v_data]).assign(vorticity=vorticity_fn)

    vorticity_data = data["vorticity"].data
    vorticity_data = torch.from_numpy(vorticity_data)

    return vorticity_data
