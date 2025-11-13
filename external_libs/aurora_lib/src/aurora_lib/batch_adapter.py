import pdb
from typing import Any

import torch
from aurora import Batch, Metadata

# # surface variables (b, t, h, w)
# surf_vars={k: torch.randn(1, 2, 721, 1440) for k in ("2t", "10u", "10v", "msl")},
# # static variables (h, w)
# static_vars={k: torch.randn(721, 1440) for k in ("lsm", "z", "slt")},
# # atmospheric variables (b, t, c, h, w)
# atmos_vars={k: torch.randn(1, 2, 4, 721, 1440) for k in ("z", "u", "v", "t", "q")},
# # metadata
# metadata=Metadata(
#     lat=torch.linspace(90, -90, 721),
#     lon=torch.linspace(0, 360, 1440 + 1)[:-1],
#     time=(datetime(2020, 6, 1, 12, 0),), # time
#     atmos_levels=(100, 250, 500, 850), # atmospheric levels
# ),

SURF_VAR_NAMES = ("2t", "10u", "10v", "msl")
ATMOS_VAR_NAMES = ("z", "u", "v", "t", "q")
STATIC_VAR_NAMES = ("lsm", "z", "slt")


class BatchAdapter:
    """Batch adapter for the aurora model."""

    def __init__(self, metadata: Metadata, static_vars: dict[str, torch.Tensor]):
        """
        Initialize the batch adapter.

        Args:
            metadata: The metadata.
            static_vars: The static variables.
        """
        self.metadata = metadata
        self.num_atmos_levels = len(metadata.atmos_levels)
        self.num_surf_vars = len(SURF_VAR_NAMES)
        self.num_atmos_vars = len(ATMOS_VAR_NAMES)
        self.num_static_vars = len(STATIC_VAR_NAMES)
        self.static_vars = static_vars

        self.surf_var_to_c_idx = {name: i for i, name in enumerate(SURF_VAR_NAMES)}

        self.atmos_var_to_c_idx = {
            name: range(
                self.num_surf_vars + i * self.num_atmos_levels,
                self.num_surf_vars + (i + 1) * self.num_atmos_levels,
            )
            for i, name in enumerate(ATMOS_VAR_NAMES)
        }

        self.static_var_to_c_idx = {name: i for i, name in enumerate(STATIC_VAR_NAMES)}

    def aurora_to_scisi(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the aurora batch format to the scisi format.

        Args:
            batch: The aurora batch format data.

        Returns:
            The scisi format data.
        """

        surf_vars = batch.surf_vars
        atmos_vars = batch.atmos_vars
        # static_vars = batch.static_vars

        b, t, h, w = surf_vars["2t"].shape

        field_history = torch.zeros(
            b, self.num_surf_vars + self.num_atmos_vars * self.num_atmos_levels, h, w, t
        )

        for t_idx in range(t):
            for key in surf_vars.keys():
                field_history[:, self.surf_var_to_c_idx[key], :, :, t_idx] = surf_vars[
                    key
                ][:, t_idx]

            for key in atmos_vars.keys():
                field_history[:, self.atmos_var_to_c_idx[key], :, :, t_idx] = (
                    atmos_vars[key][:, t_idx]
                )

        # field_cond = torch.zeros(b, self.num_static_vars, h, w)
        # for key in static_vars.keys():
        #     field_cond[:, self.static_var_to_c_idx[key], :, :] = static_vars[key]

        return field_history[..., -1], field_history

    def scisi_to_aurora(self, field_history: torch.Tensor) -> Batch:
        """
        Convert the scisi format to the aurora format.

        Args:
            field_history: The field history.

        Returns:
            The aurora batch format data.
        """
        b, _, h, w, t = field_history.shape

        surf_vars = {key: torch.zeros(b, t, h, w) for key in SURF_VAR_NAMES}
        atmos_vars = {
            key: torch.zeros(b, t, self.num_atmos_levels, h, w)
            for key in ATMOS_VAR_NAMES
        }

        for t_idx in range(t):
            for key, idx in self.surf_var_to_c_idx.items():
                surf_vars[key][:, t_idx] = field_history[:, idx, :, :, t_idx]

            for key, idx in self.atmos_var_to_c_idx.items():  # type: ignore[assignment]
                atmos_vars[key][:, t_idx] = field_history[:, idx, :, :, t_idx]

        # for key, idx in self.static_var_to_c_idx.items():
        #     static_vars[key] = field_cond[0, idx, :, :]

        return Batch(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            static_vars=self.static_vars,
            metadata=self.metadata,
        )
