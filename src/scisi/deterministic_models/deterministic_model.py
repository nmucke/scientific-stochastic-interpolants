from typing import Any, Optional

import torch
import torch.nn as nn

from scisi.models.base_model import BaseModel

DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_PHYSICAL_STEPS = 10


class DeterministicModel(BaseModel):
    """Deterministic time-stepping model.

    Predicts the next state directly, ``x_{n+1} = network(x_n, context,
    params)``, with no interpolant, pseudo-time integration, or noise. The
    network is any architecture with the repo's standard signature
    ``network(x, cond, field_history, field_cond, pars_cond)`` (e.g.
    ``scisi.architectures.u_net.UNet``).
    """

    def __init__(
        self,
        network: nn.Module,
        residual: bool = False,
        mask_path: Optional[str] = None,
    ) -> None:
        """Initialize the deterministic model.

        Args:
            network: Architecture mapping the current state to the next one.
            residual: If True, the network predicts the increment and the
                model returns ``x + network(x, ...)``; if False (default),
                the network output is the next state itself.
            mask_path: Optional path to a mask file (see ``BaseModel``).
        """
        super(DeterministicModel, self).__init__(mask_path=mask_path)

        self.network = network
        self.residual = residual

    def _step(
        self,
        x: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One-step map ``x_n -> x_{n+1}``.

        The architectures require a scalar conditional input (pseudo-time for
        the stochastic models); a zero placeholder is passed so architecture
        configs with ``cond_dim: 1`` work unchanged.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.

        Returns:
            torch.Tensor: Next-state tensor [B, C, H, W].
        """
        cond = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

        out = self.network(x, cond, field_history, field_cond, pars_cond)

        return x + out if self.residual else out

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Not defined for a direct next-step predictor."""
        raise NotImplementedError(
            "DeterministicModel predicts the next state directly and has no drift."
        )

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training: predict the next state from the base.

        Args:
            base (torch.Tensor): Base tensor [B, C, H, W] (state x_n).
            target (torch.Tensor): Target tensor [B, C, H, W] (state x_{n+1}).
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.
            **kwargs: Ignored. Absorbs batch keys deterministic training does
                not use (e.g. ``t``/``noise`` from a stochastic trainer).

        Returns:
            torch.Tensor: Predicted next state [B, C, H, W].
            torch.Tensor: Target tensor [B, C, H, W].
        """
        pred = self._step(
            base,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        return pred, target

    def sample(
        self,
        field_history: torch.Tensor,
        base: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict the next state (a "sample" is one deterministic step).

        Args:
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L].
            base (torch.Tensor): Current state [B, C, H, W]. If None, the last
                slice of the field history is used.
            batch_size (int): Batch size to repeat singleton inputs to.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.
            return_field_history (bool): If True, also return the field
                history rolled forward with the prediction as its last slice.
            **kwargs: Ignored. Absorbs sampler arguments that only apply to
                stochastic models (``num_steps``, ``stepper``,
                ``diffusion_term``, ``gaussian_base``).

        Returns:
            torch.Tensor: Predicted next state [B, C, H, W] (on CPU), plus the
                rolled field history when ``return_field_history`` is True.
        """
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )

        if base is None:
            base = field_history[:, :, :, :, -1]

        with torch.no_grad():
            pred = self._step(
                base.to(self.device),
                field_history=(
                    field_history.to(self.device)
                    if field_history is not None
                    else None
                ),
                field_cond=(
                    field_cond.to(self.device) if field_cond is not None else None
                ),
                pars_cond=(
                    pars_cond.to(self.device) if pars_cond is not None else None
                ),
            ).cpu()

        # Add the new state to the field history
        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:].cpu(), pred.unsqueeze(-1)], dim=-1
            )
            return pred, field_history

        return pred

    def sample_trajectory(
        self,
        field_history: torch.Tensor,
        base: Optional[torch.Tensor] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressively roll out a trajectory.

        Note: with ``batch_size > 1`` all ensemble members are identical (the
        model is deterministic).

        Args:
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L].
            base (torch.Tensor): Current state [B, C, H, W]. If None, the last
                slice of the field history is used.
            batch_size (int): Batch size to repeat singleton inputs to.
            num_physical_steps (int): Total trajectory length, including the
                seeded history frames.
            field_cond (torch.Tensor): Field conditional trajectory tensor
                [B, C_field_cond, H, W, T]. Can be None.
            pars_cond (torch.Tensor): pars conditional trajectory tensor
                [B, T]. Can be None.
            **kwargs: Ignored. Absorbs sampler arguments that only apply to
                stochastic models (``num_steps``, ``stepper``,
                ``diffusion_term``).

        Returns:
            torch.Tensor: Trajectory tensor [B, C, H, W, num_physical_steps].
        """
        return self._sample_trajectory(
            base=base,
            field_history=field_history,
            batch_size=batch_size,
            num_steps=1,
            num_physical_steps=num_physical_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
            gaussian_base=False,
        )
