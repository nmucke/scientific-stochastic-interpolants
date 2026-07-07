import pdb
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import tqdm

from scisi.models.base_model import BaseModel
from scisi.models.interpolations import _expand_t
from scisi.sampling.sde_solvers import euler_maruyama_step

MIN_TIME = 1e-4
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_STEPS = 100
DEFAULT_NUM_PHYSICAL_STEPS = 10


class DenoiseDiffusionModel(BaseModel):
    """Diffusion model.

    Supports two parametrisations of the underlying network, selected at
    construction:

    * **denoise mode** (``denoise_model``): the network predicts the noise
      ``eps``; the score is ``s = -eps / alpha`` (a VP/VE diffusion prior).
    * **velocity mode** (``velocity_model``): the network predicts the
      flow-matching velocity ``v``; the score is recovered from ``v`` via the
      affine-path velocity<->score identity with the diffusion anchor
      ``a0 = 0``. This lets a well-trained flow-matching model stand in for a
      (poorly trained) diffusion prior -- the reverse-SDE drift
      ``v + 1/2 g^2 s`` is identical either way. Build it with
      :meth:`from_flow_matching`.

    Exactly one of ``denoise_model`` / ``velocity_model`` must be provided.
    """

    def __init__(
        self,
        interpolation: nn.Module,
        denoise_model: Optional[nn.Module] = None,
        velocity_model: Optional[nn.Module] = None,
        diffusion_term: Optional[nn.Module] = None,
        mask_path: Optional[str] = None,
    ) -> None:
        """Initialize Diffusion model."""
        super(DenoiseDiffusionModel, self).__init__(mask_path=mask_path)

        if (denoise_model is None) == (velocity_model is None):
            raise ValueError(
                "DenoiseDiffusionModel needs exactly one of `denoise_model` "
                "(noise parametrisation) or `velocity_model` (flow-matching "
                "velocity parametrisation)."
            )

        self.interpolation = interpolation
        self.denoise_model = denoise_model
        self.velocity_model = velocity_model
        self._use_velocity = velocity_model is not None

        self.diffusion_term = diffusion_term
        if diffusion_term is None:
            self.diffusion_term = self.interpolation.alpha

    @classmethod
    def from_flow_matching(
        cls,
        flow_matching_model: nn.Module,
        diffusion_term: Optional[nn.Module] = None,
        mask_path: Optional[str] = None,
    ) -> "DenoiseDiffusionModel":
        """Build a diffusion prior from a trained flow-matching model.

        Reuses the flow-matching model's interpolation (an affine Gaussian path
        with deterministic anchor ``a0 = 0``) and its trained velocity network.
        The score is reconstructed from the velocity on the fly, so the
        resulting object behaves as a diffusion prior (``score`` / ``drift`` /
        reverse-SDE sampling) while only ever evaluating the FM velocity net.

        Args:
            flow_matching_model: A ``FlowMatchingModel`` (exposes
                ``interpolation`` and ``drift_model``).
            diffusion_term: Reverse-SDE diffusion coefficient ``g(t)``. Defaults
                to ``interpolation.alpha`` (endpoint-vanishing at the data end),
                matching the denoise-mode default.
            mask_path: Optional inpainting mask path.

        Returns:
            DenoiseDiffusionModel: A velocity-mode diffusion prior.
        """
        return cls(
            interpolation=flow_matching_model.interpolation,
            velocity_model=flow_matching_model.drift_model,
            diffusion_term=diffusion_term,
            mask_path=mask_path,
        )

    @property
    def model(self) -> nn.Module:
        """
        Get the underlying network.

        This is to ensure compatibility with the rest of the code base (e.g. the
        NFE counter). Returns the velocity net in velocity mode, else the
        denoise net.

        Returns:
            nn.Module: The active network.
        """
        return self.velocity_model if self._use_velocity else self.denoise_model

    def score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the score of the Diffusion model.

        In velocity mode the score is recovered from the flow-matching velocity
        via the shared velocity->score identity (anchor ``a0 = 0``); in denoise
        mode it is ``-eps / alpha``.
        """
        # The base-posterior main loop passes t as [B, 1] but its first-step path
        # passes a bare [1]; normalise to [., 1] so the net and _expand_t both get
        # the rank they expect.
        if t.dim() == 1:
            t = t.reshape(-1, 1)
        if self._use_velocity:
            velocity = self.velocity_model(
                x, t, field_history, field_cond, pars_cond
            )
            t_expanded = _expand_t(t, x) if t.dim() < x.dim() else t
            return self.interpolation.score_from_velocity(
                x=x,
                v=velocity,
                t=t_expanded,
                a0=torch.zeros_like(x),
            )

        return -self.denoise_model(
            x, t, field_history, field_cond, pars_cond
        ) / self.interpolation.alpha(t)

    def _get_velocity_from_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Get the velocity of the Diffusion model.

        Routes through the shared velocity<->score identity on the
        interpolation (paper Eq. general_velocity_of_score) with the diffusion
        anchor a0 = 0 and source scale sigma = alpha.
        """
        return self.interpolation.velocity_from_score(
            x=x,
            s=score,
            t=t,
            a0=torch.zeros_like(x),
        )

    def drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the drift of the Diffusion model.

        Args:
            x (torch.Tensor): Input tensor [B, C, H, W].
            t (torch.Tensor): Time tensor [B, 1].
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.
        """

        # Normalise a bare [1] (base-posterior first-step path) to [., 1].
        if t.dim() == 1:
            t = t.reshape(-1, 1)
        t_expanded = _expand_t(t, x) if t.dim() < x.dim() else t

        if self._use_velocity:
            # Velocity mode: one net eval gives v directly; derive the score from
            # it (no round-trip score->velocity, which would re-recover the same
            # v) and assemble the reverse-SDE drift.
            velocity = self.velocity_model(
                x, t, field_history, field_cond, pars_cond
            )
            score = self.interpolation.score_from_velocity(
                x=x,
                v=velocity,
                t=t_expanded,
                a0=torch.zeros_like(x),
            )
        else:
            score = self.score(x, t, field_history, field_cond, pars_cond)
            velocity = self._get_velocity_from_score(x, t, score)

        return velocity + 0.5 * self.diffusion_term(t_expanded) ** 2 * score  # type: ignore[misc]

    def forward(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        field_history: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the diffusion model when training the drift model.

        Args:
            base (torch.Tensor): Base tensor [B, C, H, W].
            target (torch.Tensor): Target tensor [B, C, H, W].
            t (torch.Tensor): Time tensor [B, 1].
            noise (torch.Tensor): Noise tensor [B, C, H, W].
            field_cond (torch.Tensor): Field conditional tensor [B, C_field_cond, H, W]. Can be None.
            field_history (torch.Tensor): Field history tensor [B, C, H, W, L]. Can be None.
            pars_cond (torch.Tensor): pars conditional tensor [B, D_pars_cond]. Can be None.

        Returns:
            torch.Tensor: Drift tensor [B, C, H, W].
            torch.Tensor: Interpolation derivative tensor [B, C, H, W].
        """
        if self._use_velocity:
            raise RuntimeError(
                "DenoiseDiffusionModel built in velocity mode "
                "(from_flow_matching) is inference-only: train the underlying "
                "flow-matching model instead of calling forward()."
            )

        interpolant = self.interpolation.forward(
            base=noise,
            target=target,
            t=t,
        )

        pred_noise = self.denoise_model(
            x=interpolant,
            cond=t,
            field_history=field_history,
            field_cond=field_cond,
            pars_cond=pars_cond,
        )

        return pred_noise, noise

    def _compute_first_step(
        self,
        base: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Optional[Callable] = euler_maruyama_step,
        mask: torch.Tensor = torch.tensor(1.0),
    ) -> torch.Tensor:
        """Compute the first step of the Follmer stochastic interpolant."""

        # drift = lambda x, t, field_history, field_cond, pars_cond: (
        #     0.5
        #     * self.diffusion_term(t) ** 2  # type: ignore[misc]
        #     * self.score(x, t, field_history, field_cond, pars_cond)
        # )

        drift = (
            0.5
            * self.diffusion_term(t) ** 2  # type: ignore[misc]
            * self.score(base, t, field_history, field_cond, pars_cond)
        )
        diffusion = self.diffusion_term(t) * torch.randn_like(base)  # type: ignore[misc]

        return base + drift * dt + diffusion * torch.sqrt(dt)  # * self.mask

    def sample(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        return_field_history: bool = False,
        stepper: Callable = euler_maruyama_step,
        diffusion_term: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Sample from the Diffusion model."""

        if diffusion_term is not None:
            self.diffusion_term = diffusion_term  # type: ignore[assignment]

        return self._sample(
            base=base,
            field_history=field_history,
            batch_size=batch_size,
            num_steps=num_steps,
            field_cond=field_cond,
            pars_cond=pars_cond,
            return_field_history=return_field_history,
            stepper=stepper,
            diffusion_term=diffusion_term,
            with_first_step=True,
        )

    def sample_trajectory(
        self,
        field_history: torch.Tensor,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_steps: int = DEFAULT_NUM_STEPS,
        num_physical_steps: int = DEFAULT_NUM_PHYSICAL_STEPS,
        base: Optional[torch.Tensor] = None,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Callable = euler_maruyama_step,
    ) -> torch.Tensor:
        """Sample a trajectory from the Flow Matching model."""

        return self._sample_trajectory(
            field_history=field_history,
            batch_size=batch_size,
            num_steps=num_steps,
            num_physical_steps=num_physical_steps,
            base=base,
            field_cond=field_cond,
            pars_cond=pars_cond,
            stepper=stepper,
            gaussian_base=True,
        )
