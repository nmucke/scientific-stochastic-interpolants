"""D-Flow SGLD likelihood holder -- BASELINE.

D-Flow (Ben-Hamu et al., ICML 2024, arXiv:2402.14017) does NOT use a per-step
guidance score: it differentiates the data cost through the WHOLE FM-ODE flow and
optimises the source latent ``z_0`` (see ``DFlowPosterior``). There is therefore
nothing for a ``score``-style likelihood to return. This class is a thin holder
that carries the linear observation operator ``H`` and the measurement variance
``R`` to the posterior (which forms the data term ``1/(2R) ||y - H Phi(z_0)||^2``
itself). It mirrors the constructor signature of the other Gaussian-likelihood
baselines so the ``build_posterior`` wiring is uniform.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator


class DFlowSGLDLikelihood(nn.Module):
    """Holder for the obs operator + measurement variance used by ``DFlowPosterior``.

    The SGLD optimisation over the source latent (data term + Gaussian source
    prior, backprop through the flow) lives in ``DFlowPosterior.sample``; this
    object only exposes ``obs_operator`` and ``original_variance`` plus the
    Langevin hyper-parameters read by ``build_posterior``.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        num_optim_steps: int = 20,
        step_size: float = 1e-2,
        noise_scale: float = 1.0,
        guidance_scale: float = 1.0,
    ) -> None:
        super(DFlowSGLDLikelihood, self).__init__()
        self.model = model
        self.obs_operator = obs_operator
        self.original_variance = float(variance)
        self.ensemble_size = ensemble_size
        self.num_optim_steps = int(num_optim_steps)
        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)
        self.guidance_scale = float(guidance_scale)
        # FM anchor a0 = 0 (source ~ N(0, I)); kept for interface symmetry.
        self.anchor = "zeros"

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Unused (D-Flow has no per-step guidance term)."""
        pass

    def score(self, *args: Any, **kwargs: Any) -> Any:
        """Not used: D-Flow differentiates the full flow in the posterior."""
        raise NotImplementedError(
            "DFlowSGLDLikelihood carries only H and R; the data term is formed and "
            "differentiated through the whole flow inside DFlowPosterior.sample."
        )
