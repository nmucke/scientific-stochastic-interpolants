"""D-Flow SGLD likelihood holder -- BASELINE.

D-Flow SGLD (Parikh, Chen & Wang, arXiv:2602.21469; building on D-Flow, Ben-Hamu
et al., ICML 2024, arXiv:2402.14017) does NOT use a per-step guidance score: it
differentiates the data cost through the WHOLE FM-ODE flow and runs pSGLD over the
source latent ``z_0`` (see ``DFlowPosterior``). There is therefore nothing for a
``score``-style likelihood to return. This class is a thin holder that carries the
observation operator ``F`` / ``H`` to the posterior (which forms the plain-SSE data
term ``||y - F(Phi(z_0))||^2`` -- Algorithm 1 L6, NO 1/(2R) factor -- itself). It
mirrors the constructor signature of the other Gaussian-likelihood baselines so the
``build_posterior`` wiring is uniform.
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
        num_optim_steps: int = 300,
        step_size: float = 5e-2,
        noise_scale: float = 1e-3,
        lambda_reg: float = 5e-6,
    ) -> None:
        super(DFlowSGLDLikelihood, self).__init__()
        self.model = model
        self.obs_operator = obs_operator
        self.original_variance = float(variance)
        self.ensemble_size = ensemble_size
        self.num_optim_steps = int(num_optim_steps)
        self.step_size = float(step_size)  # eta
        self.noise_scale = float(noise_scale)  # s (Langevin noise scale)
        self.lambda_reg = float(lambda_reg)  # lambda (source-reg weight)
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
