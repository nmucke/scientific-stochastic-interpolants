"""D-Flow SGLD posterior sampler -- BASELINE.

D-Flow (Ben-Hamu, Puny, Gat, Karrer, Singer & Lipman, "D-Flow: Differentiating
through Flows for Controlled Generation", ICML 2024, arXiv:2402.14017) controls
generation by optimising the SOURCE latent ``z_0`` so that the deterministic
FM-ODE flow ``Phi(z_0) = x_1`` matches a task cost, differentiating the cost
through the WHOLE ODE solve (``d x_1 / d z_0``). The original paper minimises

    L_tilde(z_0) = L(Phi(z_0)) + R(z_0),     R(z_0) = -log p(||z_0||)

with LBFGS + line search (a MAP-style point estimate). For a data-assimilation
*posterior* (we need an ENSEMBLE, not one MAP point) we use the natural
stochastic-gradient Langevin (SGLD) variant: each ensemble member is a Langevin
chain over its own source latent ``z_0``, targeting the posterior
``p(z_0 | y) propto N(y; H Phi(z_0), R) * N(z_0; 0, I)`` (the Gaussian source
prior of the trained FM model). The per-step update is

    z_0 <- z_0 - eta * grad_{z_0}[ 1/(2R) ||y - H Phi(z_0)||^2 + 1/2 ||z_0||^2 ]
               + sqrt(2 eta) * noise_scale * xi,      xi ~ N(0, I),

for ``K = num_optim_steps`` iterations, returning ``x_1 = Phi(z_0)``. With
``noise_scale = 0`` this reduces to plain gradient-descent D-Flow (MAP).

WHY A NEW POSTERIOR (not a likelihood).
The repo's per-step ``_one_step`` + likelihood pattern DETACHES every ODE step
to protect the autoregressive feedback, so a guidance "likelihood" only ever
sees the local state ``x_t`` and its one-step Jacobian. D-Flow is fundamentally
an OUTER optimisation that backprops through the ENTIRE multi-step rollout
``z_0 -> x_1`` -- it cannot be expressed as a per-step score correction. We
therefore override ``sample()`` to run the SGLD-over-``z_0`` loop with a
DIFFERENTIABLE FM-ODE rollout built from ``model.drift`` (NOT the no-grad
``model.sample``). ``sample_trajectory`` is inherited unchanged, so the
autoregressive windowing / field-history threading is identical to every other
posterior; only the per-window solver differs.

Time convention (forward FM): ``t : 0 -> 1``, ``t = 0`` source ``~ N(0, I)``,
``t = 1`` clean field; ``x_t = (1-t) eps + t x_1``; velocity ``v = model.drift``;
Euler step ``x_{t+dt} = x_t + v dt``. The rollout starts from ``z_0`` at ``t=0``.
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from scisi.posterior_models.base_posterior import BasePosterior
from scisi.sampling.sde_solvers import euler_maruyama_step

logger = logging.getLogger(__name__)


class DFlowPosterior(BasePosterior):
    """D-Flow SGLD posterior: SGLD over the FM source latent ``z_0``.

    Overrides ``sample()`` to optimise/sample the source latent through a
    differentiable FM-ODE rollout; ``sample_trajectory`` is inherited from
    :class:`BasePosterior` (autoregressive windowing unchanged).
    """

    def __init__(
        self,
        model: nn.Module,
        likelihood_model: nn.Module,
        diffusion_term: Optional[Callable] = None,
        num_optim_steps: int = 20,
        step_size: float = 1e-2,
        noise_scale: float = 1.0,
        guidance_scale: float = 1.0,
        precond_decay: float = 0.99,
        precond_eps: float = 1e-8,
        variance: float = 0.05,
    ) -> None:
        """Initialize the D-Flow SGLD posterior.

        Args:
            model: Trained FM model (``FlowMatchingModel``); supplies the
                deterministic FM-ODE flow via ``model.drift``.
            likelihood_model: Carries the linear observation operator ``H``
                (``likelihood_model.obs_operator``) and the measurement variance
                ``R`` (``likelihood_model.original_variance``); its ``score`` is
                NOT used (D-Flow differentiates the full rollout, not a per-step
                guidance term).
            diffusion_term: Ignored (kept for interface symmetry); the D-Flow
                flow is the deterministic FM-ODE (g = 0).
            num_optim_steps: K, number of SGLD/optimisation steps per window.
            step_size: Langevin step size eta.
            noise_scale: Multiplier on the Langevin noise sqrt(2 eta); 0 -> MAP
                preconditioned gradient descent (deterministic D-Flow), 1 ->
                standard pSGLD.
            guidance_scale: Extra multiplier on the data-term gradient (1/R is
                applied separately); lets the data pull be tempered vs. the
                source prior, mirroring the other baselines' ``guidance_scale``.
            precond_decay: RMSProp decay rho for the running second-moment
                estimate V used to build the diagonal preconditioner P (paper
                Eq. 20, "(diagonal) adaptive preconditioner from running
                second-moment statistics of the gradient").
            precond_eps: Floor lambda in P = 1 / (lambda + sqrt(V)); also the
                value at rho=... step 0. Keeps P bounded when V is tiny.
            variance: Fallback measurement variance R if the likelihood does not
                expose ``original_variance``.
        """
        # FM-ODE is deterministic; force g = 0 and the N(0, I) source init.
        super(DFlowPosterior, self).__init__(
            model=model,
            likelihood_model=likelihood_model,
            diffusion_term=lambda t: torch.zeros_like(t),
            gaussian_base=True,
        )
        self.num_optim_steps = int(num_optim_steps)
        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)
        self.guidance_scale = float(guidance_scale)
        self.precond_decay = float(precond_decay)
        self.precond_eps = float(precond_eps)
        # Gradient-checkpoint the differentiable rollout (O(1) activation memory
        # at ~2x compute); without it the backprop-through-ODE OOMs at realistic
        # ensemble size / num_steps.
        self.use_checkpoint = True

        self.obs_operator = likelihood_model.obs_operator
        self.original_variance = float(
            getattr(likelihood_model, "original_variance", variance)
        )

    def _one_step(self, *args, **kwargs):  # type: ignore[override]
        """Unused: D-Flow overrides ``sample`` with a full differentiable rollout."""
        raise NotImplementedError(
            "DFlowPosterior optimises the source latent through the whole ODE in "
            "``sample``; the per-step ``_one_step`` hook is not used."
        )

    def _flow(
        self,
        z0: torch.Tensor,
        num_steps: int,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Differentiable FM-ODE rollout x_1 = Phi(z_0) (forward Euler, g = 0).

        Builds the rollout from ``model.drift`` with grad ENABLED so
        ``d x_1 / d z_0`` flows for the SGLD gradient (the no-grad ``model.sample``
        would break the optimisation). Mirrors ``BaseModel._integrate`` /
        ``FlowMatchingPosterior._one_step`` (ODE branch: ``x += v * dt``) but
        WITHOUT the per-step ``detach``.

        MEMORY. A naive grad-enabled rollout stores every step's UNet activations
        for backward, which is O(num_steps) and OOMs at realistic E / num_steps.
        With ``self.use_checkpoint`` each Euler step is wrapped in gradient
        checkpointing: only the step INPUT is kept and the forward is recomputed
        during backward, cutting activation memory to O(1) at ~2x compute. Skipped
        automatically when ``x`` carries no grad (the final no-grad reconstruction).
        """
        t_vec = torch.linspace(0.0, 1.0, num_steps + 1, device=z0.device).unsqueeze(0)
        x = z0
        for i in range(num_steps):
            t = t_vec[:, i : i + 1]
            dt = t_vec[0, i + 1] - t_vec[0, i]

            def step(x_in: torch.Tensor, _t=t, _dt=dt) -> torch.Tensor:
                v = self.model.drift(x_in, _t, field_history, field_cond, pars_cond)
                return x_in + v * _dt

            if self.use_checkpoint and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(step, x, use_reentrant=False)
            else:
                x = step(x)
        return x

    def sample(  # type: ignore[override]
        self,
        base: torch.Tensor,
        batch_size: int,
        num_steps: int,
        field_history: torch.Tensor,
        observations: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        stepper: Callable = euler_maruyama_step,
        return_field_history: bool = False,
    ) -> torch.Tensor:
        """Sample one assimilation window via SGLD over the source latent z_0.

        For each ensemble member, initialise ``z_0 ~ N(0, I)`` then run K SGLD
        steps targeting ``p(z_0 | y) propto N(y; H Phi(z_0), R) N(z_0; 0, I)``,
        returning ``x_1 = Phi(z_0)``. The expensive backprop-through-the-flow is
        batched over members (``batch_size`` chunks) exactly like the base loop.
        """
        ensemble_size = field_history.shape[0]
        observations = observations.to(self.device)

        # Prepare the batch (broadcast a single-member window to the ensemble).
        if (batch_size > 1) and (field_history.shape[0] == 1):
            base, field_history, field_cond, pars_cond = self.model._prepare_batch(
                batch_size=batch_size,
                base=base,
                field_history=field_history,
                field_cond=field_cond,
                pars_cond=pars_cond,
            )
            ensemble_size = field_history.shape[0]

        fixed_input = lambda batch_ids: {
            "field_history": (
                field_history[batch_ids].to(self.device)
                if field_history is not None
                else None
            ),
            "field_cond": (
                field_cond[batch_ids].to(self.device)
                if field_cond is not None
                else None
            ),
            "pars_cond": (
                pars_cond[batch_ids].to(self.device) if pars_cond is not None else None
            ),
        }

        # FM source N(0, I) init (gaussian_base); independent latent per member.
        base = torch.randn_like(field_history[..., 0]) if base is None else base

        out = torch.empty_like(base)

        for batch_idx in range(0, ensemble_size, batch_size):
            batch_ids = torch.arange(
                batch_idx, min(batch_idx + batch_size, ensemble_size)
            )
            inp = fixed_input(batch_ids)
            # ``observations`` is one window's measurement [1, N_y] shared across
            # the ensemble; broadcasting in the residual handles the chunk size.
            y = (
                observations[batch_ids].to(self.device)
                if observations.shape[0] == ensemble_size
                else observations.to(self.device)
            )

            x1 = self._sgld_window(
                z0=base[batch_ids].to(self.device),
                observations=y,
                num_steps=num_steps,
                **inp,
            )
            out[batch_ids] = x1.detach().cpu()

        base = out

        if return_field_history:
            field_history = torch.cat(
                [field_history[:, :, :, :, 1:], base.unsqueeze(-1)], dim=-1
            )
            return base, field_history

        return base

    def _sgld_window(
        self,
        z0: torch.Tensor,
        observations: torch.Tensor,
        num_steps: int,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run K preconditioned-SGLD (pSGLD) steps over z_0 for one batch.

        Preconditioned SGLD (Li, Chen, Carlson & Carin, 2016), matching D-Flow
        SGLD Eq. 20: a per-element RMSProp diagonal preconditioner ``P`` built
        from the running second moment ``V`` of the energy gradient tempers the
        ``1/R``-scaled data gradient (which is otherwise ~400x the source-prior
        gradient and very anisotropic), and the Langevin noise is scaled by
        ``P^{1/2}`` so the stationary distribution is preserved:

            V   <- rho V + (1 - rho) g (.) g
            P   =  1 / (eps + sqrt(V))
            z   <- z - eta P (.) g + noise_scale sqrt(2 eta) sqrt(P) (.) xi.

        ``V`` resets each window (fresh z_0 ~ N(0, I) per window). noise_scale=0
        recovers preconditioned gradient descent (MAP D-Flow).
        """
        eta = self.step_size
        R = self.original_variance
        rho = self.precond_decay
        eps = self.precond_eps
        z = z0.detach().clone()
        V = torch.zeros_like(z)  # running second moment (per element), reset/window

        for k in range(1, max(self.num_optim_steps, 0) + 1):
            z_g = z.detach().requires_grad_(True)
            with torch.enable_grad():
                # Differentiable rollout x_1 = Phi(z_0).
                x1 = self._flow(
                    z_g, num_steps, field_history, field_cond, pars_cond
                )
                residual = observations - self.obs_operator(x1)  # [B, N_y]
                data_term = 0.5 * (residual.reshape(residual.shape[0], -1) ** 2).sum(
                    dim=1
                ) / R  # [B]
                prior_term = 0.5 * (z_g.reshape(z_g.shape[0], -1) ** 2).sum(dim=1)  # [B]
                loss = self.guidance_scale * data_term + prior_term
                grad = torch.autograd.grad(outputs=loss.sum(), inputs=z_g)[0]

            # Diagonal RMSProp preconditioner from the running 2nd moment, with
            # Adam-style bias correction (FIX 2): without it V ~ (1-rho) g^2 at
            # k=1 makes P ~ 1/(|g| sqrt(1-rho)) ~ 10x too large (cold-start
            # explosion). V_hat = V / (1 - rho^k) restores the published P form.
            V = rho * V + (1.0 - rho) * grad * grad
            V_hat = V / (1.0 - rho**k)
            P = 1.0 / (eps + V_hat.sqrt())

            # pSGLD update: preconditioned descent + P^{1/2}-scaled Langevin noise
            # (scaled by noise_scale; 0 -> preconditioned MAP descent).
            noise = (
                self.noise_scale * (2.0 * eta) ** 0.5 * P.sqrt() * torch.randn_like(z)
                if self.noise_scale != 0.0
                else torch.zeros_like(z)
            )
            z = (z_g - eta * P * grad).detach() + noise

        # Final clean reconstruction x_1 = Phi(z_0) (no grad needed).
        with torch.no_grad():
            x1 = self._flow(z, num_steps, field_history, field_cond, pars_cond)
        return x1
