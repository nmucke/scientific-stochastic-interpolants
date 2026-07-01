"""D-Flow SGLD posterior sampler -- BASELINE.

D-Flow SGLD (Parikh, Chen & Wang, "D-Flow SGLD: Source-Space Posterior Sampling
for Scientific Inverse Problems with Flow Matching", arXiv:2602.21469) turns the
D-Flow controlled-generation idea (Ben-Hamu, Puny, Gat, Karrer, Singer & Lipman,
"D-Flow: Differentiating through Flows for Controlled Generation", ICML 2024,
arXiv:2402.14017) into approximate POSTERIOR sampling. D-Flow controls generation
by optimising the SOURCE latent ``z_0`` so the deterministic FM-ODE flow
``Phi(z_0) = x_1`` matches a task cost, differentiating through the WHOLE ODE
solve (``d x_1 / d z_0``); the original paper minimises it with LBFGS for a single
MAP point. D-Flow SGLD instead runs preconditioned stochastic-gradient Langevin
dynamics (pSGLD) over ``z_0`` so each chain is a posterior sampler.

Following the paper's Algorithm 1 / Table D.3, the per-step pSGLD update over the
source-space energy is

    L1 = ||y - F(Phi(z_0))||^2                        (data misfit, Alg.1 L6)
    L  = L1 + lambda * ||z_0||^2                        (optional source reg, Eq. 31)
    V  <- omega V + (1 - omega) grad(L1) (.) grad(L1)  (RMSProp on data grad, L14)
    P  =  diag( 1 / (sqrt(V) + delta) )                (preconditioner, L15)
    z_0 <- z_0 - eta P grad(L) + xi,  xi ~ N(0, 2 eta s P)   (Langevin step, L19)

for ``Nsteps = num_optim_steps`` iterations, returning ``x_1 = Phi(z_0)``. Note
the data term is the PLAIN squared residual (no 1/(2R) factor -- the data/prior
balance is controlled by ``lambda``), the RMSProp second moment uses the DATA-term
gradient only, and there is no bias correction. With ``s = 0`` this reduces to
preconditioned gradient-descent D-Flow (MAP).

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
``t = 1`` clean field; ``x_t = (1-t) eps + t x_1``; velocity ``v = model.drift``.
The differentiable rollout uses the MIDPOINT (RK2) integrator with
``self.ode_steps`` steps (Table D.3: 6), starting from ``z_0`` at ``t=0``.
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
        num_optim_steps: int = 300,
        step_size: float = 5e-2,
        noise_scale: float = 1e-3,
        lambda_reg: float = 5e-6,
        precond_decay: float = 0.99,
        precond_eps: float = 1e-3,
        ode_steps: int = 6,
        burn: int = 100,
        guidance_scale: float = 1.0,
        variance: float = 0.05,
    ) -> None:
        """Initialize the D-Flow SGLD posterior.

        Parameter names follow the paper's Algorithm 1 / Table D.3 (Parikh, Chen &
        Wang, "D-Flow SGLD: Source-Space Posterior Sampling for Scientific Inverse
        Problems with Flow Matching", arXiv:2602.21469). The pSGLD update is::

            L1 = ||y - F(T_theta(x0))||^2                 (data term, Eq. Alg.1 L6)
            L  = L1 + lambda * ||x0||^2                    (optional source reg, Eq. 31)
            V  <- omega V + (1 - omega) grad(L1) (.) grad(L1)
            P  =  diag( 1 / (sqrt(V) + delta) )
            x0 <- x0 - eta P grad(L) + N(0, 2 eta s P)

        Args:
            model: Trained FM model (``FlowMatchingModel``); supplies the
                deterministic FM-ODE flow via ``model.drift``.
            likelihood_model: Carries the observation operator ``F`` / ``H``
                (``likelihood_model.obs_operator``); its ``score`` is NOT used
                (D-Flow differentiates the full rollout, not a per-step guidance
                term). The measurement variance ``R`` is NOT used in the loss --
                Algorithm 1's data term is the plain SSE ``||y - F(x1)||^2``.
            diffusion_term: Ignored (kept for interface symmetry); the D-Flow
                flow is the deterministic FM-ODE (g = 0).
            num_optim_steps: ``Nsteps``, number of pSGLD steps per window
                (Table D.3: 500 toy / 300 KS / 600 turb).
            step_size: pSGLD step size ``eta`` (Table D.3: 5e-2 toy/turb, 1e-2 KS).
            noise_scale: Langevin noise scale ``s``; the increment is
                ``N(0, 2 eta s P)`` (Table D.3: ``s(i)`` = 1e-2 toy, 1e-3 KS/turb).
                ``s = 0`` recovers preconditioned MAP descent (deterministic D-Flow).
            lambda_reg: ``lambda``, weight on the source-space regulariser
                ``lambda ||x0||^2`` (Eq. 31; Table D.3: 0.1/0.05 toy, 1e-3 KS,
                5e-6 turb). This is the single most important knob -- it balances
                measurement consistency against source-prior plausibility and its
                tuned value spans ~5 orders of magnitude across problems (shrinks
                steeply with dimension). ``0`` disables the regulariser (recovering
                the brittle off-manifold behaviour of deterministic D-Flow).
            precond_decay: RMSProp decay ``omega`` for the running second-moment
                ``V`` of the LIKELIHOOD (data-term) gradient only (Algorithm 1 L14;
                Table D.3: 0.99).
            precond_eps: Floor ``delta`` in ``P = 1 / (sqrt(V) + delta)``
                (Algorithm 1 L15; Table D.3: ``delta(i)`` = 1e-3).
            ode_steps: Number of steps for the differentiable FM-ODE rollout,
                integrated with the MIDPOINT (RK2) method (Table D.3, D-Flow /
                D-Flow SGLD column: "midpoint, 6 steps"). Decoupled from the global
                solver ``num_steps`` because the rollout is backpropagated through.
            burn: ``burn``, burn-in length (Table D.3: 100). Documented for
                reference; see ``sample`` for how the endpoint-per-chain collection
                maps onto the autoregressive harness.
            guidance_scale: Unused for D-Flow SGLD (the paper's ``b`` is a guidance,
                not a source-inference, hyperparameter). Kept for wiring symmetry.
            variance: Unused in the loss (see ``likelihood_model``); kept for
                interface symmetry.
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
        self.noise_scale = float(noise_scale)  # paper's ``s`` (noise scale)
        self.lambda_reg = float(lambda_reg)  # paper's ``lambda`` (source reg weight)
        self.precond_decay = float(precond_decay)  # paper's ``omega``
        self.precond_eps = float(precond_eps)  # paper's ``delta``
        self.ode_steps = int(ode_steps)  # midpoint rollout steps (Table D.3: 6)
        self.burn = int(burn)
        self.guidance_scale = float(guidance_scale)  # unused (kept for symmetry)
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
        """Differentiable FM-ODE rollout x_1 = Phi(z_0) (MIDPOINT / RK2, g = 0).

        Integrates ``dx/dt = v_theta(t, x)`` from ``t=0`` (source ``z_0``) to
        ``t=1`` with the explicit MIDPOINT method (Table D.3, D-Flow / D-Flow SGLD
        column: "midpoint, 6 steps")::

            k1 = v(t, x);  k2 = v(t + dt/2, x + dt/2 k1);  x <- x + dt k2

        Built from ``model.drift`` with grad ENABLED so ``d x_1 / d z_0`` flows for
        the SGLD gradient (the no-grad ``model.sample`` would break the
        optimisation). ``num_steps`` here is ``self.ode_steps`` (6), NOT the global
        solver step count -- the rollout is backpropagated through, so it uses the
        paper's short accurate midpoint schedule rather than the many-step Euler
        schedule of the guidance baselines.

        MEMORY. A naive grad-enabled rollout stores every step's UNet activations
        for backward, which is O(num_steps) and OOMs at realistic E / num_steps.
        With ``self.use_checkpoint`` each step is wrapped in gradient checkpointing:
        only the step INPUT is kept and the forward is recomputed during backward,
        cutting activation memory to O(1) at ~2x compute. Skipped automatically when
        ``x`` carries no grad (the final no-grad reconstruction).
        """
        t_vec = torch.linspace(0.0, 1.0, num_steps + 1, device=z0.device).unsqueeze(0)
        x = z0
        for i in range(num_steps):
            t = t_vec[:, i : i + 1]
            dt = t_vec[0, i + 1] - t_vec[0, i]

            def step(x_in: torch.Tensor, _t=t, _dt=dt) -> torch.Tensor:
                # Explicit midpoint (RK2): evaluate the drift at the interval
                # midpoint of an Euler half-step, then take the full step with it.
                k1 = self.model.drift(x_in, _t, field_history, field_cond, pars_cond)
                x_mid = x_in + 0.5 * _dt * k1
                k2 = self.model.drift(
                    x_mid, _t + 0.5 * _dt, field_history, field_cond, pars_cond
                )
                return x_in + _dt * k2

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
        """Sample one assimilation window via pSGLD over the source latent z_0.

        Each ensemble member runs an independent pSGLD chain (Algorithm 1) over
        its own source latent ``z_0 ~ N(0, I)``, conditioned on that member's own
        ``field_history``, and returns the post-burn endpoint ``x_1 = Phi(z_0)``.

        COLLECTION NOTE. Algorithm 1 runs ``Nparallel`` chains and collects EVERY
        post-burn (optionally thinned) source sample ``{x0^i}_{i >= burn}`` across
        chains (Eq. 22). That single-shot collection assumes all chains share one
        conditioning context. In this autoregressive DA harness each ensemble
        member instead carries its OWN evolving ``field_history`` forward between
        windows, so we map "Nparallel chains x collected samples" onto "one
        independent chain per member, take its endpoint": with
        ``num_optim_steps > burn`` the endpoint is a valid post-burn-in draw, and
        the per-member history threading is preserved. The rollout uses
        ``self.ode_steps`` midpoint steps, NOT the global ``num_steps``.
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
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor],
        pars_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run ``Nsteps`` preconditioned-SGLD (pSGLD) steps over z_0 for one batch.

        Preconditioned SGLD (Li, Chen, Carlson & Carin, 2016), matching D-Flow
        SGLD Algorithm 1. The data term is the plain squared measurement residual
        ``L1 = ||y - F(Phi(z_0))||^2`` (Algorithm 1 L6 -- NO 1/(2R) factor; the
        data/prior balance is set by ``lambda``, not by R). The optional source
        regulariser is ``L2 = lambda ||z_0||^2`` (Eq. 31), so the total loss is
        ``L = L1 + L2``. The full gradient ``grad L = grad L1 + 2 lambda z_0``
        drives the preconditioned descent, while the diagonal RMSProp
        preconditioner ``P`` is built from the running second moment ``V`` of the
        DATA-term gradient ``grad L1`` ONLY (L14):

            V   <- omega V + (1 - omega) grad(L1) (.) grad(L1)
            P   =  1 / (sqrt(V) + delta)
            z   <- z - eta P (.) grad(L) + xi,   xi ~ N(0, 2 eta s P).

        ``V`` resets each window (fresh z_0 ~ N(0, I) per window). ``s = 0``
        recovers preconditioned gradient descent (MAP D-Flow).
        """
        eta = self.step_size
        s = self.noise_scale
        lam = self.lambda_reg
        omega = self.precond_decay
        delta = self.precond_eps
        z = z0.detach().clone()
        V = torch.zeros_like(z)  # running second moment (per element), reset/window

        for _ in range(max(self.num_optim_steps, 0)):
            z_g = z.detach().requires_grad_(True)
            with torch.enable_grad():
                # Differentiable rollout x_1 = Phi(z_0) (midpoint, self.ode_steps).
                x1 = self._flow(
                    z_g, self.ode_steps, field_history, field_cond, pars_cond
                )
                residual = observations - self.obs_operator(x1)  # [B, N_y]
                # Data term L1 = ||y - F(Phi(z_0))||^2 (Algorithm 1 L6).
                data_term = (residual.reshape(residual.shape[0], -1) ** 2).sum(
                    dim=1
                )  # [B]
                # Data-term gradient grad L1 (drives P and the descent). The source
                # regulariser gradient is analytic (= 2 lambda z_0), so only L1 is
                # differentiated through the flow.
                data_grad = torch.autograd.grad(
                    outputs=data_term.sum(), inputs=z_g
                )[0]

            # Full loss gradient grad L = grad L1 + lambda grad(||z_0||^2), with
            # the source regulariser lambda ||z_0||^2 (Eq. 31) so grad = 2 lambda
            # z_0 (added in closed form).
            grad = data_grad + 2.0 * lam * z_g.detach()

            # Diagonal RMSProp preconditioner from the running 2nd moment of the
            # DATA-term gradient ONLY (Algorithm 1 L14-15). No bias correction --
            # canonical pSGLD / the paper use the raw running V.
            V = omega * V + (1.0 - omega) * data_grad * data_grad
            P = 1.0 / (delta + V.sqrt())

            # pSGLD update (Algorithm 1 L19): preconditioned descent + Langevin
            # noise xi ~ N(0, 2 eta s P), i.e. sqrt(2 eta s) P^{1/2} (.) standard
            # normal (s = 0 -> preconditioned MAP descent).
            noise = (
                (2.0 * eta * s) ** 0.5 * P.sqrt() * torch.randn_like(z)
                if s != 0.0
                else torch.zeros_like(z)
            )
            z = (z_g - eta * P * grad).detach() + noise

        # Final clean reconstruction x_1 = Phi(z_0) (no grad needed).
        with torch.no_grad():
            x1 = self._flow(z, self.ode_steps, field_history, field_cond, pars_cond)
        return x1
