"""TMPD (Tweedie Moment Projected Diffusions) likelihood -- BASELINE.

Faithful PyTorch port of the observation guidance of Boys et al., "Tweedie
Moment Projected Diffusions for Inverse Problems" (arXiv 2310.06721; reference
JAX code ``bb515/tmpdjax``, guidance functions in ``diffusionjax/
inverse_problems.py``), read on our FM prior as a diffusion model (DM) via the
score--velocity duality, exactly like the SDA / DPS baselines.

TMPD IN ITS OWN (VP) NOTATION
-----------------------------
VP forward marginal ``x_t = sqrt(a_t) x_0 + sqrt(v_t) eps``. Tweedie's first and
second moments of ``x_0 | x_t`` are

    m_t = E[x_0 | x_t]   = (x_t + v_t s) / sqrt(a_t),
    C_t = Cov[x_0 | x_t] = (v_t / a_t) (I + v_t grad^2 log p_t)
                         = (v_t / sqrt(a_t)) grad_{x_t} m_t,

and TMPD projects ``p(y | x_t)`` onto the Gaussian
``N(H m_t, H C_t H^T + sigma_y^2 I)``. Its guidance (C held CONSTANT, only m
differentiated) is

    f^y = (grad_{x_t} m_t)^T H^T (H C_t H^T + sigma_y^2 I)^{-1} (y - H m_t),

which REPLACES the score ``s -> s + f^y`` in the reverse SDE (no guidance
scale, no thresholding). ``v_t / sqrt(a_t)`` is ``sde.ratio`` in diffusionjax.

MAPPING ONTO OUR AFFINE PATH
----------------------------
Our path (pseudo-time ``tau: 0 -> 1``, noise -> data) is
``x_tau = alpha_tau a0 + beta_tau x_1 + sigma_tau eps`` with the FM anchor
``a0 = 0`` and ``sigma_tau = alpha_tau``. It is the VP marginal above with
``sqrt(a_t) <-> beta_tau`` and ``v_t <-> sigma_tau^2`` (data ``x_0`` of TMPD is
our ``x_1``). Derivation (anchor kept general; it is zero for FM): with
``x' = x_tau - alpha a0 = beta x_1 + sigma eps``, Tweedie gives
``E[beta x_1 | x_tau] = x' + sigma^2 s`` and
``Cov[beta x_1 | x_tau] = sigma^2 (I + sigma^2 grad^2 log p_tau)``, hence

    xhat_1 = E[x_1 | x_tau]   = (x_tau - alpha a0 + sigma^2 s) / beta,
    grad_{x_tau} xhat_1       = (I + sigma^2 grad^2 log p_tau) / beta,
    C_tau  = Cov[x_1 | x_tau] = (sigma^2 / beta^2)(I + sigma^2 grad^2 log p_tau)
                              = (sigma_tau^2 / beta_tau) grad_{x_tau} xhat_1,

i.e. the TMPD ``ratio`` is ``sigma_tau^2 / beta_tau`` (the anchor term is
constant in ``x_tau`` and drops out of the Jacobian). The score ``s`` is
recovered from the trained velocity by ``interpolation.score_from_velocity``
with ``a0 = 0`` (``DenoiseDiffusionModel`` velocity mode / FM model), or from
the noise net in denoise mode.

COVARIANCE VARIANTS (all from reverse-mode VJPs through the network; with
``J_b = grad_{x_tau} xhat_1`` of ensemble member ``b`` and
``K_b = J_b^T H^T`` its ``d_y`` VJP columns):

* ``"full"`` (TMPD, ``get_vjp_guidance``): ``d_y`` VJPs build ``K_b``;
  ``H C H^T = ratio * H K_b`` (as in the reference code, NOT symmetrised: for a
  network Jacobian it is only approximately symmetric); per-member dense solve
  ``(ratio H K_b + sigma_y^2 I) u_b = y - H xhat_1``; guidance ``K_b u_b``.
  The returned log-density uses the SYMMETRISED ``Cyy`` (a Gaussian density
  needs a symmetric covariance); the guidance solve keeps the reference's
  unsymmetrised one. Both coincide for an exact (symmetric) Jacobian.
* ``"diagonal"`` (``get_diag_vjp_guidance``): same ``d_y`` VJPs, keep only the
  diagonal of ``H K_b``; elementwise division; guidance ``K_b u_b``.
* ``"row_sum"`` (DTMPD, ``get_vjp_guidance_mask``; the variant used for all
  TMPD image experiments): one VJP ``J^T H^T 1``, diagonal
  ``ratio * H J^T H^T 1 + sigma_y^2`` (row sums of ``H C H^T``), elementwise
  division, and one more VJP for the guidance. The reference code forms the
  ones vector as ``H 1_x``, which equals ``1_y`` for the selection (mask)
  operators it targets; we use ``1_y`` (the paper's row-sum definition), which
  coincides with the code for selection operators.

TMPD is defined for linear ``H`` with Gaussian noise only (nonlinear ``h`` is
a stated limitation of the paper), so a nonlinear operator
(``obs_operator.is_linear`` False) raises ``NotImplementedError``.

This is deliberately NOT our method's closed-form guidance (no forward-mode
JVP / symmetry trick, no ensemble-shared Jacobian, no Jacobian damping, no
refresh-every-k): every step and every member gets its own reverse-mode
Jacobian, as in TMPD.

Returns ``(guidance, log_likelihood)`` like the other likelihoods; the guided
reverse SDE is run by :class:`scisi.posterior_models.tmpd_posterior.TMPDPosterior`.

COST: ``vjp_count`` counts the reverse-mode VJPs actually taken (one VJP = one
backward pass through the network over the whole batch, as one NFE is one
batched forward): ``d_y`` per guided step for ``"full"`` / ``"diagonal"`` (the
guidance reuses ``K``), 2 for ``"row_sum"``, 0 below ``MIN_TIME``. These are
NOT in the network-forward NFE the drivers report; the drivers report them
separately.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn

from scisi.likelihood_models.observation_operators import LinearObservationOperator

MIN_TIME = 1e-4

COVARIANCE_OPTIONS = ("full", "diagonal", "row_sum")


def _t_grid(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape ``t`` ([B, 1] / [1, 1] / [B]) to broadcast against ``x``."""
    if t.dim() < x.dim():
        return t.reshape(t.shape[0], *([1] * (x.dim() - 1)))
    return t


def _is_vmap_error(error: RuntimeError) -> bool:
    """True for the "no batching rule" failure of ``is_grads_batched``."""
    message = str(error).lower()
    return "vmap" in message or "batching rule" in message or "batched" in message


def prior_velocity_and_score(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    field_history: Optional[torch.Tensor],
    field_cond: Optional[torch.Tensor],
    pars_cond: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prior FM velocity ``v`` and DM score ``s`` from ONE network evaluation.

    Supports a velocity-mode ``DenoiseDiffusionModel`` (``from_flow_matching``),
    a denoise-mode ``DenoiseDiffusionModel`` (``s = -eps/alpha``) and a plain
    ``FlowMatchingModel`` (``drift`` is the velocity). The anchor is ``a0 = 0``
    throughout (diffusion / FM source ``N(0, I)``).
    """
    if t.dim() == 1:
        t = t.reshape(-1, 1)
    interp = model.interpolation
    tg = _t_grid(t, x)
    zeros = torch.zeros_like(x)
    use_velocity = getattr(model, "_use_velocity", None)
    if use_velocity is False:
        # Denoise-mode diffusion prior: the network predicts the noise.
        s = model.score(x, t, field_history, field_cond, pars_cond)
        v = interp.velocity_from_score(x=x, s=s, t=tg, a0=zeros)
        return v, s
    if use_velocity is True:
        v = model.velocity_model(x, t, field_history, field_cond, pars_cond)
    else:
        # FlowMatchingModel: drift IS the velocity.
        v = model.drift(x, t, field_history, field_cond, pars_cond)
    s = interp.score_from_velocity(x=x, v=v, t=tg, a0=zeros)
    return v, s


class TMPDLikelihood(nn.Module):
    """TMPD Gaussian moment-projection guidance (see module docstring)."""

    def __init__(
        self,
        model: nn.Module,
        obs_operator: nn.Module = LinearObservationOperator,
        variance: float = 0.05,
        ensemble_size: int = 1,
        model_class: str = "fm",
        covariance: str = "full",
        vjp_chunk_size: int = 64,
    ) -> None:
        """
        Args:
            model: Diffusion prior (``DenoiseDiffusionModel``, typically built
                with ``from_flow_matching``) or a ``FlowMatchingModel``.
            obs_operator: LINEAR observation operator ``H`` (``is_linear``).
            variance: Observation-noise variance ``sigma_y^2``.
            ensemble_size: Kept for interface parity with the other
                likelihoods; every batch member is treated independently
                (per-member covariance and solve).
            model_class: Only ``"fm"`` (FM prior read as a DM, anchor 0).
            covariance: ``"full"`` | ``"diagonal"`` | ``"row_sum"``.
            vjp_chunk_size: Number of VJP cotangents batched per autograd call
                when building ``K = J^T H^T`` (``"full"`` / ``"diagonal"``).
        """
        super(TMPDLikelihood, self).__init__()
        if model_class != "fm":
            raise ValueError(
                "TMPDLikelihood is a diffusion-model baseline run on the FM prior "
                f"(model_class='fm'); got {model_class!r}."
            )
        if covariance not in COVARIANCE_OPTIONS:
            raise ValueError(
                f"covariance must be one of {COVARIANCE_OPTIONS}, got {covariance!r}."
            )
        self._check_linear(obs_operator)
        self.model = model
        self.obs_operator = obs_operator
        self.original_variance = float(variance)
        self.ensemble_size = ensemble_size
        self.model_class = model_class
        self.anchor = "zeros"
        self.covariance = covariance
        self.vjp_chunk_size = int(vjp_chunk_size)
        self.interpolant = self.model.interpolation
        # Reverse-mode VJPs taken so far (see module docstring, COST).
        self.vjp_count = 0
        # Batched VJPs (``is_grads_batched``) until an op without a vmap rule
        # forces the per-cotangent loop; remembered so it is tried only once.
        self._batched_vjp = True
        # Linearity is a property of the operator: checked once per operator.
        self._linear_checked_id = id(obs_operator)

    @staticmethod
    def _check_linear(obs_operator: Any) -> None:
        if not getattr(obs_operator, "is_linear", False):
            raise NotImplementedError(
                "TMPD is defined for a LINEAR observation operator with Gaussian "
                "noise only (Boys et al.; nonlinear h is a stated limitation). Got "
                f"{type(obs_operator).__name__} with is_linear=False."
            )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Unused (score interface is the entry point)."""
        pass

    # ------------------------------------------------------------------
    # VJP helpers
    # ------------------------------------------------------------------
    def _vjp_columns(
        self, hm: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """``K_b = J_b^T H^T`` for every member: ``[B, d, d_y]`` via ``d_y`` VJPs.

        Cotangent ``e_i`` (broadcast over the batch) on ``H xhat_1`` gives, per
        member, ``J_b^T H^T e_i`` (members are independent in the network).
        """
        b, d_y = hm.shape
        d = x[0].numel()
        eye = torch.eye(d_y, dtype=hm.dtype, device=hm.device)
        cols = []
        for start in range(0, d_y, self.vjp_chunk_size):
            stop = min(start + self.vjp_chunk_size, d_y)
            cot = eye[start:stop].unsqueeze(1).expand(stop - start, b, d_y)
            g = None
            if self._batched_vjp:
                try:
                    g = torch.autograd.grad(
                        hm, x, grad_outputs=cot, retain_graph=True, is_grads_batched=True
                    )[0]
                except RuntimeError as error:
                    # Only an op without a vmap (batching) rule falls back; OOM
                    # and every other failure propagate.
                    if isinstance(error, torch.OutOfMemoryError) or not _is_vmap_error(
                        error
                    ):
                        raise
                    self._batched_vjp = False
            if g is None:
                # One VJP per cotangent.
                g = torch.stack(
                    [
                        torch.autograd.grad(hm, x, grad_outputs=c, retain_graph=True)[0]
                        for c in cot
                    ]
                )
            self.vjp_count += stop - start
            cols.append(g.reshape(stop - start, b, d))
        return torch.cat(cols, dim=0).permute(1, 2, 0)  # [B, d, d_y]

    def _apply_H_to_columns(self, K: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """``H K_b`` for ``K`` of shape ``[B, d, k]`` -> ``[B, d_y, k]``."""
        b, d, k = K.shape
        cols = K.permute(0, 2, 1).reshape(b * k, *x_shape[1:])
        return self.obs_operator(cols).reshape(b, k, -1).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------
    def guided_terms(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: Optional[torch.Tensor],
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """TMPD guidance plus the prior ``(v, s)`` of the SAME network call.

        Returns a dict with detached tensors ``guidance`` ``[B, ...]``,
        ``log_likelihood`` ``[B]`` (``log N(y; H xhat_1, H C H^T + R)``, the
        diagonal density for the diagonal variants), ``velocity`` and
        ``score`` (prior, for the posterior's reverse-SDE drift) and
        ``xhat1``. Below ``MIN_TIME`` (``beta -> 0``) the guidance and
        log-likelihood are zero and only the prior terms are computed.
        """
        if id(self.obs_operator) != self._linear_checked_id:
            self._check_linear(self.obs_operator)  # operator swapped after init
            self._linear_checked_id = id(self.obs_operator)
        if t.dim() == 1:
            t = t.reshape(-1, 1)
        b = x.shape[0]
        apply_guidance = float(t.reshape(-1).min()) >= MIN_TIME
        t_c = torch.clamp(t, min=MIN_TIME, max=1.0 - MIN_TIME)

        if not apply_guidance:
            with torch.no_grad():
                v, s = prior_velocity_and_score(
                    self.model, x.detach(), t_c, field_history, field_cond, pars_cond
                )
            return {
                "guidance": torch.zeros_like(x),
                "log_likelihood": torch.zeros(b, dtype=x.dtype, device=x.device),
                "velocity": v.detach(),
                "score": s.detach(),
                "xhat1": torch.full_like(x, float("nan")),
            }

        # Detached grad-enabled copy: never mutate the caller's state in place.
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v, s = prior_velocity_and_score(
                self.model, x_g, t_c, field_history, field_cond, pars_cond
            )
            tg = _t_grid(t_c, x_g)
            beta = self.interpolant.beta(tg)
            sigma = self.interpolant.sigma(tg)
            # Tweedie mean of x_1 (FM anchor a0 = 0).
            xhat1 = (x_g + sigma**2 * s) / beta
            hm = self.obs_operator(xhat1)  # [B, d_y]
            residual = observations.to(hm.dtype) - hm
            # TMPD ratio v_t / sqrt(a_t) <-> sigma_tau^2 / beta_tau (obs-space shape).
            ratio = (sigma**2 / beta).reshape(-1, 1)  # [1 or B, 1]
            r_var = self.original_variance

            if self.covariance == "row_sum":
                ones = torch.ones_like(hm)
                w = torch.autograd.grad(hm, x_g, grad_outputs=ones, retain_graph=True)[0]
                diag = ratio * self.obs_operator(w) + r_var  # [B, d_y]
                u = residual.detach() / diag
                guidance = torch.autograd.grad(hm, x_g, grad_outputs=u)[0]
                self.vjp_count += 2
                log_lik = self._diag_log_lik(residual.detach(), diag)
            else:
                K = self._vjp_columns(hm, x_g)  # [B, d, d_y]
                HCHt = ratio.unsqueeze(-1) * self._apply_H_to_columns(K, x_g.shape)
                if self.covariance == "diagonal":
                    diag = torch.diagonal(HCHt, dim1=-2, dim2=-1) + r_var
                    u = residual.detach() / diag
                    log_lik = self._diag_log_lik(residual.detach(), diag)
                else:  # full: per-member dense solve (float64 for stability)
                    # MPS has no float64: solve on the CPU there (d_y x d_y only).
                    solve_dev = (
                        torch.device("cpu") if hm.device.type == "mps" else hm.device
                    )
                    Cyy = HCHt.detach().to(solve_dev).to(torch.float64)
                    Cyy = Cyy + r_var * torch.eye(
                        Cyy.shape[-1], dtype=Cyy.dtype, device=Cyy.device
                    )
                    r64 = residual.detach().to(solve_dev).to(torch.float64)
                    # Guidance: the reference's unsymmetrised Cyy.
                    u64 = torch.linalg.solve(Cyy, r64.unsqueeze(-1)).squeeze(-1)
                    u = u64.to(hm.dtype).to(hm.device)
                    # Log-density: the symmetrised Cyy (a Gaussian covariance).
                    Cyy_sym = 0.5 * (Cyy + Cyy.transpose(-1, -2))
                    w64 = torch.linalg.solve(Cyy_sym, r64.unsqueeze(-1)).squeeze(-1)
                    _, logdet = torch.linalg.slogdet(Cyy_sym)
                    log_lik = (
                        -0.5 * (r64 * w64).sum(-1)
                        - 0.5 * logdet
                        - 0.5 * Cyy.shape[-1] * math.log(2 * math.pi)
                    ).to(hm.dtype).to(hm.device)
                # Guidance K_b u_b = J_b^T H^T u_b (reference code reuses K).
                guidance = torch.bmm(K, u.unsqueeze(-1)).reshape(x_g.shape)

        return {
            "guidance": guidance.detach(),
            "log_likelihood": log_lik.detach(),
            "velocity": v.detach(),
            "score": s.detach(),
            "xhat1": xhat1.detach(),
        }

    @staticmethod
    def _diag_log_lik(residual: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        # ``|diag|`` in the log-determinant: the DTMPD row-sum diagonal is not
        # guaranteed positive for a general (non-selection) H (row sums of
        # H C H^T can be negative); the guidance itself uses the signed value,
        # exactly as the reference code does.
        return -0.5 * (
            residual**2 / diag + torch.log(diag.abs()) + math.log(2 * math.pi)
        ).sum(-1)

    def score(
        self,
        observations: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        field_history: torch.Tensor,
        field_cond: Optional[torch.Tensor] = None,
        pars_cond: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TMPD guidance ``f^y`` (the score increment) and ``log p(y | x_tau)``."""
        out = self.guided_terms(
            observations, x, t, field_history, field_cond, pars_cond
        )
        return out["guidance"], out["log_likelihood"]


__all__ = ["TMPDLikelihood", "prior_velocity_and_score"]
