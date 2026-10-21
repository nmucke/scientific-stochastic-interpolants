"""Training for conditional deterministic and stochastic finite-time maps.

The objective is the diagonal regression plus an off-diagonal self-distillation
loss (Boffi et al., arXiv:2505.18825; Pan et al., arXiv:2606.11156), with an
EMA copy of the student providing the targets:

* ``progressive_self_distillation``: the map over [s, t] matches two composed
  EMA maps over [s, u] and [u, t] (semigroup property).
* ``lagrangian_self_distillation``: the time derivative of the map's drift
  displacement matches the EMA drift at the mapped state.
"""

import copy
import logging

import torch
import torch.autograd.forward_ad as fwAD

from scisi.training.trainer import Trainer

logger = logging.getLogger(__name__)


def _expand(time):
    """[B, 1] -> [B, 1, 1, 1] to broadcast over fields."""
    return time.reshape(-1, 1, 1, 1)


def progressive_self_distillation_loss(
    model, ema_model, loss_fn, x, s, u, t, path, conditions
):
    """X_{s,t} = X_{u,t} o X_{s,u}, written for average velocities.

    Two EMA half-steps define the target average velocity over [s, t], sharing
    one Brownian path.
    """
    with torch.no_grad():
        g1 = ema_model.average_velocity(x, s, u, brownian_path=path, **conditions)
        mid = x + _expand(u - s) * g1
        if path is not None:
            mid = mid + path.increment(s, u)
        g2 = ema_model.average_velocity(mid, u, t, brownian_path=path, **conditions)
        fraction = _expand((u - s) / (t - s))
        target = fraction * g1 + (1 - fraction) * g2
    prediction = model.average_velocity(x, s, t, brownian_path=path, **conditions)
    return loss_fn(prediction, target)


def lagrangian_self_distillation_loss(
    model, ema_model, loss_fn, x, s, u, t, path, conditions
):
    """d/dt D_{s,t}(x) = b_t(X_{s,t}(x)), with D_{s,t} = (t - s) v_{s,t}.

    D is the drift part of the map (the martingale increment is added
    separately), so d/dt D = v_{s,t} + (t - s) d/dt v_{s,t} is matched to the
    EMA diagonal drift b_t = v_{t,t} at the stop-gradient mapped state
    X_{s,t} = x + D + (M_t - M_s) (Pan et al., Alg. 1). The derivative is one
    forward-mode JVP in t, which stays differentiable for the parameter update.

    The derivative is taken with the Brownian features held fixed: path queries
    detach their times, so the tangent only flows through the explicit time
    input. With Karhunen-Loeve features, which do not depend on (s, t), this is
    the exact derivative. Local increment features move with t and are not
    differentiable in t, so for them this is a frozen-feature approximation of
    the Lagrangian condition. As for the diagonal loss, the target diagonal
    drift receives zero path features. ``u`` is unused.

    Dropout is disabled in this pass (as in Boffi et al.'s reference code):
    dropout noise in a differentiated forward makes the time derivative and
    its gradient noisy and, in bf16, unstable.
    """
    was_training = model.training
    model.eval()
    try:
        with fwAD.dual_level():
            time = fwAD.make_dual(t, torch.ones_like(t))
            velocity = model.average_velocity(
                x, s, time, brownian_path=path, **conditions
            )
            displacement, prediction = fwAD.unpack_dual(_expand(time - s) * velocity)
    finally:
        model.train(was_training)
    with torch.no_grad():
        mapped = x + displacement
        if path is not None:
            mapped = mapped + path.increment(s, t)
        target = ema_model.average_velocity(mapped, t, t, **conditions)
    return loss_fn(prediction, target)


OFF_DIAGONAL_LOSSES = {
    "progressive_self_distillation": progressive_self_distillation_loss,
    "lagrangian_self_distillation": lagrangian_self_distillation_loss,
}


class MapTrainer(Trainer):
    """Self-distillation of a map model with an EMA teacher.

    ``off_diagonal`` selects the off-diagonal loss (see OFF_DIAGONAL_LOSSES).

    Besides the base checkpoints, saves ``ema.pth`` (EMA weights of the best
    epoch) and includes the EMA model in ``last.pth``.
    """

    def __init__(
        self,
        *args,
        off_diagonal="progressive_self_distillation",
        off_diagonal_weight=1.0,
        diagonal_epochs=0,
        warmup_epochs=0,
        endpoint_probability=0.25,
        ema_decay=0.99,
        **kwargs,
    ):
        # Interpolant targets do not implement masked dynamics.
        model = kwargs.get("model", args[1] if len(args) > 1 else None)
        mask = getattr(model, "mask", None)
        if mask is not None and not torch.all(torch.as_tensor(mask) == 1):
            raise ValueError("MapTrainer requires an identically-one mask")
        super().__init__(*args, **kwargs)
        self.off_diagonal_weight = float(off_diagonal_weight)
        self.diagonal_epochs = int(diagonal_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.endpoint_probability = float(endpoint_probability)
        self.ema_decay = float(ema_decay)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.validating = False
        if off_diagonal not in OFF_DIAGONAL_LOSSES:
            raise ValueError(
                f"Unknown off_diagonal loss {off_diagonal!r}; "
                f"choose from {sorted(OFF_DIAGONAL_LOSSES)}"
            )
        self.off_diagonal_loss = OFF_DIAGONAL_LOSSES[off_diagonal]
        if (
            off_diagonal == "lagrangian_self_distillation"
            and getattr(self.model, "brownian_features", None) == "local"
        ):
            logger.warning(
                "Lagrangian self-distillation with local Brownian features holds "
                "the features fixed in t, an approximation; Karhunen-Loeve "
                "features make the time derivative exact."
            )

    def _intervals(self, batch_size):
        """Sample s < u < t, with an atom at the full interval [0, 1]."""
        length = 0.01 + 0.99 * torch.rand(batch_size, 1, device=self.device)
        s = torch.rand_like(length) * (1 - length)
        t = s + length
        endpoint = torch.rand_like(length) < self.endpoint_probability
        s = torch.where(endpoint, torch.zeros_like(s), s)
        t = torch.where(endpoint, torch.ones_like(t), t)
        u = s + (t - s) * (0.25 + 0.5 * torch.rand_like(s))
        return s, u, t

    def _current_off_diagonal_weight(self):
        """Zero for the first `diagonal_epochs`, then a linear warmup in epochs."""
        if self.validating:
            return self.off_diagonal_weight
        epoch = self.epoch - self.diagonal_epochs
        if epoch < 0:
            return 0.0
        if self.warmup_epochs == 0:
            return self.off_diagonal_weight
        return self.off_diagonal_weight * min(1.0, (epoch + 1) / self.warmup_epochs)

    def _compute_loss(self, batch):
        conditions = {
            k: batch[k]
            for k in ("field_history", "field_cond", "pars_cond")
            if k in batch
        }

        # Diagonal: the instantaneous velocity matches the interpolant drift.
        x_diag, target_diag = self.model.training_pair(
            batch["base"], batch["target"], batch["t"], batch["noise"]
        )
        diagonal = self.loss_fn(
            self.model.average_velocity(x_diag, batch["t"], batch["t"], **conditions),
            target_diag,
        )

        weight = self._current_off_diagonal_weight()
        if weight == 0:  # diagonal-only phase
            self.components = {"diagonal_loss": diagonal.item(), "off_diagonal_loss": 0.0}
            return diagonal

        # Off-diagonal: self-distillation over random intervals s < u < t.
        s, u, t = self._intervals(batch["base"].shape[0])
        x, _ = self.model.training_pair(
            batch["base"], batch["target"], s, torch.randn_like(batch["base"])
        )
        path = self.model.make_brownian_path(x)
        off_diagonal = self.off_diagonal_loss(
            self.model, self.ema_model, self.loss_fn, x, s, u, t, path, conditions
        )

        self.components = {
            "diagonal_loss": diagonal.item(),
            "off_diagonal_loss": off_diagonal.item(),
        }
        return diagonal + weight * off_diagonal

    def _after_step(self):
        with torch.no_grad():
            for average, current in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                average.lerp_(current, 1 - self.ema_decay)
            for average, current in zip(self.ema_model.buffers(), self.model.buffers()):
                average.copy_(current)

    def _validate(self):
        self.validating = True
        try:
            return super()._validate()
        finally:
            self.validating = False

    def checkpoint_state(self):
        return {**super().checkpoint_state(), "ema_model": self.ema_model.state_dict()}

    def load_checkpoint_state(self, state):
        super().load_checkpoint_state(state)
        self.ema_model.load_state_dict(state["ema_model"])

    def save_checkpoint(self):
        super().save_checkpoint()
        if self.early_stopping.save_checkpoint:
            torch.save(self.ema_model.state_dict(), self.checkpoint_path / "ema.pth")
