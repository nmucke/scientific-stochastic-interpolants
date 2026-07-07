"""Trainer for stochastic-interpolant-style models.

The module path is kept so existing configs and checkpointed ``config.yaml``s
referencing ``scisi.training.trainer.Trainer`` / ``...trainer.EarlyStopping``
keep working. All generic training machinery lives in
``scisi.training.base_trainer.BaseTrainer``.
"""

import torch

from scisi.training.base_trainer import (  # noqa: F401  (re-exports)
    SCHEDULERS_THAT_REQUIRE_LOSS,
    BaseTrainer,
    EarlyStopping,
)


class StochasticInterpolantTrainer(BaseTrainer):
    """Trainer for the stochastic interpolant.

    Serves Follmer stochastic interpolants, flow matching, and diffusion
    models: samples a pseudo-time and an interpolant noise per batch and
    delegates the model-specific math to ``model.forward``.
    """

    def _prepare_batch(self, batch: dict) -> dict[str, torch.Tensor]:
        """Prepare the batch: device transfer plus sampled time and noise."""
        batch = super()._prepare_batch(batch)

        # Sample pseudo-time
        batch["t"] = torch.rand(batch["base"].shape[0], 1, device=self.device)

        # Sample noise
        batch["noise"] = torch.randn(batch["base"].shape, device=self.device)

        return batch


# Back-compat alias: existing configs and checkpointed config.yamls reference
# scisi.training.trainer.Trainer.
Trainer = StochasticInterpolantTrainer
