"""Trainer for deterministic time-stepping models."""

from scisi.training.base_trainer import BaseTrainer


class DeterministicTrainer(BaseTrainer):
    """Trainer for deterministic next-step prediction models.

    Plain MSE training needs nothing beyond the generic machinery: the
    inherited ``_prepare_batch`` (device transfer only) and ``_compute_loss``
    (``(pred, target) = model(**batch)`` followed by ``loss_fn``) already do
    the right thing. The subclass exists for config readability and future
    divergence (e.g. rollout-based validation).
    """
