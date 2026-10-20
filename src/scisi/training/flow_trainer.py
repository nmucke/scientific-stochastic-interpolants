"""Training for flow models: flow matching, stochastic interpolants and diffusion."""

from scisi.training.trainer import Trainer


class FlowTrainer(Trainer):
    """Regress the model's drift (or denoiser) onto the interpolant target."""

    def _compute_loss(self, batch):
        prediction, target = self.model(**batch)
        return self.loss_fn(prediction, target)
