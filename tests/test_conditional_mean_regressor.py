"""The conditional-mean regressor trains with the FlowTrainer on any shape."""

import torch
from torch import nn

from scisi.models.conditional_mean_regressor import ConditionalMeanRegressor
from scisi.training.flow_trainer import FlowTrainer


class Shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(3, 1, 1))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None):
        assert torch.all(cond == 0)
        return self.shift.expand_as(x)


def test_regressor_learns_the_mean_increment(tmp_path):
    base = torch.randn(8, 3, 6, 5)  # C=3, non-square
    offset = torch.tensor([1.0, -2.0, 0.5]).reshape(1, 3, 1, 1)
    batch = {"base": base, "target": base + offset, "field_history": base[..., None]}
    model = ConditionalMeanRegressor(Shift())
    trainer = FlowTrainer(
        num_epochs=5,
        model=model,
        train_dataloader=[batch],
        val_dataloader=[batch],
        optimizer=torch.optim.SGD(model.parameters(), lr=1.5),  # exact in one step
        device="cpu",
        checkpoint_dir=str(tmp_path),
    )
    trainer.train()
    torch.testing.assert_close(model.predict(base), base + offset, rtol=0, atol=1e-3)
