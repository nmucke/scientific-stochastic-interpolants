"""White noise (IdentityCovariance) reproduces the pre-WP1 code bit for bit.

``tests/data/white_noise_reference.pt`` was written by ``reference_outputs``
running on origin/dev (commit 57a4101, before NoiseCovariance existed):

    PYTHONPATH=<origin/dev checkout>/src python tests/test_white_noise_regression.py

``reference_outputs`` therefore only uses the API of that commit; the test
reruns it on the current code (default = white noise) and with an explicit
``IdentityCovariance`` and compares with zero tolerance.
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearStochasticInterpolation,
    QuadraticStochasticInterpolation,
)
from scisi.sampling.brownian_path import (
    KarhunenLoeveBrownianPath,
    WeightedBrownianPath,
)
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step
from scisi.training.map_trainer import MapTrainer
from scisi.training.trainer import EarlyStopping

REFERENCE = Path(__file__).parent / "data" / "white_noise_reference.pt"
SHAPE = (2, 1, 6, 5)  # non-square on purpose


class TinyNet(nn.Module):
    """A fixed, state- and path-dependent network for both maps and drifts."""

    def __init__(self, path_channels=0):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.state = nn.Conv2d(1, 1, 3, padding=1)
            self.path = nn.Conv2d(path_channels, 1, 1) if path_channels else None

    def forward(
        self,
        x,
        cond,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        path_features=None,
    ):
        cond = cond.reshape(cond.shape[0], -1, 1, 1)
        out = torch.tanh(self.state(x)) * (1 + cond[:, :1]) + cond[:, -1:]
        if field_history is not None:
            out = out + 0.1 * field_history[..., -1]
        if self.path is not None and path_features is not None:
            out = out + self.path(path_features)
        return out


def _interpolation(kind, **covariance):
    if kind == "quadratic":
        return QuadraticStochasticInterpolation(gamma_multiplier=1.0, **covariance)
    return LinearStochasticInterpolation(gamma_multiplier=0.5, **covariance)


def _batch():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(SHAPE, generator=generator)
    return {
        "base": base,
        "target": base + torch.randn(SHAPE, generator=generator),
        "field_history": torch.stack([0.5 * base, base], -1),
    }


def reference_outputs(**covariance):
    """Samples, paths and losses under fixed seeds (origin/dev API only)."""
    out = {}
    x = torch.randn(SHAPE, generator=torch.Generator().manual_seed(1))
    path = WeightedBrownianPath.sample_like(
        x, 0.7, generator=torch.Generator().manual_seed(3)
    )
    out["weighted_values"] = path.value(torch.tensor([0.3, 0.7]))
    out["weighted_features"] = path.features(0.1, 0.9, 4)
    out["weighted_increment"] = path.increment(0.2, 0.6)
    kl = KarhunenLoeveBrownianPath.sample_like(
        x, 0.7, generator=torch.Generator().manual_seed(4), num_modes=3
    )
    out["kl_values"] = kl.value(torch.tensor([0.4, 0.9]))
    out["kl_features"] = kl.features(0.0, 1.0, 3)
    out["kl_increment"] = kl.increment(0.1, 0.5)

    history = _batch()["field_history"][:1]
    for kind in ("quadratic", "linear"):
        for features in ("local", "karhunen_loeve"):
            model = FollmerItoMapModel(
                _interpolation(kind, **covariance),
                TinyNet(path_channels=4),
                num_path_features=4,
                brownian_features=features,
            )
            out[f"map_{kind}_{features}"] = model.sample(
                field_history=history,
                batch_size=3,
                num_steps=4,
                generator=torch.Generator().manual_seed(5),
            )

        si = FollmerStochasticInterpolant(_interpolation(kind, **covariance), TinyNet())
        data = _batch()
        t = torch.tensor([[0.25], [0.8]])
        noise = torch.randn(SHAPE, generator=torch.Generator().manual_seed(6))
        prediction, drift_target = si(data["base"], data["target"], t, noise)
        out[f"si_{kind}_prediction"] = prediction.detach()
        out[f"si_{kind}_drift_target"] = drift_target
        for name, stepper in (("euler", euler_maruyama_step), ("heun", heun_step)):
            torch.manual_seed(7)
            out[f"si_{kind}_{name}"] = si.sample(
                base=history[..., -1],
                field_history=history,
                batch_size=3,
                num_steps=5,
                stepper=stepper,
            ).detach()

        model = FollmerItoMapModel(
            _interpolation(kind, **covariance), TinyNet(path_channels=4)
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = MapTrainer(
                num_epochs=1,
                model=model,
                train_dataloader=[_batch()],
                val_dataloader=[_batch()],
                optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                early_stopping=EarlyStopping(patience=3),
                device="cpu",
                checkpoint_dir=directory,
                warmup_epochs=0,
            )
            torch.manual_seed(8)
            batch = trainer._prepare_batch(_batch())
            out[f"trainer_{kind}_noise"] = batch["noise"]
            out[f"trainer_{kind}_loss"] = trainer._compute_loss(batch).detach()
    return out


def _assert_identical(outputs, reference):
    assert outputs.keys() == reference.keys()
    for key, value in reference.items():
        torch.testing.assert_close(outputs[key], value, rtol=0, atol=0, msg=key)


def test_default_white_noise_is_bit_identical_to_origin_dev():
    _assert_identical(reference_outputs(), torch.load(REFERENCE))


def test_explicit_identity_covariance_is_bit_identical_to_origin_dev():
    from scisi.sampling.noise_covariance import IdentityCovariance

    outputs = reference_outputs(noise_covariance=IdentityCovariance())
    _assert_identical(outputs, torch.load(REFERENCE))


if __name__ == "__main__":
    torch.set_num_threads(1)
    REFERENCE.parent.mkdir(exist_ok=True)
    torch.save(reference_outputs(), REFERENCE)
    print(f"wrote {REFERENCE} with scisi from {sys.modules['scisi'].__file__}")
