"""Invariant-closed Itô maps keep a linear invariant exactly (WP2)."""

import pytest
import torch
from torch import nn

from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.interpolations import (
    MatchedStochasticInterpolation,
    QuadraticStochasticInterpolation,
)
from scisi.models.invariants import SpatialMeanInvariant
from scisi.sampling.noise_covariance import (
    FourierDiagonalCovariance,
    IdentityCovariance,
    ProjectedCovariance,
)


class NonlinearMapNet(nn.Module):
    """A fixed nonlinear, path-dependent network that does not conserve the mean."""

    def __init__(self, channels, path_channels):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.state = nn.Conv2d(channels, channels, 3, padding=1)
            self.path = nn.Conv2d(path_channels, channels, 1)

    def forward(self, x, cond, field_history=None, path_features=None, **kwargs):
        cond = cond.reshape(cond.shape[0], -1, 1, 1)
        # Contracting (stable over long rollouts) but not mean-conserving.
        return (
            torch.tanh(self.state(x)) * (1 + cond[:, :1])
            + cond[:, 1:]
            - 0.5 * x
            + 0.1 * field_history[..., -1]
            + self.path(path_features)
        )


def model(invariant_on, interpolation_class, covariance):
    channels, features = 3, 2
    interpolation = interpolation_class(
        gamma_multiplier=0.8,
        noise_covariance=ProjectedCovariance(covariance, SpatialMeanInvariant()),
    )
    return FollmerItoMapModel(
        interpolation,
        NonlinearMapNet(channels, channels * features),
        num_path_features=features,
        invariant=SpatialMeanInvariant() if invariant_on else None,
    ).double()


COVARIANCES = {
    "identity": IdentityCovariance(),
    "fourier": FourierDiagonalCovariance(0.5 + torch.rand(3, 6, 3)),
}


@pytest.mark.parametrize("covariance", COVARIANCES)
@pytest.mark.parametrize(
    "interpolation_class",
    [QuadraticStochasticInterpolation, MatchedStochasticInterpolation],
)
def test_invariant_error_at_machine_precision_for_every_step_count(
    covariance, interpolation_class
):
    history = torch.randn(4, 3, 6, 5, 2, dtype=torch.float64)  # C=3, non-square
    conserved = history[..., -1].mean(dim=(2, 3))
    for invariant_on in (True, False):
        m = model(invariant_on, interpolation_class, COVARIANCES[covariance])
        for num_steps in (1, 2, 4, 8):
            out = m.sample(
                history,
                num_steps=num_steps,
                generator=torch.Generator().manual_seed(num_steps),
            )
            error = (out.mean(dim=(2, 3)) - conserved).abs().max().item()
            if invariant_on:
                assert error < 1e-13, (num_steps, error)
            else:
                assert error > 1e-3, (num_steps, error)


def test_invariant_holds_along_a_long_rollout():
    m = model(True, QuadraticStochasticInterpolation, COVARIANCES["fourier"])
    history = torch.randn(2, 3, 6, 5, 2, dtype=torch.float64)
    conserved = history[..., -1].mean(dim=(2, 3))
    trajectory = m.sample_trajectory(
        history,
        num_steps=2,
        num_physical_steps=502,
        generator=torch.Generator().manual_seed(1),
    )
    error = (trajectory[..., 2:].mean(dim=(2, 3)) - conserved[..., None]).abs().max()
    assert error < 1e-11


def test_diagonal_drift_and_projection_identities():
    invariant = SpatialMeanInvariant()
    x = torch.randn(2, 3, 4, 7, dtype=torch.float64)
    torch.testing.assert_close(
        invariant.value(invariant.project_out(x)), torch.zeros(2, 3, dtype=torch.float64)
    )
    v = torch.randn(2, 3, dtype=torch.float64)
    torch.testing.assert_close(invariant.value(invariant.lift(v, like=x)), v)
    # On the diagonal, L G = l'_t = alpha'_t L c + beta'_t l(h).
    m = model(True, QuadraticStochasticInterpolation, COVARIANCES["identity"])
    history = torch.randn(2, 3, 6, 5, 2, dtype=torch.float64)
    t = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    velocity = m.average_velocity(torch.randn(2, 3, 6, 5, dtype=torch.float64), t, t, history)
    lc = history[..., -1].mean(dim=(2, 3))
    torch.testing.assert_close(invariant.value(velocity), (-1 + 2 * t) * lc)
    with pytest.raises(ValueError, match="masks"):
        FollmerItoMapModel(
            QuadraticStochasticInterpolation(),
            NonlinearMapNet(1, 4),
            invariant=invariant,
            mask_path="mask.npz",
        )
