"""Checks of the WP1 exact-noise Euler diagnostic on synthetic inputs."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import MatchedStochasticInterpolation
from scisi.sampling.noise_covariance import FourierDiagonalCovariance

SCRIPT = Path(__file__).parents[1] / "experiments" / "wp1_euler_exact_noise.py"
spec = importlib.util.spec_from_file_location("wp1_euler_exact_noise", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class LinearDrift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(-0.5))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None):
        return self.rate * x


def test_shell_sums_keep_parseval_for_even_and_odd_grids():
    rng = np.random.default_rng(0)
    for shape in ((3, 8, 6), (3, 6, 5)):
        field = rng.normal(size=shape)
        power = np.abs(np.fft.rfft2(field, norm="ortho")) ** 2
        shells = diagnostic.shell_index(*shape[-2:])
        total = diagnostic.shell_sum(power, shells, shape[-1]).sum(-1)
        np.testing.assert_allclose(total, (field**2).sum((-2, -1)))


def test_euler_with_exact_noise_shares_one_coloured_path():
    covariance = FourierDiagonalCovariance(0.5 + torch.rand(1, 6, 3))
    model = FollmerStochasticInterpolant(
        MatchedStochasticInterpolation(gamma_multiplier=0.5, noise_covariance=covariance),
        LinearDrift(),
    )
    base = torch.zeros(4, 1, 6, 5)
    path = model.make_brownian_path(base, torch.Generator().manual_seed(0))
    fine = diagnostic.euler_exact_noise(model, base, None, path, 8)
    coarse = diagnostic.euler_exact_noise(model, base, None, path, 1)
    # One Euler step from 0 with linear drift adds exactly M_1.
    torch.testing.assert_close(coarse, path.increment(0.0, 1.0))
    # Same path, contracting drift: the fine endpoint is a damped sum of increments.
    increments = [path.increment(j / 8, (j + 1) / 8) for j in range(8)]
    expected = sum(inc * (1 - 0.5 / 8) ** (7 - j) for j, inc in enumerate(increments))
    torch.testing.assert_close(fine, expected)
    members = fine.double().numpy()
    metrics = diagnostic.case_metrics(members, members[0], diagnostic.shell_index(6, 5))
    assert metrics["rms_spread"] > 0 and np.isfinite(metrics["fair_crps"])
