"""Check hierarchical compression estimates against a solvable discrete SDE."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = (
    Path(__file__).parents[3] / "research/ito_maps/experiments/ns_path_compression.py"
)
spec = importlib.util.spec_from_file_location("ns_path_compression", SCRIPT)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class LinearTeacher:
    interpolation = SimpleNamespace(gamma_multiplier=1.0)

    def drift(self, x, t, **kwargs):
        return -4 * x


def test_hierarchical_variance_removes_noise_in_group_mean_estimator():
    endpoints = torch.tensor([[[-1.0], [1.0]], [[1.0], [3.0]]])
    result = diagnostic.compression_statistics(endpoints)
    assert result["within_coarse_path_variance"] == 2
    assert result["variance_of_group_means"] == 2
    assert result["explained_variance_estimate"] == 1
    assert result["total_conditional_forecast_variance_estimate"] == 3
    assert result["omitted_variance_fraction_estimate"] == 2 / 3


def test_forked_teacher_risk_matches_exact_discrete_linear_sde():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        outputs = diagnostic.conditional_endpoints(
            LinearTeacher(),
            torch.zeros(1, 1, 1, 3000, 1, dtype=torch.float64),
            features=[1, 4, 16],
            groups=4,
            replicas=4,
            teacher_steps=16,
            batch_size=4,
            seed=6592,
        )
    finally:
        torch.set_num_threads(previous_threads)
    times = np.linspace(0, 1, 17)
    variances = ((1 - times[:-1]) ** 3 - (1 - times[1:]) ** 3) / 3
    response = (1 - 4 / 16) ** np.arange(15, -1, -1)
    exact_total = np.sum(response**2 * variances)
    for count, endpoints in outputs.items():
        result = diagnostic.compression_statistics(endpoints)
        cells = 16 // count
        covariances = (response * variances).reshape(count, cells).sum(1)
        coarse_variances = variances.reshape(count, cells).sum(1)
        exact_explained = np.sum(covariances**2 / coarse_variances)
        exact_within = exact_total - exact_explained
        if count == 16:
            # Every solver increment is conditioned, so no bridge variation remains.
            assert torch.equal(endpoints[:, 0], endpoints[:, 1])
            assert result["within_coarse_path_variance"] == 0
        else:
            assert result["within_coarse_path_variance"] == pytest.approx(
                exact_within, rel=0.04
            )
        assert result["total_conditional_forecast_variance_estimate"] == pytest.approx(
            exact_total, rel=0.04
        )
