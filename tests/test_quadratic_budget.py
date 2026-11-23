"""QuadraticBudget on a non-NS shape: pixelwise Ornstein-Uhlenbeck with C=3.

The linear SPDE ``du = -lam u dtau + eps dW`` (N = 0, Q = Id per pixel) has
the exact one-step law ``y = exp(-lam D) c + s xi`` with
``s^2 = eps^2 (1 - exp(-2 lam D)) / (2 lam)``, so the conditional mean of the
trapezoid residual is known in closed form.
"""

import math

import pytest
import torch

from scisi.metrics import QuadraticBudget, u_statistic

SHAPE = (3, 5, 7)  # channels, non-square spatial grid
LAM = torch.tensor([0.2, 1.0, 3.0]).reshape(1, 3, 1, 1)
WEIGHT = torch.tensor([1.0, 0.5, 2.0]).reshape(1, 3, 1, 1)
EPS = 0.7


def budget(step, weight=True, offset=0.0):
    w = (lambda u: WEIGHT * u) if weight else None
    trace = float(WEIGHT.sum()) if weight else float(SHAPE[0])
    return QuadraticBudget(
        weight=w,
        dissipation=lambda u: LAM * u,
        injection_rate=0.5 * EPS**2 * trace + offset,
        step=step,
        # Var <u, W dxi> per unit time for per-pixel white dxi under the
        # area-mean inner product: sum_c mean_p(w^2 u^2) / P.
        forcing_quadratic_variation=lambda u: EPS**2
        * (WEIGHT**2 * u**2).flatten(2).mean(-1).sum(-1)
        / (SHAPE[1] * SHAPE[2]),
    )


def exact_step(c, step, generator):
    decay = torch.exp(-LAM * step)
    s2 = EPS**2 * (1 - torch.exp(-2 * LAM * step)) / (2 * LAM)
    noise = torch.randn(c.shape, generator=generator, dtype=c.dtype)
    return decay * c + s2.sqrt() * noise


def analytic_mean_residual(c, step):
    """E[r | c] per sample for the weighted OU budget."""
    decay2 = torch.exp(-2 * LAM * step)
    s2 = EPS**2 * (1 - decay2) / (2 * LAM)
    ey2 = decay2 * c**2 + s2
    per_pixel = WEIGHT * (
        0.5 * (1 + LAM * step) * ey2 - 0.5 * (1 - LAM * step) * c**2 - 0.5 * EPS**2 * step
    )
    return per_pixel.flatten(2).mean(-1).sum(-1)


def test_shapes_identity_weight_and_algebra():
    b = budget(0.1, weight=False)
    x = torch.randn(4, *SHAPE, dtype=torch.float64)
    r = b.residual(x, x)
    assert r.shape == (4,)
    expected = 0.1 * (LAM * x**2).flatten(2).mean(-1).sum(-1) - b.injection
    torch.testing.assert_close(r, expected)
    torch.testing.assert_close(b.q_plus(x) - b.q_minus(x), 0.1 * b.dissipation_rate(x))
    with pytest.raises(ValueError):
        b.residual(x, x[:2])


def test_stationary_ou_closes_exactly_in_mean():
    """Stationary c: the trapezoid form is exact in mean (E r = 0)."""
    step = 0.5
    b = budget(step)
    g = torch.Generator().manual_seed(0)
    sigma = EPS / torch.sqrt(2 * LAM)
    c = sigma * torch.randn(20000, *SHAPE, generator=g, dtype=torch.float64)
    y = exact_step(c, step, g)
    r = b.residual(y, c)
    se = r.std() / math.sqrt(len(r))
    assert abs(float(r.mean())) < 4 * float(se)
    assert abs(float(r.mean())) < 0.01 * b.injection


def test_conditional_mean_matches_closed_form_off_stationarity():
    step = 0.02
    b = budget(step)
    g = torch.Generator().manual_seed(1)
    c = 2.0 * torch.ones(1, *SHAPE, dtype=torch.float64).expand(40000, *SHAPE)
    y = exact_step(c, step, g)
    r = b.residual(y, c)
    expected = analytic_mean_residual(c[:1], step)[0]
    se = r.std() / math.sqrt(len(r))
    assert abs(float(r.mean() - expected)) < 4 * float(se)
    # Trapezoid error is O((lam Delta)^2) relative to the dissipation over a step.
    assert abs(float(expected)) < 0.02 * b.step * float(b.dissipation_rate(c[:1])[0])
    # Residual variance agrees with Delta eps^2 <c, W Q W c> to leading order.
    predicted = b.martingale_variance(c[:1])[0]
    assert float(r.var()) == pytest.approx(float(predicted), rel=0.1)


def test_u_statistic_unbiased_on_calibrated_ensemble():
    """Exact members from the true law; injection offset gives E[r|c] = -offset*D."""
    step, offset, members = 0.2, 0.3, 4
    b = budget(step, offset=offset)
    g = torch.Generator().manual_seed(2)
    c = torch.randn(1, *SHAPE, generator=g, dtype=torch.float64)
    target = float(analytic_mean_residual(c, step)[0]) - offset * step
    reps = 20000
    y = exact_step(c.expand(reps * members, *SHAPE), step, g)
    r = b.residual(y, c.expand_as(y)).reshape(reps, members)
    u = u_statistic(r, dim=1)
    naive = r.mean(1).square()
    se = u.std() / math.sqrt(reps)
    assert abs(float(u.mean()) - target**2) < 4 * float(se)
    # The naive square is biased upward by Var(r)/E.
    bias = float(naive.mean()) - target**2
    assert bias == pytest.approx(float(r.var()) / members, rel=0.1)
    with pytest.raises(ValueError):
        u_statistic(r[:, :1], dim=1)
