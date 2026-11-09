"""Multistep linear--Gaussian case: single-step parity, exact recursions, samplers."""
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from cases.analytical.classical_baselines import (  # noqa: E402
    GaussianSystem,
    enkf_posterior,
    particle_filter_posterior,
)
from cases.analytical.driver import AnalyticalRunner, wasserstein_distance  # noqa: E402
from cases.analytical.linear_system import (  # noqa: E402
    AnalyticalLinearFMVelocityModel,
    AnalyticalLinearSIDriftModel,
    DenseObservationOperator,
    FullFirstStepDiffusionModel,
    LinearGaussianSSM,
    OursSurgeDiffusionPosterior,
    adapted_resample,
    enkf_step,
    gaussian_samples,
    particle_filter_step,
)
from cases.analytical.multistep import (  # noqa: E402
    ADAPTED_VARIANT,
    REFERENCE_KALMAN,
    REFERENCE_MEMBER_CHAIN,
    SURGE_PREFIX,
    SURGE_UNGUIDED,
    SURGE_WRAPPED,
    MultistepRunner,
    kl_to_moments,
    sliced_distances,
)
from cases.analytical.prior_models import (  # noqa: E402
    AnalyticalFMVelocityModel,
    AnalyticalSIDriftModel,
)
from results_schema import Method  # noqa: E402
from scisi.posterior_models.surge_posterior import SurgePosteriorBase  # noqa: E402

PARTIAL = dict(
    A=[[0.9, 0.3], [-0.3, 0.9]],
    Q=[[1.0, 0.0], [0.0, 0.25]],
    H=[[1.0, 0.0]],
    R=[[0.5]],
)
# Anisotropic everything: full Q, a mixing H and a non-scalar R (whitened path).
GENERAL = dict(
    A=[[0.8, 0.4], [-0.2, 1.0]],
    Q=[[1.0, 0.3], [0.3, 0.5]],
    H=[[1.0, 0.5], [0.0, 1.0]],
    R=[[0.6, 0.2], [0.2, 0.4]],
)


def _config(**case):
    if case.get("system"):
        case.setdefault("scenario_name", "multistep:test")
    return OmegaConf.create({
        "case": {"plot_dimension": 2, "adapted_variant": False, **case},
        "ensemble_size": 64,
        "num_steps": 50,
    })


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


@pytest.fixture(scope="module")
def parity_rows():
    """Per-seed rows of the existing driver and of the multistep case at K = 1."""
    case = {"n_eval_samples": 512}
    single = AnalyticalRunner(_config(**case), seeds=(0, 1)).run(aggregate=False)
    multi = MultistepRunner(_config(num_cycles=1, **case), seeds=(0, 1)).run(aggregate=False)
    return single, multi


@pytest.mark.parametrize(
    "single_metric,multi_metric",
    [("kl_points", "kl_points"), ("sliced_w2", "sliced_w1")],
)
def test_one_cycle_reproduces_single_step_case(parity_rows, single_metric, multi_metric):
    """Every method, variant and seed. ``sliced_w1`` is the driver's ``sliced_w2``."""
    single, multi = parity_rows
    expected = {
        (r.method, r.variant, r.seed): r.value
        for r in single if r.metric == single_metric
    }
    found = {
        (r.method, r.variant, r.seed): r.value
        for r in multi if r.metric == multi_metric
    }
    assert len(expected) == 2 * (3 * 2 + 8)  # all methods incl. both Ours variants
    for key, value in expected.items():
        if math.isnan(value):
            assert math.isnan(found[key]), key
        else:
            # Only the reference draw differs, by float64 vs float32 rounding.
            assert found[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


def test_one_cycle_kalman_is_the_paper_posterior():
    system = LinearGaussianSSM.from_spec(2)
    x0, y = torch.tensor([0.3, -1.0]), torch.tensor([1.0, 2.0])
    mean, cov = system.kalman_step(
        x0.double(), torch.zeros(2, 2, dtype=torch.float64), y
    )
    exact_mean, cov_scale = GaussianSystem(d=2).exact_posterior_moments(x0, y)
    torch.testing.assert_close(mean.float(), exact_mean)
    torch.testing.assert_close(cov, float(cov_scale) * torch.eye(2, dtype=torch.float64))


@pytest.mark.parametrize("spec", [PARTIAL, GENERAL], ids=["partial", "general"])
def test_kalman_recursion_matches_batch_conditioning(spec):
    """Filter moments equal direct conditioning of the joint Gaussian over K steps."""
    system = LinearGaussianSSM.from_spec(2, **spec)
    d, m, K = system.d, system.m, 4
    x0 = torch.tensor([0.7, -0.4], dtype=torch.float64)
    ys = [torch.randn(m, generator=_generator(k), dtype=torch.float64) for k in range(K)]

    # x^k = A^k x0 + sum_j A^{k-j} w_j, so the stacked states are linear in the w_j.
    powers = [torch.matrix_power(system.A, k) for k in range(K + 1)]
    W = torch.zeros(K * d, K * d, dtype=torch.float64)  # x_stack = mean + W w
    for k in range(K):
        for j in range(k + 1):
            W[k * d:(k + 1) * d, j * d:(j + 1) * d] = powers[k - j]
    cov_x = W @ torch.block_diag(*[system.Q] * K) @ W.T
    mean_x = torch.cat([powers[k + 1] @ x0 for k in range(K)])
    Hs = torch.block_diag(*[system.H] * K)
    gain = torch.linalg.solve(
        Hs @ cov_x @ Hs.T + torch.block_diag(*[system.R] * K), Hs @ cov_x
    ).T
    post_mean = mean_x + gain @ (torch.cat(ys) - Hs @ mean_x)
    post_cov = cov_x - gain @ Hs @ cov_x

    mean, cov = x0, torch.zeros(d, d, dtype=torch.float64)
    for y in ys:
        mean, cov = system.kalman_step(mean, cov, y)
    torch.testing.assert_close(mean, post_mean[-d:])
    torch.testing.assert_close(cov, post_cov[-d:, -d:])


def test_member_chain_differs_from_the_filter_after_the_first_cycle():
    """Steady variances for A = Q = H = R = 1: 2/3 for the chain, (sqrt5 - 1)/2 for KF."""
    system = LinearGaussianSSM.from_spec(1)
    y = torch.zeros(1, dtype=torch.float64)
    zero = torch.zeros(1, 1, dtype=torch.float64)
    m_k, P_k = system.kalman_step(torch.ones(1, dtype=torch.float64), zero, y)
    m_c, P_c = system.member_chain_step(torch.ones(1, dtype=torch.float64), zero, y)
    torch.testing.assert_close((m_k, P_k), (m_c, P_c))
    for _ in range(60):
        m_k, P_k = system.kalman_step(m_k, P_k, y)
        m_c, P_c = system.member_chain_step(m_c, P_c, y)
    assert float(P_k) == pytest.approx((math.sqrt(5) - 1) / 2)
    assert float(P_c) == pytest.approx(2 / 3)


def _exact_member_update(system, x, y, generator):
    """Exact draw from ``p(x' | x_i, y)`` for every member, float64."""
    K = system.conditional_gain()
    I_KH = torch.eye(system.d, dtype=torch.float64) - K @ system.H
    mean = x.double() @ (I_KH @ system.A).T + (K @ y.double())
    return mean + gaussian_samples(
        torch.zeros(system.d, dtype=torch.float64), I_KH @ system.Q, x.shape[0], generator
    ).double()


def _moment_gap(samples, mean, cov):
    return (
        float((samples.double().mean(0) - mean).norm()),
        float((torch.cov(samples.double().T) - cov).norm()),
    )


@pytest.mark.parametrize("spec", [{}, PARTIAL], ids=["paper", "partial"])
def test_adapted_resampling_recovers_the_filter_and_plain_updates_do_not(spec):
    system = LinearGaussianSSM.from_spec(2, **spec)
    n, g = 200_000, _generator(1)
    x0 = torch.tensor([0.5, -0.5])
    plain = adapted = x0.repeat(n, 1)
    mean = x0.double()
    cov = torch.zeros(2, 2, dtype=torch.float64)
    chain_mean, chain_cov = mean, cov
    for k in range(6):
        y = torch.tensor([3.0, -2.0])[: system.m] * (-1.0) ** k
        mean, cov = system.kalman_step(mean, cov, y)
        chain_mean, chain_cov = system.member_chain_step(chain_mean, chain_cov, y)
        plain = _exact_member_update(system, plain, y, g).float()
        adapted = _exact_member_update(
            system, adapted_resample(system, adapted, y, g), y, g
        ).float()
    assert max(_moment_gap(adapted, mean, cov)) < 0.02
    assert max(_moment_gap(plain, chain_mean, chain_cov)) < 0.02
    assert float((chain_mean - mean).norm()) > 0.1


@pytest.mark.parametrize("spec", [{}, PARTIAL, GENERAL], ids=["paper", "partial", "general"])
@pytest.mark.parametrize("step", [enkf_step, particle_filter_step])
def test_classical_filters_track_the_kalman_filter(spec, step):
    system = LinearGaussianSSM.from_spec(2, **spec)
    g = _generator(2)
    x0 = torch.tensor([0.5, -0.5])
    x = x0.repeat(100_000, 1)
    truth = x0
    mean, cov = x0.double(), torch.zeros(2, 2, dtype=torch.float64)
    for _ in range(5):
        truth = system.propagate(truth, g)
        y = system.observe(truth, g)
        mean, cov = system.kalman_step(mean, cov, y)
        x = step(system, x, y, g)
    assert max(_moment_gap(x, mean, cov)) < 0.03


def test_filter_steps_match_the_single_step_filters_on_the_paper_system():
    system, reference = LinearGaussianSSM.from_spec(2), GaussianSystem(d=2)
    x0, y = torch.tensor([0.3, -1.0]), torch.tensor([1.0, 2.0])
    x = x0.repeat(64, 1)
    torch.testing.assert_close(
        enkf_step(system, x, y, _generator(3)),
        enkf_posterior(reference, x0, y, ensemble_size=64, generator=_generator(3)),
    )
    torch.testing.assert_close(
        particle_filter_step(system, x, y, _generator(3)),
        particle_filter_posterior(reference, x0, y, ensemble_size=64, generator=_generator(3)),
    )


@pytest.mark.parametrize("q", [1.0, 0.4])
def test_linear_priors_reduce_to_the_paper_priors(q):
    eye = torch.eye(3, dtype=torch.float64)
    g = _generator(4)
    x = torch.randn(5, 1, 1, 3, generator=g)
    fh = torch.randn(5, 1, 1, 3, 1, generator=g)
    for tau in (0.0, 0.05, 0.5, 0.95, 1.0):
        t = torch.full((5, 1), tau)
        torch.testing.assert_close(
            AnalyticalLinearSIDriftModel(eye, q * eye, g0=0.7)(x, t, fh),
            AnalyticalSIDriftModel(g0=0.7, prior_var=q)(x, t, fh),
            rtol=1e-4, atol=1e-4,
        )
        torch.testing.assert_close(
            AnalyticalLinearFMVelocityModel(eye, q * eye)(x, t, fh),
            AnalyticalFMVelocityModel(prior_var=q)(x, t, fh),
            rtol=1e-4, atol=1e-4,
        )


def test_dense_operator_is_adjoint_consistent():
    matrix = torch.tensor([[1.0, 0.5, 0.0], [0.0, 2.0, -1.0]])
    op = DenseObservationOperator(matrix)
    g = _generator(5)
    x, y = torch.randn(4, 1, 1, 3, generator=g), torch.randn(4, 2, generator=g)
    torch.testing.assert_close(op(x), x.reshape(4, 3) @ matrix.T)
    torch.testing.assert_close(
        (op(x) * y).sum(1), (x * op.transpose(y)).reshape(4, -1).sum(1)
    )


def test_whitening_keeps_the_likelihood():
    system = LinearGaussianSSM.from_spec(2, **GENERAL)
    y = torch.tensor([0.4, -1.2])
    op, variance, y_white = MultistepRunner._obs(system, y)
    assert variance == 1.0
    x = torch.randn(6, 1, 1, 2, generator=_generator(6))
    residual = y.double() - x.reshape(6, 2).double() @ system.H.T
    expected = (residual @ torch.linalg.inv(system.R) * residual).sum(1)
    torch.testing.assert_close(
        ((y_white - op(x)) ** 2).sum(1).double(), expected, rtol=1e-4, atol=1e-5
    )


def test_sliced_w2_of_a_shift():
    """Each slice moves by ``theta . c``, so SW2^2 = E[(theta . c)^2] = |c|^2 / d."""
    x = np.random.default_rng(0).normal(size=(2000, 2))
    _, w2 = sliced_distances(x, x + np.array([3.0, 4.0]), num_projections=4096)
    assert w2 == pytest.approx(5.0 / math.sqrt(2), rel=0.02)


def test_sliced_w1_is_the_single_step_estimator():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(500, 2)), 1.5 * rng.normal(size=(500, 2)) + 0.3
    w1, _ = sliced_distances(x, y)
    assert w1 == pytest.approx(wasserstein_distance(x, y, num_projections=128, seed=0))


def test_kl_to_moments_matches_the_closed_form():
    samples = np.random.default_rng(2).normal(size=(400_000, 2)) * [1.0, 2.0] + [0.5, 0.0]
    # KL(N(0, I) || N((0.5, 0), diag(1, 4))) = 0.5 (1 + 1/4 + 1/4 - 2 + ln 4).
    expected = 0.5 * (1.25 + 0.25 - 2 + math.log(4))
    value = kl_to_moments(np.zeros(2), np.eye(2), samples)
    assert value == pytest.approx(expected, abs=2e-3)
    assert math.isnan(kl_to_moments(np.zeros(2), np.eye(2), np.ones((10, 2))))


def _cycle_means(records, metric):
    return {(r.method, r.variant): r.value for r in records if r.metric == metric}


def _last_cycle(runner, metric):
    """``{(method, variant): value}`` at the final cycle of this runner's rollouts."""
    last = runner.num_cycles - 1
    return {
        (row["method"], row["variant"]): row["value"]
        for row in runner._per_step
        if row["metric"] == metric and row["step"] == last
    }


def _evaluate(method, *, num_steps, **case):
    runner = MultistepRunner(_config(**case), seeds=(0,))
    ctx = runner.make_context(method, None, 0)
    ctx.num_steps = num_steps
    torch.manual_seed(0)
    return runner, list(runner.evaluate(ctx))


OURS = [Method.OURS_SI_SDE, Method.OURS_DM_SDE, Method.OURS_FM_ODE]


@pytest.mark.parametrize("spec", [PARTIAL, GENERAL], ids=["partial", "general"])
@pytest.mark.parametrize("method", OURS)
def test_ours_shared_is_exact_for_one_cycle_on_general_systems(spec, method):
    """Full-covariance Ours samples the conditional posterior for any A, Q, H, R."""
    _, records = _evaluate(
        method, num_steps=250, num_cycles=1, n_eval_samples=16384, system=spec
    )
    assert _cycle_means(records, "kl_exact")[(method.value, "shared")] < 0.005


@pytest.mark.parametrize("spec", [{}, PARTIAL, GENERAL], ids=["paper", "partial", "general"])
@pytest.mark.parametrize("method", OURS)
def test_ours_ensemble_follows_the_member_chain_and_adapted_follows_the_filter(spec, method):
    """Spread anchors, every system: the plain rows hit the chain, adapted the filter."""
    runner, _ = _evaluate(
        method, num_steps=100, num_cycles=4, n_eval_samples=16384,
        adapted_variant=True, system=spec,
    )
    assert _last_cycle(runner, "kl_chain")[(method.value, "shared")] < 0.01
    assert _last_cycle(runner, "kl_exact")[(method.value, ADAPTED_VARIANT)] < 0.01


@pytest.mark.parametrize("spec", [{}, PARTIAL], ids=["paper", "partial"])
@pytest.mark.parametrize("method", [Method.SURGE_FLOWDAS, Method.SURGE_SDA])
def test_surge_targets_the_filter(spec, method):
    """Cross-member SMC weights supply the term a per-member update lacks."""
    runner, _ = _evaluate(
        method, num_steps=50, num_cycles=4, n_eval_samples=16384, system=spec
    )
    assert _last_cycle(runner, "kl_exact")[(method.value, None)] < 0.01


def _spread_input(runner, n=4096):
    """An ensemble over distinct anchors and an observation it does not centre on."""
    system = runner.system()
    x = torch.randn(n, system.d, generator=_generator(5))
    return system, x, system.H.float() @ torch.tensor([1.5, -1.0])


@pytest.mark.parametrize("method", SURGE_WRAPPED)
def test_surge_proposal_is_the_wrapped_sampler(method, monkeypatch):
    """Weights aside, a SURGE row moves the members exactly as the plain row does."""
    runner = MultistepRunner(_config(n_eval_samples=64, system=PARTIAL), seeds=(0,))
    system, x, y = _spread_input(runner)
    monkeypatch.setattr(
        SurgePosteriorBase, "_maybe_resample",
        lambda self, base, field_history, force=False: (base, field_history),
    )
    out = {}
    for surge in (False, True):
        torch.manual_seed(3)
        out[surge] = runner._ours_cycle(method, system, x, y, "inflated_shared", 50, surge=surge)
    assert torch.allclose(out[False], out[True], atol=1e-4)


def test_surge_diffusion_score_is_the_model_score():
    runner = MultistepRunner(_config(n_eval_samples=64, system=PARTIAL), seeds=(0,))
    system = runner.system()
    model = runner._dm_model(system, 1)
    posterior = OursSurgeDiffusionPosterior(
        model=model, obs_operator=DenseObservationOperator(system.H), variance=0.5,
    )
    x = torch.randn(32, 1, 1, 2, generator=_generator(1))
    history = torch.randn(32, 1, 1, 2, 1, generator=_generator(2))
    t = torch.full((1, 1), 0.3)
    drift = model.drift(x, t, history, None, None)
    kwargs = posterior._lik_kwargs(x, t, history, None, None, drift)
    assert torch.allclose(kwargs["score"], model.score(x, t, history, None, None), atol=1e-4)


def test_surge_needs_an_sde():
    runner = MultistepRunner(_config(n_eval_samples=64), seeds=(0,))
    system, x, y = _spread_input(runner, 64)
    with pytest.raises(ValueError, match="SDE"):
        runner._ours_cycle(Method.OURS_FM_ODE, system, x, y, "inflated_shared", 50, surge=True)


@pytest.mark.parametrize("method", SURGE_WRAPPED)
def test_surge_rows_at_one_cycle_are_the_single_step_posterior(method):
    """K = 1, one shared anchor: no predictive weight to add, and no harm done."""
    runner, rows = _evaluate(
        method, num_steps=50, num_cycles=1, n_eval_samples=16384, surge_only=True
    )
    assert {row.variant for row in rows} == {
        f"{SURGE_PREFIX}shared", f"{SURGE_PREFIX}jacfree", SURGE_UNGUIDED,
    }
    assert {row.nfe for row in rows} == {100.0}
    last = _last_cycle(runner, "kl_exact")
    for name in ("shared", "jacfree", "unguided"):
        assert last[(method.value, f"{SURGE_PREFIX}{name}")] < 0.01


@pytest.mark.parametrize("spec", [{}, PARTIAL, GENERAL], ids=["paper", "partial", "general"])
@pytest.mark.parametrize("method", SURGE_WRAPPED)
def test_surge_around_ours_targets_the_filter_not_the_chain(spec, method):
    """Spread anchors, non-identity systems: the row meets Kalman, not the chain.

    Whether this test can tell the two apart is checked by the exact chain row,
    whose KL to Kalman the SURGE row must beat.
    """
    runner, _ = _evaluate(
        method, num_steps=50, num_cycles=4, n_eval_samples=16384, surge_only=True,
        system=spec,
    )
    to_kalman = _last_cycle(runner, "kl_exact")[(method.value, f"{SURGE_PREFIX}shared")]
    cycle = runner.trajectory(0, 16384)[1][-1]
    chain = kl_to_moments(
        cycle.kalman_mean.numpy(), cycle.kalman_cov.numpy(),
        gaussian_samples(cycle.chain_mean, cycle.chain_cov, 16384, _generator(9)).numpy(),
    )
    assert to_kalman < 0.01
    assert to_kalman < chain


def test_surge_cells_leave_the_lineup_and_the_observations_alone():
    plain = MultistepRunner(_config(num_cycles=2, n_eval_samples=64), seeds=(0,))
    surge = MultistepRunner(
        _config(num_cycles=2, n_eval_samples=64, surge_only=True), seeds=(0,)
    )
    assert tuple(surge.methods()) == SURGE_WRAPPED
    assert Method.OURS_FM_ODE in plain.methods()
    (x_plain, plain_cycles), (x_surge, surge_cycles) = (
        runner.trajectory(0, 64) for runner in (plain, surge)
    )
    assert torch.equal(x_plain, x_surge)
    for a, b in zip(plain_cycles, surge_cycles, strict=True):
        assert torch.equal(a.y, b.y) and torch.equal(a.kalman_mean, b.kalman_mean)
    rows = surge.run(aggregate=False)
    assert {row.method for row in rows} == {method.value for method in SURGE_WRAPPED}
    ctx = plain.make_context(Method.OURS_SI_SDE, None, 0)
    assert not any(
        (row.variant or "").startswith(SURGE_PREFIX) for row in plain.evaluate(ctx)
    )


@pytest.mark.parametrize(
    "method",
    [Method.FLOWDAS, Method.SDA, Method.D_FLOW_SGLD, Method.GUIDED_FM_FIG],
)
def test_baselines_keep_each_member_on_its_own_anchor(method):
    """Spread anchors reach a baseline member by member: outputs track the input."""
    runner = MultistepRunner(_config(n_eval_samples=64, system=PARTIAL), seeds=(0,))
    system = runner.system()
    x = torch.linspace(-30.0, 30.0, 2048).unsqueeze(1).repeat(1, 2)
    y = torch.zeros(1)
    torch.manual_seed(0)
    out, _ = runner._baseline_cycle(method, system, x, y, 50, _generator(7), 1)
    assert np.corrcoef(x[:, 0].numpy(), out[:, 0].numpy())[0, 1] > 0.8


def test_diffusion_rows_do_not_depend_on_the_size_of_the_state():
    """Shifting the anchor and y together must shift the posterior by the same amount.

    The stock first step of the diffusion prior fails this, which is why cycles
    after the first use FullFirstStepDiffusionModel.
    """
    runner = MultistepRunner(_config(n_eval_samples=64), seeds=(0,))
    system = runner.system()
    assert isinstance(runner._dm_model(system, 1), FullFirstStepDiffusionModel)
    assert not isinstance(runner._dm_model(system, 0), FullFirstStepDiffusionModel)

    def offset(shift, cycle):
        x = torch.full((16384, 2), float(shift))
        y = torch.tensor([shift + 2.0, shift - 2.0])
        torch.manual_seed(0)
        out, _ = runner._baseline_cycle(
            Method.SURGE_SDA, system, x, y, 50, _generator(8), cycle
        )
        return out.double().mean(0) - shift

    assert float((offset(20.0, 1) - offset(0.0, 1)).norm()) < 0.05
    assert float((offset(20.0, 0) - offset(0.0, 0)).norm()) > 0.1


def test_one_singular_cycle_does_not_remove_the_seed():
    runner = MultistepRunner(_config(num_cycles=2, n_eval_samples=64), seeds=(0,))
    ctx = runner.make_context(Method.ENKF, None, 0)
    curves = {"kl_points": [float("nan"), 2.0], "kl_exact": [float("nan")] * 2}
    records = {r.metric: r.value for r in runner._emit("x", ctx, None, curves, 0, [])}
    assert records["kl_points"] == 2.0
    assert math.isnan(records["kl_exact"])


def test_sampler_failure_is_a_divergence_not_a_crash():
    runner = MultistepRunner(_config(num_cycles=3, n_eval_samples=64), seeds=(0,))
    ctx = runner.make_context(Method.ENKF, None, 0)

    def draw(x, y, k):
        if k == 1:
            raise AssertionError("FM-SDE prior drift is non-finite")
        return x + 1.0, 1

    list(runner._rollout(ctx, None, draw))
    values = [r["value"] for r in runner._per_step if r["metric"] == "sliced_w2"]
    assert math.isfinite(values[0]) and all(math.isnan(v) for v in values[1:])


def test_paper_label_is_reserved_for_the_paper_system():
    config = _config(system={"A": 0.5}, scenario_name="multistep:paper")
    runner = MultistepRunner(config, seeds=(0,))
    with pytest.raises(ValueError, match="reserved"):
        runner.system()


def test_per_step_file_takes_one_seed_and_ablation_is_refused(tmp_path):
    config = _config(num_cycles=1, n_eval_samples=64)
    config.per_step_file = str(tmp_path / "p.csv")
    with pytest.raises(ValueError, match="one seed"):
        MultistepRunner(config, seeds=(0, 1)).run()
    with pytest.raises(NotImplementedError):
        MultistepRunner(_config(), seeds=(0,)).run_ablation()


def test_reference_rows_and_per_cycle_curves(tmp_path):
    config = _config(num_cycles=3, n_eval_samples=4096)
    config.per_step_file = str(tmp_path / "per_step.csv")
    runner = MultistepRunner(config, seeds=(0,))
    records = list(runner._reference_rows(0))
    kl = _cycle_means(records, "kl_points")
    assert kl[(REFERENCE_KALMAN, None)] < 0.01
    assert kl[(REFERENCE_MEMBER_CHAIN, None)] > kl[(REFERENCE_KALMAN, None)]
    runner.write(records, tmp_path / "metrics.csv")
    lines = (tmp_path / "per_step.csv").read_text().splitlines()
    assert len(lines) == 1 + 2 * 3 * 7  # header + 2 references x 3 cycles x 7 metrics


def _grid(root, **env):
    subprocess.run(
        ["bash", str(EXPERIMENT / "run_analytical_multistep_grid.sh")],
        check=True, capture_output=True, text=True,
        env={
            **os.environ, "PY": sys.executable, "ROOT": str(root), "SEEDS": "0 1",
            "N_EVAL": "64", "CYCLES": "2", "M": "50", **env,
        },
    )


def _script(name, root, *args, check=True):
    return subprocess.run(
        [sys.executable, str(EXPERIMENT / name), "--root", str(root), *args],
        check=check, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def grid_root(tmp_path_factory):
    """Both YAMLs through the real grid, once."""
    root = tmp_path_factory.mktemp("grid")
    _grid(root)
    return root


def test_grid_aggregation_and_outputs_end_to_end(grid_root):
    assert sorted(p.name for p in (grid_root / "tmp").iterdir()) == []
    _script("aggregate_analytical_multistep.py", grid_root)
    _script("make_analytical_multistep_figures.py", grid_root)
    with open(grid_root / "aggregated" / "per_step.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["scenario"] for r in rows} == {"multistep:paper", "multistep:partial"}
    ok = [r for r in rows if r["method"] == "EnKF" and r["metric"] == "kl_points"]
    assert {r["n_traj"] for r in ok} == {"2"} and {r["step"] for r in ok} == {"0", "1"}
    assert (grid_root / "figures" / "analytical_multistep_error_vs_step.pdf").exists()
    assert (grid_root / "figures" / "analytical_multistep_surge_vs_step.pdf").exists()
    surge = {(r["method"], r["variant"]) for r in rows if r["variant"].startswith("surge_")}
    assert surge == {
        (method.value, f"{SURGE_PREFIX}{name}")
        for method in SURGE_WRAPPED for name in ("shared", "jacfree", "unguided")
    }
    # The added cells hold no reference rows, so the references stay one per seed.
    references = [r for r in rows if r["method"] == REFERENCE_KALMAN and r["step"] == "0"]
    assert {r["n_traj"] for r in references} == {"2"}
    assert "SURGE, SI-SDE (shared)" in (grid_root / "aggregated" / "table_paper.md").read_text()
    assert "k=2" in (grid_root / "aggregated" / "table_partial.md").read_text()


def test_a_different_ensemble_is_a_different_cell_and_figures_refuse_to_mix(tmp_path):
    root = tmp_path / "root"
    _grid(root, SETTINGS="paper", SEEDS="0", LINEUPS="base")
    _grid(root, SETTINGS="paper", SEEDS="0", LINEUPS="base", N_EVAL="96")
    assert len(list((root / "metrics").glob("*.csv"))) == 2
    _script("aggregate_analytical_multistep.py", root)
    mixed = _script("make_analytical_multistep_figures.py", root, check=False)
    assert mixed.returncode != 0 and "--M and --E" in mixed.stderr
    _script("make_analytical_multistep_figures.py", root, "--E", "96")


def test_aggregation_refuses_unpaired_files_and_mixed_horizons(tmp_path):
    root = tmp_path / "root"
    _grid(root, SETTINGS="paper", SEEDS="0", LINEUPS="base")
    _grid(root, SETTINGS="paper", SEEDS="0", LINEUPS="base", CYCLES="3")
    mixed = _script("aggregate_analytical_multistep.py", root, check=False)
    assert mixed.returncode != 0 and "horizons" in mixed.stderr
    next((root / "per_step").glob("*K3*")).unlink()
    unpaired = _script("aggregate_analytical_multistep.py", root, check=False)
    assert unpaired.returncode != 0 and "differ" in unpaired.stderr
