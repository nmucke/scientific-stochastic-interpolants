"""Black-box checks of the Karhunen-Loeve Brownian path for Itô maps.

The class is tested against its contract (shared with WeightedBrownianPath),
against the closed-form joint law of (xi_1..xi_K, M_t) and against an
independent fine-grid simulation of the standard Brownian motion W.
"""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.interpolations import QuadraticStochasticInterpolation
from scisi.sampling.brownian_path import (
    KarhunenLoeveBrownianPath,
    WeightedBrownianPath,
)

K = 4


def make_path(shape=(4, 2, 3, 3), amplitude=0.7, seed=19, modes=K, dtype=torch.float64):
    return KarhunenLoeveBrownianPath.sample_like(
        torch.zeros(shape, dtype=dtype),
        amplitude,
        generator=torch.Generator().manual_seed(seed),
        num_modes=modes,
    )


# ---------------------------------------------------------------- closed form
def q(t, a):
    return a * a * (t - t * t + t**3 / 3)


def m(n, t, a):
    """E[M_t xi_n] = int_0^t a (1-r) sqrt(2) cos(w_n r) dr."""
    w = (n - 0.5) * math.pi
    return math.sqrt(2) * a * ((1 - t) * math.sin(w * t) / w + (1 - math.cos(w * t)) / w**2)


def analytic_cov(times, a, modes=K):
    """Covariance of (xi_1..xi_K, M_t1..M_tT)."""
    d = modes + len(times)
    cov = torch.zeros(d, d, dtype=torch.float64)
    cov[:modes, :modes] = torch.eye(modes, dtype=torch.float64)
    for j, t in enumerate(times):
        for n in range(1, modes + 1):
            cov[n - 1, modes + j] = cov[modes + j, n - 1] = m(n, t, a)
        for i, s in enumerate(times):
            cov[modes + i, modes + j] = q(min(s, t), a)
    return cov


def cov_standard_error(cov, n):
    """Std. error of Gaussian sample covariances: sqrt((S_ii S_jj + S_ij^2)/n)."""
    diag = cov.diagonal()
    return ((diag[:, None] * diag[None, :] + cov**2) / n).sqrt()


def class_samples(times, a, n, seed, modes=K):
    """Rows (xi_1..xi_K, M_t for each t in query order) from n coordinates."""
    path = make_path((1, n), amplitude=a, seed=seed, modes=modes)
    values = [path.value(t).flatten() for t in times]
    xi = path.features(0.3, 0.6, modes).reshape(modes, n)
    return torch.cat([xi, torch.stack(values)])


def reference_samples(times, a, n, seed, modes=K, steps=2000, chunk=250):
    """Independent simulation: Ito sums of W on a fine grid (midpoint nodes)."""
    gen = torch.Generator().manual_seed(seed)
    dt = 1.0 / steps
    omega = (torch.arange(1, modes + 1, dtype=torch.float64) - 0.5) * math.pi
    t_vec = torch.tensor(times, dtype=torch.float64)
    xi = torch.zeros(modes, n, dtype=torch.float64)
    big_m = torch.zeros(len(times), n, dtype=torch.float64)
    for start in range(0, steps, chunk):
        r = (torch.arange(start, start + chunk, dtype=torch.float64) + 0.5) * dt
        dw = torch.randn(chunk, n, dtype=torch.float64, generator=gen) * math.sqrt(dt)
        xi += (math.sqrt(2) * torch.cos(omega[:, None] * r[None])) @ dw
        big_m += ((r[None] < t_vec[:, None]) * (a * (1 - r))[None]) @ dw
    return torch.cat([xi, big_m])


# ----------------------------------------------------------- 1. contract parity
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shapes_dtypes_devices_match_local_class(dtype):
    shape = (3, 2, 4, 5)
    x = torch.zeros(shape, dtype=dtype)
    kl = KarhunenLoeveBrownianPath.sample_like(
        x, 0.7, generator=torch.Generator().manual_seed(1), num_modes=K
    )
    local = WeightedBrownianPath.sample_like(
        x, 0.7, generator=torch.Generator().manual_seed(1)
    )
    assert isinstance(kl, KarhunenLoeveBrownianPath)
    assert kl.num_modes == K
    for attr in ("shape", "device", "dtype", "gamma_multiplier"):
        assert getattr(kl, attr) == getattr(local, attr)
    per_entry = torch.tensor([0.1, 0.5, 0.8])
    for s, t in [(0.2, 0.7), (per_entry, per_entry + 0.1), (per_entry[:, None], 1.0)]:
        for a, b in [(kl.value(t), local.value(t)), (kl.increment(s, t), local.increment(s, t))]:
            assert a.shape == b.shape == shape
            assert a.dtype == b.dtype == dtype and a.device == b.device
        for a, b in [(kl.features(s, t, K), local.features(s, t, K))]:
            assert a.shape == b.shape == (3, K * 2, 4, 5)
            assert a.dtype == dtype
        assert kl.variance(s, t).shape == (3, 1) and kl.variance(s, t).dtype == dtype
        torch.testing.assert_close(kl.variance(s, t), local.variance(s, t), rtol=0, atol=0)


def test_scalar_and_per_entry_times_zero_increment_and_repeats():
    path = make_path()
    assert torch.equal(path.value(0.0), torch.zeros(path.shape, dtype=torch.float64))
    times = torch.tensor([0.2, 0.5, 0.9, 0.35], dtype=torch.float64)
    v = path.value(times)
    # Per-entry times address each entry's own cache.
    torch.testing.assert_close(path.value(times[:, None]), v, rtol=0, atol=0)
    for b in range(4):
        assert torch.equal(path.value(float(times[b]))[b], v[b])
    assert torch.equal(path.increment(0.4, 0.4), torch.zeros_like(v))
    assert torch.equal(path.increment(times, times), torch.zeros_like(v))
    a = path.value(0.37)
    path.value(0.61), path.value(0.12), path.increment(0.05, 0.95)
    assert torch.equal(path.value(0.37), a)
    torch.testing.assert_close(
        path.increment(0.12, 0.61), path.value(0.61) - path.value(0.12), rtol=0, atol=0
    )
    f = path.features(0.1, 0.9, K)
    assert torch.equal(path.features(0.1, 0.9, K), f)


def test_validation_errors():
    path = make_path()
    for bad in [-0.1, 1.1, float("nan"), float("inf"), torch.tensor([0.1, 0.2])]:
        with pytest.raises(ValueError):
            path.value(bad)
    with pytest.raises(ValueError):
        path.increment(0.7, 0.3)
    with pytest.raises(ValueError):
        path.variance(0.7, 0.3)
    with pytest.raises(ValueError):
        path.features(0.7, 0.3, K)
    with pytest.raises(ValueError):
        path.features(0.1, 1.2, K)
    for wrong in (0, K - 1, K + 1):
        with pytest.raises(ValueError):
            path.features(0.1, 0.9, wrong)
    with pytest.raises(ValueError):
        make_path(amplitude=-1)
    with pytest.raises(ValueError):
        make_path(modes=0)


def test_fixed_generator_reproducible_and_seeds_differ():
    a, b, c = make_path(seed=3), make_path(seed=3), make_path(seed=4)
    assert torch.equal(a.features(0.1, 0.2, K), b.features(0.1, 0.2, K))
    assert torch.equal(a.value(0.42), b.value(0.42))
    assert not torch.equal(a.features(0.1, 0.2, K), c.features(0.1, 0.2, K))


def test_fork_copies_observations_and_refines_independently():
    n, a = 30000, 0.7
    parent = make_path((1, n), amplitude=a, seed=13)
    left, right = parent.value(0.2), parent.value(0.8)
    xi = parent.features(0.0, 1.0, K)
    first = parent.fork(torch.Generator().manual_seed(21))
    second = parent.fork(torch.Generator().manual_seed(22))
    assert isinstance(first, KarhunenLoeveBrownianPath)
    for child in (first, second):
        assert torch.equal(child.value(0.2), left)
        assert torch.equal(child.value(0.8), right)
        assert torch.equal(child.features(0.3, 0.4, K), xi)
    mid1, mid2 = first.value(0.5), second.value(0.5)
    assert not torch.equal(mid1, mid2)
    assert torch.equal(first.fork().value(0.5), mid1)  # refinements inherited
    mid_parent = parent.value(0.5)  # parent was not conditioned by the forks
    assert not torch.equal(mid_parent, mid1) and not torch.equal(mid_parent, mid2)
    # Exact conditional variance of M_0.5 given (xi, M_0.2, M_0.8), from the
    # closed-form joint law; independent forks differ by twice that variance.
    cov = analytic_cov([0.2, 0.8, 0.5], a)
    obs = list(range(K + 2))
    s_oo, s_om = cov[obs][:, obs], cov[obs, K + 2]
    cond_var = float(cov[K + 2, K + 2] - s_om @ torch.linalg.solve(s_oo, s_om))
    stacked = torch.cat([xi.reshape(K, n), left.reshape(1, n), right.reshape(1, n)])
    for mid in (mid1, mid2, mid_parent):
        resid = mid.flatten() - torch.linalg.solve(s_oo, s_om) @ stacked
        torch.testing.assert_close(resid.var(), torch.tensor(cond_var).double(), rtol=0.05, atol=0)
        assert resid.mean().abs() < 5 * math.sqrt(cond_var / n)
    d = (mid1 - mid2).flatten()
    torch.testing.assert_close(d.var(), torch.tensor(2 * cond_var).double(), rtol=0.05, atol=0)


def test_default_fork_generator_uses_global_stream():
    parent = make_path((2, 2), seed=5)
    torch.manual_seed(302)
    before = torch.random.get_rng_state().clone()
    a, b = parent.fork(), parent.fork()
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(a.value(0.5), b.value(0.5))


def test_zero_amplitude_is_finite_and_zero():
    path = make_path(amplitude=0.0)
    for t in (0.3, 0.9, 0.31, 1.0):
        assert torch.count_nonzero(path.value(t)) == 0
    assert torch.isfinite(path.features(0.1, 0.9, K)).all()


# ------------------------------------------------------------------ 2. features
def test_features_are_global_and_feature_major():
    path = make_path((3, 2, 3, 3))
    f = path.features(0.1, 0.9, K)
    per_entry = torch.tensor([0.0, 0.4, 0.9])
    for s, t in [(0.0, 0.0), (0.3, 0.3), (0.0, 1.0), (0.55, 0.6), (per_entry, per_entry + 0.1)]:
        assert torch.equal(path.features(s, t, K), f)
    path.value(0.77)  # querying M does not change the coefficients
    assert torch.equal(path.features(0.2, 0.3, K), f)
    # Feature-major layout: channel n*C + c of the features is the xi_n that
    # drives state channel c, i.e. Cov(M_t[c], f[n*C+c']) = m_n(t) delta_cc'.
    a, t, c_count = 0.9, 0.6, 2
    path = make_path((1, c_count, 120, 120), amplitude=a, seed=7)
    xi = path.features(0.0, 1.0, K).reshape(K, c_count, -1)
    mt = path.value(t).reshape(c_count, -1)
    n = mt.shape[1]
    for n_mode in range(K):
        for c in range(c_count):
            for c2 in range(c_count):
                emp = (xi[n_mode, c2] * mt[c]).mean()
                expected = m(n_mode + 1, t, a) if c == c2 else 0.0
                assert abs(emp - expected) < 5 * math.sqrt(q(t, a) / n)


def test_features_are_iid_standard_normal():
    path = make_path((8, 2, 50, 50), seed=11)
    xi = path.features(0.2, 0.8, K).reshape(8, K, 2, 50, 50)
    flat = xi.permute(1, 0, 2, 3, 4).reshape(K, -1)
    n = flat.shape[1]
    assert flat.mean(1).abs().max() < 5 / math.sqrt(n)
    torch.testing.assert_close(
        torch.corrcoef(flat), torch.eye(K, dtype=torch.float64), atol=5 / math.sqrt(n), rtol=0
    )
    assert (flat.var(1) - 1).abs().max() < 5 * math.sqrt(2 / n)
    kurt = ((flat - flat.mean(1, keepdim=True)) ** 4).mean(1) / flat.var(1) ** 2
    assert (kurt - 3).abs().max() < 5 * math.sqrt(24 / n)
    # Independent across batch entries and channels.
    per_entry = xi.permute(0, 1, 2, 3, 4).reshape(8, -1)
    off = torch.corrcoef(per_entry) - torch.eye(8, dtype=torch.float64)
    assert off.abs().max() < 5 / math.sqrt(per_entry.shape[1])


# ------------------------------------------------- 3. independent simulation
SCRAMBLED = [0.73, 0.2, 1.0, 0.5005, 0.05, 0.5, 0.35]  # all on the 1/2000 grid


@pytest.mark.parametrize("a", [1.0, 0.6])
def test_joint_law_matches_independent_fine_grid_simulation(a):
    n = 20000
    analytic = analytic_cov(SCRAMBLED, a)
    ref = reference_samples(SCRAMBLED, a, n, seed=101)
    got = class_samples(SCRAMBLED, a, n, seed=202)
    cov_ref, cov_got = torch.cov(ref), torch.cov(got)
    se = cov_standard_error(analytic, n)
    # The reference itself reproduces the closed form (validates the oracle).
    assert ((cov_ref - analytic).abs() / se).max() < 5
    assert ((cov_got - analytic).abs() / se).max() < 5
    assert ((cov_got - cov_ref).abs() / (math.sqrt(2) * se)).max() < 5
    assert (got.mean(1).abs() / analytic.diagonal().div(n).sqrt()).max() < 5
    # The close pair (0.5, 0.5005): increment variance against the reference.
    i, j = K + SCRAMBLED.index(0.5), K + SCRAMBLED.index(0.5005)
    inc_got, inc_ref = got[j] - got[i], ref[j] - ref[i]
    exact = q(0.5005, a) - q(0.5, a)
    for inc in (inc_got, inc_ref):
        assert abs(inc.var() / exact - 1) < 5 * math.sqrt(2 / n)


# --------------------------------------------------------- 4. exact law checks
def test_increment_variances_and_mode_couplings():
    a, n = 0.8, 20000
    path = make_path((1, n), amplitude=a, seed=31)
    pairs = [(0.9, 1.0), (0.1, 0.4), (0.4, 0.400001), (0.0, 0.1), (0.62, 0.9), (0.15, 0.62)]
    xi = path.features(0.0, 0.0, K).reshape(K, n)
    for s, t in pairs:
        inc = path.increment(s, t).flatten()
        exact = q(t, a) - q(s, a)
        torch.testing.assert_close(
            float(path.variance(s, t)), exact, rtol=1e-6, atol=1e-15
        )
        assert abs(inc.var() / exact - 1) < 5 * math.sqrt(2 / n), (s, t)
    for t in (0.1, 0.4, 0.62, 0.9, 1.0):
        mt = path.value(t).flatten()
        for k in range(K):
            emp = (mt * xi[k]).mean()
            assert abs(emp - m(k + 1, t, a)) < 5 * math.sqrt(q(t, a) / n)
    # Sum over all modes recovers Var(M_1) = a^2/3 (checks the m_n oracle).
    assert abs(sum(m(k, 1.0, a) ** 2 for k in range(1, 4000)) - a * a / 3) < 1e-4


def test_query_order_does_not_change_joint_law():
    a, n = 1.0, 20000
    order_a = [0.2, 0.35, 0.5, 0.5005, 0.73, 1.0, 0.05]
    order_b = [1.0, 0.05, 0.73, 0.5005, 0.2, 0.5, 0.35]
    got_a = class_samples(order_a, a, n, seed=41)
    got_b = class_samples(order_b, a, n, seed=42)
    perm = [K + order_b.index(t) for t in order_a]
    got_b = torch.cat([got_b[:K], got_b[perm]])
    se = cov_standard_error(analytic_cov(order_a, a), n)
    assert ((torch.cov(got_a) - torch.cov(got_b)).abs() / (math.sqrt(2) * se)).max() < 5


def test_many_queries_are_stable_and_exact():
    a, n = 1.0, 4000
    gen = torch.Generator().manual_seed(9)
    times = torch.rand(40, generator=gen, dtype=torch.float64).tolist()
    times += [times[3] + 1e-7, 1.0, 0.0]
    got = class_samples(times, a, n, seed=55)
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got[-1]) == 0  # M_0 = 0
    got, times = got[:-1], times[:-1]
    inc = got[K + 40] - got[K + 3]  # increment over a 1e-7 gap
    assert abs(inc.var() / (q(times[40], a) - q(times[3], a)) - 1) < 5 * math.sqrt(2 / n)
    analytic = analytic_cov(times, a)
    se = cov_standard_error(analytic, n)
    assert ((torch.cov(got) - analytic).abs() / se).max() < 5.5


# --------------------------------------------------------- 5. model integration
class ConstantVelocity(nn.Module):
    def __init__(self, value=0.5):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))
        self.seen = []

    def forward(self, x, cond, path_features=None, **kwargs):
        self.seen.append(path_features)
        return torch.ones_like(x) * self.value


def ito(features=None, num=3):
    kwargs = {} if features is None else {"brownian_features": features}
    return FollmerItoMapModel(
        QuadraticStochasticInterpolation(gamma_multiplier=0.5, wiener_process=True),
        ConstantVelocity(),
        num_path_features=num,
        **kwargs,
    )


def test_model_selects_path_class():
    x = torch.zeros(2, 2, 4, 4)
    default = ito()
    assert default.brownian_features == "local"
    assert type(default.make_brownian_path(x)) is WeightedBrownianPath
    model = ito("karhunen_loeve", num=3)
    assert model.brownian_features == "karhunen_loeve"
    path = model.make_brownian_path(x, torch.Generator().manual_seed(0))
    assert isinstance(path, KarhunenLoeveBrownianPath) and path.num_modes == 3
    assert path.gamma_multiplier == 0.5
    out = model.average_velocity(x, 0.2, 0.6, brownian_path=path)
    assert out.shape == x.shape
    assert torch.equal(model.map_model.seen[-1], path.features(0.0, 1.0, 3))
    assert model.map_model.seen[-1].shape == (2, 6, 4, 4)
    model.average_velocity(x, 0.3, 0.3)  # diagonal without a path -> zeros
    assert torch.count_nonzero(model.map_model.seen[-1]) == 0
    with pytest.raises(ValueError):
        ito("kl")


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="stochastic_navier_stokes", overrides=overrides)


def test_hydra_config_instantiates_kl_and_legacy_default():
    from hydra.utils import instantiate

    torch.set_num_threads(1)
    cfg = _compose(["models=ito_map", "model.brownian_features=karhunen_loeve"])
    model = instantiate(cfg.model).eval()
    assert model.brownian_features == "karhunen_loeve"
    x = torch.randn(1, 1, 32, 32)
    path = model.make_brownian_path(x, torch.Generator().manual_seed(0))
    assert isinstance(path, KarhunenLoeveBrownianPath)
    assert path.num_modes == model.num_path_features == cfg.model.num_path_features
    history = torch.randn(1, 1, 32, 32, cfg.len_field_history)
    with torch.no_grad():
        out = model.average_velocity(x, 0.2, 0.7, history, brownian_path=path)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert _compose(["models=ito_map"]).model.brownian_features == "local"
    legacy = instantiate(_compose(["models=ito_map", "~model.brownian_features"]).model)
    assert legacy.brownian_features == "local"


def test_map_trainer_step_with_kl_path(tmp_path):
    from scisi.training.map_trainer import MapTrainer
    from scisi.training.trainer import EarlyStopping

    def batch():
        gen = torch.Generator().manual_seed(11)
        base = torch.randn(3, 1, 2, 2, generator=gen)
        return {
            "base": base,
            "target": base + torch.randn(base.shape, generator=gen),
            "field_history": base.unsqueeze(-1),
        }

    torch.manual_seed(0)
    model = ito("karhunen_loeve", num=4)
    model.map_model.value.data.fill_(0.0)
    instance = MapTrainer(
        num_epochs=1,
        model=model,
        train_dataloader=[batch()],
        val_dataloader=[batch()],
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        gradient_clipper=None,
        early_stopping=EarlyStopping(patience=3),
        device="cpu",
        checkpoint_dir=str(tmp_path),
        warmup_epochs=0,
    )
    model.map_model.seen.clear()
    loss = instance._train_step(instance._prepare_batch(batch()))
    assert torch.isfinite(loss) and instance.global_step == 1
    assert instance.model.map_model.value.item() != 0.0
    seen = [f for f in instance.model.map_model.seen if f is not None]
    assert seen and all(f.shape == (3, 4, 2, 2) for f in seen)
    assert any(torch.count_nonzero(f) > 0 for f in seen)  # KL coefficients used
