"""Tests for deterministic-to-Ito-map fine-tuning (PR 3, plan section 9.4):
residual calibration, the mean-anchored residual Ito map (Method 1), the
weight-surgery warm start (Method 2), the Gaussian-shell analytic teacher
(Method 3), and the staged-fine-tuning trainer options."""

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ito_map_test_helpers import (
    HEIGHT,
    LEN_FIELD_HISTORY,
    NUM_CHANNELS,
    WIDTH,
    make_tiny_unet,
    make_trainer_kwargs,
)

from scisi.deterministic_models.deterministic_model import DeterministicModel
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    LinearStochasticInterpolation,
    QuadraticStochasticInterpolation,
    _expand_t,
)
from scisi.models.ito_maps import (
    GammaMatchedSigmaSchedule,
    GaussianShellTeacher,
    ItoMapModel,
    KLEncoder,
    NextStepDriftAdapter,
    PaperSigmaSchedule,
    ResidualItoMapModel,
    ResidualStats,
    ZeroSigmaSchedule,
    estimate_residual_stats,
)
from scisi.training.ito_map_trainer import ItoMapTrainer

BATCH = 4
NUM_KL_COEFFS = 5


class _ConstNet(nn.Module):
    """Network stub returning a constant field (F = value everywhere)."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None, **kwargs):
        return torch.full_like(x, self.value)


class _ZeroDriftNet(nn.Module):
    """Drift-net stub returning zeros (accepts the Brownian-feature kwarg)."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, cond, field_history=None, field_cond=None, pars_cond=None, brownian_features=None):
        return torch.zeros_like(x)


class _CalibrationDataset(Dataset):
    """Targets with known per-channel statistics (residual of a zero model)."""

    def __init__(self, num_samples: int, mean: float, std: float) -> None:
        generator = torch.Generator().manual_seed(0)
        self.field_history = torch.randn(
            num_samples, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY,
            generator=generator,
        )
        self.target = mean + std * torch.randn(
            num_samples, NUM_CHANNELS, HEIGHT, WIDTH, generator=generator
        )

    def __len__(self):
        return self.field_history.shape[0]

    def __getitem__(self, idx):
        return {
            "field_history": self.field_history[idx],
            "base": self.field_history[idx, :, :, :, -1],
            "target": self.target[idx],
        }


def _make_mean_model(residual: bool = False, seed: int = 0) -> DeterministicModel:
    torch.manual_seed(seed)
    return DeterministicModel(network=make_tiny_unet(cond_dim=1), residual=residual)


def _make_inner_ito_map(stochastic: bool = True, zero_net: bool = False) -> ItoMapModel:
    if stochastic:
        return ItoMapModel(
            interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
            drift_model=(
                _ZeroDriftNet()
                if zero_net
                else make_tiny_unet(
                    cond_dim=2,
                    two_time_cond=True,
                    brownian_feature_channels=NUM_KL_COEFFS * NUM_CHANNELS,
                )
            ),
            sigma_schedule="gamma_matched",
            brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
            num_kl_terms=8,
        )
    return ItoMapModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=(
            _ZeroDriftNet() if zero_net else make_tiny_unet(cond_dim=2, two_time_cond=True)
        ),
        sigma_schedule="zero",
    )


def _make_residual_model(
    std: float = 0.5,
    stochastic: bool = True,
    zero_net: bool = False,
    seed: int = 0,
) -> ResidualItoMapModel:
    mean_model = _make_mean_model(seed=seed)
    torch.manual_seed(seed + 1)
    inner = _make_inner_ito_map(stochastic=stochastic, zero_net=zero_net)
    stats = ResidualStats(mean=torch.zeros(1), std=torch.full((1,), std))
    return ResidualItoMapModel.from_deterministic(
        det_model=mean_model, ito_map=inner, residual_stats=stats
    )


def _field_history(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)


# ----------------------------------------------------------------------
# Calibration (plan 9.4-4)
# ----------------------------------------------------------------------


def test_estimate_residual_stats_recovers_injected_scale(tmp_path):
    """With a zero mean model the residual IS the target, whose mean/std are
    known; the estimate must recover them and round-trip through disk."""
    injected_mean, injected_std = 0.7, 1.9
    dataloader = DataLoader(
        _CalibrationDataset(num_samples=256, mean=injected_mean, std=injected_std),
        batch_size=32,
    )
    det_model = DeterministicModel(network=_ConstNet(0.0), residual=False)

    stats = estimate_residual_stats(det_model, dataloader)

    assert stats.mean.shape == (NUM_CHANNELS,)
    assert stats.std.shape == (NUM_CHANNELS,)
    assert abs(stats.mean.item() - injected_mean) < 0.05
    assert abs(stats.std.item() - injected_std) / injected_std < 0.05

    path = str(tmp_path / "residual_stats.pt")
    stats.save(path)
    loaded = ResidualStats.load(path)
    assert torch.allclose(loaded.mean, stats.mean)
    assert torch.allclose(loaded.std, stats.std)


def test_estimate_residual_stats_uses_the_mean_model():
    """A constant mean model shifts the residual mean by exactly its output."""
    dataloader = DataLoader(
        _CalibrationDataset(num_samples=128, mean=0.0, std=1.0), batch_size=32
    )
    det_model = DeterministicModel(network=_ConstNet(0.4), residual=False)
    stats = estimate_residual_stats(det_model, dataloader)
    assert abs(stats.mean.item() + 0.4) < 0.05


# ----------------------------------------------------------------------
# Method 1: residual Ito map (plan 9.4-1, 9.4-2)
# ----------------------------------------------------------------------


def test_residual_map_init_identity_at_full_span():
    """With a zeroed residual net (and the martingale term removed), the
    composite map at (0, 1) is exactly the deterministic prediction."""
    torch.manual_seed(0)
    model = _make_residual_model(std=0.37, zero_net=True)
    model.eval()

    field_history = _field_history()
    x_n = field_history[:, :, :, :, -1]
    s = torch.zeros(BATCH, 1)
    t = torch.ones(BATCH, 1)

    out = model.map(
        x=x_n,
        s=s,
        t=t,
        martingale_increment=torch.zeros_like(x_n),
        field_history=field_history,
    )
    with torch.no_grad():
        mean_pred = model.mean_model._step(x_n, field_history=field_history)

    assert torch.allclose(out, mean_pred, atol=1e-5)


def test_residual_sample_composes_mean_plus_scaled_residual():
    """sample at (0, 1) is F(x_n) + rho * (inner residual sample): same seed,
    same Brownian draws, exact composition (plan section 5.2 endpoint)."""
    torch.manual_seed(0)
    std = 0.37
    model = _make_residual_model(std=std)
    model.eval()

    field_history = _field_history(2)

    torch.manual_seed(123)
    x = model.sample(field_history=field_history)

    torch.manual_seed(123)
    j = model.ito_map.sample(
        field_history=field_history,
        base=torch.zeros_like(field_history[:, :, :, :, -1]),
    )

    with torch.no_grad():
        mean_pred = model.mean_model._step(
            field_history[:, :, :, :, -1], field_history=field_history
        )

    assert torch.allclose(x, mean_pred + std * j, atol=1e-5)
    assert torch.isfinite(x).all()


def test_residual_composite_drift_change_of_variables():
    """Full-state diagonal drift equals phi'(t) + rho * g((x - phi(t)) / rho)
    with the anchor path phi computed independently here (plan section 5.2)."""
    torch.manual_seed(1)
    std = 0.6
    model = _make_residual_model(std=std)
    model.eval()

    field_history = _field_history()
    x_n = field_history[:, :, :, :, -1]
    x = torch.randn_like(x_n)
    t = torch.full((BATCH, 1), 0.6)

    with torch.no_grad():
        mean_pred = model.mean_model._step(x_n, field_history=field_history)

    interpolation = model.ito_map.interpolation
    t_expanded = _expand_t(t, x)
    phi = interpolation.alpha(t_expanded) * x_n + interpolation.beta(t_expanded) * mean_pred
    phi_diff = (
        interpolation.alpha_diff(t_expanded) * x_n
        + interpolation.beta_diff(t_expanded) * mean_pred
    )

    inner_drift = model.ito_map.drift((x - phi) / std, t, field_history=field_history)
    expected = phi_diff + std * inner_drift

    out = model.drift(x, t, field_history=field_history)
    assert torch.allclose(out, expected, atol=1e-5)


def test_residual_model_rejects_nonpositive_std():
    mean_model = _make_mean_model()
    inner = _make_inner_ito_map()
    stats = ResidualStats(mean=torch.zeros(1), std=torch.zeros(1))
    with pytest.raises(ValueError, match="positive"):
        ResidualItoMapModel(mean_model=mean_model, ito_map=inner, residual_stats=stats)


def test_to_residual_batch_moves_batch_to_residual_coordinates():
    """The trainer's batch preparation replaces (base, target) by
    (0, (target - F) / rho) and leaves the conditioning untouched."""
    torch.manual_seed(2)
    std = 0.37
    model = _make_residual_model(std=std)
    trainer = ItoMapTrainer(**make_trainer_kwargs(model))

    batch = next(iter(trainer.train_dataloader))
    original_target = batch["target"].clone()
    original_base = batch["base"].clone()
    original_history = batch["field_history"].clone()

    prepared = trainer._prepare_batch(batch)

    with torch.no_grad():
        mean_pred = model.mean_model._step(
            original_base, field_history=original_history
        )

    assert torch.all(prepared["base"] == 0)
    assert torch.allclose(
        prepared["target"], (original_target - mean_pred) / std, atol=1e-6
    )
    assert torch.allclose(prepared["field_history"], original_history)
    assert prepared["brownian_sample"] is not None


def test_residual_training_freezes_mean_model_and_trains_inner_map():
    torch.manual_seed(3)
    model = _make_residual_model()
    mean_before = [p.clone() for p in model.mean_model.parameters()]
    inner_before = [p.clone() for p in model.ito_map.drift_model.parameters()]

    trainer = ItoMapTrainer(**make_trainer_kwargs(model))
    trainer.train()

    assert math.isfinite(trainer.early_stopping.best_loss)
    assert all(
        torch.equal(before, after)
        for before, after in zip(mean_before, model.mean_model.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(inner_before, model.ito_map.drift_model.parameters())
    )
    # The wrapper keeps the frozen mean model in eval mode even in train mode.
    model.train()
    assert not model.mean_model.training


# ----------------------------------------------------------------------
# Method 2: weight-surgery warm start (plan 9.4-1)
# ----------------------------------------------------------------------


def _make_warm_started(residual_flag: bool) -> tuple[ItoMapModel, DeterministicModel]:
    det_model = _make_mean_model(residual=residual_flag, seed=7)
    torch.manual_seed(8)
    student = ItoMapModel(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(
            cond_dim=2,
            two_time_cond=True,
            brownian_feature_channels=NUM_KL_COEFFS * NUM_CHANNELS,
        ),
        sigma_schedule="gamma_matched",
        brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
        num_kl_terms=8,
    )
    student.warm_start_from_deterministic(det_model)
    return student, det_model


@pytest.mark.parametrize("residual_flag", [False, True])
def test_warm_start_init_identity(residual_flag):
    """After the deterministic warm start, X_hat_{0,1}(x_n, W) - (M_1 - M_0)
    equals F(x_n) to float precision, for both output-head conventions."""
    student, det_model = _make_warm_started(residual_flag)
    student.eval()
    det_model.eval()

    field_history = _field_history()
    x_n = field_history[:, :, :, :, -1]
    s = torch.zeros(BATCH, 1)
    t = torch.ones(BATCH, 1)
    brownian_features = torch.randn(BATCH, NUM_KL_COEFFS * NUM_CHANNELS, HEIGHT, WIDTH)

    with torch.no_grad():
        mean_pred = det_model._step(x_n, field_history=field_history)
        mapped = student.map(
            x=x_n,
            s=s,
            t=t,
            brownian_features=brownian_features,
            martingale_increment=torch.zeros_like(x_n),
            field_history=field_history,
        )

    assert torch.allclose(mapped, mean_pred, atol=1e-5)
    # The adapter is present exactly when the teacher predicted the state.
    assert isinstance(student.drift_model, NextStepDriftAdapter) == (not residual_flag)


@pytest.mark.parametrize("residual_flag", [False, True])
def test_warm_start_is_blind_to_times_and_brownian_features(residual_flag):
    """At init, perturbing (s, t) or the Brownian features changes nothing:
    the new pathways are zero-initialized / pinned (plan section 6)."""
    student, det_model = _make_warm_started(residual_flag)
    student.eval()
    det_model.eval()

    field_history = _field_history()
    x = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH)

    def drift_at(seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        s = torch.rand(BATCH, 1, generator=generator)
        t = torch.rand(BATCH, 1, generator=generator)
        brownian_features = torch.randn(
            BATCH, NUM_KL_COEFFS * NUM_CHANNELS, HEIGHT, WIDTH, generator=generator
        )
        with torch.no_grad():
            return student.G(
                x=x,
                s=s,
                t=t,
                brownian_features=brownian_features,
                field_history=field_history,
            )

    g_first, g_second = drift_at(0), drift_at(1)
    assert torch.allclose(g_first, g_second, atol=1e-6)

    with torch.no_grad():
        mean_pred = det_model._step(x, field_history=field_history)
    assert torch.allclose(g_first, mean_pred - x, atol=1e-5)


def test_from_deterministic_classmethod():
    det_model = _make_mean_model(residual=False, seed=9)
    torch.manual_seed(10)
    model = ItoMapModel.from_deterministic(
        det_model=det_model,
        drift_model=make_tiny_unet(
            cond_dim=2,
            two_time_cond=True,
            brownian_feature_channels=NUM_KL_COEFFS * NUM_CHANNELS,
        ),
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        sigma_schedule="gamma_matched",
        brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
        num_kl_terms=8,
    )
    # Not a distillation: the deterministic model is not attached as teacher.
    assert model.teacher is None
    assert isinstance(model.drift_model, NextStepDriftAdapter)


def test_distill_from_rejects_deterministic_model():
    det_model = _make_mean_model()
    torch.manual_seed(11)
    model = _make_inner_ito_map()
    with pytest.raises(TypeError, match="no drift"):
        model.distill_from(det_model)


# ----------------------------------------------------------------------
# Method 3: Gaussian-shell teacher (plan 9.4-3)
# ----------------------------------------------------------------------


def test_gaussian_shell_teacher_matches_monte_carlo_diagonal_target():
    """With the gamma-matched sigma schedule, the teacher must equal
    E[forward_diff | I_t = x] - the trainer's exact regression target for the
    diagonal drift (which needs no score correction in that setting). Checked
    against a kernel Monte-Carlo estimate on a scalar toy across a t-grid."""
    torch.manual_seed(0)
    rho, mu, x_n_value = 0.5, 0.3, -0.4
    interpolation = LinearStochasticInterpolation(
        gamma_multiplier=0.8, wiener_process=True
    )
    mean_model = DeterministicModel(network=_ConstNet(mu), residual=False)
    teacher = GaussianShellTeacher(
        interpolation=interpolation,
        sigma_schedule=GammaMatchedSigmaSchedule(interpolation),
        mean_model=mean_model,
        residual_std=torch.tensor([rho]),
    )

    num_samples = 400_000
    for t_value in (0.25, 0.5, 0.8):
        x_n = torch.full((num_samples, 1, 1, 1), x_n_value)
        target = mu + rho * torch.randn_like(x_n)
        noise = torch.randn_like(x_n)
        t = torch.full((num_samples, 1), t_value)

        interpolant = interpolation.forward(base=x_n, target=target, t=t, noise=noise)
        velocity_samples = interpolation.forward_diff(
            base=x_n, target=target, t=t, noise=noise
        )

        marginal_std = interpolant.std()
        x_query = interpolant.mean() + 0.5 * marginal_std
        window = (interpolant - x_query).abs() < 0.02 * marginal_std
        assert window.sum() > 500
        mc_velocity = velocity_samples[window].mean()

        field_history = torch.full((1, 1, 1, 1, 1), x_n_value)
        prediction = teacher.drift(
            x=torch.full((1, 1, 1, 1), x_query.item()),
            t=torch.full((1, 1), t_value),
            field_history=field_history,
        )
        assert abs(prediction.item() - mc_velocity.item()) < 0.05


def test_gaussian_shell_score_correction_matches_autograd():
    """drift(sigma) - drift(sigma = 0) must equal (sigma^2 / 2) * score, with
    the score computed by autograd through the analytic Gaussian marginal."""
    torch.manual_seed(1)
    rho, mu, x_n_value, t_value = 0.5, 0.3, -0.4, 0.6
    interpolation = LinearStochasticInterpolation(
        gamma_multiplier=0.8, wiener_process=True
    )
    mean_model = DeterministicModel(network=_ConstNet(mu), residual=False)
    sigma_schedule = PaperSigmaSchedule()

    teacher_kwargs = {
        "interpolation": interpolation,
        "mean_model": mean_model,
        "residual_std": torch.tensor([rho]),
    }
    with_score = GaussianShellTeacher(sigma_schedule=sigma_schedule, **teacher_kwargs)
    without_score = GaussianShellTeacher(
        sigma_schedule=ZeroSigmaSchedule(), **teacher_kwargs
    )

    x = torch.full((1, 1, 1, 1), 0.2)
    t = torch.full((1, 1), t_value)
    field_history = torch.full((1, 1, 1, 1, 1), x_n_value)

    drift_difference = (
        with_score.drift(x, t, field_history=field_history)
        - without_score.drift(x, t, field_history=field_history)
    ).item()

    # Analytic Gaussian marginal of I_t, score via autograd.
    t_scalar = torch.tensor([[t_value]])
    alpha, beta = interpolation.alpha(t_scalar), interpolation.beta(t_scalar)
    sigma_path = interpolation.sigma(t_scalar)
    marginal_mean = alpha * x_n_value + beta * mu
    marginal_std = torch.sqrt(beta**2 * rho**2 + sigma_path**2)

    x_grad = torch.full((1, 1), 0.2, requires_grad=True)
    log_prob = torch.distributions.Normal(marginal_mean, marginal_std).log_prob(x_grad)
    (score,) = torch.autograd.grad(log_prob.sum(), x_grad)

    sigma = sigma_schedule(t_scalar)
    expected = (0.5 * sigma**2 * score).item()
    assert abs(drift_difference - expected) < 1e-6


def test_gaussian_shell_residual_mode_matches_zero_mean_full_mode():
    """mean_model = None (residual coordinates) is the F = 0, rho = 1,
    anchor = 0 case of the full-state teacher."""
    interpolation = QuadraticStochasticInterpolation(gamma_multiplier=1.0)
    sigma_schedule = PaperSigmaSchedule()

    residual_teacher = GaussianShellTeacher(
        interpolation=interpolation, sigma_schedule=sigma_schedule
    )
    full_teacher = GaussianShellTeacher(
        interpolation=interpolation,
        sigma_schedule=sigma_schedule,
        mean_model=DeterministicModel(network=_ConstNet(0.0), residual=False),
        residual_std=torch.ones(1),
    )

    torch.manual_seed(2)
    x = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH)
    t = torch.rand(BATCH, 1)
    field_history = torch.zeros(BATCH, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)

    assert torch.allclose(
        residual_teacher.drift(x, t, field_history=field_history),
        full_teacher.drift(x, t, field_history=field_history),
        atol=1e-6,
    )


# ----------------------------------------------------------------------
# Trainer options (plan 9.4-5)
# ----------------------------------------------------------------------


def test_freeze_keeps_module_fixed():
    torch.manual_seed(4)
    model = _make_inner_ito_map()
    frozen_before = [p.clone() for p in model.drift_model.init_conv.parameters()]

    trainer = ItoMapTrainer(
        **make_trainer_kwargs(model), freeze=["drift_model.init_conv"]
    )
    trainer.train()

    frozen_module = model.drift_model.init_conv
    assert all(not p.requires_grad for p in frozen_module.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(frozen_before, frozen_module.parameters())
    )
    # Everything else trained.
    assert any(
        p.grad is not None or p.requires_grad for p in model.drift_model.parameters()
    )


def test_unfreeze_at_epoch_flips_frozen_modules():
    torch.manual_seed(5)
    model = _make_inner_ito_map()
    frozen_before = [p.clone() for p in model.drift_model.init_conv.parameters()]

    trainer = ItoMapTrainer(
        **make_trainer_kwargs(model),
        freeze=["drift_model.init_conv"],
        unfreeze_at_epoch=1,
    )
    trainer.train()

    frozen_module = model.drift_model.init_conv
    assert all(p.requires_grad for p in frozen_module.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(frozen_before, frozen_module.parameters())
    )


def test_unfreeze_at_epoch_requires_freeze():
    model = _make_inner_ito_map()
    with pytest.raises(ValueError, match="unfreeze_at_epoch"):
        ItoMapTrainer(**make_trainer_kwargs(model), unfreeze_at_epoch=1)


def test_teacher_warmup_detaches_teacher():
    torch.manual_seed(6)
    model = _make_inner_ito_map()
    model.distill_from(
        GaussianShellTeacher(
            interpolation=model.interpolation, sigma_schedule=model.sigma_schedule
        )
    )
    assert model.teacher is not None

    trainer = ItoMapTrainer(**make_trainer_kwargs(model), teacher_warmup_epochs=1)
    trainer.train()

    assert model.teacher is None
    assert math.isfinite(trainer.early_stopping.best_loss)


def test_teacher_warmup_zero_keeps_teacher():
    torch.manual_seed(6)
    model = _make_inner_ito_map()
    model.distill_from(
        GaussianShellTeacher(
            interpolation=model.interpolation, sigma_schedule=model.sigma_schedule
        )
    )
    trainer = ItoMapTrainer(**make_trainer_kwargs(model), teacher_warmup_epochs=0)
    trainer.train()
    assert model.teacher is not None


# ----------------------------------------------------------------------
# Smoke fine-tunes (plan 9.4-6)
# ----------------------------------------------------------------------


def _assert_sampling_works(model) -> None:
    field_history = _field_history(1)
    sample = model.sample(field_history=field_history, batch_size=3)
    assert sample.shape == (3, NUM_CHANNELS, HEIGHT, WIDTH)
    assert torch.isfinite(sample).all()

    trajectory = model.sample_trajectory(
        field_history=field_history, batch_size=2, num_physical_steps=6
    )
    assert trajectory.shape == (2, NUM_CHANNELS, HEIGHT, WIDTH, 6)
    assert torch.isfinite(trajectory).all()


@pytest.mark.parametrize("consistency_mode", ["lsd", "psd"])
def test_residual_finetune_smoke_stochastic(consistency_mode):
    torch.manual_seed(0)
    model = _make_residual_model(stochastic=True)
    trainer = ItoMapTrainer(
        **make_trainer_kwargs(model), consistency_mode=consistency_mode
    )
    trainer.train()
    assert math.isfinite(trainer.early_stopping.best_loss)
    _assert_sampling_works(model)


def test_residual_finetune_smoke_flow_matching_variant():
    """FM variant (plan section 5.5): Gaussian residual base, sigma = 0."""
    torch.manual_seed(1)
    model = _make_residual_model(stochastic=False)
    trainer = ItoMapTrainer(**make_trainer_kwargs(model))
    trainer.train()
    assert math.isfinite(trainer.early_stopping.best_loss)
    _assert_sampling_works(model)


def test_residual_finetune_smoke_with_analytic_teacher_warmup():
    """Method 1 + Method 3: Gaussian-shell warm-up in residual coordinates,
    switching to data targets after one epoch."""
    torch.manual_seed(2)
    model = _make_residual_model(stochastic=True)
    model.ito_map.distill_from(
        GaussianShellTeacher(
            interpolation=model.ito_map.interpolation,
            sigma_schedule=model.ito_map.sigma_schedule,
        )
    )
    trainer = ItoMapTrainer(**make_trainer_kwargs(model), teacher_warmup_epochs=1)
    trainer.train()
    assert math.isfinite(trainer.early_stopping.best_loss)
    assert model.ito_map.teacher is None
    _assert_sampling_works(model)


@pytest.mark.parametrize("residual_flag", [False, True])
def test_warm_start_finetune_smoke(residual_flag):
    torch.manual_seed(3)
    model, _ = _make_warm_started(residual_flag)
    trainer = ItoMapTrainer(**make_trainer_kwargs(model))
    trainer.train()
    assert math.isfinite(trainer.early_stopping.best_loss)

    field_history = _field_history(1)
    sample = model.sample(field_history=field_history, batch_size=3)
    assert sample.shape == (3, NUM_CHANNELS, HEIGHT, WIDTH)
    assert torch.isfinite(sample).all()
