"""End-to-end smoke tests for the ItoMapTrainer (plan Phase 5.5): 2-epoch
from-scratch and distillation runs on the tiny synthetic dataset, covering
lsd/psd, stochastic and sigma = 0, jvp and finite-difference derivatives."""

import math

import pytest
import torch

from ito_map_test_helpers import (
    NUM_CHANNELS,
    make_tiny_attention_unet,
    make_tiny_unet,
    make_trainer_kwargs,
)

from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticStochasticInterpolation,
)
from scisi.models.ito_maps.brownian import KLEncoder
from scisi.models.ito_maps.ito_map_model import ItoMapModel
from scisi.training.ito_map_trainer import ItoMapTrainer

NUM_KL_COEFFS = 3


def _make_stochastic_ito_map(brownian_mode: str = "kl") -> ItoMapModel:
    return ItoMapModel(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(
            cond_dim=2,
            two_time_cond=True,
            field_cond_channels=NUM_KL_COEFFS * NUM_CHANNELS,
        ),
        sigma_schedule="gamma_matched",
        brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
        num_kl_terms=8,
        num_grid_points=32,
        brownian_mode=brownian_mode,
    )


def _make_deterministic_ito_map() -> ItoMapModel:
    return ItoMapModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(cond_dim=2, two_time_cond=True),
        sigma_schedule="zero",
    )


def _train(model: ItoMapModel, **trainer_overrides) -> ItoMapTrainer:
    kwargs = make_trainer_kwargs(model)
    kwargs.update(trainer_overrides)
    trainer = ItoMapTrainer(**kwargs)
    trainer.train()
    assert math.isfinite(trainer.early_stopping.best_loss)
    return trainer


@pytest.mark.parametrize("consistency_mode", ["lsd", "psd"])
def test_from_scratch_stochastic(consistency_mode):
    torch.manual_seed(0)
    _train(
        _make_stochastic_ito_map(),
        consistency_mode=consistency_mode,
        off_diagonal_distribution="logit_normal",
    )


@pytest.mark.parametrize("consistency_mode", ["lsd", "psd"])
def test_from_scratch_deterministic_sigma_zero(consistency_mode):
    """sigma = 0: Brownian features and martingale increments vanish and LSD
    reduces to deterministic flow-map matching - no special-case code."""
    torch.manual_seed(1)
    _train(
        _make_deterministic_ito_map(),
        consistency_mode=consistency_mode,
        off_diagonal_distribution="uniform",
    )


def test_finite_difference_fallback():
    torch.manual_seed(2)
    _train(
        _make_stochastic_ito_map(),
        consistency_mode="lsd",
        derivative_mode="finite_difference",
    )


def test_path_mode_brownian():
    torch.manual_seed(3)
    _train(_make_stochastic_ito_map(brownian_mode="path"), consistency_mode="lsd")


def test_distillation_from_follmer_teacher():
    torch.manual_seed(4)
    teacher = FollmerStochasticInterpolant(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(),
    )
    student = ItoMapModel.from_stochastic_interpolant(
        si_model=teacher,
        drift_model=make_tiny_unet(
            cond_dim=2,
            two_time_cond=True,
            field_cond_channels=NUM_KL_COEFFS * NUM_CHANNELS,
        ),
        sigma_schedule="gamma_matched",
        brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
        num_kl_terms=8,
    )
    assert student.teacher is not None

    _train(student, consistency_mode="lsd")


def test_distillation_from_flow_matching_teacher_with_ema_target():
    torch.manual_seed(5)
    teacher = FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(),
    )
    student = ItoMapModel.from_flow_matching(
        flow_matching_model=teacher,
        drift_model=make_tiny_unet(
            cond_dim=2,
            two_time_cond=True,
            field_cond_channels=NUM_KL_COEFFS * NUM_CHANNELS,
        ),
        sigma_schedule="paper",
        brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
        num_kl_terms=8,
    )
    assert student.teacher is not None

    trainer = _train(student, consistency_mode="lsd", ema_decay=0.99)
    assert trainer.ema_model is not None
    # The EMA copy shares the stashed teacher reference, not a re-registration.
    assert not any("teacher" in name for name, _ in trainer.ema_model.named_parameters())


def test_diagonal_only_training():
    torch.manual_seed(6)
    _train(_make_stochastic_ito_map(), consistency_weight=0.0)


def test_lsd_jvp_through_attention_unet():
    """Flash attention has no forward-mode AD; the trainer must force the
    math SDPA backend so jvp works through attention architectures."""
    torch.manual_seed(7)
    model = ItoMapModel(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_attention_unet(
            field_cond_channels=NUM_KL_COEFFS * NUM_CHANNELS
        ),
        sigma_schedule="gamma_matched",
        brownian_encoder=KLEncoder(num_coeffs=NUM_KL_COEFFS),
        num_kl_terms=8,
    )
    trainer = _train(model, consistency_mode="lsd", derivative_mode="jvp")
    # jvp must have worked - no silent downgrade to finite differences.
    assert trainer.derivative_mode == "jvp"
