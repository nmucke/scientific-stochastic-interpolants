"""Tests for the ItoMapModel: diagonal target, teacher conversion, weight
surgery and sampling (plan Phase 5.3 / 5.4)."""

import pytest
import torch
import torch.nn as nn
from ito_map_test_helpers import (
    HEIGHT,
    LEN_FIELD_HISTORY,
    NUM_CHANNELS,
    WIDTH,
    make_tiny_unet,
)

from scisi.architectures.embeddings import TwoTimeCondEncoder
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.models.interpolations import (
    LinearDeterministicInterpolation,
    QuadraticStochasticInterpolation,
    _expand_t,
)
from scisi.models.ito_maps.ito_map_model import (
    FlowMatchingTeacher,
    FollmerTeacher,
    ItoMapModel,
    warm_start_from_teacher,
)
from scisi.models.ito_maps.brownian import KLEncoder, PaperSigmaSchedule

BATCH = 6


def _rand_state(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, NUM_CHANNELS, HEIGHT, WIDTH)


def _rand_times(batch: int = BATCH) -> torch.Tensor:
    # Stay inside the clamped region so closed-form identities are exact.
    return 0.1 + 0.8 * torch.rand(batch, 1)


def test_diag_target_reduces_to_x1_minus_2x0_in_paper_setting():
    """Linear path, Gaussian base, sigma = sqrt(2(1-t)): the per-sample
    diagonal target is exactly X_1 - 2 X_0 with X_0 = noise (plan Phase 5.3)."""
    torch.manual_seed(0)
    model = ItoMapModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=nn.Identity(),
        sigma_schedule="paper",
    )

    base, target, noise = _rand_state(), _rand_state(), _rand_state()
    t = _rand_times()

    diag_target = model.G_diag_target(base=base, target=target, noise=noise, t=t)
    assert torch.allclose(diag_target, target - 2 * noise, atol=1e-5)


def test_diag_target_gamma_matched_equals_forward_diff():
    """With sigma = gamma the score correction vanishes: the target is the
    plain Follmer regression target forward_diff."""
    torch.manual_seed(1)
    interpolation = QuadraticStochasticInterpolation(gamma_multiplier=1.0)
    model = ItoMapModel(
        interpolation=interpolation,
        drift_model=nn.Identity(),
        sigma_schedule="gamma_matched",
    )

    base, target, noise = _rand_state(), _rand_state(), _rand_state()
    t = _rand_times()

    diag_target = model.G_diag_target(base=base, target=target, noise=noise, t=t)
    expected = interpolation.forward_diff(base=base, target=target, t=t, noise=noise)
    assert torch.allclose(diag_target, expected, atol=1e-6)


def test_flow_matching_teacher_matches_paper_shortcut():
    """For the linear FM path and sigma = sqrt(2(1-t)) the converted teacher
    drift equals the paper's shortcut (1 + t) v - x for any velocity net
    (plan Phase 5.4)."""
    torch.manual_seed(2)
    fm_model = FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(),
    )
    teacher = FlowMatchingTeacher(fm_model, PaperSigmaSchedule())

    x = _rand_state()
    t = _rand_times()
    field_history = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)

    fm_model.eval()
    with torch.no_grad():
        velocity = fm_model.drift_model(x, t, field_history, None, None)
        converted = teacher.drift(x, t, field_history)

    expected = (1 + _expand_t(t, x)) * velocity - x
    assert torch.allclose(converted, expected, atol=1e-5)


def test_follmer_teacher_is_identity_when_sigma_matches_gamma():
    """With sigma = gamma the prior-score correction is zero: the teacher
    drift is exactly the trained Follmer drift b_theta."""
    torch.manual_seed(3)
    si_model = FollmerStochasticInterpolant(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(),
    )
    ito_model = ItoMapModel(
        interpolation=si_model.interpolation,
        drift_model=nn.Identity(),
        sigma_schedule="gamma_matched",
    )
    teacher = FollmerTeacher(si_model, ito_model.sigma_schedule)

    x = _rand_state()
    t = _rand_times()
    field_history = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)

    si_model.eval()
    with torch.no_grad():
        b_theta = si_model.drift_model(x, t, field_history, None, None)
        converted = teacher.drift(x, t, field_history)

    assert torch.allclose(converted, b_theta, atol=1e-6)


def test_follmer_teacher_small_t_guard():
    """Below MIN_TIME the prior-score correction (which diverges like 1/t^2)
    is dropped per-sample, matching _drift_with_prior_score's behavior - even
    for the 'paper' sigma schedule where the correction is nonzero."""
    torch.manual_seed(30)
    si_model = FollmerStochasticInterpolant(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(),
    )
    teacher = FollmerTeacher(si_model, PaperSigmaSchedule())

    x = _rand_state()
    field_history = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)
    t_small = torch.full((BATCH, 1), 5e-5)  # below MIN_TIME = 1e-4

    si_model.eval()
    with torch.no_grad():
        b_theta = si_model.drift_model(x, t_small, field_history, None, None)
        converted = teacher.drift(x, t_small, field_history)

    assert torch.allclose(converted, b_theta, atol=1e-6)
    assert torch.isfinite(converted).all()


def test_warm_start_functional_equivalence_in_ns_layout():
    """The real NS distillation layout: teacher WITHOUT field_cond, student
    with Brownian features. Because features enter via a dedicated
    zero-initialized projection (not the init conv), the student's full
    forward pass at init must reproduce the teacher's output exactly - on
    network OUTPUTS, for any s and any Brownian features."""
    torch.manual_seed(4)
    teacher_net = make_tiny_unet(cond_dim=1)  # field_cond_channels: null
    student_net = make_tiny_unet(
        cond_dim=2, two_time_cond=True, brownian_feature_channels=5
    )

    report = warm_start_from_teacher(student_net, teacher_net)
    assert len(report["copied"]) > 0

    # The ONLY parameters allowed to stay at fresh init are student-specific:
    # the (zeroed) s-branch of the two-time embedding and the (zero-init)
    # Brownian feature projection. Everything else - including the whole
    # input stage - must transfer.
    allowed_skipped_prefixes = ("cond_encoder.s_encoder.", "brownian_proj.")
    assert all(
        name.startswith(allowed_skipped_prefixes) for name in report["skipped"]
    ), f"unexpected skipped keys: {report['skipped']}"

    # Functional equivalence of full forward passes at init.
    teacher_net.eval()
    student_net.eval()
    x = _rand_state()
    field_history = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)
    s, t = torch.rand(BATCH, 1), torch.rand(BATCH, 1)
    brownian_features = torch.randn(BATCH, 5, HEIGHT, WIDTH)

    with torch.no_grad():
        teacher_out = teacher_net(x, t, field_history)
        student_out = student_net(
            x,
            torch.cat([s, t], dim=1),
            field_history=field_history,
            brownian_features=brownian_features,
        )
    assert torch.allclose(student_out, teacher_out, atol=1e-6)

    # The two-time embedding matches the teacher embedding on the diagonal.
    assert isinstance(student_net.cond_encoder, TwoTimeCondEncoder)
    with torch.no_grad():
        student_embedding = student_net.cond_encoder(torch.cat([s, t], dim=1))
        teacher_embedding = teacher_net.cond_encoder(t)
    assert torch.allclose(student_embedding, teacher_embedding, atol=1e-6)


def test_warm_start_functional_equivalence_with_field_cond_teacher():
    """Same guarantee when the teacher itself uses field_cond (both nets have
    the identical init conv class; field_cond channel counts match)."""
    torch.manual_seed(40)
    teacher_net = make_tiny_unet(cond_dim=1, field_cond_channels=2)
    student_net = make_tiny_unet(
        cond_dim=2,
        two_time_cond=True,
        field_cond_channels=2,
        brownian_feature_channels=5,
    )

    report = warm_start_from_teacher(student_net, teacher_net)
    allowed_skipped_prefixes = ("cond_encoder.s_encoder.", "brownian_proj.")
    assert all(
        name.startswith(allowed_skipped_prefixes) for name in report["skipped"]
    ), f"unexpected skipped keys: {report['skipped']}"

    teacher_net.eval()
    student_net.eval()
    x = _rand_state()
    field_history = torch.randn(BATCH, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)
    field_cond = torch.randn(BATCH, 2, HEIGHT, WIDTH)
    s, t = torch.rand(BATCH, 1), torch.rand(BATCH, 1)
    brownian_features = torch.randn(BATCH, 5, HEIGHT, WIDTH)

    with torch.no_grad():
        teacher_out = teacher_net(x, t, field_history, field_cond)
        student_out = student_net(
            x,
            torch.cat([s, t], dim=1),
            field_history=field_history,
            field_cond=field_cond,
            brownian_features=brownian_features,
        )
    assert torch.allclose(student_out, teacher_out, atol=1e-6)


def test_deepcopy_shares_frozen_teacher():
    """deepcopy(model) (the trainer's EMA setup) must share the stashed
    teacher, not silently duplicate it in memory."""
    from copy import deepcopy

    torch.manual_seed(50)
    fm_model = FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(),
    )
    student = ItoMapModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(cond_dim=2, two_time_cond=True),
        sigma_schedule="paper",
    )
    student.distill_from(fm_model)

    copied = deepcopy(student)
    assert copied.teacher is student.teacher  # shared, not duplicated
    # The copy's own weights are independent.
    original_param = next(student.drift_model.parameters())
    copied_param = next(copied.drift_model.parameters())
    assert copied_param is not original_param
    assert torch.equal(copied_param, original_param)


def test_dyadic_encoder_with_kl_mode_requires_enough_terms():
    """K=16 KL terms cannot represent dyadic detail at scale 1/16: the
    pairing must be rejected unless num_kl_terms >> 2**depth."""
    from scisi.models.ito_maps.brownian import DyadicEncoder

    with pytest.raises(ValueError, match="DyadicEncoder"):
        ItoMapModel(
            interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
            drift_model=nn.Identity(),
            sigma_schedule="gamma_matched",
            brownian_encoder=DyadicEncoder(depth=4),
            brownian_mode="kl",
            num_kl_terms=16,
        )

    # Enough terms, or path mode, are both fine.
    ItoMapModel(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=nn.Identity(),
        sigma_schedule="gamma_matched",
        brownian_encoder=DyadicEncoder(depth=2),
        brownian_mode="kl",
        num_kl_terms=16,
    )
    ItoMapModel(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=nn.Identity(),
        sigma_schedule="gamma_matched",
        brownian_encoder=DyadicEncoder(depth=4),
        brownian_mode="path",
    )


def test_distill_from_stashes_frozen_teacher_outside_state_dict():
    torch.manual_seed(5)
    fm_model = FlowMatchingModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(),
    )
    student = ItoMapModel(
        interpolation=LinearDeterministicInterpolation(),
        drift_model=make_tiny_unet(cond_dim=2, two_time_cond=True),
        sigma_schedule="paper",
    )
    num_params_before = sum(p.numel() for p in student.parameters())
    state_keys_before = set(student.state_dict().keys())

    student.distill_from(fm_model)

    assert student.teacher is not None
    assert sum(p.numel() for p in student.parameters()) == num_params_before
    assert set(student.state_dict().keys()) == state_keys_before
    for param in fm_model.parameters():
        assert not param.requires_grad


def test_one_step_sample_and_trajectory_shapes():
    """One-step endpoint sampling and the autoregressive rollout (plan
    Phase 5.5, sampling part)."""
    torch.manual_seed(6)
    model = ItoMapModel(
        interpolation=QuadraticStochasticInterpolation(gamma_multiplier=1.0),
        drift_model=make_tiny_unet(
            cond_dim=2, two_time_cond=True, brownian_feature_channels=3 * NUM_CHANNELS
        ),
        sigma_schedule="gamma_matched",
        brownian_encoder=KLEncoder(num_coeffs=3),
        num_kl_terms=8,
    )
    model.eval()

    field_history = torch.randn(1, NUM_CHANNELS, HEIGHT, WIDTH, LEN_FIELD_HISTORY)

    sample = model.sample(field_history=field_history, batch_size=2, num_steps=1)
    assert sample.shape == (2, NUM_CHANNELS, HEIGHT, WIDTH)
    assert torch.isfinite(sample).all()

    # Few-step partition reuses the same Brownian path.
    sample_4 = model.sample(field_history=field_history, batch_size=2, num_steps=4)
    assert sample_4.shape == (2, NUM_CHANNELS, HEIGHT, WIDTH)
    assert torch.isfinite(sample_4).all()

    trajectory = model.sample_trajectory(
        field_history=field_history,
        batch_size=2,
        num_steps=1,
        num_physical_steps=LEN_FIELD_HISTORY + 2,
    )
    assert trajectory.shape == (
        2,
        NUM_CHANNELS,
        HEIGHT,
        WIDTH,
        LEN_FIELD_HISTORY + 2,
    )
    assert torch.isfinite(trajectory).all()
