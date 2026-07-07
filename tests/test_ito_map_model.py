"""Tests for the ItoMapModel: diagonal target, teacher conversion, weight
surgery and sampling (plan Phase 5.3 / 5.4)."""

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


def test_warm_start_surgery():
    """Weight surgery: matching layers copied, input-channel-grown convs
    zero-expanded, and the two-time embedding exactly reproduces the
    teacher's single-time embedding at init."""
    torch.manual_seed(4)
    teacher_net = make_tiny_unet(cond_dim=1, field_cond_channels=2)
    student_net = make_tiny_unet(
        cond_dim=2, two_time_cond=True, field_cond_channels=2 + 5
    )

    report = warm_start_from_teacher(student_net, teacher_net)
    assert len(report["copied"]) > 0
    assert len(report["expanded"]) > 0

    # Deeper blocks are copied exactly.
    teacher_state = teacher_net.state_dict()
    student_state = student_net.state_dict()
    for name in report["copied"]:
        teacher_name = name.replace("cond_encoder.t_encoder.", "cond_encoder.")
        assert torch.equal(student_state[name], teacher_state[teacher_name])

    # The two-time embedding matches the teacher embedding on the diagonal.
    assert isinstance(student_net.cond_encoder, TwoTimeCondEncoder)
    t = torch.rand(BATCH, 1)
    s = torch.rand(BATCH, 1)
    with torch.no_grad():
        student_embedding = student_net.cond_encoder(torch.cat([s, t], dim=1))
        teacher_embedding = teacher_net.cond_encoder(t)
    assert torch.allclose(student_embedding, teacher_embedding, atol=1e-6)

    # Zero-expanded convs ignore the new (Brownian-feature) input channels.
    for name in report["expanded"]:
        teacher_param = teacher_state[name]
        student_param = student_state[name]
        assert torch.equal(student_param[:, : teacher_param.shape[1]], teacher_param)
        assert torch.all(student_param[:, teacher_param.shape[1] :] == 0)


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
            cond_dim=2, two_time_cond=True, field_cond_channels=3 * NUM_CHANNELS
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
