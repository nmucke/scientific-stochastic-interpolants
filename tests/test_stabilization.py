"""Stabilization of Lagrangian self-distillation training.

Covers: explicit attention under forward-mode AD (fp32 logits, softmax
curvature cut from the gradient), the rotary positional embedding fix (and its
legacy mode), dropout-free LSD passes, and global-norm clipping with skipped
updates on nonfinite gradients.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch import nn

import scisi.architectures.attention as attention
from scisi.architectures.attention import (
    AttentionWithRotaryPositionalEmbeddings,
    SpatialAttention,
    combine_heads,
    split_heads,
)
from scisi.models.flow_map_model import FlowMapModel
from scisi.models.interpolations import LinearDeterministicInterpolation
from scisi.training.gradient_clipping import EmaGradientClipper, GlobalNormClipper
from scisi.training.map_trainer import MapTrainer, lagrangian_self_distillation_loss
from scisi.training.trainer import EarlyStopping

torch.set_num_threads(4)


# ------------------------------------------------------------------ attention


def _qkv(dtype=torch.float64, seed=0, shape=(2, 3, 5, 4)):
    generator = torch.Generator().manual_seed(seed)
    tensors = [torch.randn(shape, generator=generator, dtype=dtype) for _ in range(6)]
    # Large logits: saturated softmax, where the curvature term matters.
    tensors[0] = tensors[0] * 3
    return tensors


def _plain_attention(q, k, v):
    weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), -1)
    return weights @ v


def _explicit(q, k, v, dq, dk, dv, dropout_p=0.0):
    with fwAD.dual_level():
        out = attention.scaled_dot_product_attention(
            fwAD.make_dual(q, dq),
            fwAD.make_dual(k, dk),
            fwAD.make_dual(v, dv),
            dropout_p=dropout_p,
        )
        primal, tangent = fwAD.unpack_dual(out)
    return primal, tangent


def _manual(q, k, v, dq, dk, dv, detach_weight_tangent):
    """Explicit attention and its JVP, optionally cutting the weight tangent."""
    scale = 1 / math.sqrt(q.shape[-1])
    weights = torch.softmax(q @ k.transpose(-2, -1) * scale, -1)
    logits_tangent = (dq @ k.transpose(-2, -1) + q @ dk.transpose(-2, -1)) * scale
    weights_tangent = weights * (
        logits_tangent - (weights * logits_tangent).sum(-1, keepdim=True)
    )
    if detach_weight_tangent:
        weights_tangent = weights_tangent.detach()
    return weights @ v, weights_tangent @ v + weights @ dv


def test_explicit_attention_matches_jvp_reference_in_float32():
    q, k, v, dq, dk, dv = _qkv(torch.float32)
    primal, tangent = _explicit(q, k, v, dq, dk, dv)
    ref_primal, ref_tangent = torch.func.jvp(_plain_attention, (q, k, v), (dq, dk, dv))
    torch.testing.assert_close(primal, ref_primal, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tangent, ref_tangent, rtol=1e-5, atol=1e-5)


def test_explicit_attention_matches_jvp_reference_in_float64():
    q, k, v, dq, dk, dv = _qkv(torch.float64)
    primal, tangent = _explicit(q, k, v, dq, dk, dv)
    ref_primal, ref_tangent = torch.func.jvp(_plain_attention, (q, k, v), (dq, dk, dv))
    assert primal.dtype == torch.float64 and tangent.dtype == torch.float64
    torch.testing.assert_close(primal, ref_primal, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(tangent, ref_tangent, rtol=1e-10, atol=1e-12)


def test_explicit_attention_tangent_only_on_some_inputs():
    """Only v carries a tangent: the weights have none and the result is W dv."""
    q, k, v, _, _, dv = _qkv(torch.float32)
    with fwAD.dual_level():
        out = attention.scaled_dot_product_attention(q, k, fwAD.make_dual(v, dv))
        primal, tangent = fwAD.unpack_dual(out)
    ref_primal, ref_tangent = torch.func.jvp(
        lambda v_: _plain_attention(q, k, v_), (v,), (dv,)
    )
    torch.testing.assert_close(primal, ref_primal, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(tangent, ref_tangent, rtol=1e-5, atol=1e-6)


def test_no_tangent_path_is_fused_attention():
    q, k, v, *_ = _qkv(torch.float32)
    torch.testing.assert_close(
        attention.scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        rtol=0,
        atol=0,
    )


def test_logits_and_softmax_are_fp32_under_bf16_autocast(monkeypatch):
    q, k, v, dq, dk, dv = (x.to(torch.bfloat16) for x in _qkv(torch.float32))
    seen = []
    softmax = torch.softmax

    def recording_softmax(x, *args, **kwargs):
        seen.append((x.dtype, torch.is_autocast_enabled("cpu")))
        return softmax(x, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", recording_softmax)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        primal, tangent = _explicit(q, k, v, dq, dk, dv)
    assert seen == [(torch.float32, False)]
    assert primal.dtype == torch.bfloat16 and tangent.dtype == torch.bfloat16
    # Same as the fp32 computation up to the bf16 rounding of the result.
    ref_primal, ref_tangent = _manual(
        *(x.float() for x in (q, k, v, dq, dk, dv)), detach_weight_tangent=False
    )
    torch.testing.assert_close(primal.float(), ref_primal, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tangent.float(), ref_tangent, rtol=2e-2, atol=5e-2)


def _grads(fn, inputs, cotangents):
    inputs = [x.clone().requires_grad_(True) for x in inputs]
    primal, tangent = fn(*inputs)
    c_primal, c_tangent = cotangents
    ((c_primal * primal).sum() + (c_tangent * tangent).sum()).backward()
    return [torch.zeros_like(x) if x.grad is None else x.grad for x in inputs]


def test_gradient_drops_exactly_the_softmax_curvature_term():
    inputs = _qkv(torch.float64, seed=1)
    generator = torch.Generator().manual_seed(2)
    cotangents = [torch.randn(inputs[0].shape, generator=generator, dtype=torch.float64)
                  for _ in range(2)]
    new = _grads(_explicit, inputs, cotangents)
    cut = _grads(lambda *a: _manual(*a, detach_weight_tangent=True), inputs, cotangents)
    exact = _grads(lambda *a: _manual(*a, detach_weight_tangent=False), inputs, cotangents)
    # Exact JVP reference: the uncut manual formula is the true gradient.
    exact_func = _grads(
        lambda q, k, v, dq, dk, dv: torch.func.jvp(
            _plain_attention, (q, k, v), (dq, dk, dv)
        ),
        inputs,
        cotangents,
    )
    for a, b in zip(exact, exact_func):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
    # The new path's internal fp32 cast limits the agreement to fp32 precision.
    for name, a, b in zip(["q", "k", "v", "dq", "dk", "dv"], new, cut):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=name)
    # The dropped term only reaches q, k, dq, dk (through the weight tangent).
    for name, a, b in zip(["q", "k", "v", "dq", "dk", "dv"], cut, exact):
        if name in ("v", "dv"):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12, msg=name)
        else:
            assert (a - b).abs().max() > 1e-3, name
    # dq, dk enter only through the weight tangent: their gradient is zero (None).
    assert new[3].abs().max() == 0 and new[4].abs().max() == 0


def _explicit_exact(q, k, v, dq, dk, dv):
    with fwAD.dual_level():
        out = attention.scaled_dot_product_attention(
            fwAD.make_dual(q, dq),
            fwAD.make_dual(k, dk),
            fwAD.make_dual(v, dv),
            exact_tangent_gradient=True,
        )
        return fwAD.unpack_dual(out)


def test_exact_tangent_gradient_matches_the_full_jvp_gradient():
    inputs = _qkv(torch.float64, seed=1)
    generator = torch.Generator().manual_seed(2)
    cotangents = [torch.randn(inputs[0].shape, generator=generator, dtype=torch.float64)
                  for _ in range(2)]
    primal, tangent = _explicit_exact(*inputs)
    ref_primal, ref_tangent = torch.func.jvp(_plain_attention, tuple(inputs[:3]), tuple(inputs[3:]))
    torch.testing.assert_close(primal, ref_primal, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(tangent, ref_tangent, rtol=1e-10, atol=1e-12)
    exact = _grads(_explicit_exact, inputs, cotangents)
    reference = _grads(
        lambda q, k, v, dq, dk, dv: torch.func.jvp(
            _plain_attention, (q, k, v), (dq, dk, dv)
        ),
        inputs,
        cotangents,
    )
    for name, a, b in zip(["q", "k", "v", "dq", "dk", "dv"], exact, reference):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12, msg=name)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_attention_uses_the_exact_gradient_only_with_qk_norm(qk_norm, monkeypatch):
    seen = []
    original = attention.scaled_dot_product_attention

    def spy(*args, **kwargs):
        seen.append(kwargs.get("exact_tangent_gradient"))
        return original(*args, **kwargs)

    monkeypatch.setattr(attention, "scaled_dot_product_attention", spy)
    x = torch.randn(2, 6, 8)
    for module in (
        attention.AttentionWithRotaryPositionalEmbeddings(8, heads=2, qk_norm=qk_norm),
        attention.Attention(8, heads=2, qk_norm=qk_norm),
    ):
        module(x)
    assert seen == [qk_norm, qk_norm]


# --------------------------------------------------------------------- rotary


def _legacy_forward(module, x):
    """Attention forward as at HEAD before the rotary fix (rotation along heads)."""
    q = split_heads(module.W_q(x), module.heads, module.dim_head)  # (B, H, L, D)
    k = split_heads(module.W_k(x), module.heads, module.dim_head)
    v = split_heads(module.W_v(x), module.heads, module.dim_head)
    q = module.rotary_emb_q(q)
    k = module.rotary_emb_k(k)
    dropout_p = module.dropout_rate if module.training else 0.0
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
    out = combine_heads(out, module.heads, module.dim_head)
    return module.to_out(out)


def _rotary(legacy=False, dim=8, heads=2, seed=0):
    torch.manual_seed(seed)
    return AttentionWithRotaryPositionalEmbeddings(
        dim, heads=heads, legacy_rotary=legacy
    ).eval()


def _scores(module, x, monkeypatch):
    """q.k logits the module actually passes to attention, shape (B, H, L, L)."""
    captured = {}
    original = attention.scaled_dot_product_attention

    def capture(q, k, v, dropout_p=0.0, **kwargs):
        captured["scores"] = q @ k.transpose(-2, -1)
        return original(q, k, v, dropout_p=dropout_p)

    monkeypatch.setattr(attention, "scaled_dot_product_attention", capture)
    with torch.no_grad():
        module(x)
    monkeypatch.setattr(attention, "scaled_dot_product_attention", original)
    return captured["scores"]


def test_rotary_scores_depend_only_on_relative_position(monkeypatch):
    module = _rotary()
    token = torch.randn(1, 1, 8)
    scores = _scores(module, token.expand(1, 6, 8).contiguous(), monkeypatch)
    for shift in range(1, 6):  # Toeplitz: constant along every diagonal
        torch.testing.assert_close(
            scores[..., shift:, shift:], scores[..., :-shift, :-shift]
        )
    assert scores.std() > 1e-3  # identical content, yet position-dependent

    # Content a at m, b at n: the score depends on m - n, not on m and n.
    a, b = torch.randn(2, 8)
    def placed(m, n):
        x = torch.randn(1, 10, 8)
        x[0, m], x[0, n] = a, b
        return _scores(module, x, monkeypatch)[..., m, n]
    torch.testing.assert_close(placed(1, 3), placed(5, 7))
    assert (placed(1, 3) - placed(1, 6)).abs().max() > 1e-4


def test_rotary_attention_is_not_permutation_equivariant():
    x = torch.randn(2, 7, 8)
    perm = torch.randperm(7)
    with torch.no_grad():
        fixed = _rotary()
        assert (fixed(x[:, perm]) - fixed(x)[:, perm]).abs().max() > 1e-3
        legacy = _rotary(legacy=True)
        torch.testing.assert_close(legacy(x[:, perm]), legacy(x)[:, perm])


def test_legacy_rotary_reproduces_the_old_implementation():
    module = _rotary(legacy=True)
    x = torch.randn(3, 9, 8)
    with torch.no_grad():
        torch.testing.assert_close(module(x), _legacy_forward(module, x), rtol=0, atol=0)
        assert (_rotary(legacy=False)(x) - module(x)).abs().max() > 1e-4


def test_rotary_has_no_state_and_old_checkpoints_load():
    legacy = _rotary(legacy=True, seed=1)
    fixed = _rotary(legacy=False, seed=2)
    assert set(legacy.state_dict()) == set(fixed.state_dict())
    assert not any("rotary" in key for key in fixed.state_dict())
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randn(1, 4, 8)
    with torch.no_grad():
        torch.testing.assert_close(fixed.W_q(x), legacy.W_q(x))

    torch.manual_seed(0)
    old = SpatialAttention(channels=1, pos_embedding_type="rotary_legacy",
                           heads=2, patch_size=2)
    new = SpatialAttention(channels=1, pos_embedding_type="rotary", heads=2,
                           patch_size=2)
    assert old.attention.legacy_rotary and not new.attention.legacy_rotary
    new.load_state_dict(old.state_dict(), strict=True)


# ------------------------------------------------------------- LSD + dropout


class DropoutMapNetwork(nn.Module):
    """Small time-dependent map with attention and dropout."""

    def __init__(self, p=0.5):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.attention = SpatialAttention(
            channels=1, heads=2, patch_size=2, dropout_rate=p
        )
        self.dropout = nn.Dropout(p)
        self.time = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, cond, **kwargs):
        s, t = cond[:, :1].reshape(-1, 1, 1, 1), cond[:, 1:].reshape(-1, 1, 1, 1)
        h = self.conv(x) * (1 + self.time * t + (t - s) ** 2)
        return self.dropout(self.attention(h) * torch.sin(t))


def _dropout_model():
    torch.manual_seed(5)
    return FlowMapModel(LinearDeterministicInterpolation(), DropoutMapNetwork())


def _lsd_inputs():
    generator = torch.Generator().manual_seed(6)
    x = torch.randn(4, 1, 4, 4, generator=generator)
    s = torch.tensor([[0.0], [0.1], [0.3], [0.5]])
    t = torch.tensor([[1.0], [0.6], [0.4], [0.9]])
    return x, s, t


@pytest.mark.parametrize("training", [True, False])
def test_lsd_loss_restores_the_training_flag(training):
    model = _dropout_model().train(training)
    x, s, t = _lsd_inputs()
    lagrangian_self_distillation_loss(model, model, nn.MSELoss(), x, s, None, t, None, {})
    assert all(m.training == training for m in model.modules())


def test_lsd_loss_restores_the_training_flag_after_an_error():
    model = _dropout_model().train()
    x, s, t = _lsd_inputs()
    with pytest.raises(Exception):
        lagrangian_self_distillation_loss(
            model, model, nn.MSELoss(), x[:, :, :3], s, None, t, None, {}
        )
    assert all(m.training for m in model.modules())


def test_lsd_loss_is_deterministic_with_dropout_in_the_network():
    model = _dropout_model().train()
    ema = _dropout_model().eval().requires_grad_(False)
    x, s, t = _lsd_inputs()

    def loss_and_grads():
        model.zero_grad()
        torch.manual_seed(torch.randint(1000, ()).item() + 1)  # different RNG state
        loss = lagrangian_self_distillation_loss(
            model, ema, nn.MSELoss(), x, s, None, t, None, {}
        )
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters()]

    first, second = loss_and_grads(), loss_and_grads()
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    for a, b in zip(first[1], second[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # It equals the loss of the same model in eval mode.
    eval_loss = lagrangian_self_distillation_loss(
        model.eval(), ema, nn.MSELoss(), x, s, None, t, None, {}
    )
    torch.testing.assert_close(first[0], eval_loss.detach(), rtol=0, atol=0)
    # And dropout really is active in train mode outside the LSD pass.
    model.train()
    with torch.no_grad():
        a = model.average_velocity(x, s, t)
        b = model.average_velocity(x, s, t)
    assert (a - b).abs().max() > 1e-3


def _batch():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(3, 1, 4, 4, generator=generator)
    return {
        "base": base,
        "target": base + torch.randn(base.shape, generator=generator),
        "field_history": base.unsqueeze(-1),
    }


def _map_trainer(tmp_path, model, gradient_clipper=None, optimizer=None, **kwargs):
    return MapTrainer(
        num_epochs=1,
        model=model,
        train_dataloader=[_batch()],
        val_dataloader=[_batch()],
        optimizer=optimizer or torch.optim.AdamW(model.parameters(), lr=1e-2),
        gradient_clipper=gradient_clipper,
        early_stopping=EarlyStopping(patience=3),
        device="cpu",
        checkpoint_dir=str(tmp_path),
        off_diagonal="lagrangian_self_distillation",
        **kwargs,
    )


def test_diagonal_training_loss_still_uses_dropout(tmp_path):
    torch.manual_seed(0)
    instance = _map_trainer(tmp_path, _dropout_model(), diagonal_epochs=1)
    batch = instance._prepare_batch(_batch())
    instance._compute_loss(batch)
    first = instance.components["diagonal_loss"]
    instance._compute_loss(batch)
    assert instance.components["diagonal_loss"] != first

    # With the off-diagonal LSD term on, the model stays in training mode.
    instance.diagonal_epochs = 0
    instance._compute_loss(batch)
    assert instance.components["off_diagonal_loss"] > 0
    assert all(m.training for m in instance.model.modules())


# ------------------------------------------------------------------ clipping


def test_global_norm_clipper_clips_and_returns_the_preclip_norm():
    a = nn.Parameter(torch.zeros(2))
    b = nn.Parameter(torch.zeros(1))
    a.grad, b.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])
    norm = GlobalNormClipper(max_norm=1.0).clip_grads([a, b])
    torch.testing.assert_close(norm, torch.tensor(5.0))
    clipped = torch.cat([a.grad, b.grad]).norm()
    torch.testing.assert_close(clipped, torch.tensor(1.0), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a.grad, torch.tensor([0.6, 0.0]), rtol=1e-5, atol=1e-6)
    # Below the threshold: untouched.
    a.grad, b.grad = torch.tensor([0.3, 0.0]), torch.tensor([0.4])
    norm = GlobalNormClipper(max_norm=1.0).clip_grads([a, b])
    torch.testing.assert_close(norm, torch.tensor(0.5))
    torch.testing.assert_close(a.grad, torch.tensor([0.3, 0.0]), rtol=0, atol=0)


def test_ema_clipper_returns_the_preclip_norm():
    a = nn.Parameter(torch.zeros(2))
    a.grad = torch.tensor([3.0, 4.0])
    torch.testing.assert_close(EmaGradientClipper().clip_grads([a]), torch.tensor(5.0))


class ConstantMapNetwork(nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value))

    def forward(self, x, cond, **kwargs):
        return self.value.expand_as(x) * (1 + cond[:, 1:].reshape(-1, 1, 1, 1))


@pytest.mark.parametrize("clipper", [GlobalNormClipper, EmaGradientClipper])
@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_nonfinite_gradient_skips_the_update(tmp_path, clipper, bad):
    torch.manual_seed(0)
    model = FlowMapModel(LinearDeterministicInterpolation(), ConstantMapNetwork())
    instance = _map_trainer(tmp_path, model, gradient_clipper=clipper())
    parameter = instance.model.map_model.value
    before = parameter.detach().clone()
    ema_before = instance.ema_model.map_model.value.detach().clone()
    handle = parameter.register_hook(lambda g: torch.full_like(g, bad))
    loss = instance._train_step(instance._prepare_batch(_batch()))
    handle.remove()
    assert torch.isfinite(loss)
    assert instance.skipped_steps == 1 and instance.global_step == 0
    assert parameter.item() == before.item()
    assert instance.ema_model.map_model.value.item() == ema_before.item()
    assert parameter.grad is None
    assert not instance.optimizer.state  # AdamW moments untouched
    # The next healthy step updates normally.
    instance._train_step(instance._prepare_batch(_batch()))
    assert instance.skipped_steps == 1 and instance.global_step == 1
    assert parameter.item() != before.item()
    assert instance.ema_model.map_model.value.item() != ema_before.item()


# -------------------------------------------------------------------- config


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="stochastic_navier_stokes", overrides=overrides)


@pytest.mark.parametrize("models", ["flow_matching", "ito_map", "flow_map"])
def test_default_gradient_clipper_is_global_norm(models):
    from hydra.utils import instantiate

    cfg = _compose([f"models={models}"])
    clipper = instantiate(cfg.trainer.gradient_clipper)
    assert isinstance(clipper, GlobalNormClipper) and clipper.max_norm == 1.0


@pytest.mark.parametrize("models", ["flow_matching", "ito_map"])
def test_ema_gradient_clipper_override(models):
    from hydra.utils import instantiate

    cfg = _compose([f"models={models}", "gradient_clipping=ema"])
    clipper = instantiate(cfg.trainer.gradient_clipper)
    assert isinstance(clipper, EmaGradientClipper)
    assert "max_norm" not in cfg.trainer.gradient_clipper
