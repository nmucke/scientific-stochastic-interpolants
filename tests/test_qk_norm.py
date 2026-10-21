"""Optional QK-norm (per-head RMSNorm of queries and keys) in the attention."""

import math
from pathlib import Path

import pytest
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F

import scisi.architectures.attention as attention
from scisi.architectures.attention import (
    Attention,
    AttentionWithRotaryPositionalEmbeddings,
    QKRMSNorm,
    SpatialAttention,
    SpatialTransformerBlock,
    combine_heads,
    split_heads,
)

torch.set_num_threads(4)


def _reference_rotary_forward(module, x):
    """Rotary attention forward exactly as before QK-norm was added."""
    b, l, _ = x.size()
    q = module.W_q(x).view(b, l, module.heads, module.dim_head)
    k = module.W_k(x).view(b, l, module.heads, module.dim_head)
    v = split_heads(module.W_v(x), module.heads, module.dim_head)
    if module.legacy_rotary:
        q = module.rotary_emb_q(q.transpose(1, 2))
        k = module.rotary_emb_k(k.transpose(1, 2))
    else:
        q = module.rotary_emb_q(q).transpose(1, 2)
        k = module.rotary_emb_k(k).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return module.to_out(combine_heads(out, module.heads, module.dim_head))


def _reference_additive_forward(module, x):
    """Additive-variant attention forward exactly as before QK-norm was added."""
    q = split_heads(module.W_q(x), module.heads, module.dim_head)
    k = split_heads(module.W_k(x), module.heads, module.dim_head)
    v = split_heads(module.W_v(x), module.heads, module.dim_head)
    out = F.scaled_dot_product_attention(q, k, v)
    return module.to_out(combine_heads(out, module.heads, module.dim_head))


def _rotary(qk_norm=False, legacy=False, dim=16, heads=2, seed=0, **kwargs):
    torch.manual_seed(seed)
    return AttentionWithRotaryPositionalEmbeddings(
        dim, heads=heads, legacy_rotary=legacy, qk_norm=qk_norm, **kwargs
    ).eval()


def _captured_qk(module, x, monkeypatch):
    """q and k exactly as the module passes them to the attention kernel."""
    captured = {}
    original = attention.scaled_dot_product_attention

    def capture(q, k, v, dropout_p=0.0, **kwargs):
        captured["q"], captured["k"] = q, k
        return original(q, k, v, dropout_p=dropout_p)

    monkeypatch.setattr(attention, "scaled_dot_product_attention", capture)
    with torch.no_grad():
        module(x)
    monkeypatch.setattr(attention, "scaled_dot_product_attention", original)
    return captured["q"], captured["k"]


def _logits(q, k):
    return q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])


# ------------------------------------------------------------- flag off: no-op


@pytest.mark.parametrize("legacy", [False, True])
def test_off_is_bit_identical_to_the_old_rotary_attention(legacy):
    default = _rotary(legacy=legacy)  # the old call, without the new kwarg
    explicit = _rotary(qk_norm=False, legacy=legacy)
    assert not hasattr(default, "q_norm") and not hasattr(default, "k_norm")
    assert list(default.state_dict()) == list(explicit.state_dict()) == [
        f"{name}.{p}" for name in ("W_q", "W_k", "W_v", "to_out")
        for p in ("weight", "bias")
    ]
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        reference = _reference_rotary_forward(default, x)
        torch.testing.assert_close(default(x), reference, rtol=0, atol=0)
        torch.testing.assert_close(explicit(x), reference, rtol=0, atol=0)


def test_off_is_bit_identical_to_the_old_additive_attention():
    torch.manual_seed(0)
    default = Attention(16, heads=2).eval()
    torch.manual_seed(0)
    explicit = Attention(16, heads=2, qk_norm=False).eval()
    assert set(default.state_dict()) == set(explicit.state_dict())
    assert not any("norm" in key for key in default.state_dict())
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        reference = _reference_additive_forward(default, x)
        torch.testing.assert_close(default(x), reference, rtol=0, atol=0)
        torch.testing.assert_close(explicit(x), reference, rtol=0, atol=0)


@pytest.mark.parametrize("block", [SpatialAttention, SpatialTransformerBlock])
@pytest.mark.parametrize("pos", ["rotary", "rotary_legacy", "additive"])
def test_spatial_blocks_thread_the_flag(block, pos):
    kwargs = dict(channels=2, pos_embedding_type=pos, embedding_dim=16, heads=2,
                  patch_size=2)
    torch.manual_seed(0)
    old = block(**kwargs)
    torch.manual_seed(0)
    off = block(**kwargs, qk_norm=False)
    torch.manual_seed(0)
    on = block(**kwargs, qk_norm=True, qk_norm_eps=1e-5)
    assert list(old.state_dict()) == list(off.state_dict())
    for key, value in old.state_dict().items():
        assert torch.equal(value, off.state_dict()[key])
    norms = [m for m in on.modules() if isinstance(m, QKRMSNorm)]
    assert len(norms) == 2 and all(m.eps == 1e-5 for m in norms)
    assert not any(isinstance(m, QKRMSNorm) for m in off.modules())
    x = torch.randn(1, 2, 8, 8)
    old.eval(), off.eval()
    with torch.no_grad():
        torch.testing.assert_close(off(x), old(x), rtol=0, atol=0)
        assert torch.isfinite(on.eval()(x)).all()


# ----------------------------------------------------------------- flag on


def test_qk_rms_norm_unit_rms_and_dtype():
    norm = QKRMSNorm(8)
    x = 1e3 * torch.randn(3, 5, 2, 8, dtype=torch.float64)
    y = norm(x)
    assert y.dtype == torch.float64
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms))
    torch.testing.assert_close(norm(x.float()), y.float())


def test_normalized_q_and_k_have_unit_rms_per_head(monkeypatch):
    # Without the rotary embedding the captured q and k are the normed ones.
    torch.manual_seed(0)
    module = Attention(16, heads=2, qk_norm=True).eval()
    q, k = _captured_qk(module, 50 * torch.randn(2, 6, 16), monkeypatch)
    for t in (q, k):
        rms = t.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms))
    # RoPE is a rotation, so the per-head RMS survives it.
    q, k = _captured_qk(_rotary(qk_norm=True), 50 * torch.randn(2, 6, 16), monkeypatch)
    for t in (q, k):
        rms = t.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)


def test_logits_bounded_by_sqrt_head_dim_for_huge_inputs(monkeypatch):
    x = 1e4 * torch.randn(2, 8, 16)
    bound = math.sqrt(16 // 2)
    q, k = _captured_qk(_rotary(qk_norm=True), x, monkeypatch)
    assert _logits(q, k).abs().max() <= bound * (1 + 1e-5)
    q, k = _captured_qk(_rotary(qk_norm=False), x, monkeypatch)
    assert _logits(q, k).abs().max() > 1e6  # blows up without the norm
    # The bound scales with the gains: ||g_q||_inf ||g_k||_inf sqrt(D).
    module = _rotary(qk_norm=True)
    with torch.no_grad():
        module.q_norm.weight.uniform_(-2.0, 2.0)
        module.k_norm.weight.uniform_(-3.0, 3.0)
    q, k = _captured_qk(module, x, monkeypatch)
    gain = module.q_norm.weight.abs().max() * module.k_norm.weight.abs().max()
    assert _logits(q, k).abs().max() <= gain * bound * (1 + 1e-5)


def test_rotary_relative_position_property_holds_with_qk_norm(monkeypatch):
    module = _rotary(qk_norm=True)
    token = torch.randn(1, 1, 16)
    q, k = _captured_qk(module, token.expand(1, 6, 16).contiguous(), monkeypatch)
    scores = q @ k.transpose(-2, -1)
    for shift in range(1, 6):  # Toeplitz: constant along every diagonal
        torch.testing.assert_close(
            scores[..., shift:, shift:], scores[..., :-shift, :-shift]
        )
    assert scores.std() > 1e-3

    a, b = torch.randn(2, 16)

    def placed(m, n):
        x = torch.randn(1, 10, 16)
        x[0, m], x[0, n] = a, b
        q, k = _captured_qk(module, x, monkeypatch)
        return (q @ k.transpose(-2, -1))[..., m, n]

    torch.testing.assert_close(placed(1, 3), placed(5, 7))
    assert (placed(1, 3) - placed(1, 6)).abs().max() > 1e-4


def test_bf16_autocast_norm_computed_in_fp32():
    norm = QKRMSNorm(8)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = (300 * torch.randn(4, 3, 2, 8)).to(torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = norm(x)
    assert y.dtype == torch.bfloat16
    expected = norm(x.float()).to(torch.bfloat16)  # fp32 computation, rounded once
    torch.testing.assert_close(y, expected, rtol=0, atol=0)

    module = _rotary(qk_norm=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(1e4 * torch.randn(2, 6, 16))
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("variant", ["rotary", "additive"])
def test_forward_ad_then_backward_through_explicit_attention(variant):
    if variant == "rotary":
        module = _rotary(qk_norm=True).train()
    else:
        torch.manual_seed(0)
        module = Attention(16, heads=2, qk_norm=True).train()
    with torch.no_grad():  # non-trivial gains
        module.q_norm.weight.uniform_(0.5, 1.5)
        module.k_norm.weight.uniform_(0.5, 1.5)
    x = 100 * torch.randn(2, 6, 16)
    dx = torch.randn(2, 6, 16)
    with fwAD.dual_level():
        out = module(fwAD.make_dual(x, dx))
        primal, tangent = fwAD.unpack_dual(out)
    assert tangent is not None  # explicit attention path was taken
    (primal.pow(2).mean() + tangent.pow(2).mean()).backward()
    for name, p in module.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert module.q_norm.weight.grad.abs().sum() > 0
    assert module.k_norm.weight.grad.abs().sum() > 0

    # The tangent matches a finite difference of the primal.
    module.eval()
    eps = 1e-3
    x64, dx64 = x.double(), dx.double()
    module = module.double()
    with fwAD.dual_level():
        tangent = fwAD.unpack_dual(module(fwAD.make_dual(x64, dx64))).tangent
    with torch.no_grad():
        fd = (module(x64 + eps * dx64) - module(x64 - eps * dx64)) / (2 * eps)
    # The rotary cache is fp32, so compare relative to the tangent's scale.
    assert (tangent - fd).abs().max() <= 1e-2 * fd.abs().max()


# ------------------------------------------------------------ config / loading


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="stochastic_navier_stokes", overrides=overrides)


def test_hydra_flag_and_old_checkpoint_loading():
    from hydra.utils import instantiate

    torch.manual_seed(0)
    # Old checkpoints were trained without the norm.
    old = instantiate(
        _compose(["models=ito_map", "backbone.attention.qk_norm=false"]).model
    )
    assert not any(isinstance(m, QKRMSNorm) for m in old.modules())

    cfg = _compose(["models=ito_map"])  # qk_norm is on by default
    assert cfg.backbone.attention.qk_norm is True
    torch.manual_seed(0)
    new = instantiate(cfg.model)
    attn = [m for m in new.modules()
            if isinstance(m, AttentionWithRotaryPositionalEmbeddings)]
    assert attn and all(hasattr(m, "q_norm") and hasattr(m, "k_norm") for m in attn)

    result = new.load_state_dict(old.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert result.missing_keys and all(
        key.endswith((".q_norm.weight", ".k_norm.weight"))
        for key in result.missing_keys
    )
    assert len(result.missing_keys) == 2 * len(attn)
    for key, value in old.state_dict().items():
        assert torch.equal(new.state_dict()[key], value)
    for m in attn:
        assert torch.equal(m.q_norm.weight, torch.ones_like(m.q_norm.weight))
