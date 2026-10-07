"""Video metric contracts, without downloaded weights or GPU work."""

import numpy as np
import pytest
import torch

from scisi.metrics.video import I3DFeatures, frechet_video_distance, video_frame_metrics


def test_fvd_translation_and_different_sample_counts():
    real = np.array([[-1.0, 0.0], [1.0, 0.0]])
    assert frechet_video_distance(real, real + [3, 4]) == pytest.approx(25)
    # Both sets have covariance diag(2,0), despite different sample counts.
    fake = np.array([[-np.sqrt(2), 0], [0, 0], [np.sqrt(2), 0]])
    assert frechet_video_distance(real, fake) == pytest.approx(0, abs=1e-12)


def test_fvd_self_and_singular_covariance():
    real = np.random.default_rng(7).normal(size=(7, 50))
    assert frechet_video_distance(real, real) == pytest.approx(0, abs=1e-12)
    # Rank-one covariances in perpendicular directions have zero cross term.
    a = np.array([[-1, 0, 0], [1, 0, 0]])
    b = np.array([[0, -2, 0], [0, 2, 0]])
    assert frechet_video_distance(a, b) == pytest.approx(10)
    assert frechet_video_distance(b, a) == pytest.approx(10)


def test_fvd_matches_independent_psd_covariance_formula():
    rng = np.random.default_rng(20)
    a, b = rng.normal(size=(8, 5)), rng.normal(size=(13, 5)) * 1.5 + 0.2
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    values, vectors = np.linalg.eigh(ca)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    cross = np.linalg.eigvalsh(root @ cb @ root)
    expected = np.square(a.mean(0) - b.mean(0)).sum() + np.trace(ca + cb) - 2 * np.sqrt(np.maximum(cross, 0)).sum()
    assert frechet_video_distance(a, b) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("a,b", [
    (np.zeros((1, 3)), np.zeros((2, 3))),
    (np.zeros((2, 3)), np.zeros((2, 4))),
    (np.zeros((2, 0)), np.zeros((2, 0))),
    (np.zeros(3), np.zeros((2, 3))),
    (np.full((2, 3), np.nan), np.zeros((2, 3))),
])
def test_fvd_rejects_invalid_features(a, b):
    with pytest.raises(ValueError):
        frechet_video_distance(a, b)


@pytest.mark.parametrize("channels", [1, 3])
def test_frame_identity_and_psnr_cap(channels):
    target = torch.rand(channels, 16, 16) * 2 - 1
    result = video_frame_metrics(target[None].repeat(3, 1, 1, 1), target)
    for metric in ("crps", "energy_score", "rmse", "rms_spread", "out_of_range_fraction"):
        assert result[metric] == pytest.approx(0, abs=1e-14)
    assert result["spread_skill_ratio"] is None
    assert result["coverage_90"] == 1
    assert result["psnr_sample_mean"] == 100
    assert result["psnr_ensemble_mean"] == 100
    assert result["ssim_sample_mean"] == pytest.approx(1)
    assert result["ssim_ensemble_mean"] == pytest.approx(1)
    assert result["lpips_sample_mean"] is None


def test_unclipped_proper_scores_and_clipped_image_scores():
    target = -torch.ones(1, 16, 16)
    ensemble = torch.stack((target, torch.full_like(target, 3)))
    result = video_frame_metrics(ensemble, target)
    assert result["crps"] == pytest.approx(0)
    assert result["energy_score"] == pytest.approx(0)
    assert result["rmse"] == pytest.approx(1)
    assert result["rms_spread"] == pytest.approx(np.sqrt(2))
    assert result["spread_skill_ratio"] == pytest.approx(np.sqrt(3))
    assert result["out_of_range_fraction"] == 0.5
    assert result["coverage_90"] == 0
    assert result["psnr_sample_mean"] == 50
    assert result["psnr_oracle_best"] == 100
    assert result["psnr_ensemble_mean"] == 0


def test_single_member_and_fair_pair_normalization():
    target = torch.zeros(1, 16, 16)
    ensemble = torch.ones(1, 1, 16, 16)
    result = video_frame_metrics(ensemble, target)
    assert result["crps"] == pytest.approx(0.5)
    assert result["energy_score"] == pytest.approx(0.5)
    assert result["rms_spread"] == 0
    # Two members straddling the truth attain zero fair score here; biased
    # E**2 pair normalization would instead give a positive value.
    ensemble = torch.stack((-torch.ones_like(target), torch.ones_like(target)))
    result = video_frame_metrics(ensemble, target)
    assert result["crps"] == pytest.approx(0)
    assert result["energy_score"] == pytest.approx(0)


class FakeLPIPS(torch.nn.Module):
    def forward(self, x, y, normalize=False):
        assert normalize
        assert x.shape[1] == y.shape[1] == 3
        assert x.dtype == y.dtype == torch.float32
        assert x.min() >= 0 and x.max() <= 1
        return (x - y).square().mean((1, 2, 3), keepdim=True)


def test_lpips_grayscale_range_and_oracle_contract():
    target = -torch.ones(1, 16, 16)
    ensemble = torch.stack((target, -target))
    result = video_frame_metrics(ensemble, target, FakeLPIPS())
    assert result["lpips_sample_mean"] == 0.5
    assert result["lpips_oracle_best"] == 0


@pytest.mark.parametrize("ensemble,target", [
    (torch.zeros(1, 16, 16), torch.zeros(1, 16, 16)),
    (torch.zeros(0, 1, 16, 16), torch.zeros(1, 16, 16)),
    (torch.zeros(2, 3, 16, 16), torch.zeros(1, 16, 16)),
    (torch.zeros(2, 1, 8, 8), torch.zeros(1, 8, 8)),
    (torch.full((2, 1, 16, 16), float("inf")), torch.zeros(1, 16, 16)),
])
def test_frame_validation(ensemble, target):
    with pytest.raises(ValueError):
        video_frame_metrics(ensemble, target)


class FakeI3D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, x, rescale=False, resize=False, return_features=False):
        assert not self.training and not self.scale.requires_grad
        assert not rescale and resize and return_features
        assert x.shape == (2, 3, 16, 12, 12)
        assert x.min() == -1 and x.max() == 1
        assert torch.equal(x[:, 0], x[:, 1])
        return x.mean((1, 2, 3, 4))[:, None].expand(-1, 400)


def test_i3d_layout_grayscale_clipping_and_freezing(monkeypatch):
    monkeypatch.setattr(torch.jit, "load", lambda path, map_location: FakeI3D())
    model = I3DFeatures("unused.pt")
    x = torch.empty(2, 16, 1, 12, 12)
    x[0], x[1] = -2, 2
    result = model(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 400)
    np.testing.assert_allclose(result[:, 0], [-1, 1])
    with pytest.raises(ValueError):
        model(torch.zeros(2, 16, 2, 12, 12))
    with pytest.raises(ValueError):
        model(torch.full((2, 16, 1, 12, 12), float("nan")))


class ScriptableDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor, rescale: bool = False,
                resize: bool = False, return_features: bool = False):
        return x.mean((1, 2, 3, 4))[:, None].expand(-1, 400) * self.scale


def test_i3d_accepts_real_script_module_without_download(tmp_path):
    path = tmp_path / "detector.pt"
    torch.jit.script(ScriptableDetector()).save(str(path))
    detector = I3DFeatures(path, batch_size=2)
    assert all(not parameter.requires_grad for parameter in detector.model.parameters())
    videos = torch.arange(5).reshape(5, 1, 1, 1, 1).expand(5, 16, 3, 12, 12) / 4
    result = detector(videos)
    assert result.shape == (5, 400)
    np.testing.assert_allclose(result[:, 0], np.arange(5) / 4)
    with pytest.raises(ValueError):
        I3DFeatures(path, batch_size=0)
