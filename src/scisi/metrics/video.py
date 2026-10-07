"""Pixel-space video metrics with explicit range and ensemble conventions.

FVD uses the Google I3D conversion distributed by StyleGAN-V and used by RIVER.
See ``research/video_forecasting/notes/i3d_provenance.json`` for the detector.
Optional image dependencies are imported only when their metrics are requested.
"""

from pathlib import Path

import numpy as np
import torch

from scisi.metrics.accuracy import ensemble_mean_rmse
from scisi.metrics.calibration import crps


@torch.inference_mode()
def video_frame_metrics(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    lpips_model: torch.nn.Module | None = None,
) -> dict[str, float | None]:
    """Score normalized ``[E,C,H,W]`` forecasts against one ``[C,H,W]`` frame.

    Convert [-1,1] to unit pixels. Proper scores and calibration retain values
    outside [0,1]; image metrics clip their own copies. Fair CRPS and energy
    scores use E(E-1) pair normalization (E=1 gives deterministic MAE/L2).
    Energy uses Euclidean distances divided by sqrt(C*H*W). RMS spread uses
    unbiased ensemble variance for E>1; spread/skill additionally multiplies
    spread by sqrt((E+1)/E). A zero RMSE gives an undefined ratio (None).

    PSNR uses max(MSE, 1e-10), capping a perfect prediction at 100 dB.
    SSIM uses an 11-pixel Gaussian window, sigma=1.5, population covariance,
    and data_range=1. Image sizes must be at least 11 in each dimension.
    Ensemble-mean image metrics clip the raw ensemble mean. Oracle metrics
    choose a different best member for each metric and must not be interpreted
    as proper scores. LPIPS receives clipped RGB unit pixels, normalize=True.
    """
    from skimage.metrics import structural_similarity

    ensemble = torch.as_tensor(ensemble).detach().to(dtype=torch.float64)
    target = torch.as_tensor(target, device=ensemble.device).detach().to(torch.float64)
    if ensemble.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected ensemble [E,C,H,W] and target [C,H,W].")
    if ensemble.shape[0] < 1 or ensemble.shape[1:] != target.shape:
        raise ValueError("Ensemble must be nonempty and match the target shape.")
    if target.shape[0] not in (1, 3) or min(target.shape[-2:]) < 11:
        raise ValueError("Expected one or three channels and spatial dimensions >= 11.")
    if not torch.isfinite(ensemble).all() or not torch.isfinite(target).all():
        raise ValueError("Video frames must contain only finite values.")

    samples = (ensemble + 1.0) / 2.0
    truth = (target + 1.0) / 2.0
    count = len(samples)
    flat = samples.flatten(1)
    dimension_scale = truth.numel() ** 0.5
    energy = (flat - truth.flatten()).norm(dim=1).mean() / dimension_scale
    if count > 1:
        # pdist contains each unordered pair once: this is half of the
        # off-diagonal ordered-pair sum in the fair energy score.
        energy -= torch.pdist(flat).sum() / (count * (count - 1) * dimension_scale)
        spread = samples.var(dim=0, correction=1).mean().sqrt()
    else:
        spread = samples.new_zeros(())
    rmse = ensemble_mean_rmse(samples, truth)
    lo, hi = torch.quantile(samples, samples.new_tensor([0.05, 0.95]), dim=0)
    result = {
        "crps": float(crps(samples, truth)),
        "energy_score": float(energy),
        "rmse": float(rmse),
        "rms_spread": float(spread),
        "spread_skill_ratio": (
            float(spread * ((count + 1) / count) ** 0.5 / rmse)
            if rmse > 0 else None
        ),
        "coverage_90": float(((truth >= lo) & (truth <= hi)).double().mean()),
        "out_of_range_fraction": float(((samples < 0) | (samples > 1)).double().mean()),
    }

    clipped = samples.clamp(0, 1)
    clipped_truth = truth.clamp(0, 1)
    clipped_mean = samples.mean(dim=0).clamp(0, 1)
    mse = (clipped - clipped_truth).square().flatten(1).mean(dim=1)
    psnr = -10.0 * mse.clamp_min(1e-10).log10()
    mean_mse = (clipped_mean - clipped_truth).square().mean()
    result.update({
        "psnr_sample_mean": float(psnr.mean()),
        "psnr_oracle_best": float(psnr.max()),
        "psnr_ensemble_mean": float(-10.0 * mean_mse.clamp_min(1e-10).log10()),
    })

    reference = clipped_truth.movedim(0, -1).cpu().numpy()

    def ssim(image: torch.Tensor) -> float:
        return float(structural_similarity(
            reference, image.movedim(0, -1).cpu().numpy(), channel_axis=-1,
            data_range=1.0, gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False, win_size=11,
        ))

    sample_ssim = [ssim(frame) for frame in clipped]
    result.update({
        "ssim_sample_mean": float(np.mean(sample_ssim)),
        "ssim_oracle_best": float(max(sample_ssim)),
        "ssim_ensemble_mean": ssim(clipped_mean),
        "lpips_sample_mean": None,
        "lpips_oracle_best": None,
    })
    if lpips_model is not None:
        # Respect the caller's model device, including parameter-free test
        # doubles. LPIPS expects float32 even though scalar scores use float64.
        state = next(iter(lpips_model.parameters()), None)
        if state is None:
            state = next(iter(lpips_model.buffers()), None)
        device = state.device if state is not None else samples.device
        x = clipped.to(device=device, dtype=torch.float32)
        y = clipped_truth.unsqueeze(0).expand_as(clipped).to(device=device, dtype=torch.float32)
        if x.shape[1] == 1:
            x, y = x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)
        values = lpips_model(x, y, normalize=True).reshape(count, -1).mean(dim=1)
        if not torch.isfinite(values).all():
            raise ValueError("LPIPS returned nonfinite values.")
        result["lpips_sample_mean"] = float(values.mean())
        result["lpips_oracle_best"] = float(values.min())
    return result


class I3DFeatures:
    """Frozen Google-converted I3D logits for normalized [N,T,C,H,W] videos.

    Inputs are clipped to [-1,1], grayscale is repeated into RGB, and the
    detector performs its own spatial resize. No temporal resampling occurs.
    Returned NumPy features have shape [N,400]. Downloading is the caller's
    responsibility; missing or incompatible weights raise an error.
    """

    def __init__(self, path: str | Path, device: str | torch.device = "cpu", batch_size: int = 4):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("I3D batch_size must be a positive integer.")
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        # ScriptModule does not implement Module.requires_grad_().
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, videos: torch.Tensor) -> np.ndarray:
        videos = torch.as_tensor(videos)
        if videos.ndim != 5 or videos.shape[0] < 1 or videos.shape[2] not in (1, 3):
            raise ValueError("Expected nonempty videos [N,T,C,H,W] with C=1 or 3.")
        if min(videos.shape[1], *videos.shape[-2:]) < 1:
            raise ValueError("Video time and spatial dimensions must be nonempty.")
        if not torch.isfinite(videos).all():
            raise ValueError("Video frames must contain only finite values.")
        batches = []
        for start in range(0, len(videos), self.batch_size):
            x = videos[start:start + self.batch_size].to(device=self.device, dtype=torch.float32).clamp(-1, 1)
            if x.shape[2] == 1:
                x = x.repeat(1, 1, 3, 1, 1)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            features = self.model(x, rescale=False, resize=True, return_features=True)
            if features.shape != (len(x), 400) or not torch.isfinite(features).all():
                raise ValueError("Expected finite Google I3D logits with shape [N,400].")
            batches.append(features.cpu().numpy())
        return np.concatenate(batches, axis=0)


def frechet_video_distance(real: np.ndarray, fake: np.ndarray) -> float:
    """Gaussian Fréchet distance using unbiased covariance and a low-rank SVD.

    The covariance cross-root trace equals the nuclear norm of X @ Y.T,
    where centered feature matrices X,Y are divided by sqrt(n-1),sqrt(m-1).
    This handles singular covariance matrices without a dense matrix square
    root. It is efficient for pilot sets with fewer samples than features.
    Small negative roundoff is clipped to zero; invalid inputs are rejected.
    """
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if real.ndim != 2 or fake.ndim != 2:
        raise ValueError("Features must be matrices [N,D] and [M,D].")
    if len(real) < 2 or len(fake) < 2 or real.shape[1] != fake.shape[1] or real.shape[1] < 1:
        raise ValueError("Need >=2 samples per set and matching nonempty feature dimensions.")
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError("Features must contain only finite values.")
    real_mean, fake_mean = real.mean(axis=0), fake.mean(axis=0)
    x = (real - real_mean) / np.sqrt(len(real) - 1)
    y = (fake - fake_mean) / np.sqrt(len(fake) - 1)
    cross_trace = np.linalg.svd(x @ y.T, compute_uv=False).sum()
    distance = np.square(real_mean - fake_mean).sum() + np.square(x).sum() + np.square(y).sum() - 2 * cross_trace
    return float(max(distance, 0.0))
