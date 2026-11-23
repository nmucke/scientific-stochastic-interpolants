"""Round-trip check of the RIVER KTH autoencoder on real frames (skipped without checkpoint or data)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from river_lib import KTHAutoencoder

REPO_ROOT = Path(__file__).resolve().parents[3]
CKPT = REPO_ROOT / "data/KTH/vqvae.ckpt"
DATA = REPO_ROOT / "data/KTH/kth_64.npz"


@pytest.mark.skipif(
    not (CKPT.exists() and DATA.exists()), reason="KTH checkpoint or data not available"
)
def test_round_trip_on_real_frames():
    with np.load(DATA) as data:
        offsets = data["offsets"]
        starts = offsets[:-1][:: max(1, (len(offsets) - 1) // 8)][
            :8
        ]  # first frame of 8 videos
        frames_u8 = data["frames"][starts]
    frames = torch.from_numpy(frames_u8).float().div(255.0)[:, None]

    autoencoder = KTHAutoencoder(str(CKPT))
    assert not any(p.requires_grad for p in autoencoder.parameters())
    assert not autoencoder.training

    latents = autoencoder.encode(frames)
    assert latents.shape == (len(frames), 4, 8, 8)

    recon = autoencoder.decode(latents)
    assert recon.shape == frames.shape
    assert recon.min() >= 0.0 and recon.max() <= 1.0

    psnr = 10 * torch.log10(1.0 / ((recon - frames) ** 2).flatten(1).mean(dim=1))
    assert psnr.mean() > 30.0, f"mean round-trip PSNR {psnr.mean():.2f} dB"
