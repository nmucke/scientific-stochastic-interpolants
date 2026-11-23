"""Encode the KTH frames into the latent space of the pretrained RIVER VQGAN.

Every 64x64 grayscale frame becomes a continuous 4x8x8 latent, as in Chen et al.,
"Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes"
(arXiv:2403.13724). The whole latent KTH workflow:

    uv pip install -e external_libs/river_lib      # the autoencoder (once)
    python scripts/download_kth.py                 # data/KTH/kth_64.npz
    python scripts/encode_kth.py                   # data/KTH/kth_64_latents.npz
    python scripts/train.py --config-name kth_latent experiment_tracking.name=my_run
    python scripts/test.py project=kth_latent name=my_run num_physical_steps=40

Needs the autoencoder checkpoint <root>/vqvae.ckpt:
https://huggingface.co/cvg-unibe/river_kth_64/resolve/main/vqvae.ckpt

Output <root>/kth_64_latents.npz: the same keys as kth_64.npz (see
scripts/download_kth.py), with

    frames   float32 [N_frames, 4, 8, 8]   latents instead of pixels
    mean     float32 [4]                   per-channel mean over the training persons
    std      float32 [4]                   per-channel std over the training persons

The mean and std are logged ready to paste into the preprocesser of
config/data/kth_latent.yaml.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from river_lib import KTHAutoencoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

TRAIN_PERSONS = range(1, 19)  # persons 1-18, as in config/data/kth_latent.yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/KTH"))
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="Frames encoded at once."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Rebuild the npz if it exists."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args()

    in_path = args.root / "kth_64.npz"
    out_path = args.root / "kth_64_latents.npz"
    ckpt_path = args.root / "vqvae.ckpt"

    if out_path.exists() and not args.overwrite:
        logger.info(f"{out_path} already exists; pass --overwrite to rebuild it.")
        return

    logger.info(f"Loading {in_path}")
    with np.load(in_path) as data:
        arrays = {key: data[key] for key in data.files}
    frames = arrays["frames"]  # uint8 [N_frames, 64, 64]

    logger.info(f"Loading the autoencoder from {ckpt_path}")
    autoencoder = KTHAutoencoder(ckpt_path=str(ckpt_path)).to(args.device)

    # Encode
    latents = []
    for i in tqdm(range(0, len(frames), args.batch_size), desc="Encoding"):
        batch = torch.from_numpy(frames[i : i + args.batch_size]).to(args.device)
        batch = batch.unsqueeze(1).to(torch.float32) / 255.0  # [B, 1, 64, 64]
        latents.append(autoencoder.encode(batch).cpu().numpy())
    latents = np.concatenate(latents, axis=0)  # [N_frames, 4, 8, 8]

    # Per-channel statistics over the frames of the training persons
    lengths = np.diff(arrays["offsets"])
    is_train = np.repeat(np.isin(arrays["person"], TRAIN_PERSONS), lengths)
    mean = latents[is_train].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = latents[is_train].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)

    arrays.update(frames=latents, mean=mean, std=std)
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez(tmp_path, **arrays)
    tmp_path.rename(out_path)

    logger.info(
        f"Wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB): "
        f"latents {latents.shape}, min/max {latents.min():.3f}/{latents.max():.3f}"
    )
    logger.info(
        "Latent statistics over the training persons, for the preprocesser in "
        "config/data/kth_latent.yaml (base and target):\n"
        f"    mean: {[round(float(m), 4) for m in mean]}\n"
        f"    std: {[round(float(s), 4) for s in std]}"
    )


if __name__ == "__main__":
    main()
