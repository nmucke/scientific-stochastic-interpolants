#!/usr/bin/env python3
"""Estimate the coloured-driver spectrum C_k for a dataset config (WP1).

Writes an ``.npz`` with ``spectrum`` [C, *spatial[:-1], N//2+1] on the
orthonormal rfft grid, read by
``scisi.sampling.noise_covariance.FourierDiagonalCovariance(spectrum_path=...)``.
Periodic grids only (the Fourier-diagonal model of C).

The forecast uncertainty Sigma_k = mean |e_k|^2 is estimated from one of

* ``increment``: e = y - c on the chosen split (first pass; an upper bound on
  the conditional variance, so it over-noises),
* ``residuals``: e = y - m(h) of a deterministic MSE regressor
  (``models=conditional_mean``), on the **validation** split (preferred;
  training residuals underestimate the variance),
* ``anomalies``: an ensemble file (``ensemble`` [N, E, C, *spatial], centred
  per case with the E/(E-1) correction, or ``anomalies`` [N, C, *spatial]),
* ``power``: a precomputed per-mode power on the rfft grid (e.g. the Phase 0
  D1 arrays ``increment_train`` or ``anomaly_si128_exact``), in normalised
  units.

Then C_k = kappa * Sigma_k / q(1) with q(1) from the config's interpolation,
after light smoothing over neighbouring modes, a floor relative to the
per-channel maximum, and zeros on the spatial mean if ``--zero-mean`` (the
WP2 invariant). The full 2D spectrum is kept (the NS forcing is anisotropic).
All fields are in the model's normalised coordinates.

Example (on a compute node; NS data loading takes ~10 min):
  PYTHONPATH=$PWD/src .venv/bin/python research/ito_maps/experiments/estimate_noise_spectrum.py \
    --config-name stochastic_navier_stokes --mode increment --split train \
    --output /scratch/.../spectra/increment_kappa1.npz
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


# ---------------------------------------------------------------- numerics


def full_mode_power(fields: torch.Tensor) -> torch.Tensor:
    """Sum over the batch of |FFT e|^2 (norm="ortho"), full grid [C, *spatial]."""
    dims = tuple(range(2, fields.ndim))
    return torch.fft.fftn(fields.double(), dim=dims, norm="ortho").abs().square().sum(0)


def half_to_full(power_half: torch.Tensor, last_length: int) -> torch.Tensor:
    """Extend an rfft-grid power [C, *spatial[:-1], m] to the full grid using P(-k) = P(k)."""
    mirrored = power_half
    for axis in range(1, power_half.ndim - 1):  # k -> -k mod N on full axes
        mirrored = torch.roll(torch.flip(mirrored, (axis,)), 1, axis)
    missing = last_length - power_half.shape[-1]  # columns N-m+1 .. N-1 <- -k
    tail = torch.flip(mirrored[..., 1 : missing + 1], (-1,))
    return torch.cat([power_half, tail], dim=-1)


def smooth_circular(power: torch.Tensor, width: int, weight: torch.Tensor) -> torch.Tensor:
    """Weighted box average over (2 width + 1)^d neighbouring modes, periodic in k.

    Modes with zero weight (invariant directions) do not enter the average.
    """
    if width == 0:
        return power
    numerator, denominator = power * weight, weight.expand_as(power).clone()
    for axis in range(1, power.ndim):
        shifts = range(-width, width + 1)
        numerator = sum(torch.roll(numerator, shift, axis) for shift in shifts)
        denominator = sum(torch.roll(denominator, shift, axis) for shift in shifts)
    return numerator / denominator.clamp_min(1e-300)


def noise_spectrum(
    sigma_full: torch.Tensor,
    q1: float,
    kappa: float = 1.0,
    floor: float = 1e-4,
    smoothing: int = 1,
    zero_mean: bool = False,
) -> torch.Tensor:
    """C_k = kappa * smooth(Sigma)_k / q(1), floored, on the rfft grid.

    ``sigma_full`` is the per-mode forecast variance on the full grid
    [C, *spatial]. The floor is relative to each channel's maximum; with
    ``zero_mean`` the k = 0 mode is excluded from smoothing and set to zero.
    """
    if not 1.0 <= kappa <= 4.0:
        raise ValueError("kappa should lie in [1, 4] (aim for 1-2).")
    weight = torch.ones_like(sigma_full[:1])
    origin = (0,) * (sigma_full.ndim)
    if zero_mean:
        weight[origin] = 0.0
    smoothed = smooth_circular(sigma_full, smoothing, weight)
    peak = smoothed.flatten(1).max(1).values.reshape(-1, *([1] * (smoothed.ndim - 1)))
    floored = torch.maximum(smoothed, floor * peak)
    if zero_mean:
        floored[(slice(None),) + origin[1:]] = 0.0
    half = floored[..., : sigma_full.shape[-1] // 2 + 1]
    return kappa * half / q1


# -------------------------------------------------------------- data access


def compose_config(config_name, overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=list(overrides))


def white_variance_clock(cfg) -> float:
    """q(1) of the config's interpolation (its own covariance is not needed)."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    interpolation = OmegaConf.to_container(cfg.model.interpolation, resolve=True)
    interpolation.pop("noise_covariance", None)
    return float(instantiate(interpolation).variance_clock(0.0, 1.0))


def loader_for(cfg, split, batch_size):
    from hydra.utils import instantiate

    if split not in ("train", "val"):
        raise ValueError("Only the train and validation splits may be used.")
    preprocesser = instantiate(cfg.preprocesser) if cfg.get("preprocesser") else None
    data_cfg = cfg.train_data if split == "train" else cfg.val_data
    return instantiate(
        data_cfg,
        dataset={"preprocesser": preprocesser},
        batch_size=batch_size,
        shuffle=False,
    )


def load_regressor(checkpoint_dir, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(checkpoint_dir) / "config.yaml")
    model = instantiate(config.model)
    state = torch.load(Path(checkpoint_dir) / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def dataset_power(loader, residual, device, max_batches=None):
    """Mean full-grid |e_k|^2 over the loader, e = residual(batch)."""
    total, count = None, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if v is not None}
        power = full_mode_power(residual(batch))
        total = power if total is None else total + power
        count += batch["base"].shape[0]
    return total.cpu() / count, count


def ensemble_power(path, scale):
    """Mean full-grid anomaly power from an ensemble or anomaly file."""
    with np.load(path) as archive:
        if "ensemble" in archive:
            ensemble = torch.as_tensor(archive["ensemble"], dtype=torch.float64) / scale
            members = ensemble.shape[1]
            anomalies = (ensemble - ensemble.mean(1, keepdim=True)).flatten(0, 1)
            correction = members / (members - 1)
        else:
            anomalies = torch.as_tensor(archive["anomalies"], dtype=torch.float64) / scale
            correction = 1.0
    return correction * full_mode_power(anomalies) / anomalies.shape[0], anomalies.shape[0]


def precomputed_power(path, key, scale, last_length):
    with np.load(path) as archive:
        power = torch.as_tensor(archive[key], dtype=torch.float64) / scale**2
    if power.ndim == 2:  # single channel stored without its axis
        power = power[None]
    last_length = last_length or 2 * (power.shape[-1] - 1)
    return half_to_full(power, last_length), None


def atomic_savez(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


# --------------------------------------------------------------------- main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="stochastic_navier_stokes")
    parser.add_argument("--override", action="append", default=[], help="Hydra override")
    parser.add_argument("--mode", required=True, choices=["increment", "residuals", "anomalies", "power"])
    parser.add_argument("--split", choices=["train", "val"], help="default: train for increment, val for residuals")
    parser.add_argument("--regressor-dir", help="checkpoint dir of the conditional-mean regressor")
    parser.add_argument("--input", help="ensemble/anomaly/power .npz for --mode anomalies/power")
    parser.add_argument("--input-key", default="increment_train", help="array name for --mode power")
    parser.add_argument("--input-scale", type=float, default=1.0, help="divide input fields by this (to normalised units)")
    parser.add_argument("--last-length", type=int, help="full last-axis length for --mode power (default even)")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=1e-4)
    parser.add_argument("--smoothing", type=int, default=1, help="half-width of the mode box filter")
    parser.add_argument("--zero-mean", action="store_true", help="zero the spatial-mean mode (WP2)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new output path")

    cfg = compose_config(args.config_name, args.override)
    q1 = white_variance_clock(cfg)
    split = args.split or ("train" if args.mode == "increment" else "val")
    if args.mode == "residuals" and split != "val":
        print("warning: training-split residuals underestimate the variance", flush=True)

    if args.mode == "increment":
        loader = loader_for(cfg, split, args.batch_size)
        sigma, count = dataset_power(loader, lambda b: b["target"] - b["base"], args.device, args.max_batches)
    elif args.mode == "residuals":
        if not args.regressor_dir:
            parser.error("--mode residuals needs --regressor-dir")
        regressor = load_regressor(args.regressor_dir, args.device)

        def residual(batch):
            mean = regressor.predict(batch["base"], batch.get("field_history"), batch.get("field_cond"), batch.get("pars_cond"))
            return batch["target"] - mean

        loader = loader_for(cfg, split, args.batch_size)
        sigma, count = dataset_power(loader, residual, args.device, args.max_batches)
    elif args.mode == "anomalies":
        sigma, count = ensemble_power(args.input, args.input_scale)
        split = "from input file"
    else:
        sigma, count = precomputed_power(args.input, args.input_key, args.input_scale, args.last_length)
        split = "from input file"

    spectrum = noise_spectrum(sigma, q1, args.kappa, args.floor, args.smoothing, args.zero_mean)
    last = sigma.shape[-1] // 2 + 1
    metadata = {
        "argv": sys.argv if argv is None else argv,
        "config_name": args.config_name,
        "overrides": args.override,
        "mode": args.mode,
        "split": split,
        "num_samples": count,
        "q1": q1,
        "kappa": args.kappa,
        "floor_relative_to_channel_max": args.floor,
        "smoothing_half_width": args.smoothing,
        "zero_mean": args.zero_mean,
        "grid_shape": list(sigma.shape[1:]),
        "units": "normalised model coordinates; E|e_k|^2 of the orthonormal FFT",
        "sigma_over_white_min_max": [float((sigma / q1).min()), float((sigma / q1).max())],
    }
    atomic_savez(
        args.output,
        spectrum=spectrum.float().numpy(),
        sigma=sigma[..., :last].numpy(),
        q1=np.float64(q1),
        metadata=np.array(json.dumps(metadata)),
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
