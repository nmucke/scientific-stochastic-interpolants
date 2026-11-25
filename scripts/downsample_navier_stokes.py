"""Write a spatially downsampled copy of the stochastic Navier–Stokes archive.

    python scripts/downsample_navier_stokes.py --size 64

reads ``data/stochastic_navier_stokes/data.npz`` (array ``state``, shape
[trajectories, snapshots, H, W]) and writes
``data/stochastic_navier_stokes_<size>/data.npz`` with the same layout and
dtype, plus ``summary.json`` (train-split std for the config, and the fraction
of energy and enstrophy in the discarded modes).

Downsampling is spectral truncation (`spectral_downsample`), which assumes the
fields are periodic in both spatial axes, as the NS data are. The output is
saved uncompressed so that it loads quickly.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def spectral_downsample(fields: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Truncate the Fourier series of fields periodic in the last two axes.

    Keeps the wavenumbers |k_i| < size_i / 2 on each axis and drops everything
    else, including the coarse grid's Nyquist modes (they have no real-valued
    counterpart that preserves symmetry). A field band-limited to those
    wavenumbers is reproduced exactly at the coarse grid points, the mean is
    preserved, and every kept Fourier amplitude is unchanged.

    Args:
        fields: real array [..., H, W], periodic in H and W.
        size: target (h, w), even and at most (H, W).

    Returns:
        Array [..., h, w], float64.
    """
    (h, w), (big_h, big_w) = size, fields.shape[-2:]
    if h % 2 or w % 2 or h > big_h or w > big_w:
        raise ValueError(f"target {size} must be even and at most {(big_h, big_w)}")
    # norm="forward": coefficients are amplitudes, independent of grid size.
    spectrum = np.fft.rfft2(fields, norm="forward")
    kept = np.zeros(fields.shape[:-2] + (h, w // 2 + 1), dtype=spectrum.dtype)
    m, n = h // 2, w // 2
    kept[..., :m, :n] = spectrum[..., :m, :n]
    kept[..., h - m + 1 :, :n] = spectrum[..., big_h - m + 1 :, :n]
    return np.fft.irfft2(kept, s=(h, w), norm="forward")


def discarded_power(fields: np.ndarray, size: tuple[int, int]) -> dict:
    """Energy (|ω̂|²/|k|²) and enstrophy (|ω̂|²) sums: total and outside the kept modes.

    Treats ``fields`` [..., H, W] as vorticity on [0, 2π)²; used only for the
    summary. Returns sums so that they can be accumulated over chunks.
    """
    big_h, big_w = fields.shape[-2:]
    power = np.abs(np.fft.fft2(fields, norm="forward")) ** 2
    kx = np.fft.fftfreq(big_h, 1 / big_h)[:, None]
    ky = np.fft.fftfreq(big_w, 1 / big_w)[None, :]
    dropped = ~((np.abs(kx) < size[0] // 2) & (np.abs(ky) < size[1] // 2))
    k2 = kx**2 + ky**2
    inv_k2 = np.where(k2 > 0, 1 / np.where(k2 > 0, k2, 1), 0.0)
    return {
        "enstrophy_total": float(power.sum()),
        "enstrophy_dropped": float((power * dropped).sum()),
        "energy_total": float((power * inv_k2).sum()),
        "energy_dropped": float((power * inv_k2 * dropped).sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default="data/stochastic_navier_stokes/data.npz")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-trajectories", type=int, default=160)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir or f"data/stochastic_navier_stokes_{args.size}")
    output = output_dir / "data.npz"
    if output.exists():
        raise SystemExit(f"{output} exists; not overwriting")
    output_dir.mkdir(parents=True, exist_ok=True)

    state = np.load(args.input)["state"]
    size = (args.size, args.size)
    num_train = args.train_trajectories
    small = np.empty(state.shape[:-2] + size, dtype=state.dtype)
    sums = {}
    moments = np.zeros((2, 3))  # rows: input, output; columns: count, sum, sum of squares
    for i in range(state.shape[0]):  # one trajectory at a time bounds memory
        trajectory = state[i].astype(np.float64)
        small[i] = spectral_downsample(trajectory, size)
        if i < num_train:
            for key, value in discarded_power(trajectory, size).items():
                sums[key] = sums.get(key, 0.0) + value
            for row, x in enumerate((trajectory, small[i].astype(np.float64))):
                moments[row] += (x.size, x.sum(), (x**2).sum())

    def mean_std(row):
        count, total, squares = moments[row]
        mean = total / count
        return mean, float(np.sqrt(squares / count - mean**2))

    summary = {
        "input": str(args.input),
        "input_shape": list(state.shape),
        "output_shape": list(small.shape),
        "method": "spectral truncation, |k_i| < size/2 per axis, Nyquist dropped",
        "train_trajectories": [0, num_train],
        "train_mean": mean_std(1)[0],
        "train_std": mean_std(1)[1],
        "input_train_mean": mean_std(0)[0],
        "input_train_std": mean_std(0)[1],
        "discarded_fraction_train": {
            "enstrophy": sums["enstrophy_dropped"] / sums["enstrophy_total"],
            "energy": sums["energy_dropped"] / sums["energy_total"],
        },
    }

    # Atomic: write to a temporary name, then rename.
    tmp = output_dir / "data.tmp.npz"
    np.savez(tmp, state=small)
    os.replace(tmp, output)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
