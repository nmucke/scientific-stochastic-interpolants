#!/usr/bin/env python3
"""WP1 step 1: few-step Euler with exact integrated noise against its own N=128.

For a trained ``FollmerStochasticInterpolant`` (any noise covariance and
schedule), forecast one physical step on validation cases with

    X_{j+1} = X_j + dt b(X_j, t_j; h) + (M_{t_{j+1}} - M_{t_j}),

where M is the exact driver of the model's interpolation (its covariance C and
variance clock). All step counts of one case share one Brownian path, so the
N-step members are also compared pathwise with the N=128 members. Reported per
step count, averaged over cases: RMS spread (finite-ensemble corrected) and
its ratio to N=128, fair CRPS and ensemble-mean RMSE against the truth, the
same-path RMSE to N=128, and per-shell member and anomaly spectra (orthonormal
rfft2, integer shells) with the per-shell spread ratio to N=128. Normalised
coordinates; multiply CRPS/RMSE/spread by ``norm_std`` for physical units.
Validation trajectories (160-179) only.

  PYTHONPATH=$PWD/src .venv/bin/python research/ito_maps/experiments/wp1_euler_exact_noise.py \
    --checkpoint-dir checkpoints/stochastic_navier_stokes/<run> --output <dir>
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant

VALIDATION = range(160, 180)


@torch.no_grad()
def euler_exact_noise(model, base, history, path, num_steps):
    """Euler drift plus the exact martingale increment of each step."""
    state = base.clone()
    times = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64).tolist()
    for left, right in zip(times[:-1], times[1:]):
        t = state.new_full((state.shape[0], 1), left)
        state = state + (right - left) * model.drift(state, t, field_history=history)
        state = state + path.increment(left, right)
    return state


def shell_index(height, width):
    kx = np.fft.fftfreq(height) * height
    ky = np.fft.rfftfreq(width) * width
    return np.floor(np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) + 0.5).astype(int)


def shell_sum(power, shells, width):
    """Sum an rfft-grid power [..., H, W//2+1] over integer shells, counting
    each rfft column as often as it appears in the full grid."""
    weight = np.full(shells.shape, 2.0)
    weight[:, 0] = 1.0
    if width % 2 == 0:
        weight[:, -1] = 1.0  # Nyquist column
    flat = (power * weight).reshape(-1, shells.size)
    bins = shells.reshape(-1)
    out = np.stack([np.bincount(bins, row, minlength=bins.max() + 1) for row in flat])
    return out.reshape(*power.shape[:-2], -1)


def case_metrics(members, truth, shells):
    """members [E, C, H, W], truth [C, H, W] (float64 numpy)."""
    ensemble, width = members.shape[0], members.shape[-1]
    variance = members.var(0, ddof=1)
    spread = math.sqrt(variance.mean() * (ensemble + 1) / ensemble)
    rmse = math.sqrt(((members.mean(0) - truth) ** 2).mean())
    skill = np.abs(members - truth).mean(0)
    pairs = np.abs(members[:, None] - members[None]).sum((0, 1)) / (ensemble * (ensemble - 1))
    crps = float((skill - 0.5 * pairs).mean())
    rfft = lambda x: np.fft.rfft2(x, norm="ortho")
    member_power = (np.abs(rfft(members)) ** 2).mean(0)
    anomaly = members - members.mean(0)
    anomaly_power = (np.abs(rfft(anomaly)) ** 2).sum(0) / (ensemble - 1)
    return {
        "rms_spread": spread,
        "ensemble_mean_rmse": rmse,
        "fair_crps": crps,
        "member_shell_spectrum": shell_sum(member_power, shells, width).mean(0),
        "anomaly_shell_spectrum": shell_sum(anomaly_power, shells, width).mean(0),
    }


def load_model(checkpoint_dir, device):
    config = OmegaConf.load(Path(checkpoint_dir) / "config.yaml")
    model = instantiate(config.model)
    state = torch.load(Path(checkpoint_dir) / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval().requires_grad_(False), config


def atomic_write(path, text):
    handle, temporary = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(handle, "w") as file:
        file.write(text)
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--trajectories", nargs="+", type=int, default=list(VALIDATION))
    parser.add_argument("--origins", nargs="+", type=int, default=[5, 25, 45, 65, 85])
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--reference-steps", type=int, default=128)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(t not in VALIDATION for t in args.trajectories):
        parser.error("validation trajectories 160-179 only")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        parser.error("output already has results.json")

    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint_dir, args.device)
    if not isinstance(model, FollmerStochasticInterpolant):
        raise TypeError("needs a FollmerStochasticInterpolant checkpoint")
    preprocessor = instantiate(config.preprocesser)
    data_config = config.test_data.copy()
    first, last = min(args.trajectories), max(args.trajectories) + 1
    data_config.tajectories_ids = f"({first},{last})"
    data_config.train_or_test = "test"
    dataset = instantiate(data_config, preprocesser=preprocessor)
    length = dataset.len_field_history
    steps = sorted(set(args.steps) | {args.reference_steps})

    partial_path = args.output / "cases.jsonl"
    done = set()
    if partial_path.exists():  # resume
        for line in partial_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["trajectory"], row["origin"]))
    generator = torch.Generator(device=args.device)
    start = time.perf_counter()
    for trajectory in args.trajectories:
        sample = dataset[trajectory - first]["x"]
        for origin in args.origins:
            if (trajectory, origin) in done:
                continue
            window = sample[..., origin - length : origin + 1].unsqueeze(0)
            normalised = preprocessor.transform(field_history=window, is_batch=True)["field_history"]
            history = normalised[..., :length].expand(args.ensemble, *normalised.shape[1:-1], length)
            history = history.to(args.device, next(model.parameters()).dtype)
            truth = normalised[0, ..., length].double().numpy()
            base = history[..., -1]
            generator.manual_seed(args.seed + 1000 * trajectory + origin)
            path = model.make_brownian_path(base, generator)
            members = {}
            for num_steps in sorted(steps, reverse=True):  # reference first
                members[num_steps] = euler_exact_noise(model, base, history, path, num_steps)
            shells = shell_index(*truth.shape[-2:])
            reference = members[args.reference_steps].double().cpu().numpy()
            row = {"trajectory": trajectory, "origin": origin, "steps": {}}
            for num_steps, values in members.items():
                values = values.double().cpu().numpy()
                metrics = case_metrics(values, truth, shells)
                metrics["same_path_rmse_to_reference"] = math.sqrt(((values - reference) ** 2).mean())
                row["steps"][str(num_steps)] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in metrics.items()
                }
            with partial_path.open("a") as file:
                file.write(json.dumps(row) + "\n")
            print(f"{time.strftime('%T')} case {trajectory}/{origin} done", flush=True)

    rows = [json.loads(line) for line in partial_path.read_text().splitlines()]
    summary = {}
    reference = str(args.reference_steps)
    for num_steps in map(str, steps):
        get = lambda key: np.array([r["steps"][num_steps][key] for r in rows])
        ref = lambda key: np.array([r["steps"][reference][key] for r in rows])
        summary[num_steps] = {
            "rms_spread": float(get("rms_spread").mean()),
            "spread_ratio_to_reference": float((get("rms_spread") / ref("rms_spread")).mean()),
            "fair_crps": float(get("fair_crps").mean()),
            "ensemble_mean_rmse": float(get("ensemble_mean_rmse").mean()),
            "same_path_rmse_to_reference": float(get("same_path_rmse_to_reference").mean()),
            "shell_spread_ratio_to_reference": (
                get("anomaly_shell_spectrum").mean(0) / ref("anomaly_shell_spectrum").mean(0)
            ).tolist(),
            "member_shell_spectrum": get("member_shell_spectrum").mean(0).tolist(),
        }
    checkpoint = args.checkpoint_dir / "model.pth"
    result = {
        "argv": sys.argv if argv is None else argv,
        "split": "validation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "interpolation": type(model.interpolation).__name__,
        "noise_covariance": type(model.interpolation.noise_covariance).__name__,
        "gamma_multiplier": float(model.interpolation.gamma_multiplier),
        "sampler": "Euler drift + exact integrated noise of the model's interpolation, one shared path per case",
        "ensemble_size": args.ensemble,
        "num_cases": len(rows),
        "norm_std": OmegaConf.to_container(config.preprocesser.target, resolve=True).get("std"),
        "seconds_this_run": time.perf_counter() - start,
        "summary": summary,
    }
    atomic_write(args.output / "results.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {m: v[m] for m in ("spread_ratio_to_reference", "fair_crps")} for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
