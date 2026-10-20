#!/usr/bin/env python3
"""Stream a CPU PDE diagnostic under explicitly supplied, unverified parameters.

This is NOT verified continuation truth for the supplied forecasting dataset.
The archived physical parameters are unknown. Initial conditions and timestamps
are recovered from data_file.pt, while the caller supplies viscosity/drag/noise.
Only sparse state snapshots are stored; every lead has diagnostic metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def saved_leads(horizon: int, save_every: int) -> list[int]:
    if horizon < 1 or save_every < 1:
        raise ValueError("horizon and save_every must be positive")
    return sorted(set([0, horizon, *range(save_every, horizon + 1, save_every)]))


def validate_initial_state(states, times, trajectory_id, origin, grid):
    if states.ndim != 4 or states.shape[-2:] != (256, 256):
        raise ValueError(
            "Expected original [trajectory,time,256,256] vorticity archive"
        )
    if not 0 <= trajectory_id < len(states) or not 1 <= origin <= states.shape[1]:
        raise ValueError("Initial trajectory/origin lies outside the archive")
    if len(times) != states.shape[1] or grid not in (128, 256):
        raise ValueError("Invalid time vector or diagnostic grid")
    value = states[trajectory_id, origin - 1]
    if hasattr(value, "numpy"):
        value = value.numpy()
    initial = np.array(value, dtype=np.float32)
    if not np.isfinite(initial).all():
        raise ValueError("Nonfinite initial condition")
    return initial[:: 256 // grid, :: 256 // grid].copy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/stochastic_navier_stokes/data_file.pt"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trajectory-id", type=int, default=180)
    p.add_argument("--origin", type=int, default=5)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--members", type=int, default=1)
    p.add_argument("--grid", type=int, choices=[128, 256], default=256)
    p.add_argument("--viscosity", type=float, required=True)
    p.add_argument("--drag", type=float, required=True)
    p.add_argument("--forcing-scale", type=float, required=True)
    p.add_argument("--substep-dt", type=float, default=1e-4)
    p.add_argument("--inner-steps", type=int, default=4999)
    p.add_argument("--save-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=87301)
    p.add_argument("--cpu-cores", type=int, default=4)
    args = p.parse_args()
    leads = saved_leads(args.horizon, args.save_every)
    if min(args.members, args.inner_steps, args.cpu_cores) < 1 or args.substep_dt <= 0:
        p.error("members, inner_steps, cpu_cores and substep_dt must be positive")
    if min(args.viscosity, args.drag, args.forcing_scale) < 0:
        p.error("Physical parameters must be nonnegative")
    if args.output.exists():
        p.error("Output exists; select a new diagnostic directory")

    # This helper never initializes or reserves the GPU used by model experiments.
    os.environ["JAX_PLATFORMS"] = "cpu"
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[-args.cpu_cores :])
    import jax
    import jax.numpy as jnp
    from jax_cfd.base.grids import Grid
    import torch
    import jax_cfd_lib.navier_stokes_forward_model as solver

    states, times = torch.load(
        args.source, weights_only=True, mmap=True, map_location="cpu"
    )
    initial = validate_initial_state(
        states, times, args.trajectory_id, args.origin, args.grid
    )
    snapshot_dt = args.inner_steps * args.substep_dt
    metadata = {
        "label": "Assumed-parameter PDE diagnostic; not verified dataset continuation truth",
        "is_verified_dataset_continuation": False,
        "physical_parameters_status": "Caller supplied; not identified from archived data",
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_initial_frame_index": args.origin - 1,
        "source_initial_time": float(times[args.origin - 1]),
        "source_stored_spacing_mean": float(times.double().diff().mean()),
        "diagnostic_snapshot_dt": snapshot_dt,
        "domain": [[0.0, float(2 * np.pi)], [0.0, float(2 * np.pi)]],
        "initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
        "solver_sha256": hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
        "integrator": "Existing pseudospectral explicit Euler-Maruyama with brick-wall advection filtering",
        "forcing_modes": [[6, 0], [7, 0], [5, 5], [8, 8]],
        "rng_policy": "Independent fold_in(master_seed,member_id), then fold_in(member_key,lead); existing solver splits substeps",
        "saved_leads": leads,
        "state_storage": "state_samples.npy [member,saved_lead,128,128], raw vorticity",
        "completed_leads": 0,
    }
    args.output.mkdir(parents=True)
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    samples = np.lib.format.open_memmap(
        args.output / "state_samples.npy",
        mode="w+",
        dtype=np.float32,
        shape=(args.members, len(leads), 128, 128),
    )
    samples[:] = np.nan
    stride = args.grid // 128
    samples[:, 0] = initial[::stride, ::stride]
    equation_step = solver.set_up_forward_model(
        compile=False,
        use_true_model=False,
        stochastic=True,
        grid=Grid((args.grid, args.grid), domain=((0, 2 * np.pi), (0, 2 * np.pi))),
        viscosity=args.viscosity,
        drag=args.drag,
        smooth=True,
        hf_dt=args.substep_dt,
        inner_steps=args.inner_steps,
        stochastic_forcing_scale=args.forcing_scale,
    )
    step = jax.jit(jax.vmap(equation_step))
    state = jnp.broadcast_to(
        jnp.fft.rfft2(jnp.asarray(initial)),
        (args.members, args.grid, args.grid // 2 + 1),
    )
    member_keys = jax.vmap(
        lambda i: jax.random.fold_in(jax.random.PRNGKey(args.seed), i)
    )(jnp.arange(args.members))
    start = time.perf_counter()
    with (args.output / "diagnostics.jsonl").open("w") as stream:
        for lead in range(1, args.horizon + 1):
            tic = time.perf_counter()
            keys = jax.vmap(lambda k: jax.random.fold_in(k, lead))(member_keys)
            state, _ = step(state, keys)
            full = np.asarray(jnp.fft.irfft2(state)).astype(np.float64)
            coarse = full[:, ::stride, ::stride]
            finite = bool(np.isfinite(full).all())
            row = {
                "lead": lead,
                "elapsed_diagnostic_time": lead * snapshot_dt,
                "finite": finite,
                "step_seconds_including_first_call_compilation": time.perf_counter()
                - tic,
                "enstrophy_full": (
                    (0.5 * np.mean(full**2, axis=(-2, -1))).tolist() if finite else None
                ),
                "enstrophy_coarse": (
                    (0.5 * np.mean(coarse**2, axis=(-2, -1))).tolist()
                    if finite
                    else None
                ),
                "rms_spread_coarse": (
                    float(np.sqrt(np.var(coarse, axis=0, ddof=1).mean()))
                    if finite and args.members > 1
                    else None
                ),
                "max_abs_vorticity": float(np.max(np.abs(full))) if finite else None,
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if not finite:
                metadata["stop_reason"] = "nonfinite_state"
                break
            if lead in leads:
                samples[:, leads.index(lead)] = coarse.astype(np.float32)
                samples.flush()
            metadata["completed_leads"] = lead
            if lead % 16 == 0 or lead == args.horizon:
                print(f"Completed diagnostic lead {lead}/{args.horizon}", flush=True)
    metadata["total_seconds_including_compilation"] = time.perf_counter() - start
    metadata["complete"] = metadata["completed_leads"] == args.horizon
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    samples.flush()
    if not metadata["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
