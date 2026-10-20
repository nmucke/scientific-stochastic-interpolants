#!/usr/bin/env python3
"""CPU timing of the current PDE solver, not verified original-data physics.

Four-core affinity, one held-out initial condition, 256 substeps per timing trial.
The long-horizon costs reported in the output are linear projections, not runs.
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-4:])
import jax, jax.numpy as jnp, numpy as np, torch, time, json
from jax_cfd.base.grids import Grid
from jax_cfd_lib.navier_stokes_forward_model import set_up_forward_model

x, _ = torch.load(
    "data/stochastic_navier_stokes/data_file.pt",
    weights_only=True,
    mmap=True,
    map_location="cpu",
)
results = []
for size in [128, 256]:
    grid = Grid((size, size), domain=((0, 2 * np.pi), (0, 2 * np.pi)))
    step = set_up_forward_model(
        compile=True,
        use_true_model=False,
        stochastic=True,
        grid=grid,
        viscosity=0.001,
        drag=0.1,
        smooth=True,
        hf_dt=0.0001,
        inner_steps=256,
        stochastic_forcing_scale=1.0,
    )
    initial = jnp.fft.rfft2(
        jnp.asarray(x[180, 4, :: 256 // size, :: 256 // size].numpy())
    )
    key = jax.random.PRNGKey(87251)
    t = time.perf_counter()
    state, nextkey = step(initial, key)
    jax.block_until_ready(state)
    compile_elapsed = time.perf_counter() - t
    elapsed = []
    for i in range(4):
        t = time.perf_counter()
        state, nextkey = step(initial, jax.random.fold_in(key, i))
        jax.block_until_ready(state)
        elapsed.append(time.perf_counter() - t)
    field = np.asarray(jnp.fft.irfft2(state))
    result = {
        "grid": size,
        "substeps": 256,
        "substep_dt": 0.0001,
        "compile_and_first_call_seconds": compile_elapsed,
        "steady_seconds": elapsed,
        "finite": bool(np.isfinite(field).all()),
        "enstrophy_initial": float(
            0.5 * np.mean(np.asarray(jnp.fft.irfft2(initial)) ** 2)
        ),
        "enstrophy_after_last_trial": float(0.5 * np.mean(field**2)),
        "projected_seconds_per_stored_interval_4999_substeps": float(
            np.median(elapsed) * 4999 / 256
        ),
    }
    print(result, flush=True)
    results.append(result)
from pathlib import Path

p = Path("research/ito_maps/results/long_horizon/cpu_reference_benchmark.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(
    json.dumps(
        {
            "backend": str(jax.devices()),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "parameters_status": "Current repository simulator; parameters not verified for original dataset",
            "viscosity": 0.001,
            "drag": 0.1,
            "forcing_scale": 1.0,
            "results": results,
        },
        indent=2,
    )
    + "\n"
)
