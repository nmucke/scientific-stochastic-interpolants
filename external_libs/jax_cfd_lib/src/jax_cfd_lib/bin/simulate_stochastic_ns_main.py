import pdb
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax_cfd.base.grids as grids
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray

from jax_cfd_lib.navier_stokes_forward_model import (
    get_initial_vorticity,
    set_up_forward_model,
)

VISCOSITY = 1e-3
MAX_VELOCITY = 7
GRID = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
HF_DT = 1e-4
REDUCED_DT = 0.5
SMOOTH = True  # use anti-aliasing
FINAL_TIME = 100.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
COMPILE = True
DRAG = 0.1
NUM_TRAJECTORIES = 1


def main() -> None:
    """Main function."""
    # Check if CUDA is available
    # jax.config.update('jax_platforms', 'cpu')
    print("JAX devices:", jax.devices())

    # setup step function using crank-nicolson runge-kutta order 4
    step_repeated = set_up_forward_model(
        compile=COMPILE,
        use_true_model=False,
        stochastic=True,
        grid=GRID,
        viscosity=VISCOSITY,
        drag=DRAG,
        smooth=SMOOTH,
        hf_dt=HF_DT,
        inner_steps=INNER_STEPS,
    )

    t1 = time.time()
    # create an initial velocity field and compute the fft of the vorticity.
    vorticity_hat0 = jax.vmap(
        partial(get_initial_vorticity, grid=GRID, max_velocity=MAX_VELOCITY, k=4)
    )(jax.random.split(jax.random.PRNGKey(0), NUM_TRAJECTORIES))
    rng_key = jax.random.PRNGKey(10)
    rng_keys = jax.random.split(rng_key, vorticity_hat0.shape[0])

    # Temporarily disable JIT compilation for this section
    # with jax.disable_jit():
    trajectory = []
    for _ in range(OUTER_STEPS):
        rng_keys = jax.random.split(rng_keys[0], vorticity_hat0.shape[0])
        vorticity_hat0, rng_keys = jax.vmap(step_repeated)(vorticity_hat0, rng_keys)
        trajectory.append(vorticity_hat0)
    trajectory = jnp.stack(trajectory)
    trajectory = jnp.swapaxes(trajectory, 0, 1)
    trajectory = jnp.fft.irfftn(trajectory, axes=(-2, -1))

    trajectory_to_save = trajectory[0]
    np.savez("trajectory.npz", trajectory=np.array(trajectory_to_save))

    # transform the trajectory into real-space and wrap in xarray for plotting
    spatial_coord = (
        jnp.arange(GRID.shape[0]) * 2 * jnp.pi / GRID.shape[0]
    )  # same for x and y
    coords = {
        "time": REDUCED_DT * jnp.arange(OUTER_STEPS),
        "x": spatial_coord,
        "y": spatial_coord,
    }
    xarray.DataArray(trajectory[0], dims=["time", "x", "y"], coords=coords).plot.imshow(
        col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True
    )
    xarray.DataArray(trajectory[1], dims=["time", "x", "y"], coords=coords).plot.imshow(
        col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True
    )
    plt.show()


if __name__ == "__main__":
    main()
