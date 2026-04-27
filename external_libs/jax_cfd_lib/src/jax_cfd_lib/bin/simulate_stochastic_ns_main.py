import pdb
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax_cfd.base.grids as grids
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

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
FINAL_TIME = 50.0# 100.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
COMPILE = True
DRAG = 0.1
NUM_TRAJECTORIES = 5


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

    # Compute kinetic energy in time from vorticity.
    # In 2D: -Δψ = ω, u = ∂ψ/∂y, v = -∂ψ/∂x.
    nx, ny = GRID.shape
    vorticity_hat = jnp.fft.rfftn(trajectory, axes=(-2, -1))
    kx = jnp.fft.fftfreq(nx, d=1.0 / nx)
    ky = jnp.fft.rfftfreq(ny, d=1.0 / ny)
    kx2d, ky2d = jnp.meshgrid(kx, ky, indexing="ij")
    k_sq = kx2d**2 + ky2d**2
    k_sq = k_sq.at[0, 0].set(1.0)
    psi_hat = vorticity_hat / k_sq
    psi_hat = psi_hat.at[..., 0, 0].set(0.0)
    u_hat = 1j * ky2d * psi_hat
    v_hat = -1j * kx2d * psi_hat
    u = jnp.fft.irfftn(u_hat, s=(nx, ny), axes=(-2, -1))
    v = jnp.fft.irfftn(v_hat, s=(nx, ny), axes=(-2, -1))
    kinetic_energy = 0.5 * jnp.mean(u**2 + v**2, axis=(-2, -1))

    times = REDUCED_DT * jnp.arange(OUTER_STEPS)
    _, ax_ke = plt.subplots()
    color = "C0"
    for i in range(kinetic_energy.shape[0]):
        ax_ke.plot(times, kinetic_energy[i], color=color, linewidth=0.8, alpha=0.6)
    if kinetic_energy.shape[0] > 1:
        ax_ke.plot(
            times, jnp.mean(kinetic_energy, axis=0), color="black", linewidth=2.5
        )
    ax_ke.grid()
    ax_ke.set_xlabel("time")
    ax_ke.set_ylabel("kinetic energy")
    ax_ke.set_title("Kinetic energy over time")

    # Animate vorticity of the first trajectory.
    vorticity_anim = np.array(trajectory[0])
    vmax = float(np.max(np.abs(vorticity_anim)))
    fig_anim, ax_anim = plt.subplots()
    im = ax_anim.imshow(
        vorticity_anim[0],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=(0, 2 * np.pi, 0, 2 * np.pi),
    )
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    title = ax_anim.set_title("vorticity, t=0.00")
    fig_anim.colorbar(im, ax=ax_anim)

    def update(frame: int):
        im.set_array(vorticity_anim[frame])
        title.set_text(f"vorticity, t={frame * REDUCED_DT:.2f}")
        return im, title

    anim = animation.FuncAnimation(
        fig_anim, update, frames=vorticity_anim.shape[0], interval=50, blit=False
    )
    _ = anim

    anim.save("vorticity_animation.mp4", writer="ffmpeg", fps=10)

    # # transform the trajectory into real-space and wrap in xarray for plotting
    # spatial_coord = (
    #     jnp.arange(GRID.shape[0]) * 2 * jnp.pi / GRID.shape[0]
    # )  # same for x and y
    # coords = {
    #     "time": REDUCED_DT * jnp.arange(OUTER_STEPS),
    #     "x": spatial_coord,
    #     "y": spatial_coord,
    # }
    # xarray.DataArray(trajectory[0], dims=["time", "x", "y"], coords=coords).plot.imshow(
    #     col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True
    # )
    # xarray.DataArray(trajectory[1], dims=["time", "x", "y"], coords=coords).plot.imshow(
    #     col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True
    # )
    plt.show()




if __name__ == "__main__":
    main()
