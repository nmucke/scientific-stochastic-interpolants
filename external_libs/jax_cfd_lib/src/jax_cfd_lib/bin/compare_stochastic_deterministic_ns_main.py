from functools import partial

import jax
import jax.numpy as jnp
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
SMOOTH = True
FINAL_TIME = 150.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
COMPILE = True
DRAG = 0.1
NUM_TRAJECTORIES = 10
STOCHASTIC_FORCING_SCALES = [0.5, 1.5, 2.4]


def simulate(stochastic: bool, stochastic_forcing_scale: float = 1.0) -> jnp.ndarray:
    step_repeated = set_up_forward_model(
        compile=COMPILE,
        use_true_model=False,
        stochastic=stochastic,
        grid=GRID,
        viscosity=VISCOSITY,
        drag=DRAG,
        smooth=SMOOTH,
        hf_dt=HF_DT,
        inner_steps=INNER_STEPS,
        stochastic_forcing_scale=stochastic_forcing_scale,
    )

    vorticity_hat0 = jax.vmap(
        partial(get_initial_vorticity, grid=GRID, max_velocity=MAX_VELOCITY, k=4)
    )(jax.random.split(jax.random.PRNGKey(0), NUM_TRAJECTORIES))
    rng_keys = jax.random.split(jax.random.PRNGKey(10), vorticity_hat0.shape[0])

    trajectory = []
    for _ in range(OUTER_STEPS):
        rng_keys = jax.random.split(rng_keys[0], vorticity_hat0.shape[0])
        vorticity_hat0, rng_keys = jax.vmap(step_repeated)(vorticity_hat0, rng_keys)
        trajectory.append(vorticity_hat0)
    trajectory = jnp.stack(trajectory)
    trajectory = jnp.swapaxes(trajectory, 0, 1)
    return jnp.fft.irfftn(trajectory, axes=(-2, -1))


def compute_kinetic_energy(trajectory: jnp.ndarray) -> jnp.ndarray:
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
    return 0.5 * jnp.mean(u**2 + v**2, axis=(-2, -1))


def save_vorticity_animation(trajectory: jnp.ndarray, filename: str, title_prefix: str) -> None:
    vorticity = np.array(trajectory[0])
    vorticity = vorticity[15:]
    vmax = float(np.max(np.abs(vorticity)))
    fig, ax = plt.subplots()
    im = ax.imshow(
        vorticity[0],
        cmap="viridis",#RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=(0, 2 * np.pi, 0, 2 * np.pi),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    # title = ax.set_title(f"{title_prefix}, t=0.00")
    title = ax.set_title(f"")
    # fig.colorbar(im, ax=ax)

    def update(frame: int):
        im.set_array(vorticity[frame])
        # title.set_text(f"{title_prefix}, t={frame * REDUCED_DT:.2f}")
        title.set_text(f"")
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=vorticity.shape[0], interval=50, blit=False
    )
    anim.save(filename, writer="ffmpeg", fps=10)
    plt.close(fig)


def main() -> None:
    print("JAX devices:", jax.devices())

    stoch_trajectories: dict[float, jnp.ndarray] = {}
    for scale in STOCHASTIC_FORCING_SCALES:
        print(f"Running stochastic simulation (scale={scale})...")
        stoch_trajectories[scale] = simulate(
            stochastic=True, stochastic_forcing_scale=scale
        )
    print("Running deterministic simulation...")
    traj_det = simulate(stochastic=False)

    stoch_energies = {
        scale: compute_kinetic_energy(traj) for scale, traj in stoch_trajectories.items()
    }
    ke_det = compute_kinetic_energy(traj_det)
    ke_det = ke_det[:, 5:]

    times = REDUCED_DT * jnp.arange(OUTER_STEPS)
    times = times[5:]

    fig, ax = plt.subplots()
    color_list = ["tab:blue", "tab:green", "tab:orange"]
    det_color = "tab:red"

    # pass 1: thin ensemble members
    for idx, scale in enumerate(STOCHASTIC_FORCING_SCALES):
        ke = stoch_energies[scale][:, 5:]
        for i in range(ke.shape[0]):
            ax.plot(times, ke[i], color=color_list[idx], linewidth=0.8, alpha=0.5)
    for i in range(ke_det.shape[0]):
        ax.plot(times, ke_det[i], color=det_color, linewidth=0.8, alpha=0.5)

    # pass 2: thick means, drawn on top
    for idx, scale in enumerate(STOCHASTIC_FORCING_SCALES):
        ke = stoch_energies[scale][:, 5:]
        ax.plot(
            times,
            jnp.mean(ke, axis=0),
            color=color_list[idx],
            linewidth=2.5,
            zorder=10,
            label=f"stochastic (scale={scale})",
        )
    ax.plot(
        times,
        jnp.mean(ke_det, axis=0),
        color=det_color,
        linewidth=2.5,
        zorder=10,
        label="deterministic",
    )

    ax.grid()
    ax.set_xlabel("time")
    ax.set_ylabel("kinetic energy")
    ax.set_title("Kinetic energy: stochastic vs deterministic")
    ax.legend()
    fig.savefig("kinetic_energy_comparison.png", dpi=150, bbox_inches="tight")

    print("Saving animations...")
    for scale, traj in stoch_trajectories.items():
        save_vorticity_animation(
            traj,
            f"vorticity_stochastic_scale_{scale}.mp4",
            f"stochastic vorticity (scale={scale})",
        )
    save_vorticity_animation(
        traj_det, "vorticity_deterministic.mp4", "deterministic vorticity"
    )

    plt.show()


if __name__ == "__main__":
    main()
