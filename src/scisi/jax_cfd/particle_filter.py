import pdb
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray

from scisi.jax_cfd.navier_stokes_forward_model import (
    GRID,
    INNER_STEPS,
    OUTER_STEPS,
    REDUCED_DT,
    get_initial_vorticity,
    set_up_forward_model,
)


def get_observation_operator(
    skip_grid: int = 4,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the observation operator."""

    def observation_operator(x: jnp.ndarray) -> jnp.ndarray:
        """Observation operator."""
        x = jnp.fft.irfftn(x)
        return x[::skip_grid, ::skip_grid]

    return observation_operator


def get_likelihood_model(variance: float = 1.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the likelihood model."""

    def likelihood_model(x: jnp.ndarray, observations: jnp.ndarray) -> jnp.ndarray:
        """Likelihood model."""
        diff = observations - x
        diff = jnp.linalg.norm(diff, axis=(-2, -1)) ** 2
        return jnp.exp(-0.5 * diff / variance)

    return likelihood_model  # type: ignore[return-value]


def get_fourier_perturber() -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get a random perturbation."""

    def fourier_perturber(x: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Random perturbation."""
        noise = jax.random.normal(rng_key, x.shape) * 5.0
        return x + noise  # jnp.fft.rfftn(noise)

    return fourier_perturber  # type: ignore[return-value]


class ParticleFilter:
    """Particle filter for the Navier-Stokes equation."""

    def __init__(
        self,
        forward_model: Callable[[jnp.ndarray], jnp.ndarray],
        ensemble_size: int = 100,
        observation_operator: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
        likelihood_model: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: jnp.ones(
            x.shape[0]
        ),
        batch_size: Optional[int] = None,
        perturber: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    ) -> None:
        """Initialize the particle filter."""
        self.forward_model = forward_model
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        if batch_size is None:
            self.batch_size = ensemble_size

        self.rng_key = rng_key
        self.observation_operator = observation_operator
        self.likelihood_model = likelihood_model
        self.perturber = perturber

        self.N_threshold = self.ensemble_size * 0.5

    def compute_weight(
        self, particle: jnp.ndarray, observations: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute the weights of the particles."""
        return self.likelihood_model(self.observation_operator(particle), observations)  # type: ignore[call-arg]

    def resample_particles(
        self, particles: jnp.ndarray, weights: jnp.ndarray
    ) -> jnp.ndarray:
        """Resample the particles."""
        return jax.random.choice(
            self.rng_key, particles, (self.ensemble_size,), p=weights, replace=True
        )

    def compute_posterior_trajectories(
        self, observations: jnp.ndarray, init_condition: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute the posterior trajectories.

        Args:
            observations: observations
            init_condition: initial condition of the particles
        Returns:
            Posterior trajectories of the particles
        """

        num_obs_steps = observations.shape[0]

        if len(init_condition.shape) == 2:
            particles = jnp.stack([init_condition] * self.ensemble_size, axis=0)
        else:
            particles = init_condition

        weights = jnp.ones(particles.shape[0]) / particles.shape[0]
        num_resamples = 0
        for i in range(num_obs_steps):
            self.rng_key, _ = jax.random.split(self.rng_key)

            particles = jax.vmap(self.forward_model)(particles)
            particles = jax.vmap(self.perturber)(
                particles, jax.random.split(self.rng_key, self.ensemble_size)
            )

            observations_i = observations[i]
            observations_i = jnp.expand_dims(observations_i, axis=0)
            observations_i = jnp.repeat(observations_i, particles.shape[0], axis=0)

            # new_weights = jax.vmap(self.compute_weight)(particles, observations_i)

            particles_obs = jax.vmap(self.observation_operator)(particles)
            likelihood = jax.vmap(self.likelihood_model)(particles_obs, observations_i)

            new_weights = likelihood + 1e-4

            is_nan_or_inf = jnp.isnan(new_weights) | jnp.isinf(new_weights)

            new_weights = jnp.nan_to_num(new_weights, nan=0.0)
            new_weights = jnp.where(jnp.isinf(new_weights), 1, new_weights)

            weights = weights * new_weights
            weights = weights / weights.sum(axis=0)
            N_eff = 1 / jnp.sum(weights**2)

            print(f"N_eff: {N_eff}")
            print("================================================")

            if N_eff < self.N_threshold or jnp.any(is_nan_or_inf):
                num_resamples += 1
                particles = self.resample_particles(particles, weights)
                weights = jnp.ones(weights.shape) / weights.shape[0]

        return particles, num_resamples, N_eff


def main() -> None:
    """Main function."""

    ensemble_size = 25
    skip_grid = 16
    variance = 1.0

    # Load the trajectory
    trajectory = np.load("trajectory.npz")["trajectory"]
    init_condition = jax.vmap(get_initial_vorticity)(
        jax.random.split(jax.random.PRNGKey(0), ensemble_size)
    )
    trajectory = trajectory[10:]
    # init_condition = jnp.repeat(trajectory[0:1], ensemble_size, axis=0)
    # init_condition = jnp.fft.rfftn(init_condition, axes=(-2, -1))
    # init_condition = init_condition + jax.random.normal(jax.random.PRNGKey(0), init_condition.shape) * 5

    observation_operator = get_observation_operator(skip_grid=skip_grid)
    likelihood_model = get_likelihood_model(variance=variance)
    forward_model = set_up_forward_model(compile=True, use_true_model=False)
    fourier_perturber = get_fourier_perturber()

    observations = jax.vmap(observation_operator)(
        jnp.fft.rfftn(trajectory, axes=(-2, -1))
    )
    observations = observations + jax.random.normal(
        jax.random.PRNGKey(0), observations.shape
    ) * jnp.sqrt(variance)

    # Initialize the particle filter
    particle_filter = ParticleFilter(
        forward_model=forward_model,
        observation_operator=observation_operator,
        likelihood_model=likelihood_model,
        perturber=fourier_perturber,
        ensemble_size=ensemble_size,
        batch_size=ensemble_size,
        rng_key=jax.random.PRNGKey(0),
    )
    posterior_trajectories, num_resamples, N_eff = (
        particle_filter.compute_posterior_trajectories(
            observations[1:],
            init_condition,
        )
    )

    print(f"Number of resamples: {num_resamples}")
    print(f"N_eff: {N_eff}")

    posterior_trajectories = jnp.fft.irfftn(posterior_trajectories, axes=(-2, -1))
    posterior_trajectories = jnp.concatenate(
        [trajectory[-1:], posterior_trajectories], axis=0
    )

    spatial_coord = (
        jnp.arange(GRID.shape[0]) * 2 * jnp.pi / GRID.shape[0]
    )  # same for x and y
    coords = {
        "ensemble_index": jnp.arange(particle_filter.ensemble_size + 1),
        "x": spatial_coord,
        "y": spatial_coord,
    }
    xarray.DataArray(
        posterior_trajectories, dims=["ensemble_index", "x", "y"], coords=coords
    ).plot.imshow(col="ensemble_index", col_wrap=5, cmap=sns.cm.icefire, robust=True)
    plt.show()


if __name__ == "__main__":
    main()
