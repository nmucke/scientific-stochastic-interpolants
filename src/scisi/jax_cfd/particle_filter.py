import functools
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


def get_observation_operator_from_skip_grid(
    skip_grid: int = 4,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the observation operator."""

    def observation_operator(x: jnp.ndarray) -> jnp.ndarray:
        """Observation operator."""
        x = jnp.fft.irfftn(x)
        return x[::skip_grid, ::skip_grid]

    return observation_operator


def get_observation_operator_from_obs_matrix(
    obs_matrix: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the observation operator."""

    def observation_operator(x: jnp.ndarray) -> jnp.ndarray:
        """Observation operator."""
        x = jnp.fft.irfftn(x)
        return obs_matrix @ x

    return observation_operator


def get_proposal_model(
    observation_operator: Callable[[jnp.ndarray], jnp.ndarray],
    likelihood_model: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:

    def background_model(
        x: jnp.ndarray, particles_mean: jnp.ndarray, particles_cov: jnp.ndarray
    ) -> jnp.ndarray:
        """Background model."""
        diff = x - particles_mean
        diff = diff.flatten()
        out = jnp.linalg.solve(particles_cov, diff)
        return diff @ out

    """Get the proposal model."""

    def proposal_model(
        particles: jnp.ndarray,
        observations: jnp.ndarray,
        # particles_mean: jnp.ndarray,
        # particles_cov: jnp.ndarray
    ) -> jnp.ndarray:
        """Proposal model."""
        final_fn = lambda x: (
            likelihood_model(observation_operator(x), observations)  # type: ignore[call-arg]
            # + background_model(x, particles_mean, particles_cov)
        )
        x = particles.copy()
        for _ in range(10):
            x = x + 1e0 * jax.grad(final_fn, argnums=0)(x)
        return x

    return proposal_model  # type: ignore[return-value]


def get_likelihood_model(variance: float = 1.0) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the likelihood model."""

    def likelihood_model(x: jnp.ndarray, observations: jnp.ndarray) -> jnp.ndarray:
        """Likelihood model."""
        diff = (observations - x).flatten()
        prob = jax.scipy.stats.norm.pdf(diff, 0, jnp.sqrt(variance))
        return prob.mean()

    return likelihood_model  # type: ignore[return-value]


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
        proposal_model: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
        batch_size: Optional[int] = None,
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
        self.proposal_model = proposal_model
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

        rng_keys = jax.random.split(self.rng_key, self.ensemble_size)
        weights = jnp.ones(particles.shape[0]) / particles.shape[0]
        num_resamples = 0
        counter = 0
        for i in range(num_obs_steps):
            self.rng_key, _ = jax.random.split(self.rng_key)

            prior_particles, rng_keys = jax.vmap(self.forward_model)(
                particles, rng_keys
            )

            weight_fn = functools.partial(
                self.compute_weight, observations=observations[i]
            )
            new_weights = jax.vmap(weight_fn)(particles)
            print(f"Weight mean before proposal: {new_weights.mean()}")

            # prior_mean = prior_particles.mean(axis=0)
            # prior_cov = jnp.cov(jnp.reshape(prior_particles, (prior_particles.shape[0], -1)).T)

            proposal_fn = functools.partial(
                self.proposal_model,
                observations=observations[i],
                # particles_mean=prior_mean,
                # particles_cov=prior_cov
            )
            particles = jax.vmap(proposal_fn)(prior_particles)

            err = jnp.linalg.norm(prior_particles - particles)
            print(f"Error: {err}")

            # # Compute gradient of likelihood with respect to particles
            # weight_fn = lambda x: self.compute_weight(x, observations[i])
            # weight_grad_fn = jax.grad(weight_fn, argnums=0)
            # weight_grads = jax.vmap(weight_grad_fn)(particles)
            # particles = particles + 1e-1 + weight_grads

            weight_fn = functools.partial(
                self.compute_weight, observations=observations[i]
            )

            new_weights = jax.vmap(weight_fn)(particles)
            print(f"Weight mean after proposal: {new_weights.mean()}")

            is_nan_or_inf = jnp.isnan(new_weights) | jnp.isinf(new_weights)

            new_weights = jnp.nan_to_num(new_weights, nan=0.0)

            weights = weights * new_weights
            weights = weights / weights.sum(axis=0)
            N_eff = 1 / jnp.sum(weights**2)

            print(f"Time step {i} of {num_obs_steps}. N_eff: {N_eff}")
            print("================================================")

            # if N_eff < self.N_threshold or jnp.any(is_nan_or_inf) or counter > 5:
            num_resamples += 1
            particles = self.resample_particles(particles, weights)
            weights = jnp.ones(weights.shape) / weights.shape[0]
            counter = 0

            counter += 1
        return particles, num_resamples, N_eff


def main() -> None:
    """Main function."""

    ensemble_size = 500
    skip_grid = 4
    variance = 0.0025

    num_steps = 5

    # obs_matrix = np.load(f"obs_matrix_{skip_grid}.npz")["obs_matrix"]
    # obs_matrix = jnp.array(obs_matrix)

    # Load the trajectory
    trajectory = np.load("trajectory.npz")["trajectory"]
    # init_condition = jax.vmap(get_initial_vorticity)(
    #     jax.random.split(jax.random.PRNGKey(0), ensemble_size)
    # )
    trajectory = trajectory[100:]
    init_condition = jnp.repeat(trajectory[0:1], ensemble_size, axis=0)
    init_condition = jnp.fft.rfftn(init_condition, axes=(-2, -1))

    # observation_operator = jax.jit(get_observation_operator_from_obs_matrix(obs_matrix=obs_matrix))
    observation_operator = jax.jit(
        get_observation_operator_from_skip_grid(skip_grid=skip_grid)
    )
    likelihood_model = jax.jit(get_likelihood_model(variance=variance))
    forward_model = set_up_forward_model(
        compile=True, use_true_model=False, stochastic=True
    )
    proposal_model = jax.jit(
        get_proposal_model(
            observation_operator=observation_operator, likelihood_model=likelihood_model
        )
    )

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
        ensemble_size=ensemble_size,
        proposal_model=proposal_model,
        batch_size=ensemble_size,
        rng_key=jax.random.PRNGKey(0),
    )
    posterior_trajectories, num_resamples, N_eff = (
        particle_filter.compute_posterior_trajectories(
            observations[1 : num_steps + 1],
            init_condition,
        )
    )
    posterior_trajectories = jax.vmap(jnp.fft.irfftn)(posterior_trajectories)

    print(f"Number of resamples: {num_resamples}")
    print(f"N_eff: {N_eff}")

    prior_trajectories = init_condition
    rng_keys = jax.random.split(jax.random.PRNGKey(0), prior_trajectories.shape[0])
    for _ in range(num_steps):
        prior_trajectories, rng_keys = jax.vmap(forward_model)(
            prior_trajectories, rng_keys
        )
    prior_trajectories = jax.vmap(jnp.fft.irfftn)(prior_trajectories)

    posterior_center_point_samples = posterior_trajectories[:, 128, 128]
    prior_center_point_samples = prior_trajectories[:, 128, 128]
    true_center_point = trajectory[num_steps + 1, 128, 128]

    plt.figure()
    plt.hist(prior_center_point_samples, bins=100, label="Prior", density=True)
    plt.hist(
        posterior_center_point_samples,
        bins=100,
        label="Posterior",
        alpha=0.5,
        density=True,
    )
    plt.axvline(true_center_point, color="red", label="True", linewidth=4)
    plt.legend()
    plt.title("Point distribution")
    plt.xlabel("Vorticity")
    plt.ylabel("Frequency")
    plt.show()

    posterior_trajectories = jnp.concatenate(
        [trajectory[-1:], posterior_trajectories], axis=0
    )

    spatial_coord = (
        jnp.arange(GRID.shape[0]) * 2 * jnp.pi / GRID.shape[0]
    )  # same for x and y
    coords = {
        "ensemble_index": jnp.arange(10),
        "x": spatial_coord,
        "y": spatial_coord,
    }
    xarray.DataArray(
        posterior_trajectories[0:10], dims=["ensemble_index", "x", "y"], coords=coords
    ).plot.imshow(col="ensemble_index", col_wrap=5, cmap=sns.cm.icefire, robust=True)
    plt.show()


if __name__ == "__main__":
    main()
