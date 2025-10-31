import dataclasses
import time
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp
import jax_cfd.base as cfd
import jax_cfd.base as base
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import jax_cfd.spectral.equations as equations
import jax_cfd.spectral.time_stepping as time_stepping
import jax_cfd.spectral.types as types
import jax_cfd.spectral.utils as spectral_utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import tree_math
import xarray
from jax_cfd.base import boundaries
from jax_cfd.base.funcutils import scan

PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]

import pdb
from typing import Callable

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]  # type: ignore[valid-type]


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


# pylint: disable=invalid-name
def _get_grid_variable(
    arr: jnp.ndarray,
    grid: grids.Grid,
    bc: boundaries.BoundaryConditions = boundaries.periodic_boundary_conditions(2),
    offset: Tuple[float, float] = (0.5, 0.5),
) -> grids.GridVariable:
    """Get the grid variable.
    Args:
        arr: array
        grid: grid
        bc: boundary conditions
        offset: offset
    Returns:
        Grid variable
    """
    return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)


class NavierStokes2D:
    """Navier-Stokes equation in vorticity formulation."""

    def __init__(
        self,
        viscosity: float,
        grid: grids.Grid,
        drag: float = 0.0,
        smooth: bool = True,
        forcing_fn: Optional[Callable[[grids.Grid], Any]] = None,
    ):
        """
        Initialize the Navier-Stokes equation.

        Args:
            viscosity: viscosity of the fluid
            grid: grid of the domain
            drag: drag of the fluid
            smooth: smooth the advection term using the 2/3-rule
            forcing_fn: forcing function
            rng_key: random key
        """
        super().__init__()
        self.viscosity = viscosity
        self.grid = grid
        self.drag = drag
        self.smooth = smooth
        self.forcing_fn = forcing_fn
        self._forcing_fn_with_grid = None

        self.kx, self.ky = self.grid.rfft_mesh()
        self.laplace = (jnp.pi * 2j) ** 2 * (self.kx**2 + self.ky**2)
        self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        self.linear_term = self.viscosity * self.laplace - self.drag

        # setup the forcing function with the caller-specified grid.
        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    def explicit_terms(self, vorticity_hat: jnp.ndarray) -> jnp.ndarray:
        """Compute the explicit terms of the Navier-Stokes equation.
        Args:
            vorticity_hat: vorticity field
        Returns:
            Explicit terms
        """
        velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
        vxhat, vyhat = velocity_solve(vorticity_hat)
        vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

        grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
        grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
        grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

        advection = -(grad_x * vx + grad_y * vy)
        advection_hat = jnp.fft.rfftn(advection)

        if self.smooth is not None:
            advection_hat *= self.filter_

        terms = advection_hat

        if self.forcing_fn is not None:
            fx, fy = self._forcing_fn_with_grid(  # type: ignore[misc]
                (_get_grid_variable(vx, self.grid), _get_grid_variable(vy, self.grid))
            )
            fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
            terms += spectral_utils.spectral_curl_2d(
                (self.kx, self.ky), (fx_hat, fy_hat)
            )

        terms += self.linear_term * vorticity_hat
        return terms

    def stochastic_explicit_terms(
        self,
        vorticity_hat: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Compute the stochastic explicit terms of the Navier-Stokes equation.
        Args:
            vorticity_hat: vorticity field
        Returns:
            Stochastic explicit terms
        """

        Nx = self.grid.shape[0]
        Ny = self.grid.shape[1]

        # Initialize Fourier array
        dB_fourier = jnp.zeros((Nx, Ny // 2 + 1), dtype=jnp.complex64)

        # Sample Wiener increments
        # keys = jax.random.split(rng_key, 8)
        # dW = jnp.array([jax.random.normal(k) for k in keys])
        rng_key = jax.random.split(rng_key)[0]
        dW = jax.random.normal(rng_key, (8,))

        # Normalization factor for FFT (depends on convention)
        # For jnp.fft: forward transform has no normalization
        norm = Nx * Ny

        # Helper function to set mode
        def set_mode(arr: jnp.ndarray, kx: int, ky: int, value: float) -> jnp.ndarray:
            """Set Fourier mode at (kx, ky)"""
            # Handle negative kx (wraps around)
            idx_x = kx if kx >= 0 else Nx + kx
            # ky must be non-negative for rfftn
            if ky >= 0 and ky <= Ny // 2:
                arr = arr.at[idx_x, ky].set(value * norm)
            return arr

        # Mode (6, 0): W5 cos(6x) + W1 sin(6x)
        # cos(6x) → (δ(k-6) + δ(k+6))/2, sin(6x) → (δ(k-6) - δ(k+6))/(2i)
        dB_fourier = set_mode(dB_fourier, 6, 0, 0.5 * (dW[4] - 1j * dW[0]))
        dB_fourier = set_mode(dB_fourier, -6, 0, 0.5 * (dW[4] + 1j * dW[0]))

        # Mode (7, 0): W2 cos(7x) + W6 sin(7x)
        dB_fourier = set_mode(dB_fourier, 7, 0, 0.5 * (dW[1] - 1j * dW[5]))
        dB_fourier = set_mode(dB_fourier, -7, 0, 0.5 * (dW[1] + 1j * dW[5]))

        # Mode (5, 5): W7 cos(5(x+y)) + W3 sin(5(x+y))
        dB_fourier = set_mode(dB_fourier, 5, 5, 0.5 * (dW[6] - 1j * dW[2]))
        dB_fourier = set_mode(dB_fourier, -5, -5, 0.5 * (dW[6] + 1j * dW[2]))
        # Note: (-5, -5) is outside rfftn range, handled by Hermitian symmetry

        # Mode (8, 8): W4 cos(8(x+y)) + W8 sin(8(x+y))
        dB_fourier = set_mode(dB_fourier, 8, 8, 0.5 * (dW[3] - 1j * dW[7]))
        dB_fourier = set_mode(dB_fourier, -8, -8, 0.5 * (dW[3] + 1j * dW[7]))

        return dB_fourier


def repeated(f: Callable, steps: int) -> Callable:
    """Returns a repeatedly applied version of f()."""

    def f_repeated(
        x_initial: jnp.ndarray, rng_key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jax.random.PRNGKey]:
        rng_keys = jax.random.split(rng_key, steps)
        x_final, _ = jax.lax.scan(f, x_initial, xs=rng_keys, length=steps)
        rng_keys_final, _ = jax.random.split(rng_keys[-1])
        return x_final, rng_keys_final

    return f_repeated


def forward_euler(
    equation: NavierStokes2D,
    time_step: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Forward Euler time stepping for the Navier-Stokes equation."""
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)

    @tree_math.wrap  # type: ignore[misc]
    def step_fn(u0: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Time step the Navier-Stokes equation.
        Args:
            u0: initial vorticity field
            rng_key: random key
        Returns:
            Final vorticity field
        """
        u_final = u0 + dt * F(u0)

        return u_final, None

    return step_fn  # type: ignore[no-any-return]


def forward_euler_maruyama(
    equation: NavierStokes2D,
    time_step: float,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Forward Euler-Maruyama time stepping for the Navier-Stokes equation."""
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    G = tree_math.unwrap(equation.stochastic_explicit_terms)

    @tree_math.wrap  # type: ignore[misc]
    def step_fn(u0: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Time step the Navier-Stokes equation.
        Args:
            u0: initial vorticity field
        Returns:
            Final vorticity field
        """
        u_final = u0 + dt * F(u0)

        u_final = u_final + G(u0, rng_key) * jnp.sqrt(dt)

        return u_final, None

    return step_fn  # type: ignore[no-any-return]


def kolmogorov_forcing(
    grid: grids.Grid,
    scale: float = 1,
    k: int = 4,
    swap_xy: bool = False,
    offsets: Tuple[Tuple[float, ...], ...] = ((0, 0), (0, 0)),
) -> ForcingFn:
    """
    Compute the Kolmogorov forcing function for turbulence in 2D.
    Args:
        grid: grid of the domain
        scale: scale of the forcing
        k: wave number
        swap_xy: swap x and y
        offsets: offsets of the grid
    Returns:
        Forcing function
    """

    offsets = grid.cell_faces

    y = grid.mesh(offsets[0])[1]
    u = scale * grids.GridArray(jnp.sin(k * y), offsets[0], grid)

    v = grids.GridArray(jnp.zeros_like(u.data), (1 / 2, 1), grid)
    f = (u, v)

    def forcing(v: jnp.ndarray) -> Tuple[grids.GridArray, grids.GridArray]:
        """Compute the forcing function.
        Args:
            v: velocity field
        Returns:
            Forcing function
        """
        del v
        return f

    return forcing


def set_up_forward_model(
    compile: bool = COMPILE,
    use_true_model: bool = False,
    stochastic: bool = False,
    grid: grids.Grid = GRID,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Set up the forward model."""

    # forcing = lambda grid: kolmogorov_forcing(grid)
    forcing = None

    if use_true_model:
        step_fn = spectral.time_stepping.crank_nicolson_rk4(
            spectral.equations.NavierStokes2D(
                VISCOSITY, grid, smooth=SMOOTH, forcing_fn=forcing, drag=0.1
            ),
            HF_DT,
        )
        step_repeated = cfd.funcutils.repeated(step_fn, INNER_STEPS)
    else:
        stepper = forward_euler_maruyama if stochastic else forward_euler

        step_fn = stepper(
            NavierStokes2D(
                VISCOSITY,
                grid,
                smooth=SMOOTH,
                forcing_fn=forcing,
                drag=DRAG,
            ),
            HF_DT,
        )
        # step_repeated = cfd.funcutils.repeated(step_fn, INNER_STEPS)
        step_repeated = repeated(step_fn, INNER_STEPS)

    if compile:
        step_repeated = jax.jit(step_repeated)

    return step_repeated  # type: ignore[no-any-return]


if COMPILE:
    ifft_fn = partial(jnp.fft.irfftn, axes=(-2, -1))
    ifft_fn = jax.jit(ifft_fn)
else:
    ifft_fn = partial(jnp.fft.irfftn, axes=(-2, -1))


def get_initial_vorticity(rng_key: jax.random.PRNGKey) -> jnp.ndarray:
    """Get the initial vorticity field."""
    v0 = cfd.initial_conditions.filtered_velocity_field(rng_key, GRID, MAX_VELOCITY, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    return vorticity_hat0


def main() -> None:
    """Main function."""
    # Check if CUDA is available
    # Hide GPU from JAX
    # jax.config.update('jax_platforms', 'cpu')
    print("JAX devices:", jax.devices())

    # setup step function using crank-nicolson runge-kutta order 4
    step_repeated = set_up_forward_model(
        compile=COMPILE,
        use_true_model=False,
        stochastic=True,
    )

    t1 = time.time()
    # create an initial velocity field and compute the fft of the vorticity.
    vorticity_hat0 = jax.vmap(get_initial_vorticity)(
        jax.random.split(jax.random.PRNGKey(0), 200)
    )
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
    trajectory = ifft_fn(trajectory)

    np.savez(
        "data/stochastic_navier_stokes/data_jax.npz",
        state=np.array(trajectory[:, 100:, ::2, ::2]),  # type: ignore[call-overload]
    )
    pdb.set_trace()
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")

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
