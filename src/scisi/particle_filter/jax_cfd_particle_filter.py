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

PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]

import pdb
from typing import Callable

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


# pylint: disable=invalid-name
def _get_grid_variable(
    arr, grid, bc=boundaries.periodic_boundary_conditions(2), offset=(0.5, 0.5)
):
    return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)


VISCOSITY = 1e-3
MAX_VELOCITY = 7
GRID = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
HF_DT = 1e-4
REDUCED_DT = 0.5
SMOOTH = True  # use anti-aliasing
FINAL_TIME = 55.0
OUTER_STEPS = int(FINAL_TIME // REDUCED_DT)
INNER_STEPS = int(FINAL_TIME // HF_DT) // OUTER_STEPS
COMPILE = True


@dataclasses.dataclass
class NavierStokes2D:
    """Breaks the Navier-Stokes equation into implicit and explicit parts.

    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.

    Attributes:
      viscosity: strength of the diffusion term
      grid: underlying grid of the process
      smooth: smooth the advection term using the 2/3-rule.
      forcing_fn: forcing function, if None then no forcing is used.
      drag: strength of the drag. Set to zero for no drag.
    """

    viscosity: float
    grid: grids.Grid
    drag: float = 0.0
    smooth: bool = True
    forcing_fn: Optional[Callable[[grids.Grid], Any]] = None
    _forcing_fn_with_grid = None
    stochastic_forcing_fn: Optional[Callable[[grids.Grid], Any]] = None
    _stochastic_forcing_fn_with_grid = None

    def __post_init__(self):
        self.kx, self.ky = self.grid.rfft_mesh()
        self.laplace = (jnp.pi * 2j) ** 2 * (self.kx**2 + self.ky**2)
        self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
        self.linear_term = self.viscosity * self.laplace - self.drag
        self.key = jax.random.PRNGKey(42)

        # setup the forcing function with the caller-specified grid.
        if self.forcing_fn is not None:
            self._forcing_fn_with_grid = self.forcing_fn(self.grid)

    def explicit_terms(self, vorticity_hat):
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
            fx, fy = self._forcing_fn_with_grid(
                (_get_grid_variable(vx, self.grid), _get_grid_variable(vy, self.grid))
            )
            fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
            terms += spectral_utils.spectral_curl_2d(
                (self.kx, self.ky), (fx_hat, fy_hat)
            )

        terms += self.linear_term * vorticity_hat
        return terms

    def stochastic_explicit_terms(self, vorticity_hat):

        # Generate 8 independent Wiener increments: ΔW_i ~ N(0, dt)
        dW = jax.random.normal(self.key, shape=(8,))

        # Initialize zero array (complex)
        dQ_hat = jnp.zeros_like(vorticity_hat)

        # W₁: sin(6x) + W₅: cos(6x) affect modes (±6, 0)
        # Note: kx = -6 is at index 256 - 6 = 250 (wraparound)
        dQ_hat = dQ_hat.at[6, 0].add((-1j / 2) * dW[0] + (1 / 2) * dW[4])
        dQ_hat = dQ_hat.at[250, 0].add((1j / 2) * dW[0] + (1 / 2) * dW[4])

        # W₂: cos(7x) + W₆: sin(7x) affect modes (±7, 0)
        dQ_hat = dQ_hat.at[7, 0].add((1 / 2) * dW[1] + (-1j / 2) * dW[5])
        dQ_hat = dQ_hat.at[249, 0].add((1 / 2) * dW[1] + (1j / 2) * dW[5])

        # W₃: sin(5(x+y)) + W₇: cos(5(x+y)) affect modes (±5, ±5)
        # Only (5, 5) stored; (-5, -5) handled by Hermitian symmetry in rfftn
        dQ_hat = dQ_hat.at[5, 5].add((-1j / 2) * dW[2] + (1 / 2) * dW[6])

        # W₄: cos(8(x+y)) + W₈: sin(8(x+y)) affect modes (±8, ±8)
        dQ_hat = dQ_hat.at[8, 8].add((1 / 2) * dW[3] + (-1j / 2) * dW[7])

        self.key, subkey = jax.random.split(self.key)

        return dQ_hat

        # x = self.grid.mesh((0,0))[0]
        # y = self.grid.mesh((0,0))[1]

        # noise = []
        # for _ in range(8):
        #     self.key, subkey = jax.random.split(self.key)
        #     noise.append(jax.random.normal(subkey, x.shape))

        # out = noise[0] * jnp.sin(6.0 * x)
        # out += noise[1] * jnp.cos(7.0 * x)
        # out += noise[2] * jnp.sin(5.0 * (x + y))
        # out += noise[3] * jnp.cos(8.0 * (x + y))
        # out += noise[4] * jnp.cos(6.0 * x)
        # out += noise[5] * jnp.sin(7.0 * x)
        # out += noise[6] * jnp.cos(5.0 * (x + y))
        # out += noise[7] * jnp.sin(8.0 * (x + y))

        # # return out
        # return jnp.fft.rfftn(out)


def forward_euler(
    equation: NavierStokes2D,
    time_step: float,
) -> TimeStepFn:
    dt = time_step
    F = tree_math.unwrap(equation.explicit_terms)
    G = tree_math.unwrap(equation.stochastic_explicit_terms)

    @tree_math.wrap
    def step_fn(u0):
        u_final = u0 + dt * F(u0)

        u_final = u_final + G(u0) * jnp.sqrt(dt)

        return u_final

    return step_fn


def set_up_forward_model(
    compile: bool = COMPILE,
):
    """Set up the forward model."""
    step_fn = forward_euler(
        NavierStokes2D(VISCOSITY, GRID, smooth=SMOOTH, forcing_fn=None, drag=0.1), HF_DT
    )
    step_repeated = cfd.funcutils.repeated(step_fn, INNER_STEPS)

    if compile:
        step_repeated = jax.jit(step_repeated)

    return step_repeated


if COMPILE:
    ifft_fn = partial(jnp.fft.irfftn, axes=(1, 2))
    ifft_fn = jax.jit(ifft_fn)
else:
    ifft_fn = partial(jnp.fft.irfftn, axes=(1, 2))


def main():

    # jax.config.update("jax_platforms", "cuda")
    # Check if CUDA is available
    print("JAX devices:", jax.devices())

    t1 = time.time()
    # setup step function using crank-nicolson runge-kutta order 4
    step_repeated = set_up_forward_model(compile=COMPILE)

    # create an initial velocity field and compute the fft of the vorticity.
    # the spectral code assumes an fft'd vorticity for an initial state
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(42), GRID, MAX_VELOCITY, 4
    )
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    # with jax.disable_jit():
    trajectory = []
    for _ in range(OUTER_STEPS):
        vorticity_hat0 = step_repeated(vorticity_hat0)
        trajectory.append(vorticity_hat0)
    trajectory = jnp.stack(trajectory[-10:])
    trajectory = ifft_fn(trajectory)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")

    # transform the trajectory into real-space and wrap in xarray for plotting
    spatial_coord = (
        jnp.arange(GRID.shape[0]) * 2 * jnp.pi / GRID.shape[0]
    )  # same for x and y
    coords = {
        # 'time': REDUCED_DT * jnp.arange(OUTER_STEPS) * INNER_STEPS,
        "time": REDUCED_DT * jnp.arange(10) * INNER_STEPS,
        "x": spatial_coord,
        "y": spatial_coord,
    }
    xarray.DataArray(trajectory, dims=["time", "x", "y"], coords=coords).plot.imshow(
        col="time", col_wrap=5, cmap=sns.cm.icefire, robust=True
    )
    plt.show()


if __name__ == "__main__":
    main()
