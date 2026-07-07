"""CPU unit tests for correlation-based localization (Vossepoel et al. 2025).

Tiny synthetic data only -- no jax-cfd NS solver, no multi-step assimilation.
Run with::

    ENKF_JAX_PLATFORM=cpu JAX_PLATFORMS=cpu \
        .venv/bin/python external_libs/jax_cfd_lib/tests/test_correlation_localization.py

Checks (see the task brief):
  1. obs-error inflation E_inf rises smoothly from 1 at d_c = beta*d_t toward
     E_max near d_c = d_t (and the b-formula gives E_inf == E_max at d_c = d_t).
  2. a highly-correlated measurement is RETAINED (E_inf ~ 1, weight ~ 1) and a
     ~zero-correlation measurement is TRUNCATED (weight == 0) for a variable.
  3. correlation localization strongly updates a state var via its correlated
     obs and leaves an uncorrelated var (whose only obs is truncated) ~unchanged.
  4. the existing DISTANCE-based path is byte-for-byte unchanged: a tiny
     distance-localized EnKF analysis is identical with and without the new
     localization_type argument present (default "distance").
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("ENKF_JAX_PLATFORM", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_platform_name", "cpu")

from jax_cfd_lib.ENKF import (  # noqa: E402
    LocalizedSpectralEnKF,
    correlation_localization_weights,
    correlation_state_obs,
)
from jax_cfd_lib.ETKF import LocalizedSpectralETKF  # noqa: E402


# --------------------------------------------------------------------------- #
# Test 1: E_inf taper shape (the Eq. for b).
# --------------------------------------------------------------------------- #
def test_inflation_taper_shape() -> None:
    r_t = 0.4
    d_t = 1.0 - r_t
    beta = 0.5
    E_max = 6.0

    # Build rho values that map to chosen d_c via d_c = 1 - |rho|.
    d_c_grid = jnp.linspace(0.0, d_t, 21)
    rho_grid = (1.0 - d_c_grid).reshape(-1, 1)  # (n, 1) one obs per state var
    w = correlation_localization_weights(
        rho_grid, corr_threshold=r_t, inflation_max=E_max, inflation_beta=beta
    ).reshape(-1)
    E_inf = 1.0 / w  # w = 1 / E_inf for retained obs

    knee = beta * d_t
    # Below the knee: no inflation.
    below = d_c_grid <= knee
    assert jnp.allclose(E_inf[below], 1.0, atol=1e-6), E_inf[below]
    # Monotonically non-decreasing in d_c.
    assert bool(jnp.all(jnp.diff(E_inf) >= -1e-6)), jnp.diff(E_inf)
    # At d_c = d_t, E_inf == E_max (the b-formula is calibrated for this).
    assert abs(float(E_inf[-1]) - E_max) < 1e-4, float(E_inf[-1])
    # Strictly increasing once past the knee.
    past = d_c_grid > knee + 1e-6
    assert float(E_inf[past][0]) > 1.0
    print("test_inflation_taper_shape: OK  (E_inf 1 ->", round(float(E_inf[-1]), 3), ")")


# --------------------------------------------------------------------------- #
# Test 2: retain correlated / truncate uncorrelated.
# --------------------------------------------------------------------------- #
def test_retain_and_truncate() -> None:
    key = jax.random.PRNGKey(0)
    N = 200  # ensemble size; r_t default = 3/sqrt(200) ~ 0.212

    # Variable a strongly drives obs 0; obs 1 is independent noise.
    a = jax.random.normal(key, (N,))
    key, k2, k3 = jax.random.split(key, 3)
    obs0 = 0.95 * a + 0.05 * jax.random.normal(k2, (N,))  # highly correlated
    obs1 = jax.random.normal(k3, (N,))  # ~uncorrelated with a

    X_pert = (a - a.mean()).reshape(N, 1)  # (N, n_state=1)
    HX_pert = jnp.stack([obs0 - obs0.mean(), obs1 - obs1.mean()], axis=1)  # (N, 2)

    rho = correlation_state_obs(X_pert, HX_pert)  # (1, 2)
    r_t = 3.0 / (N**0.5)
    w = correlation_localization_weights(rho, corr_threshold=r_t)[0]  # (2,)

    assert abs(float(rho[0, 0])) > 0.9, float(rho[0, 0])
    assert abs(float(rho[0, 1])) < r_t, float(rho[0, 1])
    # Correlated obs retained with ~no inflation; uncorrelated obs truncated.
    assert float(w[0]) > 0.99, float(w[0])
    assert float(w[1]) == 0.0, float(w[1])
    print(
        "test_retain_and_truncate: OK  rho=",
        [round(float(x), 3) for x in rho[0]],
        " w=",
        [round(float(x), 3) for x in w],
    )


# --------------------------------------------------------------------------- #
# Helpers: a 1D synthetic localized EnKF on a tiny grid (no NS solver).
# --------------------------------------------------------------------------- #
class _DenseH:
    """Trivial obs operator: select grid points by index (spectral == physical).

    We bypass the FFT by giving the EnKF ``real_space=False`` and feeding it a
    physical-space ensemble cast as the "spectral" state -- fft2/ifft2 then act as
    near-identity round trips on a tiny grid, which is enough to exercise the
    analysis math. ``obs_coords`` is supplied for the distance path.
    """

    def __init__(self, sel_indices, grid_shape):
        self.sel = jnp.asarray(sel_indices)
        nx, ny = grid_shape
        self.obs_coords = jnp.stack(
            [self.sel // ny, self.sel % ny], axis=1
        ).astype(jnp.float32)

    def __call__(self, u_spectral):
        phys = jnp.fft.ifft2(u_spectral).real
        return phys.reshape(-1)[self.sel]

    def physical_observation_operator(self, u_physical):
        return u_physical.reshape(-1)[self.sel]


def _tiny_ensemble(grid_shape, N, key):
    nx, ny = grid_shape
    phys = jax.random.normal(key, (N, nx, ny))
    return jax.vmap(jnp.fft.fft2)(phys)  # complex spectral; real_space=False


# --------------------------------------------------------------------------- #
# Test 3: correlation localization updates only the correlated variable.
# --------------------------------------------------------------------------- #
def test_correlation_analysis_local_update() -> None:
    grid_shape = (4, 4)
    N = 300
    key = jax.random.PRNGKey(1)

    # Build a physical ensemble where grid point 0 is strongly correlated with
    # grid point 5 (obs there is informative for it) and point 15 is independent.
    nx, ny = grid_shape
    base = jax.random.normal(key, (N, nx * ny))
    # Make point 5 ~ point 0 (so observing 5 informs 0).
    base = base.at[:, 5].set(0.97 * base[:, 0] + 0.03 * base[:, 5])
    phys = base.reshape(N, nx, ny)
    ens_spectral = jax.vmap(jnp.fft.fft2)(phys)

    obs_op = _DenseH(sel_indices=[5], grid_shape=grid_shape)  # observe only pt 5

    enkf = LocalizedSpectralEnKF(
        grid_shape=grid_shape,
        ensemble_size=N,
        model_noise_std=0.0,
        obs_noise_std=0.01,
        localization_radius=0.0,
        observation_operator=obs_op,
        real_space=False,
        localization_type="correlation",
        corr_threshold=0.3,
    )

    # Strong observation pulling point 5 well away from its mean.
    obs = jnp.asarray([5.0])
    key, k = jax.random.split(key)
    analysis = enkf.analysis_step(ens_spectral, obs, k, inflation=1.0)

    fmean = jnp.fft.ifft2(jnp.mean(ens_spectral, axis=0)).real.reshape(-1)
    amean = jnp.fft.ifft2(jnp.mean(analysis, axis=0)).real.reshape(-1)

    d0 = abs(float(amean[0] - fmean[0]))  # correlated with obs -> should move
    d15 = abs(float(amean[15] - fmean[15]))  # uncorrelated -> truncated -> ~0
    assert d0 > 1e-2, ("correlated var not updated", d0)
    assert d15 < 1e-3, ("uncorrelated var moved", d15)
    print(
        "test_correlation_analysis_local_update: OK  |dpt0|=",
        round(d0, 4),
        " |dpt15|=",
        round(d15, 5),
    )


# --------------------------------------------------------------------------- #
# Test 4: distance path unchanged (default localization_type="distance").
# --------------------------------------------------------------------------- #
def test_distance_path_unchanged() -> None:
    grid_shape = (4, 4)
    N = 30
    key = jax.random.PRNGKey(2)
    ens = _tiny_ensemble(grid_shape, N, key)
    obs_op = _DenseH(sel_indices=[0, 5, 10], grid_shape=grid_shape)
    obs = jnp.asarray([1.0, -1.0, 0.5])

    common = dict(
        grid_shape=grid_shape,
        ensemble_size=N,
        model_noise_std=0.0,
        obs_noise_std=0.1,
        localization_radius=2.0,
        observation_operator=obs_op,
        real_space=False,
    )
    # Default (no new arg) vs explicit localization_type="distance".
    enkf_default = LocalizedSpectralEnKF(**common)
    enkf_explicit = LocalizedSpectralEnKF(**common, localization_type="distance")

    k = jax.random.PRNGKey(7)
    a_default = enkf_default.analysis_step(ens, obs, k, inflation=1.1)
    a_explicit = enkf_explicit.analysis_step(ens, obs, k, inflation=1.1)

    assert jnp.allclose(a_default, a_explicit, atol=0.0, rtol=0.0), (
        "distance path changed by new argument"
    )
    print("test_distance_path_unchanged: OK  (bitwise identical)")


# --------------------------------------------------------------------------- #
# Test 5: LETKF correlation path is the faithful per-variable local analysis.
# --------------------------------------------------------------------------- #
def test_letkf_correlation_local_update() -> None:
    grid_shape = (4, 4)
    N = 300
    key = jax.random.PRNGKey(3)

    nx, ny = grid_shape
    base = jax.random.normal(key, (N, nx * ny))
    base = base.at[:, 5].set(0.97 * base[:, 0] + 0.03 * base[:, 5])  # pt0 ~ pt5
    ens_spectral = jax.vmap(jnp.fft.fft2)(base.reshape(N, nx, ny))

    obs_op = _DenseH(sel_indices=[5], grid_shape=grid_shape)
    letkf = LocalizedSpectralETKF(
        grid_shape=grid_shape,
        ensemble_size=N,
        model_noise_std=0.0,
        obs_noise_std=0.01,
        localization_radius=0.0,
        observation_operator=obs_op,
        real_space=False,
        localization_type="correlation",
        corr_threshold=0.3,
    )

    obs = jnp.asarray([5.0])
    k = jax.random.PRNGKey(11)
    analysis = letkf.analysis_step(ens_spectral, obs, k, inflation=1.0)

    fmean = jnp.fft.ifft2(jnp.mean(ens_spectral, axis=0)).real.reshape(-1)
    amean = jnp.fft.ifft2(jnp.mean(analysis, axis=0)).real.reshape(-1)
    d0 = abs(float(amean[0] - fmean[0]))
    d15 = abs(float(amean[15] - fmean[15]))
    assert d0 > 1e-2, ("LETKF correlated var not updated", d0)
    assert d15 < 1e-3, ("LETKF uncorrelated var moved", d15)
    print(
        "test_letkf_correlation_local_update: OK  |dpt0|=",
        round(d0, 4),
        " |dpt15|=",
        round(d15, 5),
    )


if __name__ == "__main__":
    test_inflation_taper_shape()
    test_retain_and_truncate()
    test_correlation_analysis_local_update()
    test_distance_path_unchanged()
    test_letkf_correlation_local_update()
    print("\nALL CORRELATION-LOCALIZATION TESTS PASSED")
