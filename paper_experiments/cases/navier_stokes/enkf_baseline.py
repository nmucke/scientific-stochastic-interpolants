"""Classical data-assimilation baselines for Case 2 (Navier--Stokes).

This module provides two *true-solver* baselines that propagate ensemble members
with the real ``jax-cfd`` stochastic Navier--Stokes solver (NOT a learned prior):

* :func:`run_enkf_baseline`            -- (localized) spectral Ensemble Kalman
  Filter.
* :func:`run_particle_filter_baseline` -- bootstrap particle filter.

Both share the SAME truth + observations + sensor mask + observation noise ``R``
as the torch-side generative methods (they consume the driver's ``TruthAndObs``
and the torch ``LinearObservationOperator`` verbatim), and both return an
:class:`~cases.navier_stokes._ns_pipeline.AssimResult` that the existing
``compute_metrics`` consumes.

Design (see the task brief):
* The solver runs at 256^2 in RAW vorticity (the resolution the dataset was
  simulated at). For every observation comparison and for all metrics, the
  256^2 state is downsampled to 128^2 by spectral truncation (irfft2 of the low
  128x65 block of the rfft2), normalised (divide by the preprocesser std), then
  fed through the SAME torch ``H`` -- so the observation points are identical to
  the generative methods (the obs mask / operator is reused verbatim, never
  remapped to 256^2).
* The whole EnKF observation comparison therefore lives in NORMALISED 128^2
  space; ``obs_noise_std`` passed to the EnKF is in that same normalised space
  (= ``sqrt(variance)``, with ``variance`` the case-config ``R``).

jax device: GPU by default (jax-cfd runs natively on the GPU, where the corrected
5000-sub-step forecast is far faster than on CPU). Override with the env var
``ENKF_JAX_PLATFORM=cpu`` when a torch job needs the whole GPU. We disable XLA memory
pre-allocation so jax coexists with torch on the same device instead of grabbing ~75%
of GPU memory up front.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np
import torch

# --- Select the jax platform BEFORE importing jax (default GPU; set ----------- #
# --- ENKF_JAX_PLATFORM=cpu to force CPU, e.g. alongside a torch GPU job). ------ #
_JAX_PLATFORM = os.environ.get("ENKF_JAX_PLATFORM", "cuda").lower()
os.environ.setdefault("JAX_PLATFORMS", _JAX_PLATFORM)
# Don't let jax pre-grab the whole GPU (so torch can share it).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax  # noqa: E402

jax.config.update(
    "jax_platform_name", "gpu" if _JAX_PLATFORM in ("cuda", "gpu") else "cpu"
)
import jax.numpy as jnp  # noqa: E402

from jax_cfd_lib.ENKF import LocalizedSpectralEnKF, SpectralEnKF  # noqa: E402
from jax_cfd_lib.ETKF import LocalizedSpectralETKF  # noqa: E402
from jax_cfd_lib.navier_stokes_forward_model import (  # noqa: E402
    set_up_forward_model,
)

import jax_cfd.base.grids as grids  # noqa: E402

from cases.navier_stokes._ns_pipeline import AssimResult, TruthAndObs  # noqa: E402
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearObservationOperator,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Physical / solver constants (documented in bin/simulate_stochastic_ns_main.py)
# --------------------------------------------------------------------------- #
HIGH_RES = 256
LOW_RES = 128
DOMAIN = ((0.0, 2.0 * float(jnp.pi)), (0.0, 2.0 * float(jnp.pi)))
VISCOSITY = 1e-3
DRAG = 0.1
SMOOTH = True
# Forecast interval per assimilation step MUST equal the data/training interval so
# the EnKF/PF advance the same physical time the SI/FM prior was trained to predict.
# The dataset was generated (simulate_stochastic_ns_main.py) with REDUCED_DT = 0.5,
# HF_DT = 1e-4 => INNER_STEPS = REDUCED_DT / HF_DT = 5000 sub-steps per saved state.
# (A previous value of 100 advanced only 0.01 -- a 50x under-propagation that
# invalidated the EnKF/PF comparison; fixed.)
HF_DT = 1e-4
REDUCED_DT = 0.5  # physical time between consecutive observations (= data save interval)
INNER_STEPS = int(round(REDUCED_DT / HF_DT))  # = 5000; one forecast = one assimilation interval
STOCHASTIC_FORCING_SCALE = 1.0

# Normalisation: raw = normalised * NORM_STD (preprocesser base std, mean 0).
NORM_STD = 3.09969


# --------------------------------------------------------------------------- #
# Up/down-sample between 128^2 and 256^2 in PHYSICAL space (stride-2 subsample).
# --------------------------------------------------------------------------- #
#
# The 128^2 and 256^2 grids are co-located: the 128^2 grid point (i, j) is
# exactly the 256^2 grid point (2i, 2j). So down-sampling for the observation
# operator and for all metric comparisons is a plain stride-2 subsample
# ``field_256[::2, ::2]`` -- this reads the SAME physical grid locations the
# torch ``H`` reads (no interpolation, no spectral aliasing concerns). The
# 128->256 up-sample is needed only ONCE, to lift the IC into the solver grid;
# the even points (2i, 2j) hold the denormalised IC values exactly and the
# intervening points are filled by Fourier interpolation (zero-pad of the rfft2).
# Any up-sample whose even points equal the source leaves ``subsample2`` an exact
# left-inverse, which the obs-identity self-test verifies.


def subsample2(field_256: jnp.ndarray) -> jnp.ndarray:
    """256^2 physical field -> 128^2 physical field via stride-2 subsample."""
    return field_256[::2, ::2]


def downsample_physical_256_to_128(field_256: jnp.ndarray) -> jnp.ndarray:
    """256^2 physical field -> 128^2 physical field (alias for :func:`subsample2`)."""
    return subsample2(field_256)


def upsample_physical_128_to_256(field_128: jnp.ndarray) -> jnp.ndarray:
    """128^2 physical field -> 256^2 physical field by periodic bilinear fill.

    Even grid points ``(2i, 2j)`` hold the source values EXACTLY; the intervening
    odd points are filled by averaging the neighbouring even points (periodic
    bilinear interpolation). This guarantees ``subsample2(upsample(f)) == f``
    exactly, which the obs-identity self-test relies on. (Fourier zero-pad would
    perturb the node values whenever the source carries Nyquist energy.)
    """
    f = field_128
    # Place source at the even sublattice of a 256^2 grid.
    up = jnp.zeros((HIGH_RES, HIGH_RES), dtype=f.dtype)
    up = up.at[::2, ::2].set(f)
    # Horizontal fill: odd columns = mean of left/right even neighbours (periodic).
    f_right = jnp.roll(f, -1, axis=1)
    up = up.at[::2, 1::2].set(0.5 * (f + f_right))
    # Vertical fill: odd rows on even columns = mean of up/down even neighbours.
    f_down = jnp.roll(f, -1, axis=0)
    up = up.at[1::2, ::2].set(0.5 * (f + f_down))
    # Centre (odd, odd) = mean of the four diagonal even neighbours.
    f_diag = jnp.roll(jnp.roll(f, -1, axis=0), -1, axis=1)
    up = up.at[1::2, 1::2].set(0.25 * (f + f_right + f_down + f_diag))
    return up


# --------------------------------------------------------------------------- #
# torch <-> jax conversion.
# --------------------------------------------------------------------------- #


def _torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# jax observation operator wrapping the torch H (downsample + normalise).
# --------------------------------------------------------------------------- #


class TorchHObservationOperator:
    """jax-callable observation operator wrapping the torch ``H``.

    The EnKF/PF carry the state as a 256^2 ``rfft2`` spectrum. This operator maps
    such a spectrum to the observation vector by:
        256^2 spectrum -> 256^2 raw physical -> 128^2 raw physical (truncation)
        -> 128^2 NORMALISED physical (/NORM_STD) -> torch ``H`` (selection / pool).
    The result lives in the SAME normalised observation space as the torch-side
    generative methods, using the exact same obs points / operator matrix.

    Two entry points are exposed to match the EnKF base class' expectations:
    ``__call__(u_spectral_256)`` (used by :class:`SpectralEnKF`) and
    ``physical_observation_operator(u_physical_256)`` (used by
    :class:`LocalizedSpectralEnKF`, which works in physical space).
    """

    def __init__(self, obs_operator: LinearObservationOperator):
        self.obs_operator = obs_operator
        self.C, self.H, self.W = obs_operator.C, obs_operator.H, obs_operator.W
        # Dense observation matrix in jax (num_obs, C*H*W), normalised-space H.
        self._H = jnp.asarray(_torch_to_np(obs_operator.obs_matrix))
        # Observation coordinates in the 256^2 SOLVER frame, for the localized
        # EnKF's Gaspari-Cohn distance matrix. The 128^2 obs point (h, w) is the
        # co-located 256^2 point (2h, 2w). For the block-average super-res
        # operator every 128^2 cell contributes, so obs_coords spans the full
        # (sub-sampled) 128^2 grid lifted to 256^2 -- distances stay coherent.
        try:
            chw = _torch_to_np(obs_operator.obs_indices_c_h_w)  # (num_obs, 3)
            self.obs_coords = jnp.asarray(chw[:, 1:3] * 2)  # (num_obs, 2) in 256 frame
        except Exception:  # pragma: no cover - defensive
            self.obs_coords = None

    def _apply_H_on_norm128(self, field_norm_128: jnp.ndarray) -> jnp.ndarray:
        """Apply the torch H (as a dense matrix) to a normalised 128^2 field."""
        flat = field_norm_128.reshape(-1)
        return self._H @ flat

    def physical_observation_operator(
        self, u_physical_256: jnp.ndarray
    ) -> jnp.ndarray:
        """256^2 raw physical field -> normalised-128^2 observation vector."""
        raw_128 = downsample_physical_256_to_128(u_physical_256)
        norm_128 = raw_128 / NORM_STD
        return self._apply_H_on_norm128(norm_128)

    def __call__(self, u_spectral_256: jnp.ndarray) -> jnp.ndarray:
        """256^2 rfft2 spectrum -> normalised-128^2 observation vector."""
        u_physical_256 = jnp.fft.irfft2(u_spectral_256, s=(HIGH_RES, HIGH_RES))
        return self.physical_observation_operator(u_physical_256)


# --------------------------------------------------------------------------- #
# Solver setup + ensemble initialisation.
# --------------------------------------------------------------------------- #


def _build_dynamics():  # type: ignore[no-untyped-def]
    """Return a jitted ``pde(u_spectral_256, rng_key) -> u_spectral_256`` step."""
    grid = grids.Grid((HIGH_RES, HIGH_RES), domain=DOMAIN)
    forward_model = set_up_forward_model(
        compile=True,
        use_true_model=False,
        stochastic=True,
        grid=grid,
        viscosity=VISCOSITY,
        drag=DRAG,
        smooth=SMOOTH,
        hf_dt=HF_DT,
        inner_steps=INNER_STEPS,
        stochastic_forcing_scale=STOCHASTIC_FORCING_SCALE,
    )

    def pde(u_spectral: jnp.ndarray, rng_key: jax.random.PRNGKey) -> jnp.ndarray:
        u_final, _ = forward_model(u_spectral, rng_key)
        return u_final

    return pde


def _init_ensemble_spectral(
    init_base_norm_128: torch.Tensor,
    ensemble_size: int,
    key: jax.random.PRNGKey,
    perturb_std: float,
) -> jnp.ndarray:
    """Build the initial 256^2 spectral ensemble from the normalised IC.

    The IC ``init_base`` is the normalised 128^2 vorticity at the last history
    step. We denormalise (xNORM_STD), spectrally up-sample to 256^2, replicate
    across the ensemble, and add a small physical-space perturbation so the
    ensemble has non-zero initial spread.
    """
    raw_128 = jnp.asarray(_torch_to_np(init_base_norm_128)[0, 0]) * NORM_STD
    raw_256 = upsample_physical_128_to_256(raw_128)  # (256, 256)

    ensemble_physical = jnp.broadcast_to(
        raw_256[None], (ensemble_size, HIGH_RES, HIGH_RES)
    )
    if perturb_std > 0:
        noise = perturb_std * jax.random.normal(
            key, (ensemble_size, HIGH_RES, HIGH_RES)
        )
        ensemble_physical = ensemble_physical + noise

    return jax.vmap(jnp.fft.rfft2)(ensemble_physical)


def _spectral_ensemble_to_norm128_torch(
    ensemble_spectral: jnp.ndarray,
) -> torch.Tensor:
    """(E, 256, 129) spectrum -> torch (E, 1, 128, 128) NORMALISED field."""
    phys_256 = jax.vmap(lambda h: jnp.fft.irfft2(h, s=(HIGH_RES, HIGH_RES)))(
        ensemble_spectral
    )
    phys_128 = jax.vmap(downsample_physical_256_to_128)(phys_256)
    norm_128 = np.asarray(phys_128) / NORM_STD
    return torch.from_numpy(norm_128).float().unsqueeze(1)  # (E, 1, 128, 128)


# --------------------------------------------------------------------------- #
# EnKF baseline.
# --------------------------------------------------------------------------- #


def run_enkf_baseline(
    truth_obs: TruthAndObs,
    obs_operator: LinearObservationOperator,
    ensemble_size: int,
    variance: float,
    num_physical_steps: int,
    len_field_history: int,
    seed: int,
    localization_radius: Optional[float] = None,
    inflation: float = 1.0,
    model_noise_std: float = 0.0,
    init_perturb_std: float = 0.05,
    localization_type: str = "distance",
    corr_threshold: Optional[float] = None,
    corr_inflation_max: float = 4.0,
    corr_inflation_beta: float = 0.5,
) -> AssimResult:
    """Run a true-solver spectral EnKF on one NS trajectory.

    Propagates ``ensemble_size`` members with the real jax-cfd stochastic NS
    solver at 256^2, assimilating ``truth_obs.observations`` (normalised, shared
    with the generative methods) at every physical step after the history prefix.

    Returns an :class:`AssimResult` whose ``posterior_trajectory`` is
    ``[E, C, H, W, T]`` (normalised 128^2). The first ``len_field_history``
    columns hold the (broadcast) IC so the array is full-length; only columns
    ``>= len_field_history`` are scored by ``compute_metrics``.
    """
    obs_noise_std = float(variance) ** 0.5
    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)

    pde = _build_dynamics()
    jax_H = TorchHObservationOperator(obs_operator)

    grid_shape = (LOW_RES, LOW_RES)  # localization distances are in 128-grid points

    # Correlation-based localization (opt-in) ALSO requires the localized EnKF
    # class -- it is a local-analysis scheme, not a covariance Schur taper, so it
    # is selected whenever localization_type == "correlation" even if no distance
    # radius was given. ``localization_radius`` is then ignored by that path.
    use_localized = localization_radius is not None or localization_type == "correlation"
    if use_localized:
        enkf = LocalizedSpectralEnKF(
            grid_shape=(HIGH_RES, HIGH_RES),
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            localization_radius=float(localization_radius)
            if localization_radius is not None
            else 0.0,
            observation_operator=jax_H,
            real_space=True,
            adaptive_localization=False,
            localization_type=localization_type,
            corr_threshold=corr_threshold,
            corr_inflation_max=corr_inflation_max,
            corr_inflation_beta=corr_inflation_beta,
        )
    else:
        enkf = SpectralEnKF(
            grid_shape=(HIGH_RES, HIGH_RES),
            ensemble_size=ensemble_size,
            model_noise_std=model_noise_std,
            obs_noise_std=obs_noise_std,
            observation_operator=jax_H,
            real_space=True,
        )
    del grid_shape

    ensemble_spectral = _init_ensemble_spectral(
        truth_obs.init_base, ensemble_size, init_key, init_perturb_std
    )

    # observations[:, :, t] is the observation for physical step t (0-based). The
    # generative methods assimilate y at steps L .. T-1 (truth_obs.observations is
    # filled for all T but only [:, :, L:] are used). We mirror that here.
    obs_np = _torch_to_np(truth_obs.observations)[0]  # (num_obs, T)
    T = num_physical_steps
    C, H, W = obs_operator.C, obs_operator.H, obs_operator.W

    # Pre-fill the posterior trajectory with the (downsampled) ensemble IC so the
    # history columns are finite; assimilated columns overwrite L.. .
    posterior = torch.zeros(ensemble_size, C, H, W, T)
    init_norm = _spectral_ensemble_to_norm128_torch(ensemble_spectral)  # (E,1,128,128)
    for t in range(len_field_history):
        posterior[..., t] = init_norm

    n_assim = T - len_field_history
    start = time.time()
    for t in range(len_field_history, T):
        key, subkey = jax.random.split(key)
        obs_t = jnp.asarray(obs_np[:, t])
        ensemble_spectral, _ = enkf.assimilate(
            ensemble_spectral=ensemble_spectral,
            observations=obs_t,
            dynamics=pde,
            key=subkey,
            inflation=inflation,
        )
        analysis_norm = _spectral_ensemble_to_norm128_torch(ensemble_spectral)
        posterior[..., t] = analysis_norm

        mean_t = analysis_norm.mean(dim=0)  # (1, 128, 128)
        true_t = truth_obs.true_trajectory[0, ..., t]  # (1, 128, 128)
        rmse = float(torch.sqrt(torch.mean((mean_t - true_t) ** 2)))
        logger.info("[EnKF] step %d/%d  RMSE(norm)=%.4f", t, T - 1, rmse)
    elapsed = time.time() - start

    return AssimResult(
        posterior_trajectory=posterior,
        true_trajectory=truth_obs.true_trajectory,
        nfe_per_step=float("nan"),
        seconds_per_step=elapsed / max(n_assim, 1),
    )


# --------------------------------------------------------------------------- #
# LETKF baseline (Local Ensemble Transform Kalman Filter, true solver).
# --------------------------------------------------------------------------- #

# Default localization radius (in 128-grid points) used when the config passes
# None. LETKF is inherently a LOCAL filter, so -- unlike the EnKF -- there is no
# meaningful non-localized variant to fall back to; we pick a sensible default.
DEFAULT_LETKF_LOCALIZATION_RADIUS = 20.0


def run_letkf_baseline(
    truth_obs: TruthAndObs,
    obs_operator: LinearObservationOperator,
    ensemble_size: int,
    variance: float,
    num_physical_steps: int,
    len_field_history: int,
    seed: int,
    localization_radius: Optional[float] = None,
    inflation: float = 1.0,
    model_noise_std: float = 0.0,
    init_perturb_std: float = 0.05,
    localization_type: str = "distance",
    corr_threshold: Optional[float] = None,
    corr_inflation_max: float = 4.0,
    corr_inflation_beta: float = 0.5,
) -> AssimResult:
    """Run a true-solver Localized ETKF (LETKF) on one NS trajectory.

    Near-exact copy of :func:`run_enkf_baseline`, but the analysis is the
    deterministic local ensemble transform of :class:`LocalizedSpectralETKF`
    instead of the (stochastic) localized EnKF. The forecast (jax-cfd stochastic
    NS solver at 256^2), observation handling (normalised 128^2 obs space), and
    :class:`AssimResult` return are identical.

    LETKF is inherently localized, so ``localization_radius`` is REQUIRED: if the
    config passes ``None`` it is defaulted to
    :data:`DEFAULT_LETKF_LOCALIZATION_RADIUS` (in 128-grid points) rather than
    building a non-localized variant.
    """
    obs_noise_std = float(variance) ** 0.5
    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)

    pde = _build_dynamics()
    jax_H = TorchHObservationOperator(obs_operator)

    if localization_radius is None:
        localization_radius = DEFAULT_LETKF_LOCALIZATION_RADIUS

    letkf = LocalizedSpectralETKF(
        grid_shape=(HIGH_RES, HIGH_RES),
        ensemble_size=ensemble_size,
        model_noise_std=model_noise_std,
        obs_noise_std=obs_noise_std,
        localization_radius=float(localization_radius),
        observation_operator=jax_H,
        real_space=True,
        adaptive_localization=False,
        localization_type=localization_type,
        corr_threshold=corr_threshold,
        corr_inflation_max=corr_inflation_max,
        corr_inflation_beta=corr_inflation_beta,
    )

    ensemble_spectral = _init_ensemble_spectral(
        truth_obs.init_base, ensemble_size, init_key, init_perturb_std
    )

    obs_np = _torch_to_np(truth_obs.observations)[0]  # (num_obs, T)
    T = num_physical_steps
    C, H, W = obs_operator.C, obs_operator.H, obs_operator.W

    posterior = torch.zeros(ensemble_size, C, H, W, T)
    init_norm = _spectral_ensemble_to_norm128_torch(ensemble_spectral)
    for t in range(len_field_history):
        posterior[..., t] = init_norm

    n_assim = T - len_field_history
    start = time.time()
    for t in range(len_field_history, T):
        key, subkey = jax.random.split(key)
        obs_t = jnp.asarray(obs_np[:, t])
        ensemble_spectral, _ = letkf.assimilate(
            ensemble_spectral=ensemble_spectral,
            observations=obs_t,
            dynamics=pde,
            key=subkey,
            inflation=inflation,
        )
        analysis_norm = _spectral_ensemble_to_norm128_torch(ensemble_spectral)
        posterior[..., t] = analysis_norm

        mean_t = analysis_norm.mean(dim=0)  # (1, 128, 128)
        true_t = truth_obs.true_trajectory[0, ..., t]  # (1, 128, 128)
        rmse = float(torch.sqrt(torch.mean((mean_t - true_t) ** 2)))
        logger.info("[LETKF] step %d/%d  RMSE(norm)=%.4f", t, T - 1, rmse)
    elapsed = time.time() - start

    return AssimResult(
        posterior_trajectory=posterior,
        true_trajectory=truth_obs.true_trajectory,
        nfe_per_step=float("nan"),
        seconds_per_step=elapsed / max(n_assim, 1),
    )


# --------------------------------------------------------------------------- #
# EnSF baseline (Ensemble Score Filter, TRUE-SOLVER forecast).
# --------------------------------------------------------------------------- #


def run_ensf_baseline(
    truth_obs: TruthAndObs,
    obs_operator: LinearObservationOperator,
    ensemble_size: int,
    variance: float,
    num_physical_steps: int,
    len_field_history: int,
    seed: int,
    analysis_steps: int = 20,
    init_perturb_std: float = 0.05,
) -> AssimResult:
    """Run a true-solver Ensemble Score Filter (EnSF) on one NS trajectory.

    Same structure as :func:`run_enkf_baseline` -- the ensemble is forecast with
    the real jax-cfd stochastic NS solver at 256^2, init from the truth IC, and
    is autoregressive (the analysis re-seeds the next forecast). The ONLY
    difference from the EnKF is the ANALYSIS update: instead of the Kalman gain,
    we use the score-based analysis of
    :meth:`scisi.particle_filter.ensemble_score_filter.EnsembleScoreFilter._analysis_update`
    (empirical-Gaussian prior score + tempered Gaussian observation-likelihood
    score, integrated by a short reverse diffusion).

    Forecast <-> analysis coupling (the up/down-sampling, see module docstring):
      * forecast the spectral ensemble with the jax solver (256^2 rfft2);
      * downsample 256^2 -> 128^2 (irfft2 then ``subsample2``) and normalise
        (/NORM_STD) -> torch ``(E, 1, 128, 128)`` -- this is the score filter's
        forecast ensemble, in the SAME normalised 128^2 space as ``H`` and the
        observations;
      * run the torch score-based ``_analysis_update`` on that 128^2 ensemble;
      * the analysis ensemble IS the posterior for this step (recorded);
      * to forecast the next step, denormalise (xNORM_STD) + upsample 128^2 ->
        256^2 + rfft2 to re-seed the spectral ensemble (autoregressive feedback).

    We instantiate an :class:`EnsembleScoreFilter` purely to reuse its
    ``_analysis_update`` (its learned-prior ``run`` is NOT used here).
    """
    from scisi.particle_filter.ensemble_score_filter import EnsembleScoreFilter

    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)

    pde = _build_dynamics()
    jax_H = TorchHObservationOperator(obs_operator)
    score_filter = EnsembleScoreFilter(analysis_steps=int(analysis_steps))

    ensemble_spectral = _init_ensemble_spectral(
        truth_obs.init_base, ensemble_size, init_key, init_perturb_std
    )

    obs_np = _torch_to_np(truth_obs.observations)[0]  # (num_obs, T)
    T = num_physical_steps
    C, H, W = obs_operator.C, obs_operator.H, obs_operator.W

    posterior = torch.zeros(ensemble_size, C, H, W, T)
    init_norm = _spectral_ensemble_to_norm128_torch(ensemble_spectral)
    for t in range(len_field_history):
        posterior[..., t] = init_norm

    n_assim = T - len_field_history
    start = time.time()
    for t in range(len_field_history, T):
        # --- Forecast: propagate every member with the true solver. ---------- #
        key, prop_key = jax.random.split(key)
        prop_keys = jax.random.split(prop_key, ensemble_size)
        ensemble_spectral = jax.vmap(pde)(ensemble_spectral, prop_keys)

        # Downsample 256^2 -> normalised 128^2 torch tensor (E, 1, 128, 128).
        forecast_norm = _spectral_ensemble_to_norm128_torch(ensemble_spectral)

        # --- Analysis: score-based observation update (normalised 128^2). ---- #
        obs_t = torch.from_numpy(np.asarray(obs_np[:, t])).float()
        obs_t = obs_t.unsqueeze(0).expand(ensemble_size, -1)  # (E, N_y)
        analysis_norm = score_filter._analysis_update(
            forecast=forecast_norm,
            observations=obs_t,
            obs_operator=obs_operator,
            variance=variance,
            generator=gen,
        )  # (E, 1, 128, 128) normalised 128^2

        posterior[..., t] = analysis_norm

        # --- Autoregressive feedback: re-seed the spectral ensemble. --------- #
        # Denormalise + upsample 128^2 -> 256^2 + rfft2 for the next forecast.
        analysis_raw_128 = np.asarray(analysis_norm[:, 0]) * NORM_STD  # (E, 128, 128)
        analysis_raw_256 = jax.vmap(upsample_physical_128_to_256)(
            jnp.asarray(analysis_raw_128)
        )
        ensemble_spectral = jax.vmap(jnp.fft.rfft2)(analysis_raw_256)

        mean_t = analysis_norm.mean(dim=0)
        true_t = truth_obs.true_trajectory[0, ..., t]
        rmse = float(torch.sqrt(torch.mean((mean_t - true_t) ** 2)))
        logger.info("[EnSF] step %d/%d  RMSE(norm)=%.4f", t, T - 1, rmse)
    elapsed = time.time() - start

    return AssimResult(
        posterior_trajectory=posterior,
        true_trajectory=truth_obs.true_trajectory,
        nfe_per_step=float("nan"),
        seconds_per_step=elapsed / max(n_assim, 1),
    )


# --------------------------------------------------------------------------- #
# Bootstrap particle filter baseline.
# --------------------------------------------------------------------------- #


def _systematic_resample(
    weights: np.ndarray, key: jax.random.PRNGKey
) -> np.ndarray:
    """Systematic resampling: returns the resampled particle indices."""
    n = weights.shape[0]
    positions = (np.arange(n) + float(jax.random.uniform(key))) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # guard against fp drift
    return np.searchsorted(cumsum, positions).astype(np.int64)


def run_particle_filter_baseline(
    truth_obs: TruthAndObs,
    obs_operator: LinearObservationOperator,
    ensemble_size: int,
    variance: float,
    num_physical_steps: int,
    len_field_history: int,
    seed: int,
    init_perturb_std: float = 0.05,
) -> AssimResult:
    """Run a bootstrap particle filter with the true jax-cfd NS solver.

    Each particle is propagated with the real stochastic solver at 256^2, then
    weighted by ``N(y; H(downsample(x)), R)`` in normalised 128^2 space, and the
    particle set is resampled (systematic) every step. Weight degeneracy in this
    high-dimensional setting is expected -- this is an acceptable baseline result;
    the routine just needs to run and produce finite metrics.
    """
    obs_noise_std = float(variance) ** 0.5
    key = jax.random.PRNGKey(int(seed))
    key, init_key = jax.random.split(key)

    pde = _build_dynamics()
    jax_H = TorchHObservationOperator(obs_operator)

    particles_spectral = _init_ensemble_spectral(
        truth_obs.init_base, ensemble_size, init_key, init_perturb_std
    )

    obs_np = _torch_to_np(truth_obs.observations)[0]  # (num_obs, T)
    T = num_physical_steps
    C, H, W = obs_operator.C, obs_operator.H, obs_operator.W

    posterior = torch.zeros(ensemble_size, C, H, W, T)
    init_norm = _spectral_ensemble_to_norm128_torch(particles_spectral)
    for t in range(len_field_history):
        posterior[..., t] = init_norm

    n_assim = T - len_field_history
    start = time.time()
    for t in range(len_field_history, T):
        # Propagate every particle with the true solver.
        key, prop_key = jax.random.split(key)
        prop_keys = jax.random.split(prop_key, ensemble_size)
        particles_spectral = jax.vmap(pde)(particles_spectral, prop_keys)

        # Weight by the Gaussian observation likelihood in normalised 128^2 space.
        Hx = jax.vmap(jax_H)(particles_spectral)  # (E, num_obs)
        resid = np.asarray(Hx) - obs_np[None, :, t]  # (E, num_obs)
        log_w = -0.5 * np.sum(resid**2, axis=1) / (obs_noise_std**2)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum <= 0:
            w = np.ones(ensemble_size) / ensemble_size
        else:
            w = w / w_sum

        # Record the weighted particle set BEFORE resampling (the posterior
        # ensemble is the resampled set, so resample then store).
        key, rs_key = jax.random.split(key)
        idx = _systematic_resample(w, rs_key)
        particles_spectral = particles_spectral[jnp.asarray(idx)]

        analysis_norm = _spectral_ensemble_to_norm128_torch(particles_spectral)
        posterior[..., t] = analysis_norm

        mean_t = analysis_norm.mean(dim=0)
        true_t = truth_obs.true_trajectory[0, ..., t]
        rmse = float(torch.sqrt(torch.mean((mean_t - true_t) ** 2)))
        ess = float(1.0 / np.sum(w**2))
        logger.info(
            "[PF] step %d/%d  RMSE(norm)=%.4f  ESS=%.1f/%d",
            t,
            T - 1,
            rmse,
            ess,
            ensemble_size,
        )
    elapsed = time.time() - start

    return AssimResult(
        posterior_trajectory=posterior,
        true_trajectory=truth_obs.true_trajectory,
        nfe_per_step=float("nan"),
        seconds_per_step=elapsed / max(n_assim, 1),
    )


__all__ = [
    "run_enkf_baseline",
    "run_letkf_baseline",
    "run_ensf_baseline",
    "run_particle_filter_baseline",
    "TorchHObservationOperator",
    "subsample2",
    "upsample_physical_128_to_256",
    "downsample_physical_256_to_128",
]
