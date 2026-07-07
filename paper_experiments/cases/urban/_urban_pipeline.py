"""Urban (uDALES) assimilation pipeline helpers (case-local; not in common/).

This module owns the heavy lifting the :class:`UrbanRunner` delegates to and is a
close mirror of ``cases/navier_stokes/_ns_pipeline.py`` -- it loads the trained
prior + dataset, builds the per-scenario observation operator, runs one
autoregressive assimilation with a chosen sampler, and computes the urban metric
set on the posterior ensemble. The differences from the NS pipeline are all
driven by the urban case being GENERATIVE-ONLY and MULTI-CHANNEL with solid cells:

* the state is ``(u, v, w, thl)`` -- 4 channels (the trained UNet has
  ``in_channels=out_channels=4``); the temperature channel is ``thl`` (index 3).
  Accuracy is reported as a PER-VARIABLE RMSE: a velocity RMSE over the ``(u, v,
  w)`` channels and a temperature RMSE over ``thl`` (spec Section 6).
* a SOLID-CELL MASK (``data/udales/mask.npz``, ``mask==1`` = fluid / keep,
  ``mask==0`` = building interior) is excluded from EVERY metric (RMSE, CRPS,
  spread--skill) and -- for the sparse scenario -- from the sensor pool, so no
  sensor lands inside a building. The mask is the dataset's ``field_cond`` and is
  threaded into the model exactly as the NS history is.
* NO KL-at-points: urban has only a ground-truth STATE, not a ground-truth
  posterior, so there is no reference ensemble to draw and ``compute_metrics``
  does not emit ``kl_points`` (author decision, archive/PROJECT_HANDOFF.md §B.4).
* NO conventional / true-solver baselines: there is no in-repo CFD solver for the
  urban data, so EnKF / LETKF / PF / EnSF are excluded (the driver's
  ``URBAN_METHODS`` already omits them). Every wired method shares the learned
  prior, so the same ``build_posterior`` dispatch as NS is reused verbatim.

The trained priors live under ``checkpoints/udales/`` and -- unlike the NS case --
there ARE dedicated SI and FM checkpoints (see ``configs/case/urban.yaml``), so
the FM prior is loaded from its own ``model.pth`` (no SI-drift reuse fallback).
The mean/std normalisation is the per-channel ``preprocesser`` stored in each
checkpoint's ``config.yaml`` (4 means/stds, with the ~295 K temperature mean on
channel 3); ``load_prior`` reads it the same way the NS loader does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from cases.navier_stokes._ns_pipeline import (
    AssimResult,
    LoadedPrior,
    TruthAndObs,
    attach_nfe_counter,
    build_posterior,
    load_prior,
    run_assimilation,
)
from scisi.likelihood_models.observation_operators import LinearObservationOperator
from scisi.metrics.accuracy import ensemble_mean_rmse
from scisi.metrics.calibration import crps, spread_skill

logger = logging.getLogger(__name__)

# Channel layout of the uDALES state, matching scisi.data.datasets.UDalesDataset
# (``torch.stack([u, v, w, thl])``). Velocity is the first three channels; the
# temperature variable ``thl`` (potential temperature, ~295 K) is the last.
VELOCITY_CHANNELS: tuple[int, ...] = (0, 1, 2)
TEMPERATURE_CHANNELS: tuple[int, ...] = (3,)


# --------------------------------------------------------------------------- #
# Model + dataset loading -- reuse the NS loader verbatim.
#
# ``load_prior`` is generic over the project: it reads ``case_cfg.project``
# (``udales`` here), resolves ``checkpoints.si_run`` / ``checkpoints.fm_run``,
# instantiates each checkpoint's own ``model`` + ``preprocesser`` + ``test_data``
# from its ``config.yaml``, and loads the ``model.pth`` weights. The urban
# checkpoints carry dedicated SI and FM ``model.pth`` files, so both priors load
# their own trained weights (the SI-drift-reuse fallback only triggers when
# ``fm_run`` is null, which is not the case for urban).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Solid-cell mask.
# --------------------------------------------------------------------------- #


def fluid_keep_mask(
    prior: LoadedPrior, num_channels: int, device: str
) -> torch.Tensor:
    """Return the boolean fluid keep-mask ``[C, H, W]`` (``True`` = keep).

    The dataset stores the solid-cell mask as a ``[1, H, W]`` float tensor with
    ``1`` = fluid and ``0`` = solid (building interior). All metrics take a
    boolean KEEP-mask where ``True`` marks a cell to score, so fluid -> ``True``.
    The single spatial mask is broadcast over the ``num_channels`` state channels
    so the same fluid geometry excludes solid cells in every variable.

    ``num_channels`` is the STATE channel count (4 = ``u, v, w, thl``), taken
    from the loaded test trajectory by the caller -- NOT
    ``test_dataset.num_channels`` (that attribute is hardcoded to 5 in
    ``UDalesDataset`` but only 4 channels are actually stacked; the trained UNet
    has ``in_channels=4``).
    """
    mask = prior.test_dataset.mask  # [1, H, W] float, 1 = fluid
    keep = mask.to(torch.bool)  # [1, H, W]
    return keep.expand(num_channels, -1, -1).to(device)


# --------------------------------------------------------------------------- #
# Observation operator per scenario (seeded, shared across methods).
# --------------------------------------------------------------------------- #


def build_obs_operator(
    scenario_cfg: Any,
    data_size: tuple[int, int, int],
    mask_seed: int,
    fluid_mask: Optional[torch.Tensor] = None,
) -> LinearObservationOperator:
    """Build the scenario's observation operator with a shared seeded mask.

    Mirrors ``_ns_pipeline.build_obs_operator`` but, for the SPARSE scenario,
    restricts the sensor pool to FLUID grid cells: a building-interior cell is
    not a physically plausible sensor location and is excluded from every metric
    anyway, so it must not consume a sensor budget. The block-average super-res
    operator is unchanged (it pools the whole grid; solid cells are masked out of
    the metrics downstream).

    ``data_size`` is the model-grid ``(C, H, W)``; ``fluid_mask`` is the boolean
    ``[C, H, W]`` keep-mask from :func:`fluid_keep_mask` (only consulted for the
    sparse operator).
    """
    from omegaconf import OmegaConf

    obs_cfg = OmegaConf.to_container(scenario_cfg.obs_operator, resolve=True)
    obs_cfg.pop("_target_", None)

    if obs_cfg.get("type") == "random":
        obs_cfg["seed"] = int(mask_seed)
        operator = LinearObservationOperator(data_size=tuple(data_size), **obs_cfg)
        if fluid_mask is not None:
            _restrict_sensors_to_fluid(
                operator, fluid_mask, percent_obs=float(obs_cfg["percent_obs"]),
                seed=int(mask_seed),
            )
        return operator

    return LinearObservationOperator(data_size=tuple(data_size), **obs_cfg)


def _restrict_sensors_to_fluid(
    operator: LinearObservationOperator,
    fluid_mask: torch.Tensor,
    percent_obs: float,
    seed: int,
) -> None:
    """Re-draw the sparse operator's sensors within the fluid cells, in place.

    The default ``random`` operator permutes ALL ``C*H*W`` dofs; here we restrict
    the permutation to the fluid dofs (``fluid_mask == True``) so no sensor lands
    inside a building. The sensor count is ``percent_obs`` of the FULL grid (the
    same budget as NS / the scenario config), placed entirely in fluid cells.
    Seeded by the same ``mask_seed`` as the scenario so the mask is identical
    across methods (reproducibility Section 9). Rebuilds ``obs_matrix`` /
    ``obs_indices`` / ``num_obs`` to match the selection-operator contract.
    """
    fluid_flat = fluid_mask.reshape(-1).to(torch.bool).cpu()
    fluid_dofs = torch.nonzero(fluid_flat).reshape(-1)
    num_obs = int(operator.num_dofs * percent_obs)
    num_obs = min(num_obs, int(fluid_dofs.numel()))

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(int(fluid_dofs.numel()), generator=generator)
    obs_indices = torch.sort(fluid_dofs[perm[:num_obs]]).values

    obs_matrix = torch.zeros(num_obs, operator.num_dofs)
    obs_matrix[torch.arange(num_obs), obs_indices] = 1.0

    operator.obs_indices = obs_indices
    operator.num_obs = num_obs
    operator.obs_matrix = obs_matrix
    operator._obs_matrix_cache = None  # invalidate the device cache (load_mask peer)


# --------------------------------------------------------------------------- #
# One trajectory's truth + observations (seeded, shared across methods).
#
# Reuse the NS ``prepare_truth_and_obs`` verbatim: it is already multi-channel
# (it normalises the whole ``[1, C, H, W, T]`` trajectory through the
# preprocesser, which carries the per-channel mean/std) and already threads
# ``field_cond`` (the urban mask) through the preprocesser. The urban
# preprocesser sets ``field_cond`` mean/std to 0/1, so the mask passes through
# unchanged and reaches the UNet as the solid-cell conditioning channel.
# --------------------------------------------------------------------------- #

from cases.navier_stokes._ns_pipeline import prepare_truth_and_obs  # noqa: E402


# --------------------------------------------------------------------------- #
# Metrics on the posterior ensemble (urban set; NO KL).
# --------------------------------------------------------------------------- #


def _channel_keep_mask(
    fluid_mask: torch.Tensor, channels: tuple[int, ...]
) -> torch.Tensor:
    """Boolean keep-mask ``[C, H, W]`` that is fluid on ``channels`` else False.

    Used to score a single variable group (velocity or temperature) over fluid
    cells only: cells outside ``channels`` are dropped (``False``) and solid
    cells within ``channels`` are dropped too.
    """
    keep = torch.zeros_like(fluid_mask, dtype=torch.bool)
    for c in channels:
        keep[c] = fluid_mask[c]
    return keep


def compute_metrics(
    result: AssimResult,
    obs_operator: LinearObservationOperator,
    fluid_mask: torch.Tensor,
    len_field_history: int,
) -> dict[str, float]:
    """Compute the urban metric set, averaged over assimilation steps.

    Returns a dict keyed by the canonical metric strings: ``rmse`` (all-channel,
    fluid only), ``rmse_velocity`` (``u, v, w`` channels, fluid only),
    ``rmse_temperature`` (``thl`` channel, fluid only), ``crps``,
    ``crps_observed`` / ``crps_unobserved`` (the observed/unobserved split, both
    fluid only), ``spread_skill`` (reported as ``|1-ratio|``), plus ``nfe`` /
    ``seconds``. NO ``kl_points`` -- urban has no ground-truth posterior.

    Solid cells (``fluid_mask == False``) are excluded from EVERY metric. The
    observed/unobserved split is taken on the fluid cells only: a cell is
    "observed" if it is both fluid AND a sensor location; "unobserved" if fluid
    and not a sensor. (For the block-average super-res operator every cell feeds
    an observation, so the unobserved set is empty and ``crps_unobserved`` is
    NaN, as in NS.)
    """
    post = result.posterior_trajectory  # [E, C, H, W, T]
    true = result.true_trajectory  # [1, C, H, W, T]
    T = post.shape[-1]
    E = post.shape[0]
    C, H, W = post.shape[1], post.shape[2], post.shape[3]

    fluid = fluid_mask.to(post.device).to(torch.bool)  # [C, H, W]

    # Observed-cell mask on the grid, intersected with the fluid keep-mask so the
    # observed/unobserved split scores only fluid cells (solid cells are never
    # scored). ``obs_indices_on_grid`` marks sensor cells (sparse) / all cells
    # (super-res).
    obs_grid = obs_operator.obs_indices_on_grid.reshape(C, H, W).to(torch.bool)
    obs_grid = obs_grid.to(post.device)
    obs_keep = fluid & obs_grid  # observed AND fluid
    unobs_keep = fluid & (~obs_grid)  # unobserved AND fluid
    has_obs = bool(obs_keep.any())
    has_unobs = bool(unobs_keep.any())

    vel_keep = _channel_keep_mask(fluid, VELOCITY_CHANNELS)
    temp_keep = _channel_keep_mask(fluid, TEMPERATURE_CHANNELS)

    rmse_steps: list[float] = []
    rmse_vel_steps: list[float] = []
    rmse_temp_steps: list[float] = []
    crps_steps: list[float] = []
    crps_obs_steps: list[float] = []
    crps_unobs_steps: list[float] = []
    ss_steps: list[float] = []

    # Only score the assimilated steps (skip the seeded history prefix).
    for t in range(len_field_history, T):
        ens_t = post[..., t].reshape(E, C, H, W)  # [E, C, H, W]
        true_t = true[0, ..., t]  # [C, H, W]

        rmse_steps.append(float(ensemble_mean_rmse(ens_t, true_t, mask=fluid)))
        rmse_vel_steps.append(
            float(ensemble_mean_rmse(ens_t, true_t, mask=vel_keep))
        )
        rmse_temp_steps.append(
            float(ensemble_mean_rmse(ens_t, true_t, mask=temp_keep))
        )
        crps_steps.append(float(crps(ens_t, true_t, mask=fluid)))
        if has_obs:
            crps_obs_steps.append(float(crps(ens_t, true_t, mask=obs_keep)))
        if has_unobs:
            crps_unobs_steps.append(float(crps(ens_t, true_t, mask=unobs_keep)))
        # spread-skill is undefined for a single-member ensemble (needs E >= 2).
        if E >= 2:
            ss = spread_skill(ens_t, true_t, mask=fluid)
            ss_steps.append(float(ss["deviation"]))

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "rmse": _mean(rmse_steps),
        "rmse_velocity": _mean(rmse_vel_steps),
        "rmse_temperature": _mean(rmse_temp_steps),
        "crps": _mean(crps_steps),
        "crps_observed": _mean(crps_obs_steps),
        "crps_unobserved": _mean(crps_unobs_steps),
        "spread_skill": _mean(ss_steps),
        "nfe": result.nfe_per_step,
        "seconds": result.seconds_per_step,
        # Per-(assimilation-)step metric curves (one value per scored step), so
        # the driver can persist metric-vs-time curves for every trajectory
        # without saving the raw ensembles. Mirrors the NS pipeline.
        "per_step": {
            "rmse": list(rmse_steps),
            "rmse_velocity": list(rmse_vel_steps),
            "rmse_temperature": list(rmse_temp_steps),
            "crps": list(crps_steps),
            "crps_observed": list(crps_obs_steps),
            "crps_unobserved": list(crps_unobs_steps),
            "spread_skill": list(ss_steps),
        },
    }


__all__ = [
    "LoadedPrior",
    "TruthAndObs",
    "AssimResult",
    "load_prior",
    "attach_nfe_counter",
    "build_posterior",
    "run_assimilation",
    "prepare_truth_and_obs",
    "build_obs_operator",
    "fluid_keep_mask",
    "compute_metrics",
    "VELOCITY_CHANNELS",
    "TEMPERATURE_CHANNELS",
]
