"""Shared helpers for the Phase 0 diagnostics (D1-D7).

Contents:

* ``load_model``: load a solver (SI/FM) or map checkpoint with its config, and
  ``describe_model``: the settings Phase 0 must record per checkpoint.
* ``load_trajectories``: whole trajectories in archive units from a dataset
  config, with an optional ``.npy`` cache (loading the NS ``data.npz`` costs
  minutes, so jobs cache the slices they need once).
* SI steppers: left-point Euler-Maruyama (as the existing solver) and Euler
  drift with exact integrated noise, both optionally sharing one noise draw
  across the batch.
* ``SharedBrownianPath`` / ``ZeroFeaturePath``: wrappers that give a map one
  Brownian path for every batch member, or hide the path from the network.
* ``sample_next``: one physical forecast step for any model with either the
  solver ``sample`` or the map ``sample`` interface.

Everything here works on ``[B, C, *spatial]`` states and ``[B, C, *spatial, L]``
histories in the model's normalised coordinates, unless stated otherwise.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scisi.bin.main_map_model_train import load_teacher
from scisi.models.base_map_model import BaseMapModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant

# Validation split of the NS archive (whole trajectories). Test is 180-199 and
# must not be used for anything in Phase 0.
NS_TRAIN = (0, 160)
NS_VALIDATION = (160, 180)


def finite_or_none(value):
    """Recursively replace non-finite floats by ``None`` (strict JSON)."""
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(finite_or_none(value), indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(tmp, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------- loading


def load_model(checkpoint_dir, device="cpu"):
    """Return ``(model, config)`` for a solver or map checkpoint (eval mode)."""
    model = load_teacher(checkpoint_dir, device)
    config = OmegaConf.load(Path(checkpoint_dir) / "config.yaml")
    return model.eval(), config


def describe_model(model, config):
    """Record what Phase 0 must state per checkpoint (HANDOVER section 3)."""
    interpolation = getattr(model, "interpolation", None)
    trainer = config.get("trainer") or {}
    return {
        "model_class": type(model).__name__,
        "interpolation": type(interpolation).__name__ if interpolation else None,
        "gamma_multiplier": (
            float(interpolation.gamma_multiplier)
            if hasattr(interpolation, "gamma_multiplier")
            else None
        ),
        "beta": (
            "t^2"
            if type(interpolation).__name__.startswith("Quadratic")
            else "t" if type(interpolation).__name__.startswith("Linear") else None
        ),
        "brownian_features": getattr(model, "brownian_features", None),
        "num_path_features": getattr(model, "num_path_features", None),
        "training_mode": (
            str(trainer.get("off_diagonal"))
            if hasattr(trainer, "get") and trainer.get("off_diagonal") is not None
            else "solver (diagonal drift only)"
        ),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


def preprocessing_affine(config):
    """``(mean, std)`` float64 arrays of the target normalisation, per channel."""
    prep = config.get("preprocesser")
    if prep is None:
        return np.zeros(1), np.ones(1)
    mean = np.asarray(prep.target.mean, dtype=np.float64)
    std = np.asarray(prep.target.std, dtype=np.float64)
    base_mean = np.asarray(prep.base.mean, dtype=np.float64)
    base_std = np.asarray(prep.base.std, dtype=np.float64)
    if not (np.array_equal(mean, base_mean) and np.array_equal(std, base_std)):
        raise ValueError("Autoregressive diagnostics need equal base/target scaling")
    return mean, std


def load_trajectories(config, ids, cache_dir=None):
    """Whole trajectories ``[n, C, *spatial, T]`` (float32, archive units).

    ``config`` is a run config whose ``test_data`` entry describes the dataset;
    only the trajectory range is overridden. With ``cache_dir`` the slice is
    stored as ``.npy`` keyed by the dataset file hash prefix and the range.
    """
    lo, hi = int(ids[0]), int(ids[1])
    data_config = config.test_data.copy()
    data_config.tajectories_ids = f"({lo}, {hi})"
    data_config.train_or_test = "test"
    cache = None
    if cache_dir is not None:
        files = "_".join(Path(f).stem for f in data_config.files)
        cache = Path(cache_dir) / f"{files}_traj{lo:03d}_{hi:03d}.npy"
        if cache.exists():
            return torch.from_numpy(np.load(cache, mmap_mode="r").copy())
    dataset = instantiate(data_config, preprocesser=None)
    data = torch.stack([dataset[i]["x"] for i in range(len(dataset))]).float()
    if cache is not None:
        atomic_npy(cache, data.numpy())
    return data


def normalise(x, mean, std, channel_axis=1):
    """Archive units -> model units with per-channel affine ``(mean, std)``."""
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    m = torch.as_tensor(mean, dtype=x.dtype).reshape(shape)
    s = torch.as_tensor(std, dtype=x.dtype).reshape(shape)
    return (x - m) / s


# -------------------------------------------------------------- SI steppers


def integrated_variance(diffusion, s, t):
    """``int_s^t diffusion(r)^2 dr`` by 3-point Gauss-Legendre quadrature.

    Exact when ``diffusion`` is a polynomial of degree <= 2 in ``r`` (the
    repository's ``gamma = a (1 - t)`` is linear). ``s``/``t`` are tensors.
    """
    nodes = (-math.sqrt(0.6), 0.0, math.sqrt(0.6))
    weights = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    half, mid = 0.5 * (t - s), 0.5 * (t + s)
    total = 0.0
    for node, weight in zip(nodes, weights):
        total = total + weight * diffusion(mid + half * node) ** 2
    return half * total


def make_si_stepper(noise="exact", shared=False, generator=None):
    """Stepper for ``FollmerStochasticInterpolant.sample``.

    ``noise="em"``: ``x + b dt + gamma(t) sqrt(dt) xi`` (the existing left-point
    Euler-Maruyama solver; its endpoint variance is inflated by about
    ``1/(2n)`` relative to ``a^2/3`` at zero drift). ``noise="exact"``: the
    same Euler drift plus the exact integrated noise
    ``N(0, int_t^{t+dt} gamma^2)``. With ``shared=True`` one Gaussian draw is
    used for every batch member (same-noise diagnostics).
    """
    if noise not in ("em", "exact"):
        raise ValueError("noise must be 'em' or 'exact'")

    def step(
        drift_model,
        diffusion_term,
        x,
        t,
        dt,
        field_history=None,
        field_cond=None,
        pars_cond=None,
        mask=None,
    ):
        shape = (1, *x.shape[1:]) if shared else x.shape
        xi = torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
        drift = drift_model(x, t, field_history, field_cond, pars_cond)
        if noise == "em":
            # Same floating-point order as sde_solvers.euler_maruyama_step.
            return x + drift * dt + diffusion_term(t) * (xi * torch.sqrt(dt))
        scale = integrated_variance(diffusion_term, t, t + dt).sqrt()
        return x + drift * dt + scale * xi

    return step


# ------------------------------------------------------------ map wrappers


def _single_time(t):
    value = torch.as_tensor(t, dtype=torch.float64).reshape(-1)
    if not torch.all(value == value[0]):
        raise ValueError("A shared path needs one common time for the batch")
    return value[:1]


class SharedBrownianPath:
    """Present one Brownian path (batch 1) to a map evaluated on batch ``B``.

    Every member then sees the identical realisation: increments and path
    features are those of the wrapped path, expanded over the batch. Only
    common (scalar) query times are allowed.
    """

    def __init__(self, path, batch_size):
        if path.shape[0] != 1:
            raise ValueError("Wrap a path with batch size 1")
        self.path = path
        self.batch_size = batch_size
        self.shape = (batch_size, *path.shape[1:])

    def _expand(self, value):
        return value.expand(self.batch_size, *value.shape[1:])

    def value(self, t):
        return self._expand(self.path.value(_single_time(t)))

    def increment(self, s, t):
        return self._expand(self.path.increment(_single_time(s), _single_time(t)))

    def features(self, s, t, num_features):
        return self._expand(
            self.path.features(_single_time(s), _single_time(t), num_features)
        )


class ZeroFeaturePath:
    """Keep the path's explicit increments but feed zero path features (D3)."""

    def __init__(self, path):
        self.path = path
        self.shape = path.shape

    def value(self, t):
        return self.path.value(t)

    def increment(self, s, t):
        return self.path.increment(s, t)

    def features(self, s, t, num_features):
        f = self.path.features(s, t, num_features)
        return torch.zeros_like(f)


# --------------------------------------------------------------- sampling


def model_kind(model):
    if isinstance(model, FollmerStochasticInterpolant):
        return "si"
    if isinstance(model, BaseMapModel):
        return "map"
    raise TypeError(f"Unsupported model type {type(model).__name__}")


@torch.no_grad()
def sample_next(
    model,
    history,
    num_steps,
    *,
    si_noise="exact",
    shared=False,
    generator=None,
    brownian_path=None,
    path_wrapper=None,
    return_path=False,
    base=None,
):
    """One physical step for every history row: ``[B,C,*s,L] -> [B,C,*s]``.

    Solvers integrate ``num_steps`` steps with :func:`make_si_stepper`. Maps
    use ``num_steps`` equal sub-intervals of one path; ``shared=True`` gives
    every row the same path (or the same Gaussian base for flow maps).
    ``path_wrapper`` (e.g. :class:`ZeroFeaturePath`) wraps the path before
    use. With ``return_path`` the (unwrapped) path object is returned too.
    ``base`` fixes the Gaussian base of a flow map (ignored by solvers).
    """
    device = model.device
    dtype = next(model.parameters()).dtype
    history = history.to(device, dtype)
    batch = history.shape[0]
    if model_kind(model) == "si":
        stepper = make_si_stepper(si_noise, shared, generator)
        out = model.sample(
            base=history[..., -1],
            field_history=history,
            batch_size=batch,
            num_steps=num_steps,
            stepper=stepper,
        )
        return (out, None) if return_path else out
    if base is not None:
        base = base.to(device, dtype)
    path = brownian_path
    if path is None:
        x0 = history[..., -1]
        if shared:
            if model.gaussian_base:
                base = torch.randn(
                    (1, *x0.shape[1:]), device=device, dtype=dtype, generator=generator
                ).expand_as(x0)
            single = model.make_brownian_path(x0[:1], generator)
            path = None if single is None else SharedBrownianPath(single, batch)
        else:
            path = model.make_brownian_path(x0, generator)
    used = path_wrapper(path) if (path_wrapper is not None and path is not None) else path
    out = model.sample(
        field_history=history,
        batch_size=batch,
        num_steps=num_steps,
        base=base,
        brownian_path=used,
        generator=generator,
    )
    return (out, path) if return_path else out


@torch.no_grad()
def rollout(model, history, num_steps, leads, *, si_noise="exact", generator=None):
    """Autonomous rollout ``[B,C,*s,L] -> [B,C,*s,leads]`` (normalised units)."""
    states = []
    for _ in range(leads):
        y = sample_next(
            model, history, num_steps, si_noise=si_noise, generator=generator
        ).to(history.dtype)
        states.append(y)
        history = torch.cat([history[..., 1:], y.unsqueeze(-1).to(history.device)], -1)
    return torch.stack(states, -1)


def make_generator(device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def validation_cases(trajectories, origins, history_length):
    """``(trajectory_index, origin)`` pairs; origin = index of the first target."""
    cases = []
    for i in range(trajectories):
        for origin in origins:
            if origin < history_length:
                raise ValueError("origin must leave a full history")
            cases.append((i, origin))
    return cases
