#!/usr/bin/env python
"""NS instances of :class:`scisi.metrics.QuadraticBudget` and their archive check.

Builds the enstrophy (``W = Id``) and energy (``W = (-Lap)^{-1}``, zero on
the mean) budgets with ``D = drag - viscosity Lap`` from
``research/ito_maps/configs/ns_budget.yaml``. All operators are Fourier
multipliers on the periodic ``[0, L]^2`` grid; inputs are physical vorticity
``[B, 1, H, W]``. The inner product is the area mean (no domain area).

``validate`` (CLI) applies the budget to consecutive archive frames on the
training and validation splits (never test) and reports:

* stationary balance ``mean <w, W D w>`` against the injection rate ``I``;
* per-pair residual ``r = q_+(y) - q_-(c) - Delta I``: mean (and its
  trajectory-bootstrap 90% band) relative to ``Delta I``, its correlation with
  ``q(c)``, and its std against the predicted martingale std.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from scisi.metrics import QuadraticBudget

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/ns_budget.yaml"


def wavenumbers(height, width, length, dtype=torch.float64, device="cpu"):
    """Angular wavenumbers ``(kx [H,1], ky [1,W])`` of the full FFT grid."""
    scale = 2 * math.pi / length
    kx = torch.fft.fftfreq(height, 1.0 / height, dtype=dtype, device=device) * scale
    ky = torch.fft.fftfreq(width, 1.0 / width, dtype=dtype, device=device) * scale
    return kx[:, None], ky[None, :]


def fourier_multiplier(symbol):
    """Linear operator ``u -> ifft2(symbol * fft2(u))`` on the last two axes."""

    def apply(u):
        return torch.fft.ifft2(torch.fft.fft2(u) * symbol.to(u.dtype)).real

    return apply


def build_ns_budgets(config=None, height=128, width=128, device="cpu"):
    """``{"enstrophy": QuadraticBudget, "energy": QuadraticBudget}`` for NS."""
    cfg = OmegaConf.load(config or DEFAULT_CONFIG)
    kx, ky = wavenumbers(height, width, cfg.domain_length, device=device)
    k2 = kx**2 + ky**2
    dissipation = fourier_multiplier(cfg.drag + cfg.viscosity * k2)
    inverse_laplacian = torch.where(k2 > 0, 1.0 / torch.where(k2 > 0, k2, 1.0), 0.0)
    weights = {"enstrophy": torch.ones_like(k2), "energy": inverse_laplacian}
    scale = 2 * math.pi / cfg.domain_length
    forced = [(int(a), int(b)) for a, b in cfg.forcing_modes]
    budgets = {}
    for name, symbol in weights.items():
        w_forced = [float(symbol[a % height, b % width]) for a, b in forced]
        trace = cfg.forcing_mode_variance * sum(w_forced)
        rate = 0.5 * cfg.forcing_scale**2 * trace

        def quadratic_variation(u, symbol=symbol):
            # eps^2 <u, W Q W u>: per forced mode, |(W u)_k|^2 (forward FFT)
            # times the per-mode coefficient variance (cos and sin parts).
            wu_hat = torch.fft.fft2(fourier_multiplier(symbol)(u), norm="forward")
            total = sum(wu_hat[..., a % height, b % width].abs() ** 2 for a, b in forced)
            return cfg.forcing_scale**2 * total.flatten(1).sum(-1)

        budgets[name] = QuadraticBudget(
            weight=None if name == "enstrophy" else fourier_multiplier(symbol),
            dissipation=dissipation,
            injection_rate=rate,
            step=float(cfg.snapshot_spacing),
            forcing_quadratic_variation=quadratic_variation,
        )
    budgets["_config"] = OmegaConf.to_container(cfg)
    budgets["_scale"] = scale
    return budgets


def frame_terms(budget, frames, chunk=512):
    """``(q, <w,WDw>, q_+, q_-, qv)`` per frame for ``[N, 1, H, W]`` fields."""
    out = {k: [] for k in ("q", "dis", "qp", "qm", "qv")}
    for i in range(0, len(frames), chunk):
        x = frames[i : i + chunk].double()
        q, d = budget.quadratic(x), budget.dissipation_rate(x)
        out["q"].append(q)
        out["dis"].append(d)
        out["qp"].append(q + 0.5 * budget.step * d)
        out["qm"].append(q - 0.5 * budget.step * d)
        out["qv"].append(budget.martingale_variance(x))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def trajectory_bootstrap(per_traj, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(per_traj), size=(n_boot, len(per_traj)))
    boot = per_traj[draws].mean(1)
    return float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))


def validate_archive(budget, trajectories):
    """Budget statistics on consecutive frames of ``[n, 1, H, W, T]`` archive data."""
    n, T = trajectories.shape[0], trajectories.shape[-1]
    frames = trajectories.permute(0, 4, 1, 2, 3).reshape(n * T, *trajectories.shape[1:-1])
    terms = {k: v.reshape(n, T) for k, v in frame_terms(budget, frames).items()}
    inj = budget.injection
    r = terms["qp"][:, 1:] - terms["qm"][:, :-1] - inj
    rel = r / inj
    per_traj = rel.mean(1)
    lo, hi = trajectory_bootstrap(per_traj)
    predicted_std = float(np.sqrt(0.5 * (terms["qv"][:, 1:] + terms["qv"][:, :-1]).mean() * budget.step))
    stationary = terms["dis"].mean()
    return {
        "num_trajectories": int(n),
        "num_pairs": int(r.size),
        "mean_quadratic": float(terms["q"].mean()),
        "stationary_dissipation_mean": float(stationary),
        "injection_rate": budget.injection_rate,
        "stationary_balance_ratio": float(stationary / budget.injection_rate),
        "injection_per_step": inj,
        "residual_mean": float(r.mean()),
        "residual_mean_over_injection": float(rel.mean()),
        "residual_mean_over_injection_q05_q95": [lo, hi],
        "residual_mean_over_injection_trajectory_se": float(per_traj.std(ddof=1) / np.sqrt(n)),
        "residual_std": float(r.std()),
        "predicted_martingale_std": predicted_std,
        "residual_std_over_predicted": float(r.std() / predicted_std),
        "corr_residual_with_q_prev": float(np.corrcoef(r.ravel(), terms["q"][:, :-1].ravel())[0, 1]),
        "corr_residual_with_dissipation_prev": float(
            np.corrcoef(r.ravel(), terms["dis"][:, :-1].ravel())[0, 1]
        ),
    }


def main(argv=None):
    from phase0_common import NS_TRAIN, NS_VALIDATION, atomic_json, load_trajectories

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint-dir", required=True, help="supplies the dataset config")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--data-cache")
    p.add_argument("--output", required=True)
    p.add_argument("--max-trajectories", type=int, help="smoke tests only")
    args = p.parse_args(argv)
    run_config = OmegaConf.load(Path(args.checkpoint_dir) / "config.yaml")
    result = {
        "diagnostic": "WP7 budget closure on the data archive (train and validation only)",
        "argv": sys.argv if argv is None else argv,
        "budget_config": OmegaConf.to_container(OmegaConf.load(args.config)),
        "derivation": "r = q_+(y) - q_-(c) - Delta*I, q_pm(x) = 1/2 <x, W(Id +- Delta D) x>, area-mean inner product, trapezoid rule",
        "splits": {},
    }
    for split, ids in (("train", NS_TRAIN), ("validation", NS_VALIDATION)):
        hi = ids[1] if args.max_trajectories is None else ids[0] + args.max_trajectories
        t0 = time.time()
        data = load_trajectories(run_config, (ids[0], hi), args.data_cache)
        budgets = build_ns_budgets(args.config, *data.shape[2:4])
        result["splits"][split] = {
            "ids": [ids[0], hi],
            **{name: validate_archive(budgets[name], data) for name in ("enstrophy", "energy")},
            "seconds": time.time() - t0,
        }
        print(split, {k: {kk: round(vv, 5) for kk, vv in v.items() if isinstance(vv, float)} for k, v in result["splits"][split].items() if isinstance(v, dict)}, flush=True)
    atomic_json(args.output, result)


if __name__ == "__main__":
    main()
