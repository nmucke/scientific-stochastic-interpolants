#!/usr/bin/env python
"""D1: forecast-uncertainty spectrum against the white driver of the model.

Per basis mode ``k`` (NS: 2D real FFT) this computes, in the MODEL's
normalised units and with an orthonormal transform,

* ``increment``: ``E |(y - c)_k|^2`` over one-step pairs ``(c, y)`` of the
  data (raw second moment, so an upper bound on the conditional variance
  ``E Var(y_k | h)``), on the training and the validation split;
* ``anomaly`` (optional): the unbiased ensemble-member variance
  ``E_cases [ (1/(E-1)) sum_e |(y_e - ybar)_k|^2 ]`` of a multi-step SI
  reference on validation cases (an estimate of ``E Var(y_k | h)`` under the
  model), plus the ensemble-mean error power ``E |(ybar - y)_k|^2``;
* ``climatology``: per-mode variance of single validation fields;
* the white driver level ``Var(M_1)_k = a^2/3`` (``a`` = gamma_multiplier).

With an orthonormal transform a field with per-pixel variance ``s^2`` has
variance ``s^2`` in every mode, so ``power / (a^2/3)`` is the factor by which
a mode's uncertainty exceeds what the white martingale supplies.

Assumptions: additive state-independent driver with per-pixel variance
``a^2/3`` at ``t = 1`` (``gamma = a (1 - t)``); stationarity is not assumed,
everything is a pooled average over cases. Uncertainty bands bootstrap whole
trajectories.

Outputs: a compact JSON (shell tables, gate verdict, file format) and an
``.npz`` with the full per-mode arrays (format documented in the JSON).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phase0_common import (
    NS_TRAIN,
    NS_VALIDATION,
    atomic_json,
    atomic_npy,
    atomic_npz,
    config_gamma_multiplier,
    describe_model,
    load_model,
    load_run_config,
    load_trajectories,
    make_generator,
    normalise,
    preprocessing_affine,
    sample_next,
    sha256,
)

REPORT_SHELLS = (1, 4, 7, 11, 16, 24, 32, 48, 64)  # shells beyond the grid are skipped


# ------------------------------------------------------------------ bases


class FourierBasis:
    """Real 2D FFT over the last two axes with ``norm="ortho"``.

    Modes live on the rfft grid ``[H, W//2 + 1]``. ``multiplicity`` counts how
    often each rfft mode appears in the full FFT grid (1 or 2), so full-grid
    sums and shell means are exact. Shells are nearest-integer radii
    ``floor(sqrt(kx^2 + ky^2) + 0.5)`` in cycles per domain along each axis
    (key_findings section 4). Requires a periodic grid; non-square is allowed.
    """

    name = "fft2_rfft_grid_ortho"

    def __init__(self, height, width):
        self.height, self.width = int(height), int(width)
        kx = np.fft.fftfreq(self.height) * self.height
        ky = np.fft.rfftfreq(self.width) * self.width
        radius = np.hypot(kx[:, None], ky[None, :])
        self.shell = np.floor(radius + 0.5).astype(np.int64)
        mult = np.full(ky.shape, 2.0)
        mult[0] = 1.0
        if self.width % 2 == 0:
            mult[-1] = 1.0
        self.multiplicity = np.broadcast_to(mult, self.shell.shape).copy()
        self.mode_shape = self.shell.shape
        self.kx, self.ky = kx, ky

    def power(self, x):
        """``|x_k|^2`` (float64) for real ``x`` ``[..., H, W]``."""
        coeff = np.fft.rfft2(np.asarray(x, dtype=np.float64), norm="ortho")
        return coeff.real**2 + coeff.imag**2

    def coefficients(self, x):
        return np.fft.rfft2(np.asarray(x, dtype=np.float64), norm="ortho")

    def num_shells(self):
        return int(self.shell.max()) + 1

    def shell_means(self, modal):
        """Multiplicity-weighted mean over each shell: ``[..., modes] -> [..., K]``."""
        w = self.multiplicity.ravel()
        flat = np.asarray(modal, dtype=np.float64).reshape(*modal.shape[:-2], -1)
        labels = self.shell.ravel()
        k = self.num_shells()
        onehot = np.zeros((labels.size, k))
        onehot[np.arange(labels.size), labels] = w
        counts = onehot.sum(0)
        return (flat @ onehot) / counts, counts

    def shell_sums(self, modal):
        means, counts = self.shell_means(modal)
        return means * counts


BASES = {"fft2": FourierBasis}


# --------------------------------------------------------------- spectra


def increment_power(trajectories, basis, history_length):
    """Per-trajectory mean ``|(y - c)_k|^2``: ``[n, C, T, ...] -> [n, C, modes]``.

    Pairs follow the dataset's window convention: targets at indices
    ``L .. T-2`` and ``c`` the frame before (``T - L - 1`` pairs).
    """
    out = []
    for traj in trajectories:  # [C, *s, T] normalised
        x = np.asarray(traj, dtype=np.float64)
        T = x.shape[-1]
        y = x[..., history_length : T - 1]
        c = x[..., history_length - 1 : T - 2]
        inc = np.moveaxis(y - c, -1, 0)  # [pairs, C, H, W]
        out.append(basis.power(inc).mean(0))
    return np.stack(out)


def climatology_power(trajectories, basis):
    """Per-trajectory mean of ``|(x - xbar_pool)_k|^2`` over single fields.

    The pooled mean field over all given trajectories and times is removed.
    """
    x = np.asarray(trajectories, dtype=np.float64)
    pooled_mean = x.mean(axis=(0, -1))
    out = []
    for traj in x:
        fields = np.moveaxis(traj, -1, 0) - pooled_mean
        out.append(basis.power(fields).mean(0))
    return np.stack(out)


def anomaly_power(ensemble, truth, basis):
    """Per-case member variance and ensemble-mean error power per mode.

    ``ensemble`` ``[cases, E, C, H, W]``, ``truth`` ``[cases, C, H, W]``.
    Returns ``(variance, error)`` each ``[cases, C, modes]``: the unbiased
    ``(1/(E-1)) sum_e |(y_e - ybar)_k|^2`` and ``|(ybar - y)_k|^2``. For a
    calibrated ensemble ``E error = (1 + 1/E) E variance``.
    """
    coeff = basis.coefficients(ensemble)
    mean = coeff.mean(1, keepdims=True)
    dev = coeff - mean
    var = (dev.real**2 + dev.imag**2).sum(1) / (ensemble.shape[1] - 1)
    err = mean[:, 0] - basis.coefficients(truth)
    return var, err.real**2 + err.imag**2


def bootstrap_shell_ratio(per_group, basis, white, groups, n_boot=1000, seed=0):
    """Shell ratio ``mean / white`` with a whole-group (trajectory) bootstrap.

    ``per_group`` ``[n, C, modes]`` with one row per case and ``groups`` the
    trajectory id of each row. Returns ``(ratio [C, K], q05 [C, K], q95)``.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    by_group = np.stack([per_group[groups == g].mean(0) for g in unique])
    shell_group, _ = basis.shell_means(by_group)  # [G, C, K]
    ratio = shell_group.mean(0) / white
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    boot = shell_group[draws].mean(1) / white
    return ratio, np.quantile(boot, 0.05, 0), np.quantile(boot, 0.95, 0)


def gate(ratios, shells, factor=3.0):
    """Gate of HANDOVER section 3 on shells 1..max complete annulus.

    ``within_white`` asks whether every shell ratio lies in
    ``[1/factor, factor]`` of the model's own white level. ``range`` is
    ``max/min`` over shells: a white driver of ANY amplitude can be within
    ``factor`` of every shell only if ``range <= factor^2``.
    """
    r = np.asarray(ratios)[shells]
    rng = float(r.max() / r.min())
    return {
        "shells_used": [int(shells[0]), int(shells[-1])],
        "max_ratio": float(r.max()),
        "argmax_shell": int(shells[int(r.argmax())]),
        "min_ratio": float(r.min()),
        "argmin_shell": int(shells[int(r.argmin())]),
        "fraction_of_shells_within_factor_of_white": float(
            np.mean((r >= 1 / factor) & (r <= factor))
        ),
        "dynamic_range_max_over_min": rng,
        "within_factor_of_white_everywhere": bool(r.max() <= factor and r.min() >= 1 / factor),
        "any_white_level_within_factor": bool(rng <= factor**2),
    }


# ------------------------------------------------------------- ensembles


def generate_reference(
    model, trajectories, ids, origins, history_length, mean, std, args, out_dir, noise
):
    """Sample and cache the SI reference ensembles, one file per trajectory.

    File ``ens_<noise>_s<steps>_traj<id>.npy``: ``[origins, E, C, H, W]``
    float32 normalised model output. Existing files are reused (resumable).
    """
    files = []
    for traj_index, traj_id in enumerate(ids):
        path = out_dir / f"ens_{noise}_s{args.steps}_E{args.ensemble}_traj{traj_id:03d}.npy"
        files.append(path)
        if path.exists():
            continue
        x = normalise(trajectories[traj_index : traj_index + 1], mean, std)
        histories = torch.cat(
            [x[..., o - history_length : o] for o in origins], 0
        ).repeat_interleave(args.ensemble, 0)
        seed = args.seed + 1000 * traj_id + (0 if noise == "em" else 1)
        gen = make_generator(args.device, seed)
        outs = []
        for start in range(0, len(histories), args.batch):
            outs.append(
                sample_next(
                    model,
                    histories[start : start + args.batch],
                    args.steps,
                    si_noise=noise,
                    generator=gen,
                ).float()
            )
        ens = torch.cat(outs).reshape(len(origins), args.ensemble, *x.shape[1:-1])
        atomic_npy(path, ens.numpy())
        print(f"{time.strftime('%T')} saved {path.name}", flush=True)
    return files


# ------------------------------------------------------------------- main


def shell_table(ratio, q05, q95, shells=REPORT_SHELLS, channel=0):
    return {
        str(k): {
            "ratio": float(ratio[channel, k]),
            "q05": float(q05[channel, k]),
            "q95": float(q95[channel, k]),
        }
        for k in shells
        if k < ratio.shape[-1]
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", help="SI checkpoint; also supplies the dataset config and normalisation")
    source.add_argument(
        "--config-name",
        help="data-only run (needs --no-ensemble): Hydra case supplying dataset, normalisation and gamma_multiplier",
    )
    p.add_argument("--basis", default="fft2", choices=sorted(BASES))
    p.add_argument("--train-ids", nargs=2, type=int, default=list(NS_TRAIN))
    p.add_argument("--val-ids", nargs=2, type=int, default=list(NS_VALIDATION))
    p.add_argument("--origins", nargs="+", type=int, default=[5, 25, 45, 65, 85])
    p.add_argument("--ensemble", type=int, default=16)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--si-noise", nargs="*", default=["em", "exact"], choices=["em", "exact"])
    p.add_argument("--no-ensemble", action="store_true")
    p.add_argument("--max-val-trajectories", type=int, help="smoke tests only")
    p.add_argument("--max-train-trajectories", type=int, help="smoke tests only")
    p.add_argument("--batch", type=int, default=80)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-cache", help="directory for .npy trajectory caches")
    p.add_argument("--arrays-dir", required=True, help="large outputs (ensembles, npz)")
    p.add_argument("--output", required=True, help="JSON summary path")
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args(argv)

    if args.checkpoint_dir is not None:
        model, config = load_model(args.checkpoint_dir, args.device)
        info = describe_model(model, config)
        checkpoint = Path(args.checkpoint_dir) / "model.pth"
    else:
        if not args.no_ensemble:
            raise ValueError("--config-name runs need --no-ensemble (no trained model)")
        model, config = None, load_run_config(config_name=args.config_name)
        info = {
            "model_class": None,
            "config_name": args.config_name,
            "gamma_multiplier": config_gamma_multiplier(config),
            "note": "no checkpoint: white level from the case config's SI interpolation",
        }
        checkpoint = None
    a = info["gamma_multiplier"]
    if a is None:
        raise ValueError("D1 needs a stochastic model with a gamma_multiplier")
    white = a**2 / 3.0
    L = int(config.len_field_history)
    mean, std = preprocessing_affine(config)
    arrays = Path(args.arrays_dir)
    arrays.mkdir(parents=True, exist_ok=True)

    val_ids = list(range(*args.val_ids))[: args.max_val_trajectories]
    train_ids = list(range(*args.train_ids))[: args.max_train_trajectories]
    t0 = time.time()
    val = load_trajectories(config, (val_ids[0], val_ids[-1] + 1), args.data_cache)
    print(f"{time.strftime('%T')} loaded validation {tuple(val.shape)} in {time.time()-t0:.0f}s", flush=True)
    basis = BASES[args.basis](*val.shape[2:-1])
    val_n = normalise(val, mean, std).numpy()

    result = {
        "diagnostic": "D1 uncertainty spectrum vs white driver",
        "argv": sys.argv if argv is None else argv,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "dataset": [str(Path(d) / f) for d, f in zip(config.test_data.paths, config.test_data.files)],
        "grid": [basis.height, basis.width],
        "model": info,
        "white_driver_level_per_mode": white,
        "normalisation": {"mean": mean.tolist(), "std": std.tolist()},
        "basis": basis.name,
        "units": (
            "All per-mode powers are E|x_k|^2 of the rfft2 with norm='ortho' of fields in "
            "NORMALISED model units (archive / std). A white field with per-pixel variance s^2 "
            "has s^2 in every mode, so the white driver is a^2/3 per mode. To convert to "
            "archive units multiply by std^2; to key_findings' enstrophy convention "
            "(norm='forward', 0.5|w_k|^2) multiply by 0.5*std^2/(H*W)."
        ),
        "splits": {"train": [train_ids[0], train_ids[-1] + 1], "validation": [val_ids[0], val_ids[-1] + 1]},
        "history_length": L,
    }

    inc_val = increment_power(val_n, basis, L)
    clim_val = climatology_power(val_n, basis)
    del val_n
    t0 = time.time()
    train = load_trajectories(config, (train_ids[0], train_ids[-1] + 1), args.data_cache)
    print(f"{time.strftime('%T')} loaded train {tuple(train.shape)} in {time.time()-t0:.0f}s", flush=True)
    inc_train = np.concatenate(
        [
            increment_power(normalise(train[i : i + 16], mean, std).numpy(), basis, L)
            for i in range(0, len(train), 16)
        ]
    )
    del train

    K = basis.num_shells()
    complete = np.arange(1, min(basis.height, basis.width) // 2 + 1)
    tables, arrays_out = {}, {}
    for name, per_traj, ids in (
        ("increment_train", inc_train, train_ids),
        ("increment_validation", inc_val, val_ids),
        ("climatology_validation", clim_val, val_ids),
    ):
        ratio, q05, q95 = bootstrap_shell_ratio(per_traj, basis, white, ids, args.bootstrap)
        tables[name] = {
            "shell_table_ratio_to_white": shell_table(ratio, q05, q95),
            "ratio_all_shells": ratio[0].tolist(),
            "ratio_q05_all_shells": q05[0].tolist(),
            "ratio_q95_all_shells": q95[0].tolist(),
            "gate": gate(ratio[0], complete),
        }
        arrays_out[name] = per_traj.mean(0)
        arrays_out[name + "_per_trajectory"] = per_traj.astype(np.float32)

    if not args.no_ensemble:
        if info["model_class"] != "FollmerStochasticInterpolant":
            raise ValueError("The D1 ensemble reference is the multi-step SI solver")
        truth_cases = np.stack(
            [normalise(val[i : i + 1], mean, std)[0, ..., o].numpy() for i in range(len(val_ids)) for o in args.origins]
        )
        groups = np.repeat(val_ids, len(args.origins))
        ens_dir = arrays / "ensembles"
        ens_dir.mkdir(exist_ok=True)
        for noise in args.si_noise:
            t0 = time.time()
            files = generate_reference(model, val, val_ids, args.origins, L, mean, std, args, ens_dir, noise)
            ens = np.concatenate([np.load(f) for f in files])
            var, err = anomaly_power(ens, truth_cases, basis)
            name = f"anomaly_si{args.steps}_{noise}"
            ratio, q05, q95 = bootstrap_shell_ratio(var, basis, white, groups, args.bootstrap)
            eratio, e05, e95 = bootstrap_shell_ratio(err, basis, white, groups, args.bootstrap)
            v_sh, _ = basis.shell_means(var.mean(0))
            e_sh, _ = basis.shell_means(err.mean(0))
            corr = (1 + 1 / args.ensemble)
            tables[name] = {
                "sampler": "Euler-Maruyama (existing solver)" if noise == "em" else "Euler drift + exact integrated noise",
                "ensemble_size": args.ensemble,
                "origins": args.origins,
                "num_cases": int(len(ens)),
                "seconds": time.time() - t0,
                "shell_table_ratio_to_white": shell_table(ratio, q05, q95),
                "ratio_all_shells": ratio[0].tolist(),
                "ratio_q05_all_shells": q05[0].tolist(),
                "ratio_q95_all_shells": q95[0].tolist(),
                "gate": gate(ratio[0], complete),
                "error_over_corrected_spread_by_shell": (e_sh[0] / (corr * v_sh[0])).tolist(),
                "error_ratio_to_white_table": shell_table(eratio, e05, e95),
                "ensemble_files": [str(f) for f in files],
            }
            arrays_out[name] = var.mean(0)
            arrays_out[name + "_error"] = err.mean(0)
            arrays_out[name + "_per_case"] = var.astype(np.float32)

    shells_note = {
        "shell": "rfft-grid integer shell index floor(sqrt(kx^2+ky^2)+0.5) [H, W//2+1]",
        "multiplicity": "count of each rfft mode in the full FFT grid (1 or 2)",
        "kx": "fftfreq(H)*H (rows)",
        "ky": "rfftfreq(W)*W (columns)",
    }
    npz = arrays / f"d1_spectra_{args.basis}.npz"
    atomic_npz(
        npz,
        shell=basis.shell,
        multiplicity=basis.multiplicity,
        kx=basis.kx,
        ky=basis.ky,
        white_level=np.float64(white),
        norm_std=std,
        **{k: v for k, v in arrays_out.items()},
    )
    result["tables"] = tables
    result["arrays_file"] = str(npz)
    result["arrays_format"] = {
        **shells_note,
        "white_level": "a^2/3, the per-mode variance of M_1 in the same units",
        "norm_std": "per-channel std of the normalisation",
        "<name>": "[C, H, W//2+1] float64 per-mode mean power (pooled over all cases), units above",
        "<name>_per_trajectory": "[n_traj, C, H, W//2+1] float32 per-trajectory means (increment/climatology)",
        "<name>_per_case": "[cases, C, H, W//2+1] float32 per case (anomaly); case order = trajectory-major, origins inner",
        "anomaly_si<steps>_<noise>_error": "[C, H, W//2+1] ensemble-mean error power |ybar - y|^2",
        "names": sorted(arrays_out),
        "use_for_WP1": "Sigma_k estimate = anomaly_* (model ensemble) or increment_validation (upper bound); C_k = kappa * Sigma_k / q(1) with q(1) = a^2/3",
    }
    result["mode_counts_by_shell"] = basis.shell_means(np.zeros((1, *basis.mode_shape)))[1].tolist()
    result["incomplete_annuli_from_shell"] = int(min(basis.height, basis.width) // 2 + 1)
    primary = "anomaly_si%d_exact" % args.steps if f"anomaly_si{args.steps}_exact" in tables else "increment_validation"
    g = tables[primary]["gate"]
    result["gate_verdict"] = {
        "primary_spectrum": primary,
        **g,
        "diagnosis_confirmed": not g["any_white_level_within_factor"],
        "statement": (
            f"{primary}: shell ratio to white ranges {g['min_ratio']:.3g} (shell {g['argmin_shell']}) to "
            f"{g['max_ratio']:.3g} (shell {g['argmax_shell']}), dynamic range {g['dynamic_range_max_over_min']:.3g}. "
            + (
                "No white level is within 3x across scales: spectral mismatch CONFIRMED."
                if not g["any_white_level_within_factor"]
                else "A white level within 3x exists: the central diagnosis is NOT supported; re-plan before WP1."
            )
        ),
    }
    atomic_json(args.output, result)
    print(result["gate_verdict"]["statement"], flush=True)
    for name, table in tables.items():
        print(name, {k: round(v["ratio"], 4) for k, v in table["shell_table_ratio_to_white"].items()}, flush=True)


if __name__ == "__main__":
    main()
