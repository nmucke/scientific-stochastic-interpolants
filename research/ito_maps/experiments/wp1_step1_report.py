#!/usr/bin/env python
"""WP1 step 1 report: compare SI processes from their eval_si.sbatch outputs.

Reads only finished outputs (no model is loaded):

* ``<eval-root>/<run>/path_compression/*/results.json``: omitted endpoint-
  variance fraction at K path features per validation history. Uncertainty:
  standard error across histories, and the within-variance standard error of
  each history propagated to the fraction (the sampling error of the
  group-mean variance is not stored, so that per-history error is a lower
  bound). Process differences are paired by history.
* ``<eval-root>/<run>/euler_exact_noise/cases.jsonl``: Euler + exact noise at
  N steps against the same model's reference N. Spread ratio (mean over cases
  of rms_spread_N / rms_spread_ref), fair CRPS, spread/skill and anomaly
  shell-band power ratios, with a bootstrap over validation trajectories (all
  origins of a trajectory resampled together); model differences use the same
  resampled trajectories (paired).
* the residual spectrum files (``sigma`` = E|e_k|^2 of the orthonormal FFT,
  ``q1``): Sigma/white min/max and the shell profile; a second spectrum at
  another resolution is compared after rescaling its per-mode power by the
  grid-size ratio (orthonormal FFT of a band-limited field: per-mode power
  scales with H*W).
* training logs (optional): per-epoch train/val loss and the best epoch.

    python research/ito_maps/experiments/wp1_step1_report.py \
      --eval-root $WP1/stochastic_navier_stokes_64/eval \
      --run white_t2=wp1_si_white_t2_e150_seed0 --run coloured_t=... \
      --spectrum $WP1/stochastic_navier_stokes_64/spectra/residuals_kappa1.npz \
      --reference-spectrum $WP1/spectra/residuals_kappa1.npz \
      --log regressor=<log> --log white_t2=<log> --output report.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

BANDS = [(1, 4), (5, 10), (11, 20), (21, 32), (33, 1000)]
EPOCH_LINE = re.compile(r"Epoch (\d+) \| lr ([0-9.e+-]+), train_loss ([0-9.e+-]+), val_loss ([0-9.e+-]+)")


# ------------------------------------------------------------------ helpers


def shell_index(height, width):
    kx = np.fft.fftfreq(height) * height
    ky = np.fft.rfftfreq(width) * width
    return np.floor(np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) + 0.5).astype(int)


def rfft_weights(shape, width):
    weight = np.full(shape, 2.0)
    weight[:, 0] = 1.0
    if width % 2 == 0:
        weight[:, -1] = 1.0
    return weight


def shell_mean(power, width):
    """Mean per-mode power [..., H, W//2+1] on integer shells (full-grid counts)."""
    shells = shell_index(power.shape[-2], width)
    weight = rfft_weights(shells.shape, width)
    bins = shells.reshape(-1)
    count = np.bincount(bins, weight.reshape(-1))
    flat = (power * weight).reshape(-1, shells.size)
    total = np.stack([np.bincount(bins, row, minlength=count.size) for row in flat])
    return (total / count).reshape(*power.shape[:-2], -1)


def summary(values):
    values = np.asarray(values, dtype=float)
    n = values.size
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(n),
    }


def bootstrap(stat, groups, draws, rng):
    """stat(indices into groups) -> float array; returns estimate and 95% interval."""
    groups = list(groups)
    point = np.asarray(stat(np.arange(len(groups))), dtype=float)
    samples = np.stack(
        [np.asarray(stat(rng.integers(0, len(groups), len(groups))), dtype=float) for _ in range(draws)]
    )
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    return point, low, high


def interval(point, low, high):
    if np.ndim(point) == 0:
        return {"estimate": float(point), "ci95": [float(low), float(high)]}
    return {"estimate": point.tolist(), "ci95_low": low.tolist(), "ci95_high": high.tolist()}


# --------------------------------------------------------- path compression


def load_path_compression(run_dir):
    rows = {}
    for path in sorted(Path(run_dir, "path_compression").glob("*/results.json")):
        data = json.loads(path.read_text())
        key = f"{data['trajectory']}:{data['origin']}"
        rows[key] = {
            "interpolation": data["interpolation"],
            "noise_covariance": data["noise_covariance"],
            "teacher_steps": data["teacher_steps"],
            "by_k": {
                str(r["num_path_features"]): {
                    "omitted": r["omitted_variance_fraction_estimate"],
                    "omitted_se_within_only": r["within_variance_standard_error_across_groups"]
                    / r["total_conditional_forecast_variance_estimate"],
                    "within": r["within_coarse_path_variance"],
                    "total": r["total_conditional_forecast_variance_estimate"],
                }
                for r in data["results"]
            },
        }
    return rows


def path_compression_report(runs, reference):
    out = {}
    for name, rows in runs.items():
        ks = sorted({k for row in rows.values() for k in row["by_k"]}, key=int)
        first = next(iter(rows.values()))
        entry = {
            "interpolation": first["interpolation"],
            "noise_covariance": first["noise_covariance"],
            "histories": sorted(rows),
        }
        for k in ks:
            per = {h: rows[h]["by_k"][k] for h in sorted(rows)}
            entry[f"K{k}"] = {
                "omitted_fraction_across_histories": summary([v["omitted"] for v in per.values()]),
                "total_variance_across_histories": summary([v["total"] for v in per.values()]),
                "per_history": per,
            }
            if name != reference and reference in runs:
                common = sorted(set(rows) & set(runs[reference]))
                diff = [rows[h]["by_k"][k]["omitted"] - runs[reference][h]["by_k"][k]["omitted"] for h in common]
                entry[f"K{k}"][f"paired_difference_vs_{reference}"] = summary(diff)
        out[name] = entry
    return out


# ------------------------------------------------------------ Euler cases


def load_cases(run_dir):
    path = Path(run_dir, "euler_exact_noise", "cases.jsonl")
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(c["trajectory"], c["origin"]): c["steps"] for c in cases}


def band_power(spectrum):
    spectrum = np.asarray(spectrum, dtype=float)
    return np.array([spectrum[lo : min(hi, spectrum.size - 1) + 1].sum() for lo, hi in BANDS])


def case_arrays(cases, keys, steps, reference, ensemble):
    """Per-case arrays for one model, in the order of keys."""
    corr = np.sqrt((ensemble + 1) / ensemble)
    out = {}
    for n in steps + [reference]:
        rows = [cases[k][str(n)] for k in keys]
        out[n] = {
            "spread": np.array([r["rms_spread"] for r in rows]),
            "crps": np.array([r["fair_crps"] for r in rows]),
            "rmse": np.array([r["ensemble_mean_rmse"] for r in rows]),
            "bands": np.stack([band_power(r["anomaly_shell_spectrum"]) for r in rows]),
        }
    out["corr"] = corr
    return out


def euler_statistic(arrays, steps, reference):
    """Vector of statistics on a subset of cases (closure over case indices)."""

    def stat(case_index):
        ref = arrays[reference]
        values = []
        for n in steps + [reference]:
            a = arrays[n]
            values.append(np.mean(a["spread"][case_index] / ref["spread"][case_index]))  # spread ratio
            values.append(np.mean(a["crps"][case_index]))
            values.append(np.mean(a["crps"][case_index] - ref["crps"][case_index]))
            spread_rms = np.sqrt(np.mean(a["spread"][case_index] ** 2))
            rmse_rms = np.sqrt(np.mean(a["rmse"][case_index] ** 2))
            values.append(arrays["corr"] * spread_rms / rmse_rms)  # spread/skill
            values.extend(a["bands"][case_index].mean(0) / ref["bands"][case_index].mean(0))
        return np.array(values)

    return stat


STAT_NAMES = ["spread_ratio_to_reference", "fair_crps", "fair_crps_minus_reference", "spread_skill_ratio"] + [
    f"anomaly_band_power_ratio_k{lo}-{hi if hi < 1000 else 'max'}" for lo, hi in BANDS
]


def euler_report(case_sets, steps, reference, ensemble, reference_model, draws, seed):
    keys = sorted(set.intersection(*(set(c) for c in case_sets.values())))
    trajectories = sorted({k[0] for k in keys})
    by_traj = [[i for i, k in enumerate(keys) if k[0] == t] for t in trajectories]
    arrays = {m: case_arrays(c, keys, steps, reference, ensemble) for m, c in case_sets.items()}

    def expand(traj_index):
        return np.concatenate([by_traj[t] for t in traj_index])

    out = {
        "num_cases": len(keys),
        "trajectories": trajectories,
        "origins": sorted({k[1] for k in keys}),
        "bootstrap": f"{draws} resamples of the {len(trajectories)} validation trajectories, all origins together",
        "spread_skill_ratio": "sqrt((E+1)/E) * rms spread / rms ensemble-mean error over cases (1 = calibrated)",
        "models": {},
        f"paired_difference_vs_{reference_model}": {},
    }
    layout = [(n, s) for n in steps + [reference] for s in STAT_NAMES]
    for model, a in arrays.items():
        rng = np.random.default_rng(seed)
        stat = euler_statistic(a, steps, reference)
        point, low, high = bootstrap(lambda t: stat(expand(t)), trajectories, draws, rng)
        out["models"][model] = {
            str(n): {s: interval(point[i], low[i], high[i]) for i, (m, s) in enumerate(layout) if m == n}
            for n in steps + [reference]
        }
        if model != reference_model and reference_model in arrays:
            ref_stat = euler_statistic(arrays[reference_model], steps, reference)
            rng = np.random.default_rng(seed)
            point, low, high = bootstrap(
                lambda t: stat(expand(t)) - ref_stat(expand(t)), trajectories, draws, rng
            )
            out[f"paired_difference_vs_{reference_model}"][model] = {
                str(n): {s: interval(point[i], low[i], high[i]) for i, (m, s) in enumerate(layout) if m == n}
                for n in steps + [reference]
            }
    return out


# ----------------------------------------------------------------- spectra


def spectrum_report(path, reference_path=None):
    def load(p):
        data = np.load(p)
        sigma = np.asarray(data["sigma"], dtype=np.float64)
        q1 = float(data["q1"])
        width = sigma.shape[-2]  # square grids: full width equals height
        profile = shell_mean(sigma / q1, width)[0]
        return sigma, q1, width, profile

    sigma, q1, width, profile = load(path)
    out = {
        "file": str(path),
        "grid": [sigma.shape[-2], width],
        "q1": q1,
        "sigma_over_white_min_max": [float((sigma / q1).min()), float((sigma / q1).max())],
        "shell_profile_sigma_over_white": profile.tolist(),
        "variance_fraction_by_band": None,
    }
    shells = shell_index(sigma.shape[-2], width)
    weight = rfft_weights(shells.shape, width)
    total = float((sigma[0] * weight).sum())
    out["variance_fraction_by_band"] = {
        f"k{lo}-{hi if hi < 1000 else 'max'}": float((sigma[0] * weight)[(shells >= lo) & (shells <= hi)].sum() / total)
        for lo, hi in [(0, 0)] + BANDS
    }
    out["total_residual_variance_per_pixel"] = total / (sigma.shape[-2] * width)
    if reference_path is not None:
        ref_sigma, ref_q1, ref_width, ref_profile = load(reference_path)
        scale = (sigma.shape[-2] * width) / (ref_sigma.shape[-2] * ref_width)
        common = min(profile.size, ref_profile.size)
        out["reference"] = {
            "file": str(reference_path),
            "grid": [ref_sigma.shape[-2], ref_width],
            "sigma_over_white_min_max": [float((ref_sigma / ref_q1).min()), float((ref_sigma / ref_q1).max())],
            "shell_profile_sigma_over_white": ref_profile.tolist(),
            "per_mode_power_rescale_to_this_grid": scale,
            "total_residual_variance_per_pixel": float(
                (ref_sigma[0] * rfft_weights(ref_sigma.shape[-2:], ref_width)).sum() / (ref_sigma.shape[-2] * ref_width)
            ),
            "shell_ratio_this_over_rescaled_reference": (profile[:common] / (scale * ref_profile[:common])).tolist(),
        }
    return out


# ------------------------------------------------------------- train logs


def parse_log(path):
    text = Path(path).read_text(errors="replace").replace("\r", "\n")
    epochs = [
        {"epoch": int(m[1]), "lr": float(m[2]), "train_loss": float(m[3]), "val_loss": float(m[4])}
        for m in EPOCH_LINE.finditer(text)
    ]
    if not epochs:
        return {"file": str(path), "epochs": []}
    best = min(epochs, key=lambda e: e["val_loss"])
    return {
        "file": str(path),
        "num_epochs": len(epochs),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "train_loss_at_best": best["train_loss"],
        "last": epochs[-1],
        "early_stopped": "Early stopping triggered" in text,
        "epochs": epochs,
    }


# -------------------------------------------------------------------- main


def pairs(values):
    out = {}
    for item in values:
        name, _, value = item.partition("=")
        out[name] = value
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], help="name=run directory under --eval-root")
    parser.add_argument("--reference-model", default="white_t2")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--reference-steps", type=int, default=128)
    parser.add_argument("--spectrum", type=Path)
    parser.add_argument("--reference-spectrum", type=Path)
    parser.add_argument("--log", action="append", default=[], help="name=training log file")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    runs = pairs(args.run)
    report = {"argv": argv, "runs": runs}
    report["path_compression"] = path_compression_report(
        {m: load_path_compression(args.eval_root / r) for m, r in runs.items()}, args.reference_model
    )
    ensembles = {}
    for m, r in runs.items():
        meta = json.loads((args.eval_root / r / "euler_exact_noise" / "results.json").read_text())
        ensembles[m] = meta["ensemble_size"]
    if len(set(ensembles.values())) != 1:
        raise SystemExit(f"ensemble sizes differ: {ensembles}")
    report["euler_exact_noise"] = euler_report(
        {m: load_cases(args.eval_root / r) for m, r in runs.items()},
        args.steps,
        args.reference_steps,
        next(iter(ensembles.values())),
        args.reference_model,
        args.bootstrap,
        args.seed,
    )
    if args.spectrum:
        report["spectrum"] = spectrum_report(args.spectrum, args.reference_spectrum)
    report["training"] = {name: parse_log(path) for name, path in pairs(args.log).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
