"""Compute metrics and produce plots from saved posterior trajectories.

Reads everything written by `generate_posterior_samples.py`. Run for a single
method or several:

    python paper/scripts/evaluate_posterior_samples.py case=stochastic_navier_stokes
    python paper/scripts/evaluate_posterior_samples.py case=stochastic_navier_stokes \\
        +eval.methods='[si,fm]'

Aggregates metrics across all `sample_*_steps_*.pt` files it finds per method,
so a multi-test-id run gives mean ± std across test ids. When the generation
script was run with multiple `case.num_steps` values, results are split per
(method, num_steps): each appears as its own line in the metric curves and its
own row in the summary table. Filter to a subset of step counts via:

    +eval.num_steps='[100,250]'
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scisi.metrics.lsim import LSiM_distance
from scisi.metrics.spectral import compute_enstrophy_error
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.spectrum import plot_enstrophy_spectrum

logger = logging.getLogger(__name__)

DEFAULT_METHODS = ["si", "fm", "flowdas"]

_SAMPLE_RE = re.compile(r"^sample_(\d+)_steps_(\d+)\.pt$")


@contextmanager
def _quiet_nan_warnings():
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield
COLORS = {
    "si": "tab:blue",
    "fm": "tab:orange",
    "flowdas": "tab:green",
    "prior": "tab:gray",
}

LINESTYLES = ["-", (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1))]


def _to_hwt(traj: torch.Tensor) -> torch.Tensor:
    """Reduce a saved trajectory tensor to [H, W, T].

    Saved layouts:
      true / prior -> [B=1, C=1, H, W, T]
      posterior    -> [ensemble, C=1, H, W, T]
    Ensemble dim is averaged out for spectra and visual panels.
    """
    if traj.dim() != 5:
        raise ValueError(f"Expected 5D trajectory tensor, got {traj.shape}")
    return traj[:, 0].mean(dim=0)


def _list_samples_by_num_steps(method_dir: Path) -> dict[int, list[Path]]:
    """Group `sample_<id>_steps_<n>.pt` files by their `num_steps` suffix."""
    by_steps: dict[int, list[Path]] = defaultdict(list)
    for f in sorted(method_dir.glob("sample_*_steps_*.pt")):
        m = _SAMPLE_RE.match(f.name)
        if m is None:
            continue
        num_steps = int(m.group(2))
        by_steps[num_steps].append(f)
    return dict(by_steps)


def _crps_grid(ensemble_field: torch.Tensor, truth: torch.Tensor) -> float:
    """Energy-form ensemble CRPS, averaged over all grid points.

    For ensemble {x_1, ..., x_M} and truth y at each grid point:
        CRPS = (1/M) Σ_i |x_i - y| - (1/(2 M²)) Σ_i Σ_j |x_i - x_j|

    With M=1 the pairwise term vanishes, so CRPS reduces to MAE — fine for
    the prior baseline which is single-sample.
    """
    M = ensemble_field.shape[0]
    term1 = (ensemble_field - truth.unsqueeze(0)).abs().mean(dim=0)
    diffs = ensemble_field.unsqueeze(0) - ensemble_field.unsqueeze(1)
    term2 = diffs.abs().sum(dim=(0, 1)) / (2.0 * M * M)
    return float((term1 - term2).mean().item())


def _per_step_metrics(
    true_hwt: torch.Tensor,
    pred_traj: torch.Tensor,
    len_field_history: int,
    num_physical_steps: int,
) -> dict[str, np.ndarray]:
    """Per-timestep metrics, shape [n_samples, T] each.

    For RMSE/LSiM, n_samples = ensemble size (one row per member).
    For CRPS, n_samples = 1 (it's already an ensemble-level scalar).
    Stacking across test ids later concatenates on the sample axis, so the
    eventual shaded band reflects ensemble spread + test-case spread.
    """
    ensemble = pred_traj.shape[0]
    n_steps = num_physical_steps - len_field_history
    mse_loss = nn.MSELoss()

    rmse = np.zeros((ensemble, n_steps), dtype=np.float64)
    lsim = np.zeros((ensemble, n_steps), dtype=np.float64)
    crps = np.zeros((1, n_steps), dtype=np.float64)

    for i, t_idx in enumerate(range(len_field_history, num_physical_steps)):
        true_t = true_hwt[:, :, t_idx]
        ensemble_t = pred_traj[:, 0, :, :, t_idx]
        for m in range(ensemble):
            pred_t = ensemble_t[m]
            rmse[m, i] = torch.sqrt(mse_loss(true_t, pred_t)).item()
            lsim[m, i] = LSiM_distance(true_t, pred_t).item()
        crps[0, i] = _crps_grid(ensemble_t, true_t)

    return {"RMSE": rmse, "LSiM": lsim, "CRPS": crps}


def _enstrophy_error_curve(
    true_hwt: torch.Tensor,
    pred_traj: torch.Tensor,
    len_field_history: int,
    num_physical_steps: int,
) -> np.ndarray:
    """Returns shape [1, T] so it stacks consistently with the other metrics."""
    pred_hwt = pred_traj[:, 0].mean(dim=0)
    _total, per_step = compute_enstrophy_error(
        true_hwt[:, :, len_field_history:num_physical_steps],
        pred_hwt[:, :, len_field_history:num_physical_steps],
        dx=2 * torch.pi / 128,
    )
    return np.asarray([float(v) for v in per_step], dtype=np.float64)[None, :]


def _aggregate_curves(
    per_test: list[dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Concatenate per-(test_id) metric arrays along the sample axis.

    Each metric arrives as shape [n_samples_i, T]; the result has shape
    [Σ n_samples_i, T]. mean/std use nanmean/nanstd so individual NaN
    cells (failed members or timesteps) drop out without poisoning the rest.
    """
    if not per_test:
        return {}
    keys = per_test[0].keys()
    out: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        stack = np.concatenate([m[key] for m in per_test], axis=0)
        with _quiet_nan_warnings():
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
        out[key] = {"mean": mean, "std": std, "all": stack}
    return out


def _load_method_results(
    files: list[Path], case_name: str
) -> dict[str, Any] | None:
    if not files:
        return None

    posterior_curves: list[dict[str, np.ndarray]] = []
    prior_curves: list[dict[str, np.ndarray]] = []
    samples = []
    skips: list[dict[str, Any]] = []

    for f in files:
        payload = torch.load(f, map_location="cpu", weights_only=False)
        meta = payload["meta"]
        test_id = meta["test_sample_index"]
        true_hwt = payload["true_trajectory"][0, 0]  # [H, W, T]
        post = payload["posterior_trajectory"]  # [ens, C, H, W, T]
        prior = payload["prior_trajectory"]  # [1, C, H, W, T]

        post_curves = _per_step_metrics(
            true_hwt=true_hwt,
            pred_traj=post,
            len_field_history=meta["len_field_history"],
            num_physical_steps=meta["num_physical_steps"],
        )
        prior_curves_one = _per_step_metrics(
            true_hwt=true_hwt,
            pred_traj=prior,
            len_field_history=meta["len_field_history"],
            num_physical_steps=meta["num_physical_steps"],
        )

        if case_name == "stochastic_navier_stokes":
            post_curves["Enstrophy error"] = _enstrophy_error_curve(
                true_hwt, post, meta["len_field_history"], meta["num_physical_steps"]
            )
            prior_curves_one["Enstrophy error"] = _enstrophy_error_curve(
                true_hwt, prior, meta["len_field_history"], meta["num_physical_steps"]
            )

        for kind, curves in (("posterior", post_curves), ("prior", prior_curves_one)):
            for metric, arr in curves.items():
                nan_mask = np.isnan(arr)
                n_nan = int(nan_mask.sum())
                if n_nan:
                    skips.append(
                        {
                            "file": str(f),
                            "test_id": int(test_id),
                            "kind": kind,
                            "metric": metric,
                            "n_nan_cells": n_nan,
                            "n_total_cells": int(arr.size),
                        }
                    )

        posterior_curves.append(post_curves)
        prior_curves.append(prior_curves_one)
        samples.append(payload)

    return {
        "samples": samples,
        "posterior": _aggregate_curves(posterior_curves),
        "prior": _aggregate_curves(prior_curves),
        "skips": skips,
    }


def _log_summary(
    method: str, kind: str, agg: dict[str, dict[str, np.ndarray]]
) -> None:
    pieces = []
    for metric, vals in agg.items():
        with _quiet_nan_warnings():
            m = np.nanmean(vals["mean"])
            s = np.nanstd(vals["mean"])
        pieces.append(f"{metric}={m:.4f}±{s:.4f}")
    logger.info(f"[{method}/{kind}] " + " | ".join(pieces))


def _linestyle_for(num_steps: int, all_num_steps: list[int]):
    """Stable linestyle per num_steps, indexed by sorted-unique position."""
    idx = all_num_steps.index(num_steps) % len(LINESTYLES)
    return LINESTYLES[idx]


def _plot_metric_curves(
    results: dict[str, dict[str, Any] | None],
    time_range: range,
    out_dir: Path,
    case_name: str,
) -> None:
    metric_names = ["RMSE", "LSiM", "CRPS"]
    if case_name == "stochastic_navier_stokes":
        metric_names.append("Enstrophy error")

    valid = [(label, res) for label, res in results.items() if res is not None]
    all_num_steps = sorted({res["num_steps"] for _, res in valid})

    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 4))
    if len(metric_names) == 1:
        axes = [axes]
    x = list(time_range)

    for ax, metric in zip(axes, metric_names):
        for label, res in valid:
            agg = res["posterior"].get(metric)
            if agg is None:
                continue
            color = COLORS.get(res["method"])
            linestyle = _linestyle_for(res["num_steps"], all_num_steps)
            mean = agg["mean"]
            std = agg["std"]
            ax.plot(
                x, mean, label=label, color=color, linewidth=2, linestyle=linestyle
            )
            ax.fill_between(
                x,
                np.maximum(mean - std, 1e-12),
                mean + std,
                color=color,
                alpha=0.15,
                linewidth=0,
            )

        for _, res in valid:
            prior_agg = res["prior"].get(metric)
            if prior_agg is None:
                continue
            ax.plot(
                x,
                prior_agg["mean"],
                label="prior",
                color=COLORS["prior"],
                linestyle="--",
                linewidth=2,
            )
            break

        ax.set_title(metric)
        ax.set_xlabel("timestep")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6)
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_vs_time.png", dpi=150)
    plt.close(fig)


def _plot_final_state_grid(
    results: dict[str, dict[str, Any] | None],
    out_dir: Path,
) -> None:
    """Final-state image grid using the first test id from each (method, num_steps)."""
    valid = [(label, res) for label, res in results.items() if res is not None]
    if not valid:
        return

    first_label, first_res = valid[0]
    first = first_res["samples"][0]
    true_hwt = first["true_trajectory"][0, 0]  # [H, W, T]
    obs_mask = first["obs_indices_on_grid"][0]
    final_idx = first["meta"]["num_physical_steps"] - 1

    true_state = true_hwt[:, :, final_idx]
    vmin = float(true_state.min())
    vmax = float(true_state.max())

    n_cols = 2 + len(valid) + 1  # true, observed, (method, num_steps) entries, prior
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    axes[0, 0].imshow(true_state, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("True")
    axes[1, 0].axis("off")

    axes[0, 1].imshow(obs_mask * true_state, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Observed")
    axes[1, 1].axis("off")

    for j, (label, res) in enumerate(valid, start=2):
        sample = res["samples"][0]
        post_hwt = sample["posterior_trajectory"][:, 0].mean(dim=0)  # [H, W, T]
        post_state = post_hwt[:, :, final_idx]
        axes[0, j].imshow(post_state, vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"{label} (mean)")
        axes[1, j].imshow((post_state - true_state).abs())
        axes[1, j].set_title(f"{label} |err|")

    prior_hwt = first["prior_trajectory"][:, 0].mean(dim=0)
    prior_state = prior_hwt[:, :, final_idx]
    axes[0, -1].imshow(prior_state, vmin=vmin, vmax=vmax)
    axes[0, -1].set_title("Prior (mean)")
    axes[1, -1].imshow((prior_state - true_state).abs())
    axes[1, -1].set_title("Prior |err|")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_dir / "final_state_grid.png", dpi=150)
    plt.close(fig)


def _save_animation(
    results: dict[str, dict[str, Any] | None],
    out_dir: Path,
) -> None:
    valid = [(label, res) for label, res in results.items() if res is not None]
    if not valid:
        return

    first = valid[0][1]["samples"][0]
    true_hwt = first["true_trajectory"][0, 0]
    num_physical = first["meta"]["num_physical_steps"]

    sliced_true = true_hwt[:, :, :num_physical]
    tensors = [sliced_true]
    titles = ["True"]
    vmin = float(sliced_true.min())
    vmax = float(sliced_true.max())

    for label, res in valid:
        sample = res["samples"][0]
        post_hwt = sample["posterior_trajectory"][:, 0].mean(dim=0)
        tensors.append(post_hwt)
        titles.append(label)

    prior_hwt = first["prior_trajectory"][:, 0].mean(dim=0)
    tensors.append(prior_hwt)
    titles.append("Prior")

    create_animation_from_tensors(
        tensors,
        fps=10,
        file_name=str(out_dir / "trajectory.mp4"),
        colormaps="viridis",
        titles=titles,
        normalize=False,
        vmin=vmin,
        vmax=vmax,
    )


def _build_final_metrics_table(
    results: dict[str, dict[str, Any] | None],
    case_name: str,
) -> tuple[list[str], list[str], list[list[str]]]:
    """Aggregate each (method, num_steps) posterior curves to mean ± std per metric.

    `mean` averages each test id's metric over the predicted timesteps, then
    averages across test ids. `std` is the across-test-id spread of the same
    per-test time-averaged scalar (so it reflects test-case variability).

    Returns (row_labels, metric_names, cells) where `cells[i][j]` is the
    formatted "mean ± std" string for row i / metric j. One row per
    (method, num_steps), plus a final row for the prior baseline.
    """
    valid_labels = [label for label, r in results.items() if r is not None]
    metric_names = ["RMSE", "LSiM", "CRPS"]
    if case_name == "stochastic_navier_stokes":
        metric_names.append("Enstrophy error")

    def _row(agg_dict: dict[str, dict[str, np.ndarray]]) -> list[str]:
        row: list[str] = []
        for metric in metric_names:
            agg = agg_dict.get(metric)
            if agg is None:
                row.append("--")
                continue
            with _quiet_nan_warnings():
                per_test_scalars = np.nanmean(agg["all"], axis=1)
                m = np.nanmean(per_test_scalars)
                s = np.nanstd(per_test_scalars)
            if np.isnan(m):
                row.append("nan")
            else:
                row.append(f"{m:.4f} ± {s:.4f}")
        return row

    row_labels = list(valid_labels)
    cells = [_row(results[label]["posterior"]) for label in valid_labels]

    if valid_labels:
        baseline_label = valid_labels[0]
        row_labels.append(f"prior ({baseline_label})")
        cells.append(_row(results[baseline_label]["prior"]))

    return row_labels, metric_names, cells


def _log_metrics_table(
    methods: list[str],
    metric_names: list[str],
    cells: list[list[str]],
) -> None:
    """Pretty-print the final metrics table to the logger."""
    method_w = max(len("Method"), max((len(m) for m in methods), default=0))
    col_widths = [
        max(len(metric_names[j]), max((len(row[j]) for row in cells), default=0))
        for j in range(len(metric_names))
    ]

    def fmt_row(label: str, values: list[str]) -> str:
        parts = [label.ljust(method_w)] + [
            v.ljust(col_widths[j]) for j, v in enumerate(values)
        ]
        return " | ".join(parts)

    sep = "-+-".join(["-" * method_w] + ["-" * w for w in col_widths])
    logger.info("Final metrics (mean ± std across test ids):")
    logger.info(fmt_row("Method", metric_names))
    logger.info(sep)
    for method, row in zip(methods, cells):
        logger.info(fmt_row(method, row))


def _save_metrics_table_markdown(
    methods: list[str],
    metric_names: list[str],
    cells: list[list[str]],
    out_path: Path,
) -> None:
    header = "| Method | " + " | ".join(metric_names) + " |"
    sep = "|" + "|".join(["---"] * (len(metric_names) + 1)) + "|"
    body = [
        "| " + method + " | " + " | ".join(row) + " |"
        for method, row in zip(methods, cells)
    ]
    out_path.write_text("\n".join([header, sep, *body]) + "\n")


def _save_metrics_table_figure(
    methods: list[str],
    metric_names: list[str],
    cells: list[list[str]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(2 + 2.5 * len(metric_names), 0.6 + 0.5 * len(methods))
    )
    ax.axis("off")
    table = ax.table(
        cellText=cells,
        rowLabels=methods,
        colLabels=metric_names,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for j in range(len(metric_names)):
        table[(0, j)].set_facecolor("#dfe6f0")
        table[(0, j)].set_text_props(weight="bold")
    for i, method in enumerate(methods, start=1):
        if method.startswith("prior"):
            for j in range(len(metric_names)):
                table[(i, j)].set_facecolor("#f3f3f3")
    ax.set_title("Posterior metrics — mean ± std across test ids")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_enstrophy_spectrum(
    results: dict[str, dict[str, Any] | None],
    out_dir: Path,
) -> None:
    valid = [(label, res) for label, res in results.items() if res is not None]
    if not valid:
        return

    first = valid[0][1]["samples"][0]
    trajectories = [first["true_trajectory"][0, 0]]
    titles = ["True"]
    for label, res in valid:
        sample = res["samples"][0]
        post_hwt = sample["posterior_trajectory"][:, 0].mean(dim=0)
        trajectories.append(post_hwt)
        titles.append(label)

    plot_enstrophy_spectrum(
        trajectories=trajectories,
        titles=titles,
        figure_path=str(out_dir),
    )


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="benchmark",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    methods = OmegaConf.select(cfg, "eval.methods", default=None) or DEFAULT_METHODS
    methods = list(methods)

    num_steps_filter_cfg = OmegaConf.select(cfg, "eval.num_steps", default=None)
    num_steps_filter = (
        {int(n) for n in num_steps_filter_cfg} if num_steps_filter_cfg else None
    )

    case_dir = Path(cfg.results_root) / cfg.case.name
    out_dir = case_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Case  : {cfg.case.name}")
    logger.info(f"Methods evaluated: {methods}")
    if num_steps_filter is not None:
        logger.info(f"num_steps filter: {sorted(num_steps_filter)}")
    logger.info(f"Reading from: {case_dir}")

    results: dict[str, dict[str, Any] | None] = {}
    for method in methods:
        method_dir = case_dir / method
        if not method_dir.exists():
            logger.warning(f"No directory {method_dir}, skipping.")
            continue
        by_steps = _list_samples_by_num_steps(method_dir)
        if not by_steps:
            logger.warning(f"No samples found in {method_dir}, skipping.")
            continue
        for num_steps in sorted(by_steps):
            if num_steps_filter is not None and num_steps not in num_steps_filter:
                continue
            label = f"{method} ({num_steps} steps)"
            res = _load_method_results(by_steps[num_steps], cfg.case.name)
            if res is not None:
                res["method"] = method
                res["num_steps"] = num_steps
                _log_summary(label, "posterior", res["posterior"])
                _log_summary(label, "prior", res["prior"])
            results[label] = res

    if not any(r for r in results.values()):
        logger.error("No results loaded — nothing to plot.")
        return

    first_valid = next(r for r in results.values() if r is not None)
    meta = first_valid["samples"][0]["meta"]
    time_range = range(meta["len_field_history"], meta["num_physical_steps"])


    table_methods, table_metric_names, table_cells = _build_final_metrics_table(
        results, cfg.case.name
    )
    _log_metrics_table(table_methods, table_metric_names, table_cells)
    _save_metrics_table_markdown(
        table_methods, table_metric_names, table_cells, out_dir / "metrics_table.md"
    )
    _save_metrics_table_figure(
        table_methods, table_metric_names, table_cells, out_dir / "metrics_table.png"
    )
    logger.info(f"Wrote {out_dir / 'metrics_table.md'} and metrics_table.png")


    _plot_metric_curves(results, time_range, out_dir, cfg.case.name)
    _plot_final_state_grid(results, out_dir)
    _save_animation(results, out_dir)
    if cfg.case.name == "stochastic_navier_stokes":
        _plot_enstrophy_spectrum(results, out_dir)

    summary_lines = [
        "method,num_steps,kind,metric,mean_over_time,std_over_time,std_across_test_ids"
    ]
    for label, res in results.items():
        if res is None:
            continue
        method = res["method"]
        num_steps = res["num_steps"]
        for kind in ("posterior", "prior"):
            for metric, agg in res[kind].items():
                with _quiet_nan_warnings():
                    mean_over_time = float(np.nanmean(agg["mean"]))
                    std_over_time = float(np.nanstd(agg["mean"]))
                    std_across = float(np.nanmean(agg["std"]))
                summary_lines.append(
                    f"{method},{num_steps},{kind},{metric},"
                    f"{mean_over_time:.6f},"
                    f"{std_over_time:.6f},"
                    f"{std_across:.6f}"
                )
    (out_dir / "summary.csv").write_text("\n".join(summary_lines) + "\n")
    logger.info(f"Wrote {out_dir / 'summary.csv'}")

    _report_skips(results, out_dir)


def _report_skips(
    results: dict[str, dict[str, Any] | None],
    out_dir: Path,
) -> None:
    """Log every (label, test_id, kind, metric) that contained NaNs.

    NaN cells were dropped from mean/std via nanmean/nanstd, so reported
    numbers are valid — this just tells you which inputs were partial.
    """
    rows: list[tuple[str, int, str, str, int, int]] = []
    for label, res in results.items():
        if res is None:
            continue
        for s in res.get("skips", []):
            rows.append(
                (
                    label,
                    s["test_id"],
                    s["kind"],
                    s["metric"],
                    s["n_nan_cells"],
                    s["n_total_cells"],
                )
            )

    if not rows:
        logger.info("No NaN entries detected in any sample.")
        return

    logger.warning(
        f"NaN entries skipped in {len(rows)} (label, test_id, kind, metric) groups:"
    )
    rows.sort()
    label_w = max(len("label"), max(len(r[0]) for r in rows))
    metric_w = max(len("metric"), max(len(r[3]) for r in rows))
    header = (
        f"{'label'.ljust(label_w)}  test_id  {'kind'.ljust(9)}  "
        f"{'metric'.ljust(metric_w)}  nan/total"
    )
    logger.warning(header)
    logger.warning("-" * len(header))
    for label, test_id, kind, metric, n_nan, n_total in rows:
        logger.warning(
            f"{label.ljust(label_w)}  {test_id:>7d}  {kind.ljust(9)}  "
            f"{metric.ljust(metric_w)}  {n_nan}/{n_total}"
        )

    skip_path = out_dir / "skipped_nan_entries.csv"
    lines = ["label,test_id,kind,metric,n_nan_cells,n_total_cells"]
    for label, test_id, kind, metric, n_nan, n_total in rows:
        lines.append(f"{label},{test_id},{kind},{metric},{n_nan},{n_total}")
    skip_path.write_text("\n".join(lines) + "\n")
    logger.warning(f"Wrote {skip_path}")


if __name__ == "__main__":
    main()
