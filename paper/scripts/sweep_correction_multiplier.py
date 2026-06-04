"""Sweep `correction_multiplier` in `InterpolantGaussianLikelihood`.

The likelihood-score correction in `InterpolantGaussianLikelihood` includes a
scalar prefactor on the (uncorrected) score (default 1.5). This script loads
the SI model once and runs the posterior sampler for every combination of
test trajectory, multiplier, and the `apply_multiplier_to_full_expression`
flag, recording per-step error metrics so the hyperparameter can be picked
from data.

Run (uses `paper/configs/benchmark.yaml`, defaults to method=si):

    python paper/scripts/sweep_correction_multiplier.py case=stochastic_navier_stokes \\
        case.test_sample_indices=[1,2,3] \\
        +sweep.multipliers='[0.5,1.0,1.5,2.0,3.0,5.0]' \\
        +sweep.apply_full='[false,true]' \\
        +sweep.num_steps='[100,250]'

`+sweep.apply_full` toggles the second hyperparameter on the likelihood:
  * False — `correction_multiplier` scales only the score term (current default).
  * True  — `correction_multiplier` scales the full corrected expression.
Defaults to `[false]` to preserve current behavior.

`+sweep.num_steps` overrides the SDE/ODE solver step count per cell. Defaults
to `[case.num_steps]` (the value from the case config), so a single-element
sweep matches the prior behavior.

Outputs land in `paper/results/<case>/multiplier_sweep/`:
  * config.yaml                                            — composed hydra config
  * sample_<test_id>_mult_<m>_full_<bool>_steps_<n>.pt     — per-cell metrics + meta
  * summary.csv                                            — long-form rows per cell
  * metric_vs_multiplier.png                               — curves grouped by mode/steps
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from paper.scripts.evaluate_posterior_samples import (
    _enstrophy_error_curve,
    _per_step_metrics,
)
from paper.scripts.generate_posterior_samples import (
    _STEPPERS,
    _build_diffusion_term,
    _load_pretrained,
    _make_observations,
    _prepare_inputs,
    _resolve_checkpoint_name,
)
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)

DEFAULT_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
DEFAULT_APPLY_FULL = [False]


def _instantiate_posterior_with_multiplier(
    cfg: DictConfig,
    model: Any,
    obs_operator: Any,
    multiplier: float,
    apply_full: bool,
) -> Any:
    """Build the SI likelihood with a specific correction_multiplier and wrap a posterior."""

    likelihood_model = hydra.utils.instantiate(
        cfg.method.likelihood_model,
        obs_operator=obs_operator,
        model=model,
        variance=cfg.case.variance,
        ensemble_size=cfg.case.likelihood_ensemble_size,
        correction_multiplier=multiplier,
        apply_multiplier_to_full_expression=apply_full,
    )

    diffusion_term = _build_diffusion_term(cfg.method, model)

    posterior_model = hydra.utils.instantiate(
        cfg.method.posterior_model,
        model=model,
        likelihood_model=likelihood_model,
        diffusion_term=diffusion_term,
    )
    return posterior_model


def _sample_posterior(
    cfg: DictConfig,
    model: Any,
    posterior_model: Any,
    init_data: dict[str, torch.Tensor | None],
    observations: torch.Tensor,
    len_field_history: int,
    seed: int,
    num_steps: int,
) -> torch.Tensor:
    stepper = _STEPPERS[cfg.method.stepper]
    common_input = {
        "base": (
            init_data["base"]
            if isinstance(model, FollmerStochasticInterpolant)
            else None
        ),
        "field_history": init_data["field_history"],
        "field_cond": init_data["field_cond"],
        "pars_cond": init_data["pars_cond"],
        "stepper": stepper,
        "num_physical_steps": cfg.case.num_physical_steps,
    }

    mp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.case.mixed_precision
        else contextlib.nullcontext()
    )

    # Re-seed before each sample so cells see identical sampler noise.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with mp_ctx:
        return posterior_model.sample_trajectory(
            **common_input,
            ensemble_size=cfg.case.ensemble_size,
            num_steps=int(num_steps),
            observations=observations[:, :, len_field_history:],
        )


def _scalarize(curves: dict[str, np.ndarray]) -> dict[str, float]:
    """Reduce per-step metric arrays to a single scalar per metric."""
    return {k: float(np.nanmean(v)) for k, v in curves.items()}


def _format_multiplier(m: float) -> str:
    s = f"{m:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


_CONFIG_KEYS = {"test_id", "multiplier", "apply_full", "num_steps"}


def _write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    keys = ["test_id", "multiplier", "apply_full", "num_steps"] + [
        k for k in rows[0].keys() if k not in _CONFIG_KEYS
    ]
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in keys))
    out_path = out_dir / "summary.csv"
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {out_path}")


def _plot_metric_vs_multiplier(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    metric_names = [k for k in rows[0].keys() if k not in _CONFIG_KEYS]
    multipliers = sorted({r["multiplier"] for r in rows})
    apply_full_modes = sorted({bool(r["apply_full"]) for r in rows})
    num_steps_values = sorted({int(r["num_steps"]) for r in rows})

    fig, axes = plt.subplots(
        1, len(metric_names), figsize=(5 * len(metric_names), 4), squeeze=False
    )
    axes = axes[0]

    for ax, metric in zip(axes, metric_names):
        for apply_full in apply_full_modes:
            for ns in num_steps_values:
                means: list[float] = []
                stds: list[float] = []
                for m in multipliers:
                    vals = [
                        r[metric]
                        for r in rows
                        if r["multiplier"] == m
                        and bool(r["apply_full"]) == apply_full
                        and int(r["num_steps"]) == ns
                    ]
                    if not vals:
                        means.append(np.nan)
                        stds.append(np.nan)
                        continue
                    means.append(float(np.nanmean(vals)))
                    stds.append(float(np.nanstd(vals)))
                full_tag = (
                    "scale full expression" if apply_full else "scale score term only"
                )
                label_parts = []
                if len(apply_full_modes) > 1:
                    label_parts.append(full_tag)
                if len(num_steps_values) > 1:
                    label_parts.append(f"num_steps={ns}")
                label = " | ".join(label_parts) if label_parts else full_tag
                ax.errorbar(
                    multipliers,
                    np.asarray(means),
                    yerr=np.asarray(stds),
                    marker="o",
                    capsize=3,
                    label=label,
                )
        ax.set_title(metric)
        ax.set_xlabel("correction_multiplier")
        ax.set_ylabel(metric)
        ax.grid(True, linestyle=":", linewidth=0.6)
        if len(apply_full_modes) > 1 or len(num_steps_values) > 1:
            ax.legend(fontsize="small")

    fig.suptitle(
        "Posterior metric vs correction_multiplier (mean ± std across test_ids)"
    )
    fig.tight_layout()
    out_path = out_dir / "metric_vs_multiplier.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Wrote {out_path}")


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="benchmark",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.method.name != "si":
        raise ValueError(
            f"sweep_correction_multiplier requires method=si (got "
            f"method={cfg.method.name}). The correction_multiplier hyperparameter "
            f"only exists on InterpolantGaussianLikelihood."
        )

    multipliers_cfg = OmegaConf.select(cfg, "sweep.multipliers", default=None)
    multipliers = (
        [float(m) for m in multipliers_cfg]
        if multipliers_cfg is not None
        else list(DEFAULT_MULTIPLIERS)
    )

    apply_full_cfg = OmegaConf.select(cfg, "sweep.apply_full", default=None)
    apply_full_modes = (
        [bool(v) for v in apply_full_cfg]
        if apply_full_cfg is not None
        else list(DEFAULT_APPLY_FULL)
    )

    num_steps_cfg = OmegaConf.select(cfg, "sweep.num_steps", default=None)
    num_steps_values = (
        [int(v) for v in num_steps_cfg]
        if num_steps_cfg is not None
        else [int(cfg.case.num_steps)]
    )

    test_indices = cfg.case.test_sample_indices
    test_indices = (
        [test_indices] if isinstance(test_indices, int) else list(test_indices)
    )

    logger.info(f"Case  : {cfg.case.name}")
    logger.info(f"Method: {cfg.method.name}")
    logger.info(f"Test sample indices: {test_indices}")
    logger.info(f"Multipliers: {multipliers}")
    logger.info(f"apply_multiplier_to_full_expression modes: {apply_full_modes}")
    logger.info(f"num_steps values: {num_steps_values}")

    torch.manual_seed(42)

    checkpoint_name = _resolve_checkpoint_name(cfg)
    logger.info(f"Checkpoint: {cfg.case.project}/{checkpoint_name}")

    model, train_cfg, len_field_history = _load_pretrained(
        cfg.case.project, checkpoint_name
    )
    preprocesser = hydra.utils.instantiate(train_cfg.preprocesser)
    test_dataset = hydra.utils.instantiate(train_cfg.test_data)

    out_dir = Path(cfg.results_root) / cfg.case.name / "multiplier_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, out_dir / "config.yaml")

    enstrophy_extra = cfg.case.name == "stochastic_navier_stokes"
    rows: list[dict[str, Any]] = []

    for test_id in test_indices:
        prepped = _prepare_inputs(
            test_dataset=test_dataset,
            test_sample_index=int(test_id),
            preprocesser=preprocesser,
            len_field_history=len_field_history,
        )
        init_data = prepped["init_data"]
        trajectory = prepped["trajectory"]

        obs_operator = hydra.utils.instantiate(
            cfg.case.obs_operator,
            data_size=init_data["base"][0].shape,
        )
        logger.info(
            f"obs operator: num_obs={obs_operator.num_obs} "
            f"({obs_operator.num_obs / obs_operator.num_dofs * 100:.2f}% observed)"
        )

        obs_seed = 1000 * int(test_id) + 7
        observations = _make_observations(
            trajectory=trajectory,
            obs_operator=obs_operator,
            num_physical_steps=cfg.case.num_physical_steps,
            variance=cfg.case.variance,
            seed=obs_seed,
        )

        true_trajectory = preprocesser.inverse_transform(
            base=trajectory.cpu(), is_batch=True, is_trajectory=True
        )["base"]
        true_hwt = true_trajectory[0, 0]  # [H, W, T]

        for multiplier in multipliers:
            for apply_full in apply_full_modes:
                for num_steps in num_steps_values:
                    logger.info(
                        f"Sampling posterior  test_id={test_id}  "
                        f"multiplier={multiplier}  apply_full={apply_full}  "
                        f"num_steps={num_steps}"
                    )
                    posterior_model = _instantiate_posterior_with_multiplier(
                        cfg, model, obs_operator, multiplier, apply_full
                    )

                    posterior_trajectory = _sample_posterior(
                        cfg=cfg,
                        model=model,
                        posterior_model=posterior_model,
                        init_data=init_data,
                        observations=observations,
                        len_field_history=len_field_history,
                        seed=42 + int(test_id),
                        num_steps=num_steps,
                    )

                    posterior_trajectory = preprocesser.inverse_transform(
                        base=posterior_trajectory, is_batch=True, is_trajectory=True
                    )["base"].cpu()

                    curves = _per_step_metrics(
                        true_hwt=true_hwt,
                        pred_traj=posterior_trajectory,
                        len_field_history=len_field_history,
                        num_physical_steps=cfg.case.num_physical_steps,
                    )
                    if enstrophy_extra:
                        curves["Enstrophy error"] = _enstrophy_error_curve(
                            true_hwt,
                            posterior_trajectory,
                            len_field_history,
                            cfg.case.num_physical_steps,
                        )

                    scalars = _scalarize(curves)

                    payload = {
                        "meta": {
                            "case": cfg.case.name,
                            "method": cfg.method.name,
                            "checkpoint_name": checkpoint_name,
                            "test_sample_index": int(test_id),
                            "correction_multiplier": float(multiplier),
                            "apply_multiplier_to_full_expression": bool(apply_full),
                            "num_steps": int(num_steps),
                            "ensemble_size": int(cfg.case.ensemble_size),
                            "len_field_history": int(len_field_history),
                            "num_physical_steps": int(cfg.case.num_physical_steps),
                            "variance": float(cfg.case.variance),
                            "obs_seed": int(obs_seed),
                            "sampler_seed": int(42 + int(test_id)),
                        },
                        "metrics_per_step": {
                            k: np.asarray(v) for k, v in curves.items()
                        },
                        "metric_means": scalars,
                    }
                    mult_str = _format_multiplier(multiplier)
                    full_str = "true" if apply_full else "false"
                    out_path = (
                        out_dir
                        / f"sample_{int(test_id)}_mult_{mult_str}"
                        f"_full_{full_str}_steps_{int(num_steps)}.pt"
                    )
                    torch.save(payload, out_path)
                    scalars_str = ", ".join(
                        f"{k}={v:.4g}" for k, v in scalars.items()
                    )
                    logger.info(f"  Saved {out_path.name}  {scalars_str}")

                    row: dict[str, Any] = {
                        "test_id": int(test_id),
                        "multiplier": float(multiplier),
                        "apply_full": bool(apply_full),
                        "num_steps": int(num_steps),
                    }
                    row.update(scalars)
                    rows.append(row)

    _write_summary(rows, out_dir)
    _plot_metric_vs_multiplier(rows, out_dir)


if __name__ == "__main__":
    main()
