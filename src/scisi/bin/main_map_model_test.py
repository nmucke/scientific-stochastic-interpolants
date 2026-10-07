"""Reproducible ensemble forecast evaluation for map and solver models."""

import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from scisi.bin.main_map_model_train import seed_everything
from scisi.metrics import (
    crps,
    ensemble_mean_rmse,
    radial_enstrophy_spectrum,
    rank_histogram,
    spread_skill,
)
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


def forecast_metrics(ensemble, target):
    """Metrics in the given coordinates, with ensemble on the first axis."""
    flat = ensemble.flatten(1)
    norm = flat.shape[1] ** 0.5
    energy = (
        torch.linalg.vector_norm(flat - target.flatten()[None], dim=1).mean() / norm
    )
    if len(ensemble) > 1:
        energy -= torch.pdist(flat).mean() / (2 * norm)
    spread = (
        spread_skill(ensemble, target)
        if len(ensemble) > 1
        else {"spread": torch.tensor(0.0), "ratio": torch.tensor(0.0)}
    )
    lower, upper = torch.quantile(
        ensemble, torch.tensor([0.05, 0.95], device=ensemble.device), dim=0
    )
    _, truth_spectrum = radial_enstrophy_spectrum(target)
    spectra = torch.stack([radial_enstrophy_spectrum(member)[1] for member in ensemble])
    spectral_relative = torch.linalg.vector_norm(
        spectra.mean(0) - truth_spectrum
    ) / torch.linalg.vector_norm(truth_spectrum).clamp_min(1e-12)
    rmse = ensemble_mean_rmse(ensemble, target)
    rms_spread = (
        ensemble.var(dim=0, unbiased=True).mean().sqrt()
        if len(ensemble) > 1
        else ensemble.new_zeros(())
    )
    rms_ratio = ((len(ensemble) + 1) / len(ensemble)) ** 0.5 * rms_spread / rmse
    ranks = rank_histogram(ensemble, target)
    extreme_fraction = (ranks[0] + ranks[-1]).double() / ranks.sum()
    return {
        "crps": float(crps(ensemble, target)),
        "rmse": float(rmse),
        "energy_score": float(energy),
        "spread": float(spread["spread"]),
        "spread_skill": float(spread["ratio"]),
        "coverage90": float(((target >= lower) & (target <= upper)).float().mean()),
        "relative_enstrophy_error": float(spectral_relative),
        "rank_counts": ranks.tolist(),
        "rms_spread": float(rms_spread),
        "rms_spread_skill": float(rms_ratio),
        "rank_extreme_fraction": float(extreme_fraction),
    }


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def aggregate_rows(rows, bootstrap_seed=712, n_bootstrap=2000):
    """Average origins within a trajectory; resample entire test trajectories."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["num_steps"], row["lead"])].append(row)
    rng = np.random.default_rng(bootstrap_seed)
    extra_rng = np.random.default_rng(bootstrap_seed + 1)
    optional_metrics = ("rms_spread", "rms_spread_skill", "rank_extreme_fraction")
    result = []
    for (steps, lead), items in sorted(groups.items()):
        trajectories = sorted({r["trajectory"] for r in items})
        summary = {
            "num_steps": steps,
            "lead": lead,
            "num_trajectories": len(trajectories),
            "num_forecasts": len(items),
        }
        for key in (
            "crps",
            "rmse",
            "energy_score",
            "spread",
            "spread_skill",
            "coverage90",
            "relative_enstrophy_error",
            "seconds",
            "nfe",
        ) + tuple(key for key in optional_metrics if all(key in r for r in items)):
            per_traj = np.array(
                [
                    np.mean([r[key] for r in items if r["trajectory"] == tr])
                    for tr in trajectories
                ]
            )
            # Preserve the legacy metrics' bootstrap draws when optional fields
            # are added; their uncertainty estimates remain byte-for-byte stable.
            metric_rng = extra_rng if key in optional_metrics else rng
            boot = per_traj[
                metric_rng.integers(0, len(per_traj), (n_bootstrap, len(per_traj)))
            ].mean(1)
            summary[key] = float(per_traj.mean())
            summary[key + "_ci95"] = [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ]
        summary["rank_counts"] = np.sum(
            [r["rank_counts"] for r in items], axis=0
        ).tolist()
        result.append(summary)
    return result


def evaluate_model(
    model,
    dataset,
    preprocesser,
    *,
    num_steps=(1, 2, 4, 8),
    ensemble_size=16,
    origins=(5, 25, 45, 65, 85),
    num_trajectories=None,
    forecast_horizon=1,
    seed=20261007,
    device="cuda",
    output_dir=None,
    trajectory_offset=0,
):
    if ensemble_size < 2 or forecast_horizon < 1:
        raise ValueError(
            "Evaluation requires at least two ensemble members and a positive horizon"
        )
    num_steps, origins = tuple(num_steps), tuple(origins)
    if not num_steps or any(not isinstance(n, int) or n < 1 for n in num_steps):
        raise ValueError("num_steps must contain positive integer sampling budgets")
    if not origins or any(not isinstance(origin, int) for origin in origins):
        raise ValueError("origins must contain integer forecast origins")
    if not len(dataset):
        raise ValueError("The evaluation dataset must not be empty")
    if num_trajectories is not None and (
        not isinstance(num_trajectories, int) or num_trajectories < 1
    ):
        raise ValueError("num_trajectories must be a positive integer or None")
    model.to(device).eval()
    length = dataset.len_field_history
    count = min(
        num_trajectories if num_trajectories is not None else len(dataset), len(dataset)
    )
    rows = []
    nfe = [0]

    def count_call(_module, _args, _output):
        nfe[0] += 1

    hook = model.model.register_forward_hook(count_call)
    try:
        # Warm up without including initialization/kernel selection in latency.
        warm = dataset[0]["x"][..., :length].unsqueeze(0)
        warm = (
            preprocesser.transform(field_history=warm, is_batch=True)["field_history"]
            if preprocesser is not None
            else warm
        )
        with torch.no_grad():
            model.sample(
                field_history=warm,
                batch_size=ensemble_size,
                num_steps=1,
                base=(
                    warm[..., -1]
                    if isinstance(model, FollmerStochasticInterpolant)
                    else None
                ),
            )
        _sync(device)
        for steps in num_steps:
            for trajectory in range(count):
                truth = dataset[trajectory]["x"]
                for origin in origins:
                    if origin < length or origin + forecast_horizon > truth.shape[-1]:
                        raise ValueError(
                            "Forecast origin/horizon falls outside available data"
                        )
                    history = truth[..., origin - length : origin].unsqueeze(0)
                    target = truth[..., origin : origin + forecast_horizon]
                    if preprocesser is not None:
                        history = preprocesser.transform(
                            field_history=history, is_batch=True
                        )["field_history"]
                        target = preprocesser.transform(
                            target=target, is_trajectory=True
                        )["target"]
                    # Repeat the same seed across numerical budgets. This is a
                    # common random seed, not a claim of shared Brownian paths.
                    seed_everything(seed + 10000 * trajectory + 100 * origin)
                    for lead in range(1, forecast_horizon + 1):
                        nfe[0] = 0
                        _sync(device)
                        start = time.perf_counter()
                        with torch.no_grad():
                            ensemble, history = model.sample(
                                field_history=history,
                                batch_size=ensemble_size,
                                num_steps=int(steps),
                                base=(
                                    history[..., -1]
                                    if isinstance(model, FollmerStochasticInterpolant)
                                    else None
                                ),
                                return_field_history=True,
                            )
                        _sync(device)
                        seconds = time.perf_counter() - start
                        # Rank ties use random draws; scoring must not alter the
                        # random stream used by the next physical forecast.
                        with torch.random.fork_rng(devices=[]):
                            metrics = forecast_metrics(ensemble, target[..., lead - 1])
                        rows.append(
                            {
                                "trajectory": trajectory + trajectory_offset,
                                "origin": origin,
                                "lead": lead,
                                "num_steps": int(steps),
                                "ensemble_size": ensemble_size,
                                "seconds": seconds,
                                "nfe": nfe[0],
                                **metrics,
                            }
                        )
                if output_dir is not None:
                    destination = Path(output_dir)
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / "forecasts.json").write_text(
                        json.dumps(rows, indent=2)
                    )
                    print(
                        f"evaluation steps={steps} trajectory={trajectory+1}/{count}",
                        flush=True,
                    )
    finally:
        hook.remove()
    summaries = aggregate_rows(rows)
    result = {
        "coordinate_system": (
            "dataset normalized coordinates"
            if preprocesser is not None
            else "dataset original coordinates"
        ),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "device": str(device),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "gpu": (
                torch.cuda.get_device_name(device)
                if torch.device(device).type == "cuda"
                else None
            ),
        },
        "torch_version": str(torch.__version__),
        "seed": seed,
        "ensemble_size": ensemble_size,
        "forecast_horizon": forecast_horizon,
        "rows": rows,
        "summary": summaries,
    }
    if output_dir is not None:
        (Path(output_dir) / "evaluation.json").write_text(json.dumps(result, indent=2))
    return result


def run_evaluation(cfg):
    torch.set_num_threads(int(cfg.get("num_threads", 4)))
    evaluation = cfg.evaluation
    directory = Path(evaluation.checkpoint_dir)
    saved = OmegaConf.load(directory / "config.yaml")
    model = instantiate(saved.model)
    checkpoint = directory / evaluation.get("checkpoint_name", "model.pth")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    preprocesser = (
        instantiate(saved.preprocesser)
        if saved.get("preprocesser") is not None
        else None
    )
    dataset = instantiate(cfg.test_data, preprocesser=preprocesser)
    kwargs = {
        k: OmegaConf.to_container(v) if OmegaConf.is_config(v) else v
        for k, v in evaluation.items()
        if k not in ("checkpoint_dir", "checkpoint_name")
    }
    if "trajectory_offset" not in kwargs:
        trajectory_ids = cfg.test_data.get("tajectories_ids")
        if isinstance(trajectory_ids, str):
            kwargs["trajectory_offset"] = int(trajectory_ids.strip("()").split(",")[0])
        elif trajectory_ids is not None:
            # An arbitrary explicit selection has no meaningful constant offset;
            # preserve the exact source IDs after evaluating local dataset indices.
            kwargs["trajectory_offset"] = 0
    result = evaluate_model(model, dataset, preprocesser, **kwargs)
    trajectory_ids = cfg.test_data.get("tajectories_ids")
    if (
        trajectory_ids is not None
        and not isinstance(trajectory_ids, str)
        and "trajectory_offset" not in evaluation
    ):
        for row in result["rows"]:
            row["trajectory"] = int(trajectory_ids[row["trajectory"]])
        result["summary"] = aggregate_rows(result["rows"])
    result["checkpoint"] = str(checkpoint)
    result["tested_config"] = OmegaConf.to_container(cfg, resolve=True)
    result["model_config"] = OmegaConf.to_container(saved.model, resolve=True)
    result["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = Path(evaluation.output_dir)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    (output / "forecasts.json").write_text(json.dumps(result["rows"], indent=2))
    OmegaConf.save(cfg, output / "evaluation_config.yaml", resolve=True)
    return result


@hydra.main(
    config_path="../../../config",
    config_name="follmer_ito_map_stochastic_navier_stokes",
    version_base=None,
)
def main_map_model_test(cfg: DictConfig):
    return run_evaluation(cfg)


main = main_map_model_test

if __name__ == "__main__":
    main_map_model_test()
