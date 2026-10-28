import argparse
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
import contextlib
import logging
import os
import pdb

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sympy.integrals.laplace import I

from scisi.bin.main_map_model_train import seed_everything
from scisi.metrics import (
    crps,
    ensemble_mean_rmse,
    radial_enstrophy_spectrum,
    rank_histogram,
    spread_skill,
)
from scisi.metrics.lsim import LSiM_distance
from scisi.metrics.spectral import compute_enstrophy_error, get_enstrophy_spectrum
from scisi.plotting.animation import create_animation_from_tensors
from scisi.plotting.plot_fields import plot_fields
from scisi.plotting.plot_point_distributions import plot_point_distributions
from scisi.plotting.spectrum import plot_enstrophy_spectrum
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step
from scisi.utils.device_utils import set_device
from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

SHOW_PLOTS = False
VERBOSE = True
MIXED_PRECISION = True

DEFAULT_PROJECT = "stochastic_navier_stokes"
DEFAULT_NAME = "follmer_ito_map"


NUM_PHYSICAL_STEPS = 100
NUM_STEPS = 5
BATCH_SIZE = 5
PLOTTING_TIMES = [5, NUM_PHYSICAL_STEPS // 2, NUM_PHYSICAL_STEPS - 1]
TEST_SAMPLE_INDEX = 0

mixed_precision_context = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if MIXED_PRECISION
    else contextlib.nullcontext()
)


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


def main(cfg: DictConfig, project: str, name: str) -> None:
    """Main function."""
    logger.info(f"Model is {cfg.model._target_}...")

    len_field_history = cfg.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info(f"Instantiating model...")
    model = hydra.utils.instantiate(cfg.model)
    logger.info(f"Loading model from checkpoint...")
    model.load_state_dict(
        torch.load(f"checkpoints/{project}/{name}/model.pth", map_location="cpu")
    )
    model.eval()
    model.to(cfg.trainer.device)

    logger.info(f"Preparing and preprocessing trajectory...")
    trajectory = test_dataset[TEST_SAMPLE_INDEX]["x"].unsqueeze(0)
    try:
        field_cond = test_dataset[TEST_SAMPLE_INDEX]["field_cond"].unsqueeze(0)
    except:
        field_cond = None
    try:
        pars_cond = test_dataset[TEST_SAMPLE_INDEX]["pars_cond"].unsqueeze(0)
    except:
        pars_cond = None
    init_data = preprocesser.transform(
        base=trajectory[..., len_field_history - 1],
        field_history=trajectory[..., 0:len_field_history],
        is_batch=True,
    )
    init_data["field_cond"] = preprocesser.transform(
        field_cond=field_cond if field_cond is not None else None,
        is_batch=True,
        is_trajectory=True,
    )["field_cond"]
    init_data["pars_cond"] = preprocesser.transform(
        pars_cond=pars_cond if pars_cond is not None else None,
        is_batch=True,
        is_trajectory=True,
    )["pars_cond"]

    logger.info(f"Moving data to the correct device...")
    init_data = {
        k: v.to(cfg.trainer.device) for k, v in init_data.items() if v is not None
    }

    if not isinstance(model, FollmerItoMapModel):
        logger.info(f"Model is a {type(model)}. Setting base to None...")
        init_data["base"] = None

    # Use mixed precision if available
    logger.info(
        f"Sampling from the model using mixed precision..."
        if MIXED_PRECISION
        else f"Sampling from the model using full precision..."
    )
    with mixed_precision_context:
        predicted_trajectory = model.sample_trajectory(
            **init_data,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS,
            num_physical_steps=NUM_PHYSICAL_STEPS,
        )

    true_trajectory = trajectory.cpu()
    predicted_trajectory = predicted_trajectory.cpu()

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory,
        field_cond=field_cond,
        is_batch=True,
        is_trajectory=True,
    )["base"]

    figure_path = f"figures/{project}"
    os.makedirs(figure_path, exist_ok=True)

    logger.info(f"Creating animation...")
    
    create_animation_from_tensors(
        [true_trajectory[0, 0, :, :, 0:NUM_PHYSICAL_STEPS]]
        + [predicted_trajectory[i, 0] for i in range(BATCH_SIZE)],
        fps=10,
        file_name=f"{figure_path}/predicted_trajectory.mp4",
        colormaps="viridis",
        titles=["True"] + [f"Ensemble member {i}" for i in range(BATCH_SIZE)],
        vmin=np.min(true_trajectory.numpy()),
        vmax=np.max(true_trajectory.numpy()),
        normalize=False,
    )

    #### Compute metrics ####
    predicted_trajectory = predicted_trajectory[0, 0]
    true_trajectory = trajectory[0, 0].cpu()
    logger.info(f"Computing metrics...")
    lsim = [
        LSiM_distance(true_trajectory[:, :, i], predicted_trajectory[:, :, i])
        for i in range(NUM_PHYSICAL_STEPS)
    ]
    logger.info(f"LSiM: {np.mean(lsim):.4f} ± {np.std(lsim):.4f}")

    rmse = [
        torch.sqrt(
            nn.MSELoss()(true_trajectory[:, :, i], predicted_trajectory[:, :, i])
        )
        for i in range(NUM_PHYSICAL_STEPS)
    ]
    logger.info(f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")

    plot_enstrophy_spectrum(
        trajectories=[true_trajectory, predicted_trajectory],
        titles=["True", "Predicted"],
        figure_path=figure_path,
        show=SHOW_PLOTS,
    )
    ens_error, ens_error_array = compute_enstrophy_error(
        true_trajectory, predicted_trajectory, 2 * torch.pi / 128
    )
    logger.info(f"Enstrophy error: {ens_error:.4f} ± {ens_error_array.std():.4f}")


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse command line arguments for project and name
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help="Project name (default: %(default)s)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help="Model name (default: %(default)s)",
    )
    args, unknown = parser.parse_known_args()

    # Construct config directory path
    config_dir = os.path.abspath(f"checkpoints/{args.project}/{args.name}")

    logger.info(f"Loading config from: {config_dir}")
    logger.info(f"Project: {args.project}")
    logger.info(f"Name: {args.name}")

    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose the config, allowing overrides from command line
        cfg = compose(config_name="config", overrides=unknown)
        main(cfg, args.project, args.name)
