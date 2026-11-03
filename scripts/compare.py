"""Compare trained models on one test trajectory, across numbers of steps.

Each model in config/compare.yaml forecasts the same test trajectory as an
ensemble, once per number of steps. Per time step it computes the metrics of
scripts/test.py (RMSE and LSiM of the first member), CRPS, the RMSE of the
ensemble mean, the ensemble spread and the spread-skill ratio. The averages
over the forecast are printed and saved to figure_dir/metrics.csv.

figure_dir/steps_<n>/ compares all models sampled with n steps, plus the
reference forecast:

* metrics.png: the metrics against time
* kinetic_energy.png: domain-averaged kinetic energy against time
* enstrophy_spectrum.png: enstrophy spectrum averaged over the forecast
* vorticity_distribution.png: distribution of the forecast vorticity values
* vorticity.mp4: the truth next to the first member of each model
* std.mp4: the pixel-wise ensemble standard deviation of each model

figure_dir/summary/ compares across numbers of steps:

* metrics_vs_steps.png: forecast-averaged metrics against the number of steps
* cost.png: CRPS against sampling time
* crps_vs_time.png, kinetic_energy.png, enstrophy_spectrum.png: one panel per
  model, one line per number of steps

A step is one network evaluation for map models and one solver step for the
others; a Heun step of the stochastic interpolant costs two evaluations, so
compare costs in summary/cost.png.

    python scripts/compare.py
    python scripts/compare.py test_sample_index=3 weights=ema
"""

import contextlib
import csv
import logging
import time
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from scisi.metrics import crps, radial_enstrophy_spectrum, spread_skill
from scisi.metrics.lsim import LSiM_distance
from scisi.models.base_map_model import BaseMapModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.plotting.animation import create_animation_from_tensors

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

TIME_METRICS = ["CRPS", "RMSE", "RMSE member", "spread", "spread-skill", "LSiM"]
SCALARS = TIME_METRICS + ["KE error", "spectrum error"]


# ------------------------------------------------------------------ forecast


def load_model(run_dir, weights, device):
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    model = hydra.utils.instantiate(cfg.model)
    path = Path(run_dir) / f"{weights}.pth"
    if not path.exists():  # e.g. no EMA weights for models not trained by MapTrainer
        logger.warning(f"{path} not found; using model.pth")
        path = Path(run_dir) / "model.pth"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval().to(device), cfg


def forecast(model, cfg, trajectory, test_cfg, num_steps):
    """Ensemble forecast [B, H, W, T] in physical units (the history included)."""
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)
    length = cfg.len_field_history
    init_data = preprocesser.transform(
        base=trajectory[..., length - 1],
        field_history=trajectory[..., :length],
        is_batch=True,
    )
    init_data = {
        k: v.to(test_cfg.device) for k, v in init_data.items() if v is not None
    }
    # Only the Föllmer models start from the current state; the rest from noise.
    if not isinstance(model, (FollmerStochasticInterpolant, FollmerItoMapModel)):
        init_data["base"] = None
    sampling_kwargs = {"num_steps": num_steps}
    if not isinstance(model, BaseMapModel):
        stepper = (
            test_cfg.ode_stepper
            if isinstance(model, FlowMatchingModel)
            else test_cfg.sde_stepper
        )
        sampling_kwargs["stepper"] = hydra.utils.get_method(stepper)

    use_mixed_precision = test_cfg.mixed_precision and test_cfg.device.startswith(
        "cuda"
    )
    torch.manual_seed(test_cfg.seed)
    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_mixed_precision
        else contextlib.nullcontext()
    ), torch.no_grad():
        predicted = model.sample_trajectory(
            **init_data,
            batch_size=test_cfg.batch_size,
            num_physical_steps=test_cfg.num_physical_steps,
            **sampling_kwargs,
        )
    predicted = preprocesser.inverse_transform(
        base=predicted.float().cpu(), is_batch=True, is_trajectory=True
    )["base"]
    return predicted[:, 0]


# ------------------------------------------------------------------- metrics


def kinetic_energy(vorticity):
    """Domain-averaged kinetic energy of vorticity [..., N, N] on [0, 2pi]^2."""
    n = vorticity.shape[-1]
    k = torch.fft.fftfreq(n, d=1.0 / n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    k2[0, 0] = float("inf")  # the mean mode carries no kinetic energy
    w_hat = torch.fft.fft2(vorticity, norm="forward")
    return 0.5 * (w_hat.abs() ** 2 / k2).sum(dim=(-2, -1))


def evaluate(predicted, true):
    """Statistics of a forecast; predicted [B, H, W, T], true [H, W, T]."""
    curves = {name: [] for name in TIME_METRICS}
    for t in range(true.shape[-1]):
        ensemble, target = predicted[..., t], true[..., t]
        skill = spread_skill(ensemble, target)
        curves["CRPS"].append(crps(ensemble, target).item())
        curves["RMSE"].append(float(skill["skill"]))
        curves["RMSE member"].append((ensemble[0] - target).pow(2).mean().sqrt().item())
        curves["spread"].append(float(skill["spread"]))
        curves["spread-skill"].append(float(skill["ratio"]))
        curves["LSiM"].append(float(LSiM_distance(target, ensemble[0])))
    curves = {name: np.array(values, dtype=float) for name, values in curves.items()}

    energy = kinetic_energy(predicted.permute(0, 3, 1, 2))  # [B, T]
    true_energy = kinetic_energy(true.permute(2, 0, 1))
    k, spectrum = radial_enstrophy_spectrum(predicted.permute(0, 3, 1, 2))
    _, true_spectrum = radial_enstrophy_spectrum(true.permute(2, 0, 1))
    valid = (spectrum > 0) & (true_spectrum > 0)
    scalars = {name: values.mean() for name, values in curves.items()}
    scalars["KE error"] = ((energy.mean(0) - true_energy).abs() / true_energy).mean().item()
    scalars["spectrum error"] = (
        (spectrum[valid].log10() - true_spectrum[valid].log10()).pow(2).mean().sqrt().item()
    )
    return {
        "curves": curves,
        "scalars": scalars,
        "energy": energy.numpy(),
        "spectrum": (k.numpy(), spectrum.numpy()),
    }


def print_table(results):
    width = max(len(name) for name, _ in results) + 2
    header = f"{'model':<{width}}{'steps':>6}" + "".join(f"{m:>15}" for m in SCALARS)
    print("\nMean over the forecast (excluding the history)")
    print(header + f"{'time [s]':>10}")
    print("-" * (len(header) + 10))
    previous = None
    for (name, steps), result in results.items():
        if previous is not None and name != previous:
            print()
        previous = name
        values = "".join(f"{result['scalars'][m]:>15.4f}" for m in SCALARS)
        print(f"{name:<{width}}{steps:>6}{values}{result['seconds']:>10.1f}")
    print()


def save_csv(results, path):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "steps", *SCALARS, "seconds"])
        for (name, steps), result in results.items():
            writer.writerow(
                [name, steps, *(result["scalars"][m] for m in SCALARS), result["seconds"]]
            )


# --------------------------------------------------- figures per step count


def plot_metrics(results, times, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, metric in zip(axes.flat, TIME_METRICS):
        for label, result in results.items():
            ax.plot(times, result["curves"][metric], label=label)
        if metric == "spread-skill":
            ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel("time step")
    axes.flat[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kinetic_energy(true, results, times, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, kinetic_energy(true.permute(2, 0, 1)), "k", linewidth=2, label="True")
    for label, result in results.items():
        mean, std = result["energy"].mean(0), result["energy"].std(0)
        (line,) = ax.plot(times, mean, label=label)
        ax.fill_between(times, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    ax.set_xlabel("time step")
    ax.set_ylabel("kinetic energy")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_enstrophy_spectrum(true, results, path):
    """Spectra averaged over the forecast steps and the ensemble."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    k, spectrum = radial_enstrophy_spectrum(true.permute(2, 0, 1))
    ax.loglog(k, spectrum, "k", linewidth=2, label="True")
    for label, result in results.items():
        k, spectrum = result["spectrum"]
        ax.loglog(k, spectrum, label=label)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("enstrophy Z(k)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vorticity_distribution(true, predictions, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    limit = 1.5 * true.abs().max().item()
    bins = np.linspace(-limit, limit, 100)
    ax.hist(true.flatten(), bins=bins, density=True, histtype="step", color="k",
            linewidth=2, label="True")
    for label, predicted in predictions.items():
        ax.hist(predicted.flatten(), bins=bins, density=True, histtype="step", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("vorticity")
    ax.set_ylabel("density")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def animate(true, predictions, directory):
    labels = list(predictions)
    create_animation_from_tensors(
        [true] + [predictions[label][0] for label in labels],
        file_name=str(directory / "vorticity.mp4"),
        fps=10,
        titles=["True"] + labels,
        vmin=true.min().item(),
        vmax=true.max().item(),
        normalize=False,
        colorbar_label="vorticity",
    )
    stds = [predictions[label].std(0) for label in labels]
    create_animation_from_tensors(
        stds,
        file_name=str(directory / "std.mp4"),
        fps=10,
        titles=labels,
        vmin=0.0,
        vmax=max(std.max().item() for std in stds),
        normalize=False,
        colorbar_label="ensemble std",
    )


def plot_step_count(true, predictions, results, length, directory):
    """All figures comparing the models at one number of steps."""
    directory.mkdir(parents=True, exist_ok=True)
    times = np.arange(true.shape[-1])
    forecast_times = times[length:]
    true_forecast = true[..., length:]
    forecasts = {label: p[..., length:] for label, p in predictions.items()}
    plot_metrics(results, forecast_times, directory / "metrics.png")
    plot_kinetic_energy(true_forecast, results, forecast_times, directory / "kinetic_energy.png")
    plot_enstrophy_spectrum(true_forecast, results, directory / "enstrophy_spectrum.png")
    plot_vorticity_distribution(true_forecast, forecasts, directory / "vorticity_distribution.png")
    animate(true, predictions, directory)


# ------------------------------------------------- summary across step counts


def plot_metrics_vs_steps(results, reference, path):
    names = list(dict.fromkeys(name for name, _ in results))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, metric in zip(axes.flat, SCALARS):
        for name in names:
            steps = sorted(s for n, s in results if n == name)
            ax.plot(steps, [results[name, s]["scalars"][metric] for s in steps], "o-",
                    label=name)
        if reference is not None:
            ax.axhline(reference["scalars"][metric], color="k", linestyle="--",
                       linewidth=1, label="reference")
        if metric == "spread-skill":
            ax.axhline(1.0, color="k", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_title(metric)
        ax.set_xlabel("number of steps")
    axes.flat[0].legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(results, reference, path):
    names = list(dict.fromkeys(name for name, _ in results))
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        steps = sorted(s for n, s in results if n == name)
        seconds = [results[name, s]["seconds"] for s in steps]
        scores = [results[name, s]["scalars"]["CRPS"] for s in steps]
        (line,) = ax.plot(seconds, scores, "o-", label=name)
        for s, x, y in zip(steps, seconds, scores):
            ax.annotate(str(s), (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color=line.get_color())
    if reference is not None:
        ax.plot(reference["seconds"], reference["scalars"]["CRPS"], "k*", markersize=12,
                label="reference")
    ax.set_xscale("log")
    ax.set_xlabel("sampling time of the ensemble forecast [s]")
    ax.set_ylabel("CRPS (forecast mean)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_model(results, reference, true, length, kind, path):
    """One panel per model, one line per number of steps (kind: crps|energy|spectrum)."""
    names = list(dict.fromkeys(name for name, _ in results))
    columns = min(4, len(names))
    rows = -(-len(names) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.5 * rows),
                             sharex=True, squeeze=False)
    times = np.arange(length, length + true.shape[-1])
    for ax, name in zip(axes.flat, names):
        steps = sorted(s for n, s in results if n == name)
        colours = plt.cm.viridis(np.linspace(0, 0.9, len(steps)))
        if kind == "energy":
            ax.plot(times, kinetic_energy(true.permute(2, 0, 1)), "k", linewidth=2,
                    label="True")
        elif kind == "spectrum":
            k, spectrum = radial_enstrophy_spectrum(true.permute(2, 0, 1))
            ax.loglog(k, spectrum, "k", linewidth=2, label="True")
        for s, colour in zip(steps, colours):
            draw(ax, results[name, s], kind, times, colour, f"{s} steps")
        if reference is not None:
            draw(ax, reference, kind, times, "k", "reference", linestyle="--")
        ax.set_title(name, fontsize="medium")
        ax.legend(fontsize="x-small")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw(ax, result, kind, times, colour, label, linestyle="-"):
    if kind == "crps":
        ax.plot(times, result["curves"]["CRPS"], color=colour, linestyle=linestyle,
                label=label)
        ax.set_xlabel("time step")
        ax.set_ylabel("CRPS")
    elif kind == "energy":
        ax.plot(times, result["energy"].mean(0), color=colour, linestyle=linestyle,
                label=label)
        ax.set_xlabel("time step")
        ax.set_ylabel("kinetic energy")
    else:
        k, spectrum = result["spectrum"]
        ax.loglog(k, spectrum, color=colour, linestyle=linestyle, label=label)
        ax.set_xlabel("wavenumber k")
        ax.set_ylabel("enstrophy Z(k)")


# ---------------------------------------------------------------------- main


@hydra.main(config_path="../config", config_name="compare", version_base=None)
def main(test_cfg: DictConfig) -> None:
    figure_dir = Path(test_cfg.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models...")
    models = {}
    for name in [*test_cfg.models, *([test_cfg.reference.name] if test_cfg.reference else [])]:
        if name not in models:
            models[name] = load_model(
                f"checkpoints/{test_cfg.project}/{name}", test_cfg.weights, test_cfg.device
            )
    cfg = models[next(iter(models))][1]
    length = cfg.len_field_history
    if any(c.len_field_history != length for _, c in models.values()):
        raise ValueError("All models must use the same history length")
    trajectory = hydra.utils.instantiate(cfg.test_data)[test_cfg.test_sample_index]["x"]
    trajectory = trajectory.unsqueeze(0)
    true = trajectory[0, 0, ..., : test_cfg.num_physical_steps]

    def run(name, steps):
        logger.info(f"Forecasting with {name}, {steps} steps...")
        model, model_cfg = models[name]
        start = time.perf_counter()
        predicted = forecast(model, model_cfg, trajectory, test_cfg, steps)
        seconds = time.perf_counter() - start
        predicted = predicted[..., : true.shape[-1]]
        result = evaluate(predicted[..., length:], true[..., length:])
        result["seconds"] = seconds
        return predicted, result

    reference, reference_prediction, reference_label = None, None, None
    if test_cfg.reference:
        reference_label = f"{test_cfg.reference.name} ({test_cfg.reference.num_steps}, ref.)"
        reference_prediction, reference = run(
            test_cfg.reference.name, test_cfg.reference.num_steps
        )

    results = {}
    all_steps = sorted({s for steps in test_cfg.models.values() for s in steps})
    for steps in all_steps:
        predictions, step_results = {}, {}
        for name, model_steps in test_cfg.models.items():
            if steps in model_steps:
                predictions[name], results[name, steps] = run(name, steps)
                step_results[name] = results[name, steps]
        if reference is not None:
            predictions[reference_label] = reference_prediction
            step_results[reference_label] = reference
        logger.info(f"Plotting the {steps}-step comparison...")
        plot_step_count(true, predictions, step_results, length, figure_dir / f"steps_{steps}")

    order = list(test_cfg.models)
    results = dict(sorted(results.items(), key=lambda item: (order.index(item[0][0]), item[0][1])))
    table = dict(results)
    if reference is not None:
        table[reference_label, test_cfg.reference.num_steps] = reference
    print_table(table)
    save_csv(table, figure_dir / "metrics.csv")

    logger.info("Plotting the summary across numbers of steps...")
    summary = figure_dir / "summary"
    summary.mkdir(exist_ok=True)
    true_forecast = true[..., length:]
    plot_metrics_vs_steps(results, reference, summary / "metrics_vs_steps.png")
    plot_cost(results, reference, summary / "cost.png")
    plot_per_model(results, reference, true_forecast, length, "crps", summary / "crps_vs_time.png")
    plot_per_model(results, reference, true_forecast, length, "energy", summary / "kinetic_energy.png")
    plot_per_model(results, reference, true_forecast, length, "spectrum", summary / "enstrophy_spectrum.png")
    logger.info(f"Figures in {figure_dir}")


if __name__ == "__main__":
    main()
