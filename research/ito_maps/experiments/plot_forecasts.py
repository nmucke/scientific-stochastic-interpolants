"""Plot a predefined held-out forecast case without selecting samples by quality.

Example (each checkpoint has an explicit map/solver step budget):
    python research/ito_maps/experiments/plot_forecasts.py \\
      --checkpoints checkpoints/stochastic_navier_stokes/follmer_ito_map \\
                    checkpoints/stochastic_navier_stokes/stochastic_interpolant \\
      --steps 8 64 --output research/ito_maps/results/qualitative

Defaults are fixed in advance: test trajectory 180, origin 45, history 5,
16 ensemble members, first member displayed, seed 20261007. The target field
is used only for plotting, never as a forecast condition.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scisi.bin.main_map_model_train import load_teacher, seed_everything
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant


def render_forecasts(observed, truth, ensembles, labels, output, *, units, case_label):
    """Render one scalar channel with scales shared across all displayed models."""
    means = [members.mean(0) for members in ensembles]
    stds = [members.std(0, ddof=1) for members in ensembles]
    first = [members[0] for members in ensembles]
    fields = [observed, truth, *means, *first]
    limit = max(float(np.abs(field).max()) for field in fields)
    limit = max(limit, 1e-12)
    std_limit = max(max(float(field.max()) for field in stds), 1e-12)
    field_norm, std_norm = Normalize(-limit, limit), Normalize(0, std_limit)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(
        len(ensembles) + 1,
        3,
        figsize=(7.1, 2.05 * (len(ensembles) + 1)),
        layout="constrained",
    )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    field_image = axes[0, 0].imshow(
        observed, cmap="RdBu_r", norm=field_norm, origin="lower"
    )
    axes[0, 0].set_title("Latest observed field")
    axes[0, 1].imshow(truth, cmap="RdBu_r", norm=field_norm, origin="lower")
    axes[0, 1].set_title("Next-state truth")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.05,
        1.0,
        case_label + "\nFirst member; no sample selection.",
        transform=axes[0, 2].transAxes,
        va="top",
        fontsize=9,
    )
    for row, (label, mean, std, sample) in enumerate(
        zip(labels, means, stds, first), start=1
    ):
        axes[row, 0].imshow(mean, cmap="RdBu_r", norm=field_norm, origin="lower")
        std_image = axes[row, 1].imshow(
            std, cmap="viridis", norm=std_norm, origin="lower"
        )
        axes[row, 2].imshow(sample, cmap="RdBu_r", norm=field_norm, origin="lower")
        axes[row, 0].set_title("Ensemble mean")
        axes[row, 1].set_title("Ensemble std.")
        axes[row, 2].set_title("First sample")
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.colorbar(
        field_image,
        cax=axes[0, 2].inset_axes([0.05, 0.28, 0.85, 0.055]),
        orientation="horizontal",
        label="Vorticity (" + units + ")",
    )
    fig.colorbar(
        std_image,
        cax=axes[0, 2].inset_axes([0.05, -0.08, 0.85, 0.055]),
        orientation="horizontal",
        label="Std. (" + units + ")",
    )
    fig.savefig(output / "forecast_fields.pdf", bbox_inches="tight")
    fig.savefig(output / "forecast_fields.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "field_color_limits": [-limit, limit],
        "std_color_limits": [0, std_limit],
        "field_colormap": "RdBu_r",
        "std_colormap": "viridis",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--trajectory", type=int, default=180)
    parser.add_argument("--origin", type=int, default=45)
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--units", choices=["physical", "normalized"], default="physical"
    )
    parser.add_argument(
        "--save-ensembles",
        action="store_true",
        help="Save optional ensembles.npz (ignored by repository git rules)",
    )
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.steps) or any(n < 1 for n in args.steps):
        parser.error("Each checkpoint requires exactly one positive step budget")
    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        parser.error("Provide exactly one label per checkpoint")
    if not 180 <= args.trajectory < 200:
        parser.error(
            "This qualitative script uses only the predefined held-out test split [180,200)"
        )
    if args.ensemble < 2 or args.history < 1:
        parser.error("Need at least two ensemble members and positive history")
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference_cfg = OmegaConf.load(Path(args.checkpoints[0]) / "config.yaml")
    data_cfg = reference_cfg.test_data.copy()
    data_cfg.tajectories_ids = f"({args.trajectory},{args.trajectory+1})"
    data_cfg.len_field_history = args.history
    data_cfg.train_or_test = "test"
    dataset = instantiate(data_cfg, preprocesser=None)
    raw = dataset[0]["x"]
    if raw.shape[0] != 1:
        raise ValueError("This Navier–Stokes figure expects a scalar vorticity channel")
    if args.origin < args.history or args.origin >= raw.shape[-1]:
        raise ValueError("Forecast origin is outside the available trajectory")
    raw_history = raw[..., args.origin - args.history : args.origin].unsqueeze(0)
    truth = raw[..., args.origin]
    observed = raw[..., args.origin - 1]
    ensembles, labels, sources = [], [], []
    normalization = None
    for index, (checkpoint, steps) in enumerate(zip(args.checkpoints, args.steps)):
        model, cfg = load_teacher(checkpoint, args.device)
        if int(cfg.len_field_history) != args.history:
            raise ValueError("Every checkpoint must use the specified history length")
        norm_cfg = (
            OmegaConf.to_container(cfg.preprocesser, resolve=True)
            if cfg.get("preprocesser") is not None
            else None
        )
        if index == 0:
            normalization = norm_cfg
        elif norm_cfg != normalization:
            raise ValueError(
                "Checkpoint preprocessing differs; use matched normalization"
            )
        preprocessor = instantiate(cfg.preprocesser) if norm_cfg is not None else None
        history = (
            preprocessor.transform(field_history=raw_history, is_batch=True)[
                "field_history"
            ]
            if preprocessor is not None
            else raw_history.clone()
        )
        # The same seed across budgets is not a claim of pathwise coupling.
        case_seed = args.seed + 10000 * (args.trajectory - 180) + 100 * args.origin
        seed_everything(case_seed)
        with torch.no_grad():
            members = model.sample(
                field_history=history,
                batch_size=args.ensemble,
                num_steps=steps,
                base=(
                    history[..., -1]
                    if isinstance(model, FollmerStochasticInterpolant)
                    else None
                ),
            )
        if not torch.isfinite(members).all():
            raise FloatingPointError(f"Nonfinite ensemble from {checkpoint}")
        if args.units == "physical" and preprocessor is not None:
            members = preprocessor.inverse_transform(target=members, is_batch=True)[
                "target"
            ]
        if index == 0 and args.units == "normalized" and preprocessor is not None:
            # Display all fields using the target coordinate transform.
            observed = preprocessor.transform(target=observed)["target"]
            truth = preprocessor.transform(target=truth)["target"]
        ensembles.append(members[:, 0].cpu().numpy())
        label = args.labels[index] if args.labels is not None else type(model).__name__
        labels.append(f"{label}\n{steps} steps")
        path = Path(checkpoint) / "model.pth"
        sources.append(
            {
                "label": label,
                "steps": steps,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
        del model
    observed_array, truth_array = observed[0].cpu().numpy(), truth[0].cpu().numpy()
    units = "physical units" if args.units == "physical" else "normalized units"
    case_label = f"Test trajectory {args.trajectory}\nForecast origin {args.origin}; history {args.history}\nE={args.ensemble}; base seed {args.seed}"
    scales = render_forecasts(
        observed_array,
        truth_array,
        ensembles,
        labels,
        output,
        units=units,
        case_label=case_label,
    )
    metadata = {
        "split": "test",
        "trajectory": args.trajectory,
        "origin": args.origin,
        "history_length": args.history,
        "ensemble_size": args.ensemble,
        "base_seed": args.seed,
        "effective_case_seed": case_seed,
        "units": args.units,
        "sample_index": 0,
        "selection": "Fixed trajectory, origin, seed, and first sample; no quality-based selection",
        "conditioning_indices": list(range(args.origin - args.history, args.origin)),
        "truth_index": args.origin,
        "preprocesser": normalization,
        "sources": sources,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "hardware": (
            torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else platform.machine()
        ),
        **scales,
    }
    (output / "forecast_fields.json").write_text(json.dumps(metadata, indent=2))
    if args.save_ensembles:
        np.savez_compressed(
            output / "ensembles.npz",
            observed=observed_array,
            truth=truth_array,
            **{f"ensemble_{index}": members for index, members in enumerate(ensembles)},
        )
    print(f"Saved predefined test-case figure and provenance to {output}")


if __name__ == "__main__":
    main()
