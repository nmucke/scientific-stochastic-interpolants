#!/usr/bin/env python3
"""Estimate path information lost by local features of a frozen NS teacher.

For one fixed validation history, draw G coarse Brownian paths and R independent
bridge refinements of each. The variability of teacher endpoints *within* each
coarse group estimates E[Var(X_1 | path features)], the irreducible same-path
squared-error risk of the finite feature representation for the chosen teacher
and numerical solver. No map architecture is trained or evaluated here.

Example (run only when an accelerator has been assigned):
 .venv/bin/python research/ito_maps/experiments/ns_path_compression.py \
   --trajectory 160 --origin 45 --features 1 4 16 --groups 8 --replicas 4 \
   --teacher-steps 64 --batch-size 8 --device cuda

All coordinates are dataset-normalized. Defaults use validation data only.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from hydra.utils import instantiate

from scisi.bin.main_map_model_train import load_teacher
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.sampling.brownian_path import WeightedBrownianPath


@torch.no_grad()
def teacher_endpoint(teacher, base, history, path, num_steps):
    """Euler drift with exact, shared weighted-Brownian interval increments."""
    state = base.clone()
    dt = 1.0 / num_steps
    for step in range(num_steps):
        left, right = step / num_steps, (step + 1) / num_steps
        t = state.new_full((state.shape[0], 1), left)
        state = state + dt * teacher.drift(state, t, field_history=history)
        state = state + path.increment(left, right)
    return state


def compression_statistics(endpoints):
    """Unbiased hierarchical variance estimates averaged over state coordinates.

    endpoints has shape [G,R,...]. The variance of the G group means contains
    within/R Monte Carlo noise. Subtract this to estimate explained variance;
    the resulting total estimator is between_means + (1-1/R)*within. Unlike
    treating all G*R correlated members as independent, this respects nesting.
    """
    if endpoints.shape[0] < 2 or endpoints.shape[1] < 2:
        raise ValueError("At least two groups and two replicas are required")
    samples = endpoints.double().flatten(2)
    replicas = samples.shape[1]
    per_group_within = samples.var(dim=1, unbiased=True).mean(dim=1)
    within = per_group_within.mean()
    between_means = samples.mean(dim=1).var(dim=0, unbiased=True).mean()
    explained = between_means - within / replicas
    total = explained + within
    return {
        "within_coarse_path_variance": float(within),
        "variance_of_group_means": float(between_means),
        "explained_variance_estimate": float(explained),
        "total_conditional_forecast_variance_estimate": float(total),
        "omitted_variance_fraction_estimate": (
            float(within / total) if total > 0 else None
        ),
        "retained_std_fraction_estimate": (
            math.sqrt(float(explained / total))
            if explained >= 0 and total > 0
            else None
        ),
        "irreducible_same_path_rmse_estimate": math.sqrt(float(within)),
        "per_group_within_variance": per_group_within.tolist(),
        "within_variance_standard_error_across_groups": float(
            per_group_within.std(unbiased=True) / math.sqrt(len(per_group_within))
        ),
        "num_coarse_groups": samples.shape[0],
        "num_bridge_replicas": replicas,
    }


@torch.no_grad()
def conditional_endpoints(
    teacher, history, features, groups, replicas, teacher_steps, batch_size, seed
):
    """Draw nested coarse paths with independently seeded conditional bridges."""
    counts = sorted(set(features))
    if not counts or any(k < 1 or teacher_steps % k for k in counts):
        raise ValueError("Feature counts must be positive divisors of teacher_steps")
    if any(right % left for left, right in zip(counts[:-1], counts[1:])):
        raise ValueError("Feature counts must be nested integer refinements")
    if groups < 2 or replicas < 2 or batch_size < 1:
        raise ValueError("Need groups>=2, replicas>=2 and batch_size>=1")
    gamma_multiplier = float(teacher.interpolation.gamma_multiplier)
    device = history.device
    coarse_generator = torch.Generator(device=device).manual_seed(seed)
    # Independent deterministic seed streams for each K/replica pair; no resetting
    # of the global RNG and no duplicate generator states across sibling forks.
    generators = {
        (k, replica): torch.Generator(device=device).manual_seed(
            seed + 1000003 * (index + 1) + 9176 * (replica + 1)
        )
        for index, k in enumerate(counts)
        for replica in range(replicas)
    }
    outputs = {k: [] for k in counts}
    for start in range(0, groups, batch_size):
        count = min(batch_size, groups - start)
        context = history.expand(count, *history.shape[1:])
        base = context[..., -1]
        coarse_path = WeightedBrownianPath.sample_like(
            base, gamma_multiplier, coarse_generator
        )
        for k in counts:
            # The parent accumulates only requested coarse observations. Fine
            # solver queries occur on forks and never leak into later conditioning.
            coarse_path.features(0.0, 1.0, k)
            conditional = []
            for replica in range(replicas):
                fork = coarse_path.fork(generator=generators[k, replica])
                endpoint = teacher_endpoint(teacher, base, context, fork, teacher_steps)
                conditional.append(endpoint.cpu())
            outputs[k].append(torch.stack(conditional, dim=1))
    return {k: torch.cat(chunks, dim=0) for k, chunks in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/stochastic_navier_stokes/stochastic_interpolant"),
    )
    parser.add_argument("--trajectory", type=int, default=160)
    parser.add_argument("--origin", type=int, default=45)
    parser.add_argument("--features", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=83671)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/ito_maps/results/ns_path_compression"),
    )
    parser.add_argument("--save-endpoints", action="store_true")
    args = parser.parse_args()
    if not 160 <= args.trajectory < 180:
        parser.error(
            "This development diagnostic accepts validation trajectories160--179 only"
        )
    if args.teacher_steps < 1:
        parser.error("--teacher-steps must be positive")
    if (args.output / "results.json").exists():
        parser.error(
            "Output already contains results.json; select a new output directory"
        )
    torch.set_num_threads(4)
    teacher, config = load_teacher(args.checkpoint_dir, args.device)
    if not isinstance(teacher, FollmerStochasticInterpolant):
        raise TypeError(
            "This experiment requires a frozen FollmerStochasticInterpolant teacher"
        )
    teacher.eval().requires_grad_(False)
    preprocessor = instantiate(config.preprocesser)
    data_config = config.test_data.copy()
    data_config.tajectories_ids = f"({args.trajectory},{args.trajectory+1})"
    data_config.train_or_test = "test"
    dataset = instantiate(data_config, preprocesser=preprocessor)
    truth = dataset[0]["x"]
    length = dataset.len_field_history
    if not length <= args.origin < truth.shape[-1]:
        parser.error(
            "origin must follow a complete history and precede the trajectory end"
        )
    history = truth[..., args.origin - length : args.origin].unsqueeze(0)
    history = preprocessor.transform(field_history=history, is_batch=True)[
        "field_history"
    ].to(args.device)
    start = time.perf_counter()
    samples = conditional_endpoints(
        teacher,
        history,
        args.features,
        args.groups,
        args.replicas,
        args.teacher_steps,
        args.batch_size,
        args.seed,
    )
    elapsed = time.perf_counter() - start
    results = []
    for count, endpoints in samples.items():
        row = {"num_path_features": count, **compression_statistics(endpoints)}
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "argv": sys.argv,
        "split": "validation",
        "trajectory": args.trajectory,
        "origin": args.origin,
        "history_length": length,
        "state_shape": list(history.shape[1:-1]),
        "coordinate_system": "dataset normalized coordinates",
        "checkpoint": str(args.checkpoint_dir / "model.pth"),
        "checkpoint_sha256": hashlib.sha256(
            (args.checkpoint_dir / "model.pth").read_bytes()
        ).hexdigest(),
        "teacher_steps": args.teacher_steps,
        "solver": "Euler drift with exact integrated gamma(t)=a*(1-t) Brownian increments",
        "gamma_multiplier": float(teacher.interpolation.gamma_multiplier),
        "device": args.device,
        "hardware": (
            torch.cuda.get_device_name(args.device)
            if str(args.device).startswith("cuda")
            else "CPU"
        ),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "interpretation": "Nested Monte Carlo estimate of irreducible same-path squared-error risk from Brownian-feature compression for this fixed history and finite-step teacher. Not a trained-map performance bound or a guarantee across histories.",
        "finite_sample_caveat": "Explained variance subtracts within/R from group-mean variance; sampling noise can make it negative. Raw estimates are not clipped. Pixels are not treated as independent uncertainty replicates.",
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.save_endpoints:
        torch.save(samples, args.output / "conditional_endpoints.pth")
    print(f"Saved {args.output/'results.json'} after {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
