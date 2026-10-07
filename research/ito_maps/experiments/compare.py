"""Run identical held-out forecast cases for solver and learned-map checkpoints.

Example: .venv/bin/python research/ito_maps/experiments/compare.py --split validation
All outputs include raw per-trajectory scores. Run validation before test; do not
select hyperparameters on these test results.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scisi.bin.main_map_model_test import evaluate_model
from scisi.bin.main_map_model_train import load_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--checkpoints", nargs="+", default=["stochastic_interpolant", "flow_matching"]
    )
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 64])
    parser.add_argument("--origins", nargs="+", type=int, default=[5, 45, 85])
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", default="research/ito_maps/results/development")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = OmegaConf.load(
        "checkpoints/stochastic_navier_stokes/stochastic_interpolant/config.yaml"
    )
    preprocessor = instantiate(config.preprocesser)
    data_config = config.test_data.copy()
    offset = 160 if args.split == "validation" else 180
    data_config.tajectories_ids = f"({offset}, {offset+20})"
    dataset = instantiate(data_config, preprocesser=preprocessor)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "command.json").write_text(
        json.dumps(
            {
                "argv": sys.argv,
                "split": args.split,
                "offset": offset,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    for name in args.checkpoints:
        path = (
            Path(name)
            if "/" in name
            else Path("checkpoints/stochastic_navier_stokes") / name
        )
        model, cfg = load_teacher(path, "cuda")
        result = evaluate_model(
            model,
            dataset,
            preprocessor,
            num_steps=args.steps,
            origins=args.origins,
            num_trajectories=args.trajectories,
            ensemble_size=args.ensemble,
            forecast_horizon=args.horizon,
            seed=args.seed,
            device="cuda",
            output_dir=root / path.name,
            trajectory_offset=offset,
        )
        result["checkpoint"] = str(path / "model.pth")
        result["checkpoint_sha256"] = hashlib.sha256(
            (path / "model.pth").read_bytes()
        ).hexdigest()
        result["model_class"] = cfg.model._target_
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["split"] = args.split
        (root / path.name / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(
            name,
            [
                (s["num_steps"], round(s["crps"], 4), round(s["seconds"], 3))
                for s in result["summary"]
            ],
            flush=True,
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
