"""CPU persistence sanity baseline on the identical frozen forecast cases.

Pass the actual experiment freeze manifest before evaluating test data. The
baseline copies each path's latest field; it performs zero neural evaluations.
Its CPU latency must not be interpreted as a GPU generative-model speedup.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from scisi.bin.main_map_model_test import evaluate_model


class PersistenceModel(nn.Module):
    """Repeat the latest observed/generated field independently in each history."""

    def __init__(self):
        super().__init__()
        # evaluate_model hooks this unused module: measured NFE remains exactly 0.
        self.model = nn.Identity()

    def sample(
        self,
        field_history,
        batch_size=1,
        num_steps=1,
        base=None,
        return_field_history=False,
    ):
        history = field_history
        if history.shape[0] == 1 and batch_size > 1:
            history = history.expand(batch_size, *history.shape[1:])
        state = history[..., -1].detach().cpu().clone()
        if return_field_history:
            return state, torch.cat(
                [history[..., 1:].cpu(), state.unsqueeze(-1)], dim=-1
            )
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument(
        "--reference-config",
        default="checkpoints/stochastic_navier_stokes/stochastic_interpolant/config.yaml",
    )
    parser.add_argument("--origins", nargs="+", type=int, default=[5, 45, 85])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--trajectories", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.freeze_manifest)
    if not manifest.is_file():
        parser.error("The frozen experiment manifest must exist before test evaluation")
    json.loads(manifest.read_text())
    if not 1 <= args.trajectories <= 20:
        parser.error("Use one to twenty test trajectories")
    torch.set_num_threads(4)
    cfg = OmegaConf.load(args.reference_config)
    preprocessor = instantiate(cfg.preprocesser)
    data_cfg = cfg.test_data.copy()
    data_cfg.tajectories_ids = "(180,200)"
    dataset = instantiate(data_cfg, preprocesser=preprocessor)
    result = evaluate_model(
        PersistenceModel(),
        dataset,
        preprocessor,
        num_steps=[1],
        ensemble_size=args.ensemble,
        origins=args.origins,
        num_trajectories=args.trajectories,
        forecast_horizon=args.horizon,
        seed=args.seed,
        device="cpu",
        output_dir=args.output,
        trajectory_offset=180,
    )
    result.update(
        split="test",
        parameters=0,
        model_class="persistence_baseline.PersistenceModel",
        baseline="deterministic persistence",
        latency_comparison="CPU bookkeeping only; excluded from generative-model speedup claims",
        freeze_manifest=str(manifest.resolve()),
        freeze_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        reference_config=args.reference_config,
    )
    (Path(args.output) / "evaluation.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
