#!/usr/bin/env python3
"""Refine local Brownian features without changing the initial map function.

For raw martingale increments, each coarse feature is the sum of its nested
fine features. Repeating the coarse projection weight on every child therefore
preserves the projection exactly up to floating-point arithmetic. This exports
model weights and resolved configuration, never optimizer/resume state.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECTION_KEY = "map_model.backbone.init_conv.projection.weight"


def refine_weights(state, old_features, new_features, channels):
    """Return a copied state dictionary with a nested local path projection."""
    if new_features <= old_features or new_features % old_features:
        raise ValueError(
            "new feature count must be a larger integer multiple of the old count"
        )
    if PROJECTION_KEY not in state:
        raise ValueError(
            f"Checkpoint lacks the expected path projection: {PROJECTION_KEY}"
        )
    weight = state[PROJECTION_KEY]
    if weight.ndim != 4 or weight.shape[1] != old_features * channels:
        raise ValueError(
            "Path projection dimensions disagree with the source configuration"
        )
    if tuple(weight.shape[2:]) != (1, 1):
        raise ValueError("Expected the existing 1x1 local path projection")
    expanded = weight.reshape(weight.shape[0], old_features, channels, 1, 1)
    expanded = expanded.repeat_interleave(new_features // old_features, dim=1)
    result = dict(state)
    result[PROJECTION_KEY] = expanded.reshape(
        weight.shape[0], new_features * channels, 1, 1
    )
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refine_checkpoint(checkpoint_dir, output_dir, num_path_features):
    source, output = Path(checkpoint_dir), Path(output_dir)
    if (output / "model.pth").exists() or (output / "config.yaml").exists():
        raise FileExistsError(
            "Output already contains model.pth or config.yaml; choose a new directory"
        )
    config = OmegaConf.load(source / "config.yaml")
    # Materialize interpolation expressions before changing paths or dimensions.
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if (
        config.model.get("_target_")
        != "scisi.models.follmer_ito_map_model.FollmerItoMapModel"
    ):
        raise ValueError("Feature refinement requires a FollmerItoMapModel checkpoint")
    old_features = int(config.model.num_path_features)
    old_path_channels = int(config.model.map_model.path_channels)
    if old_path_channels % old_features:
        raise ValueError("Path channels must be divisible by num_path_features")
    channels = old_path_channels // old_features
    state = torch.load(source / "model.pth", map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError(
            "model.pth must be a model state dictionary, not a training checkpoint"
        )
    refined = refine_weights(state, old_features, int(num_path_features), channels)
    output.mkdir(parents=True, exist_ok=True)
    config.model.num_path_features = int(num_path_features)
    config.model.map_model.path_channels = int(num_path_features) * channels
    # The exported weights are initialization; fine-tuning gets a fresh run dir.
    config.initial_student_checkpoint = str((output / "model.pth").resolve())
    if config.get("trainer") is not None:
        config.trainer.resume_from = None
        config.trainer.checkpoint_dir = str((output / "finetune").resolve())
    if config.get("evaluation") is not None:
        config.evaluation.checkpoint_dir = str(output.resolve())
        config.evaluation.checkpoint_name = "model.pth"
    torch.save(refined, output / "model.pth")
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    provenance = {
        "source_checkpoint_dir": str(source.resolve()),
        "source_model_sha256": sha256(source / "model.pth"),
        "source_config_sha256": sha256(source / "config.yaml"),
        "output_model_sha256": sha256(output / "model.pth"),
        "source_num_path_features": old_features,
        "num_path_features": int(num_path_features),
        "physical_channels": channels,
        "refinement_factor": int(num_path_features) // old_features,
        "operation": "Repeat each coarse raw-increment projection weight on each nested child, with no division; all other weights unchanged",
        "preservation": "Identical projected features for nested subdivisions of the same Brownian path, up to floating-point arithmetic",
        "optimizer_state": "Not transferred; initialize a fresh optimizer",
        "training_status": "Untrained feature refinement of an existing trained map; count the source run in cumulative training cost",
    }
    for name, key in (
        ("metadata.json", "source_metadata"),
        ("history.json", "source_history"),
    ):
        if (source / name).exists():
            provenance[key] = json.loads((source / name).read_text())
    (output / "refinement_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-path-features", type=int, required=True)
    args = parser.parse_args()
    info = refine_checkpoint(
        args.checkpoint_dir, args.output_dir, args.num_path_features
    )
    print(
        f"Refined {info['source_num_path_features']} to {info['num_path_features']} features in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
