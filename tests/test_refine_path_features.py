"""Scientific ablation initialization preserves an existing map's path inputs."""

import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from scisi.sampling.brownian_path import WeightedBrownianPath

SCRIPT = (
    Path(__file__).parents[1] / "research/ito_maps/experiments/refine_path_features.py"
)
spec = importlib.util.spec_from_file_location("refine_path_features", SCRIPT)
refine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refine)


def test_nested_refinement_preserves_nonzero_multichannel_path_projection():
    torch.manual_seed(551)
    state = {
        refine.PROJECTION_KEY: torch.randn(5, 8, 1, 1, dtype=torch.float64),
        "other": torch.randn(3),
    }
    expanded = refine.refine_weights(state, 4, 16, 2)
    assert torch.equal(expanded["other"], state["other"])
    path = WeightedBrownianPath.sample_like(
        torch.zeros(3, 2, 4, 4, dtype=torch.float64)
    )
    s = torch.tensor([0.1, 0.25, 0.4], dtype=torch.float64)
    t = torch.tensor([0.5, 0.75, 1.0], dtype=torch.float64)
    coarse = path.features(s, t, 4)
    fine = path.features(s, t, 16)
    a = torch.nn.functional.conv2d(coarse, state[refine.PROJECTION_KEY])
    b = torch.nn.functional.conv2d(fine, expanded[refine.PROJECTION_KEY])
    torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


def test_export_has_resolved_configuration_provenance_and_no_optimizer(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    config = OmegaConf.create(
        {
            "count": 4,
            "model": {
                "_target_": "scisi.models.follmer_ito_map_model.FollmerItoMapModel",
                "num_path_features": "${count}",
                "map_model": {"path_channels": 8},
            },
            "trainer": {"resume_from": "old/last.pth", "checkpoint_dir": "old"},
            "evaluation": {"checkpoint_dir": "old"},
        }
    )
    OmegaConf.save(config, source / "config.yaml")
    original = {refine.PROJECTION_KEY: torch.randn(3, 8, 1, 1), "other": torch.ones(2)}
    torch.save(original, source / "model.pth")
    result = refine.refine_checkpoint(source, output, 16)
    loaded = torch.load(output / "model.pth", weights_only=True)
    assert loaded[refine.PROJECTION_KEY].shape == (3, 32, 1, 1)
    assert torch.equal(loaded["other"], original["other"])
    assert "optimizer" not in loaded and not (output / "last.pth").exists()
    saved = OmegaConf.load(output / "config.yaml")
    assert (
        saved.model.num_path_features == 16
        and saved.model.map_model.path_channels == 32
    )
    assert saved.trainer.resume_from is None
    assert "${" not in (output / "config.yaml").read_text()
    assert result["source_model_sha256"] == refine.sha256(source / "model.pth")
    assert (output / "refinement_provenance.json").exists()
    with pytest.raises(FileExistsError):
        refine.refine_checkpoint(source, output, 16)
    with pytest.raises(ValueError):
        refine.refine_weights(original, 4, 6, 2)
