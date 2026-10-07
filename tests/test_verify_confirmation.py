"""Integrity audit is tested on synthetic artifacts, never incomplete live runs."""

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

SCRIPT = Path(__file__).parents[1] / "research/ito_maps/experiments/verify_confirmation.py"
spec = importlib.util.spec_from_file_location("verify_confirmation", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from archive_training import archive_run


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def artifacts(tmp_path):
    checkpoints, results = tmp_path / "checkpoints", tmp_path / "results"
    names = [f"{kind}_rollout_seed{seed}" for seed in (0, 1, 2) for kind in ("ito", "flow")]
    baselines = ["stochastic_interpolant", "flow_matching"]
    for name in baselines:
        directory = checkpoints / name
        directory.mkdir(parents=True)
        (directory / "model.pth").write_bytes(name.encode())
        (directory / "config.yaml").write_text("model: {}\n")
    for seed in (0, 1, 2):
        for kind in ("ito", "flow"):
            for stage in ("psd", "rollout"):
                name = f"{kind}_{stage}_seed{seed}"
                directory = checkpoints / name
                directory.mkdir()
                epochs, batches = (20, 100) if stage == "psd" else (10, 50)
                teacher = checkpoints / (baselines[0] if kind == "ito" else baselines[1])
                initial = str(checkpoints / f"{kind}_psd_seed{seed}" / "model.pth") if stage == "rollout" else None
                config = {key: {} for key in ("len_field_history", "preprocesser", "train_data", "val_data", "test_data", "optimizer", "loss_fn", "scheduler")}
                config.update(seed=seed, initial_student_checkpoint=initial,
                              model={"num_path_features": 4, "map_model": {"path_channels": 4}},
                              teacher={"checkpoint_dir": str(teacher)},
                              trainer={"num_epochs": epochs, "max_train_batches": batches,
                                       "objective": "progressive" if stage == "psd" else "teacher_rollout",
                                       "teacher_steps": 32, "checkpoint_dir": str(directory)})
                OmegaConf.save(OmegaConf.create(config), directory / "config.yaml")
                write_json(directory / "metadata.json", {"seed": seed, "initial_student_checkpoint": initial, "teacher_checkpoint": str(teacher)})
                write_json(directory / "history.json", [{"epoch": i, "step": (i + 1) * batches, "val_loss": 0.1, "train_loss": 0.2, "elapsed_seconds": i + 1.0} for i in range(epochs)])
                for filename in ("model.pth", "ema.pth"):
                    (directory / filename).write_bytes((name + filename).encode())
                archive_run(directory, results / "training")
    protocol = {"training_seeds": [0, 1, 2], "selected_path_features": 4,
                "checkpoints": {},
                "one_step": {"trajectories": [180, 182], "origins": [5, 45, 85], "ensemble_size": 16, "seed": 20261007, "map_steps": [1, 2, 4, 8], "solver_steps": [1, 4, 16, 64, 128], "replication_steps": [1, 8]},
                "rollout": {"trajectories": [180, 182], "origins": [45], "ensemble_size": 16, "seed": 20261007, "map_steps": [1, 8], "solver_steps": [64], "horizon": 10}}
    for name in names[:2] + baselines:
        directory = checkpoints / name
        protocol["checkpoints"][name] = {"path": str(directory), "model_sha256": audit.hash_file(directory / "model.pth"), "config_sha256": audit.hash_file(directory / "config.yaml")}
    for mode in ("one_step", "rollout"):
        config = protocol[mode]
        for name in names + baselines:
            budgets = config["solver_steps"] if name in baselines else config["replication_steps"] if mode == "one_step" and not name.endswith("seed0") else config["map_steps"]
            data = {"split": "test", "seed": config["seed"], "ensemble_size": 16, "forecast_horizon": config.get("horizon", 1), "coordinate_system": "dataset normalized coordinates", "checkpoint": str(checkpoints / name / "model.pth"), "checkpoint_sha256": audit.hash_file(checkpoints / name / "model.pth"), "rows": []}
            for trajectory, origin, lead, budget in audit.product(range(180, 182), config["origins"], range(1, config.get("horizon", 1) + 1), budgets):
                row = {metric: 0.25 for metric in audit.METRICS + audit.OPTIONAL_METRICS}
                row.update(trajectory=trajectory, origin=origin, lead=lead, num_steps=budget, nfe=budget, ensemble_size=16, rank_counts=[16384] + [0] * 16)
                data["rows"].append(row)
            write_json(results / f"test_{mode}" / name / "evaluation.json", data)
    write_json(results / "frozen_protocol.json", protocol)
    return results / "frozen_protocol.json", results, checkpoints


def test_complete_confirmation_has_auditable_success(artifacts):
    result = audit.verify(*artifacts)
    assert result["success"]
    assert len(result["frozen_artifacts"]) == 4
    assert len(result["evaluations"]) == 16
    assert len(result["training_runs"]) == 12
    assert set(result["exact_lead_one_matches_by_student"].values()) == {4}
    assert result["total_forecast_rows"] == 436


@pytest.mark.parametrize("corruption,match", [
    ("frozen_hash", "Frozen artifact changed"),
    ("missing_case", "Missing, duplicate, or extra"),
    ("lead_one_score", "Lead-one mismatch"),
    ("nonfinite", "Nonfinite number"),
    ("training_seed", "Wrong training seed"),
    ("lineage", "Wrong warm-start checkpoint"),
])
def test_confirmation_rejects_scientific_integrity_errors(artifacts, corruption, match):
    protocol, results, checkpoints = artifacts
    if corruption == "frozen_hash":
        (checkpoints / "flow_matching" / "model.pth").write_bytes(b"changed")
    elif corruption in ("missing_case", "lead_one_score", "nonfinite"):
        path = results / "test_rollout" / "ito_rollout_seed2" / "evaluation.json"
        data = audit.read_json(path)
        if corruption == "missing_case":
            data["rows"].pop()
        else:
            data["rows"][0]["crps"] = 0.251 if corruption == "lead_one_score" else float("nan")
        write_json(path, data)
    else:
        path = checkpoints / "ito_rollout_seed2" / "config.yaml"
        config = OmegaConf.load(path)
        if corruption == "training_seed":
            config.seed = 1
        else:
            config.initial_student_checkpoint = str(checkpoints / "ito_psd_seed1" / "model.pth")
        OmegaConf.save(config, path)
    with pytest.raises(ValueError, match=match):
        audit.verify(*artifacts)
