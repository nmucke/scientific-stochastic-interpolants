"""Hydra entry point for map training on the existing forecast datasets."""

import json
import logging
import random
import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(checkpoint_dir, device):
    """Load an existing SI/FM checkpoint without changing its configuration."""
    directory = Path(checkpoint_dir)
    config = OmegaConf.load(directory / "config.yaml")
    model = instantiate(config.model).to(device).eval()
    state = torch.load(directory / "model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model, config


def run_training(cfg):
    seed_everything(int(cfg.get("seed", 0)))
    torch.set_num_threads(int(cfg.get("num_threads", 4)))
    device = str(cfg.trainer.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; set trainer.device=cpu explicitly"
        )
    output = Path(cfg.trainer.checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (
        (output / "model.pth").exists()
        and not cfg.get("overwrite", False)
        and not cfg.trainer.get("resume_from")
    ):
        raise FileExistsError(
            f"Run already has a checkpoint: {output}; choose a new directory or overwrite=true"
        )
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    preprocesser = (
        instantiate(cfg.preprocesser) if cfg.get("preprocesser") is not None else None
    )
    train_loader = instantiate(cfg.train_data, dataset={"preprocesser": preprocesser})
    val_loader = instantiate(cfg.val_data, dataset={"preprocesser": preprocesser})
    model = instantiate(cfg.model).to(device)
    teacher = None
    teacher_cfg = cfg.get("teacher") or {}
    if teacher_cfg.get("checkpoint_dir"):
        teacher, teacher_config = load_teacher(teacher_cfg.checkpoint_dir, device)
        if teacher_cfg.get("initialize_student", True):
            model.map_model.load_from_teacher(teacher)
        # A teacher with a different interpolant/base cannot anchor this map.
        if type(model.interpolation) is not type(teacher.interpolation):
            raise ValueError("Student and teacher interpolation classes differ")
        if getattr(model.interpolation, "gamma_multiplier", None) != getattr(
            teacher.interpolation, "gamma_multiplier", None
        ):
            raise ValueError("Student and teacher diffusion amplitudes differ")
    if cfg.get("initial_student_checkpoint"):
        model.load_state_dict(
            torch.load(
                cfg.initial_student_checkpoint, map_location=device, weights_only=True
            )
        )
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = (
        instantiate(cfg.scheduler, optimizer=optimizer)
        if cfg.get("scheduler") is not None
        else None
    )
    tracker = None
    tracking = cfg.get("experiment_tracking") or {}
    if tracking.get("enabled", False):
        import trackio

        tracker = trackio.init(
            project=tracking.project,
            name=tracking.get("name"),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    trainer = instantiate(
        cfg.trainer,
        model=model,
        teacher=teacher,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=instantiate(cfg.loss_fn),
        tracker=tracker,
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = {
        "seed": int(cfg.get("seed", 0)),
        "git_revision": revision,
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "device": device,
        "gpu": (
            torch.cuda.get_device_name(device) if device.startswith("cuda") else None
        ),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_examples": len(train_loader.dataset),
        "val_examples": len(val_loader.dataset),
        "teacher_checkpoint": teacher_cfg.get("checkpoint_dir"),
        "initial_student_checkpoint": cfg.get("initial_student_checkpoint"),
        "validation_objective": (
            "fixed_noise_full_range_teacher_rollout_with_weighted_diagonal"
            if teacher is not None
            else "fixed_noise_full_range_self_consistency_with_weighted_diagonal"
        ),
        "validation_intervals": {
            "description": "Intervals sampled across [0,1], with an atom at the full interval [0,1]",
            "endpoint_probability": float(
                cfg.trainer.get("endpoint_probability", 0.25)
            ),
            "diagonal_weight": float(cfg.trainer.get("diagonal_weight", 1.0)),
            "consistency_weight": float(cfg.trainer.get("consistency_weight", 1.0)),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Training map with %d parameters, %d train examples",
        metadata["parameters"],
        metadata["train_examples"],
    )
    try:
        trainer.train()
    finally:
        if tracker is not None:
            tracker.finish()
    return trainer


@hydra.main(
    config_path="../../../config",
    config_name="follmer_ito_map_stochastic_navier_stokes",
    version_base=None,
)
def main_map_model_train(cfg: DictConfig):
    return run_training(cfg)


main = main_map_model_train

if __name__ == "__main__":
    main_map_model_train()
