"""Train pixel-space video teachers or maps with the existing model families."""

import json
import logging
import subprocess
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from scisi.bin.main_map_model_train import run_training as run_map_training
from scisi.bin.main_map_model_train import seed_everything

logger = logging.getLogger(__name__)


def run_training(cfg):
    if cfg.model._target_.endswith(("FollmerItoMapModel", "FlowMapModel")):
        return run_map_training(cfg)
    seed_everything(int(cfg.get("seed", 0)))
    torch.set_num_threads(int(cfg.get("num_threads", 4)))
    device = str(cfg.trainer.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; set trainer.device=cpu")
    directory = Path(cfg.trainer.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (
        (directory / "model.pth").exists()
        and not cfg.get("overwrite", False)
        and not cfg.trainer.get("resume_from")
    ):
        raise FileExistsError(
            f"Checkpoint already exists: {directory}; choose another directory or overwrite=true"
        )
    OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
    preprocesser = (
        instantiate(cfg.preprocesser) if cfg.get("preprocesser") is not None else None
    )
    train_loader = instantiate(cfg.train_data, dataset={"preprocesser": preprocesser})
    val_loader = instantiate(cfg.val_data, dataset={"preprocesser": preprocesser})
    model = instantiate(cfg.model).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = (
        instantiate(cfg.scheduler, optimizer=optimizer)
        if cfg.get("scheduler") is not None
        else None
    )
    trainer = instantiate(
        cfg.trainer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=instantiate(cfg.loss_fn),
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        tracker=None,
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
        "validation_objective": "fixed_noise_interpolant_drift_regression",
        "data_manifest": str(cfg.manifest_path),
        "pixel_space": True,
        "len_field_history": int(cfg.len_field_history),
        "initialization": (
            "random" if cfg.trainer.get("resume_from") is None else "resume"
        ),
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Training video teacher with %d parameters, %d train examples",
        metadata["parameters"],
        metadata["train_examples"],
    )
    trainer.train()
    return trainer


@hydra.main(
    config_path="../../../config",
    config_name="follmer_stochastic_interpolant_kth",
    version_base=None,
)
def main_video_model_train(cfg: DictConfig):
    return run_training(cfg)


main = main_video_model_train

if __name__ == "__main__":
    main_video_model_train()
