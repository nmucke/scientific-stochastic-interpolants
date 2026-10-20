"""Train any model: flow models (FlowTrainer) and map models (MapTrainer).

Pick a case with --config-name and a model with models=<name>, e.g.

    python scripts/train.py --config-name udales models=flow_matching
    python scripts/train.py --config-name stochastic_navier_stokes models=ito_map

Every run is saved to checkpoints/<project>/<run name>/. Continue a run with
trainer.resume_from=checkpoints/<project>/<run name>/last.pth.
"""

import logging
import random

import hydra
import numpy as np
import torch

# Enable flash attention
import torch.backends.cuda
import trackio
from omegaconf import DictConfig, OmegaConf

from scisi.architectures.architecture_utils import count_model_parameters
from scisi.utils.device_utils import set_device

torch.set_default_dtype(torch.float32)

# Disable slow attention warnings
import warnings

warnings.filterwarnings("ignore", message=".*The operator.*is not optimized.*")

logger = logging.getLogger(__name__)

# Suppress httpx logs to avoid cluttering the logs from Trackio initialization
logging.getLogger("httpx").setLevel(logging.WARNING)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Load a trained flow model from its checkpoint directory."""
    config = OmegaConf.load(f"{checkpoint_dir}/config.yaml")
    teacher = hydra.utils.instantiate(config.model)
    teacher.load_state_dict(
        torch.load(
            f"{checkpoint_dir}/model.pth", map_location=device, weights_only=True
        )
    )
    return teacher.to(device).eval()


@hydra.main(  # type: ignore[misc]
    config_path="../config",
    config_name="stochastic_navier_stokes",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    set_device(cfg)

    if cfg.get("seed") is not None:
        logger.info(f"Seeding with {cfg.seed}")
        seed_everything(cfg.seed)

    is_map_model = cfg.trainer._target_.endswith("MapTrainer")
    logger.info(f"Model: {cfg.model_name} ({'map' if is_map_model else 'flow'} model)")

    logger.info(f"Instantiating experiment tracking...")
    tracker = trackio.init(
        name=cfg.experiment_tracking.get("name"),
        project=cfg.experiment_tracking.project,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger.info(f"Tracker instantiated with properties:")
    logger.info(f"Project: {tracker.project}")
    logger.info(f"URL: {tracker.url}")
    logger.info(f"Name: {tracker.name}")

    # Every run lives in checkpoints/<project>/<run name>, where the test script looks.
    checkpoint_dir = f"checkpoints/{tracker.project}/{tracker.name}"
    cfg.trainer.checkpoint_dir = checkpoint_dir

    logger.info(f"Instantiating preprocesser...")
    if cfg.preprocesser is not None:
        preprocesser = hydra.utils.instantiate(cfg.preprocesser)
    else:
        preprocesser = None

    logger.info(f"Preparing train dataloader...")
    train_dataloader = hydra.utils.instantiate(
        cfg.train_data,
        dataset={"preprocesser": preprocesser},
    )

    logger.info(f"Preparing val dataloader...")
    val_dataloader = hydra.utils.instantiate(
        cfg.val_data,
        dataset={"preprocesser": preprocesser},
    )

    logger.info(f"Instantiating model: {cfg.model._target_}")
    model = hydra.utils.instantiate(cfg.model).to(cfg.trainer.device)

    if cfg.get("teacher_checkpoint_dir"):
        logger.info(f"Warm-starting from teacher: {cfg.teacher_checkpoint_dir}")
        teacher = load_teacher(cfg.teacher_checkpoint_dir, cfg.trainer.device)
        if type(model.interpolation) is not type(teacher.interpolation):
            raise ValueError("Student and teacher interpolation classes differ")
        if getattr(model.interpolation, "gamma_multiplier", None) != getattr(
            teacher.interpolation, "gamma_multiplier", None
        ):
            raise ValueError("Student and teacher diffusion amplitudes differ")
        model.map_model.load_from_teacher(teacher)
        del teacher

    logger.info(f"Model parameters: {count_model_parameters(model)/1e6:.2f}M")

    logger.info(f"Instantiating optimizer: {cfg.optimizer._target_}")
    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.parameters(),
    )

    logger.info(f"Instantiating scheduler: {cfg.scheduler._target_}")
    scheduler = hydra.utils.instantiate(
        cfg.scheduler,
        optimizer=optimizer,
    )

    logger.info(f"Instantiating loss function: {cfg.loss_fn._target_}")
    loss_fn_kwargs = {}
    if "LatitudeWeightedMSELoss" in cfg.loss_fn._target_:
        loss_fn_kwargs = {"latitudes": train_dataloader.dataset.lat}
    loss_fn = hydra.utils.instantiate(
        cfg.loss_fn,
        **loss_fn_kwargs,
    )

    logger.info(f"Instantiating trainer: {cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tracker=tracker,
    )

    # Save the resolved config (including the final checkpoint_dir) with the run.
    OmegaConf.save(cfg, f"{checkpoint_dir}/config.yaml", resolve=True)
    logger.info(f"Checkpoints: {checkpoint_dir}")

    logger.info(f"Training...")
    try:
        trainer.train()
    finally:
        logger.info(f"Closing tracker...")
        tracker.finish()


if __name__ == "__main__":
    main()
