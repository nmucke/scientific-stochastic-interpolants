# rsync -avz --ignore-existing "delftblue:/home/ntmucke/processed_data/" /Users/ntmucke/Code/scientific-stochastic-interpolants/data/udales/
# scp /Users/ntmucke/Code/scientific-stochastic-interpolants/data/udales/* "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/data/udales/"

# scp "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/figures/udales/*" /Users/ntmucke/
# rsync -avz --ignore-existing /Users/ntmucke/Code/scientific-stochastic-interpolants/data/udales/ "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/data/udales/"


# scp "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/checkpoints/knmi/proud-rain-12/*" /Users/ntmucke/proud-rain-12
# scp
# scp "squamish:/export/scratch1/ntm/postdoc/SciGenML/data/knmi/*" "/Users/ntmucke/code/scientific-stochastic-interpolants/data/knmi"

# rsync -avz --ignore-existing squamish:/export/scratch1/ntm/postdoc/SciGenML/data/knmi/*" "/Users/ntmucke/code/scientific-stochastic-interpolants/data/knmi"


import logging
import os
import pdb

import hydra
import torch

# Enable flash attention
import torch.backends.cuda
import torch.nn as nn
import trackio
from omegaconf import DictConfig, OmegaConf

from scisi.architectures.architecture_utils import count_model_parameters
from scisi.models.ito_maps import (
    RESIDUAL_STATS_FILENAME,
    GaussianShellTeacher,
    ResidualItoMapModel,
    ResidualStats,
    estimate_residual_stats,
)
from scisi.utils.device_utils import set_device

torch.set_default_dtype(torch.float32)

# Disable slow attention warnings
import warnings

warnings.filterwarnings("ignore", message=".*The operator.*is not optimized.*")

logger = logging.getLogger(__name__)

# Suppress httpx logs to avoid cluttering the logs from Trackio initialization
logging.getLogger("httpx").setLevel(logging.WARNING)

VERBOSE = True
CONTINUE_FROM_CHECKPOINT = False
CHECKPOINT_PROJECT = "udales"
CHECKPOINT_NAME = "kind-sky-9"
CHECKPOINT_PATH = f"checkpoints/{CHECKPOINT_PROJECT}/{CHECKPOINT_NAME}/model.pth"


def _load_or_estimate_residual_stats(det_model, teacher_dir, dataloader):
    """Stage-0 residual calibration: load persisted stats or estimate them.

    Stats are persisted next to the deterministic checkpoint so repeated
    fine-tuning runs skip the calibration pass.
    """
    stats_path = f"{teacher_dir}/{RESIDUAL_STATS_FILENAME}"
    if os.path.exists(stats_path):
        logger.info(f"Loading residual statistics from {stats_path}")
        return ResidualStats.load(stats_path)

    logger.info("Estimating residual statistics (Stage 0 calibration)...")
    stats = estimate_residual_stats(det_model, dataloader)
    stats.save(stats_path)
    logger.info(f"Saved residual statistics to {stats_path}")
    return stats


def _build_from_deterministic(cfg, model, det_model, teacher_dir, dataloader):
    """Turn an Ito map + a trained deterministic model into a fine-tunable model.

    ``init_mode: residual`` (Method 1) wraps the Ito map into a
    ResidualItoMapModel around the frozen mean model; ``init_mode:
    warm_start`` (Method 2) copies the mean model's weights into the drift
    net. ``analytic_teacher: true`` additionally attaches the Gaussian-shell
    teacher (Method 3) - combine with ``trainer.teacher_warmup_epochs``.
    """
    init_mode = cfg.pre_trained_model.get("init_mode", "residual")
    analytic_teacher = cfg.pre_trained_model.get("analytic_teacher", False)
    device = cfg.trainer.get("device", "cpu")

    if init_mode == "residual":
        det_model.to(device)
        stats = _load_or_estimate_residual_stats(det_model, teacher_dir, dataloader)
        if analytic_teacher:
            # Residual coordinates: F = 0, rho = 1 inside the inner map.
            model.distill_from(
                GaussianShellTeacher(
                    interpolation=model.interpolation,
                    sigma_schedule=model.sigma_schedule,
                )
            )
        logger.info("Building ResidualItoMapModel around the frozen mean model...")
        return ResidualItoMapModel.from_deterministic(
            det_model=det_model, ito_map=model, residual_stats=stats
        )

    if init_mode == "warm_start":
        logger.info("Warm-starting the Ito map from the deterministic model...")
        model.warm_start_from_deterministic(det_model)
        if analytic_teacher:
            det_model.to(device)
            stats = _load_or_estimate_residual_stats(det_model, teacher_dir, dataloader)
            model.distill_from(
                GaussianShellTeacher(
                    interpolation=model.interpolation,
                    sigma_schedule=model.sigma_schedule,
                    mean_model=det_model,
                    residual_std=stats.std,
                )
            )
        return model

    raise ValueError(
        f"Unknown pre_trained_model.init_mode: {init_mode} "
        "(expected 'residual' or 'warm_start')."
    )


@hydra.main(  # type: ignore[misc]
    config_path="../../../config",
    # config_name="diffusion_stochastic_navier_stokes.yaml",
    # config_name="flow_matching_udales.yaml",
    # config_name="udales.yaml",
    # config_name="xie_and_castro.yaml",
    # config_name="udales_pde_transformer.yaml",
    # config_name="knmi_pde_transformer_flow_matching.yaml",
    # config_name="stochastic_navier_stokes_pde_transformer.yaml",
    config_name="stochastic_navier_stokes.yaml",
    # config_name="deterministic_navier_stokes.yaml",
    # config_name="knmi_pde_transformer.yaml",
    # config_name="knmi.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    set_device(cfg)

    if CONTINUE_FROM_CHECKPOINT:
        logger.info(f"Loading config from checkpoint:")
        logger.info(f"Project: {CHECKPOINT_PROJECT}")
        logger.info(f"Name: {CHECKPOINT_NAME}")
        cfg = OmegaConf.load(
            f"checkpoints/{CHECKPOINT_PROJECT}/{CHECKPOINT_NAME}/config.yaml"
        )
        cfg.pop("_Username"), cfg.pop("_Created"), cfg.pop("_Group")

        # cfg.train_data.batch_size = 16

    logger.info(f"Instantiating experiment tracking...")
    tracker = trackio.init(
        name="stochastic_interpolant_big_gamma1",
        project=cfg.experiment_tracking.project,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    logger.info(f"Tracker instantiated with properties:")
    logger.info(f"Project: {tracker.project}")
    logger.info(f"URL: {tracker.url}")
    logger.info(f"Name: {tracker.name}")

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
    model = hydra.utils.instantiate(cfg.model)
    if CONTINUE_FROM_CHECKPOINT:

        logger.info(f"Loading model from checkpoint:")
        logger.info(f"{CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    logger.info(f"Model parameters: {count_model_parameters(model)/1e6:.2f}M")

    # Distillation: if the config points at a pre-trained teacher checkpoint
    # (same block shape as the posterior configs), load it and warm-start /
    # attach it to the Ito map before constructing the trainer.
    if ("pre_trained_model" in cfg) and (cfg.pre_trained_model is not None):
        teacher_project = cfg.pre_trained_model.project
        teacher_name = cfg.pre_trained_model.name
        teacher_dir = f"checkpoints/{teacher_project}/{teacher_name}"

        logger.info(f"Loading teacher model from checkpoint: {teacher_dir}")
        teacher_cfg = OmegaConf.load(f"{teacher_dir}/config.yaml")
        teacher_model = hydra.utils.instantiate(teacher_cfg.model)
        teacher_model.load_state_dict(
            torch.load(f"{teacher_dir}/model.pth", map_location="cpu")
        )

        if cfg.pre_trained_model.get("type", None) == "deterministic":
            model = _build_from_deterministic(
                cfg, model, teacher_model, teacher_dir, val_dataloader
            )
        else:
            logger.info(f"Distilling into {cfg.model._target_}...")
            model.distill_from(teacher_model)

    if ("drift_model" in cfg.model) and (
        "AuroraWrapper" in cfg.model.drift_model._target_
    ):
        model.drift_model.batch_adapter = train_dataloader.dataset.batch_adapter

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
        latitudes = locals()["train_dataloader"].dataset.lat
        loss_fn_kwargs = {"latitudes": latitudes}
    loss_fn = hydra.utils.instantiate(
        cfg.loss_fn,
        **loss_fn_kwargs,
    )

    logger.info(f"Instantiating trainer...")
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

    logger.info(f"Training...")
    trainer.train()

    logger.info(f"Closing tracker...")
    tracker.finish()


if __name__ == "__main__":
    main()
