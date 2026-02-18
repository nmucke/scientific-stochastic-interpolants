# rsync -avz --ignore-existing "delftblue:/home/ntmucke/processed_data/" /Users/ntmucke/Code/scientific-stochastic-interpolants/data/udales/
# scp /Users/ntmucke/Code/scientific-stochastic-interpolants/data/udales/* "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/data/udales/"

# scp "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/figures/udales/*" /Users/ntmucke/
# rsync -avz --ignore-existing /Users/ntmucke/Code/scientific-stochastic-interpolants/data/udales/ "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/data/udales/"


# scp "squamish:/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants/checkpoints/knmi/proud-rain-12/*" /Users/ntmucke/proud-rain-12
# scp
# scp "squamish:/export/scratch1/ntm/postdoc/SciGenML/data/knmi/*" "/Users/ntmucke/code/scientific-stochastic-interpolants/data/knmi"

# rsync -avz --ignore-existing squamish:/export/scratch1/ntm/postdoc/SciGenML/data/knmi/*" "/Users/ntmucke/code/scientific-stochastic-interpolants/data/knmi"


import logging
import pdb

import hydra
import torch

# Enable flash attention
import torch.backends.cuda
import torch.nn as nn
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

VERBOSE = True
CONTINUE_FROM_CHECKPOINT = True
CHECKPOINT_PROJECT = "stochastic_navier_stokes"
CHECKPOINT_NAME = "artful-hare-68"
CHECKPOINT_PATH = f"checkpoints/{CHECKPOINT_PROJECT}/{CHECKPOINT_NAME}/model.pth"


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
    # config_name="knmi_pde_transformer.yaml",
    config_name="knmi.yaml",
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

        cfg.train_data.batch_size = 16

    logger.info(f"Instantiating experiment tracking...")
    tracker = trackio.init(
        name="artful-hare-68_continued",
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

    if "AuroraWrapper" in cfg.model.drift_model._target_:
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
