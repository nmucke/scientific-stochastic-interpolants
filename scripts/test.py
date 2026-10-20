"""Test any trained model: flow models and map models.

Loads checkpoints/<project>/<name>/{config.yaml,model.pth}, forecasts one test
trajectory, logs metrics and makes the case's figures (scisi.plotting.cases).
Settings are in config/test.yaml:

    python scripts/test.py project=stochastic_navier_stokes name=my_run
    python scripts/test.py project=knmi name=proud-rain-12 \
        +run_overrides.test_data.paths=[data/knmi/]
"""

import contextlib
import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from scisi.metrics.lsim import LSiM_distance
from scisi.models.base_map_model import BaseMapModel
from scisi.models.flow_matching_model import FlowMatchingModel
from scisi.models.follmer_ito_map_model import FollmerItoMapModel
from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.plotting.cases import plot_case
from scisi.utils.device_utils import set_device

torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name="test", version_base=None)
def main(test_cfg: DictConfig) -> None:
    project, name = test_cfg.project, test_cfg.name
    run_dir = f"checkpoints/{project}/{name}"

    logger.info(f"Loading config from {run_dir}...")
    cfg = OmegaConf.merge(
        OmegaConf.load(f"{run_dir}/config.yaml"), test_cfg.run_overrides
    )
    logger.info(f"Model is {cfg.model._target_}...")

    set_device(cfg)
    device = cfg.trainer.device
    len_field_history = cfg.len_field_history

    logger.info(f"Instantiating preprocesser...")
    preprocesser = hydra.utils.instantiate(cfg.preprocesser)

    logger.info(f"Instantiating test data...")
    test_dataset = hydra.utils.instantiate(cfg.test_data)

    logger.info(f"Loading model from checkpoint...")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(
        torch.load(
            f"{run_dir}/model.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()
    model.to(device)

    logger.info(f"Preparing and preprocessing trajectory...")
    sample = test_dataset[test_cfg.test_sample_index]
    trajectory = sample["x"].unsqueeze(0)
    field_cond, pars_cond = sample.get("field_cond"), sample.get("pars_cond")
    field_cond = field_cond.unsqueeze(0) if field_cond is not None else None
    pars_cond = pars_cond.unsqueeze(0) if pars_cond is not None else None
    init_data = preprocesser.transform(
        base=trajectory[..., len_field_history - 1],
        field_history=trajectory[..., 0:len_field_history],
        is_batch=True,
    )
    # Conditioning for the i-th generated step sits at index i, so drop the history.
    for key, value in (("field_cond", field_cond), ("pars_cond", pars_cond)):
        if value is None:
            continue
        if key == "pars_cond" or value.ndim == 5:  # time-dependent
            value = value[..., len_field_history:]
        # Only normalize conditioning the preprocesser has statistics for.
        if getattr(preprocesser, f"{key}_mean", None) is not None:
            value = preprocesser.transform(
                **{key: value}, is_batch=True, is_trajectory=True
            )[key]
        init_data[key] = value
    init_data = {k: v.to(device) for k, v in init_data.items() if v is not None}

    # Only the Föllmer models start from the current state; the rest from noise.
    if not isinstance(model, (FollmerStochasticInterpolant, FollmerItoMapModel)):
        init_data["base"] = None

    if isinstance(model, BaseMapModel):
        sampling_kwargs = {"num_steps": test_cfg.num_map_steps}
    else:
        stepper = (
            test_cfg.ode_stepper
            if isinstance(model, FlowMatchingModel)
            else test_cfg.sde_stepper
        )
        sampling_kwargs = {
            "num_steps": test_cfg.num_flow_steps,
            "stepper": hydra.utils.get_method(stepper),
        }

    batch_size = test_cfg.batch_size
    use_mixed_precision = test_cfg.mixed_precision and device.startswith("cuda")
    logger.info(
        f"Sampling {batch_size} trajectories with {sampling_kwargs['num_steps']} "
        f"steps using {'mixed' if use_mixed_precision else 'full'} precision..."
    )
    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_mixed_precision
        else contextlib.nullcontext()
    ), torch.no_grad():
        predicted_trajectory = model.sample_trajectory(
            **init_data,
            batch_size=batch_size,
            num_physical_steps=test_cfg.num_physical_steps,
            **sampling_kwargs,
        )

    logger.info(f"Inverse transforming predicted trajectory...")
    predicted_trajectory = preprocesser.inverse_transform(
        base=predicted_trajectory.float().cpu(),
        is_batch=True,
        is_trajectory=True,
    )["base"]

    num_steps = min(trajectory.shape[-1], predicted_trajectory.shape[-1])
    true_trajectory = trajectory[0, ..., :num_steps].cpu()  # [C, H, W, T]
    predicted_trajectory = predicted_trajectory[..., :num_steps]  # [B, C, H, W, T]

    logger.info(f"Computing metrics on the first channel and ensemble member...")
    true_field, predicted_field = true_trajectory[0], predicted_trajectory[0, 0]
    lsim = [
        LSiM_distance(true_field[..., t], predicted_field[..., t])
        for t in range(num_steps)
    ]
    rmse = [
        torch.sqrt(torch.mean((true_field[..., t] - predicted_field[..., t]) ** 2))
        for t in range(num_steps)
    ]
    logger.info(f"LSiM: {np.mean(lsim):.4f} ± {np.std(lsim):.4f}")
    logger.info(f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")

    figure_path = test_cfg.figure_dir
    logger.info(f"Plotting to {figure_path}...")
    plot_case(
        project,
        true_trajectory,
        predicted_trajectory,
        test_dataset,
        figure_path,
        show=test_cfg.show_plots,
    )


if __name__ == "__main__":
    main()
