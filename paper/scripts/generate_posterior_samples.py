"""Generate posterior trajectory samples for one method on one case.

Composes a Hydra config from `paper/configs/benchmark.yaml`. Run a single method:

    python paper/scripts/generate_posterior_samples.py method=si

Switch case (e.g. udales):

    python paper/scripts/generate_posterior_samples.py case=udales method=si

Sweep all methods (one process per method, separate Hydra run dirs):

    python paper/scripts/generate_posterior_samples.py --multirun method=si,fm,flowdas

Sweep over solver step counts (cross-product with methods is supported too):

    python paper/scripts/generate_posterior_samples.py --multirun \\
        method=si,fm,flowdas case.num_steps=50,100,250

To sweep test ids cheaply (model loads once), put a list in the case config:

    case.test_sample_indices=[0,1,2,3,4]

Outputs land in `paper/results/<case>/<method>/sample_<test_id>_steps_<num_steps>.pt`.
The checkpoint loaded for each method is `case.checkpoints[method.name]` under
`checkpoints/<case.project>/`.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Make `paper` importable when run as a script.
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scisi.models.follmer_stochastic_interpolant import FollmerStochasticInterpolant
from scisi.posterior_models.flow_matching_posterior import FlowMatchingPosterior
from scisi.sampling.ode_solvers import euler_step
from scisi.sampling.sde_solvers import euler_maruyama_step, heun_step

logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.float32)

_STEPPERS = {"sde": euler_maruyama_step, "ode": euler_step}


def _resolve_checkpoint_name(cfg: DictConfig) -> str:
    """Look up the checkpoint for the active method in the case's checkpoints map."""
    method_name = cfg.method.name
    try:
        return cfg.case.checkpoints[method_name]
    except Exception as exc:
        raise KeyError(
            f"Case '{cfg.case.name}' has no checkpoint configured for method "
            f"'{method_name}'. Add it under case.checkpoints."
        ) from exc


def _load_pretrained(project: str, checkpoint_name: str) -> tuple[Any, DictConfig, int]:
    """Load training-time config + model weights for the given checkpoint."""

    cfg = OmegaConf.load(f"checkpoints/{project}/{checkpoint_name}/config.yaml")
    logger.info(f"Loaded checkpoint config: project={project} name={checkpoint_name}")

    try:
        len_field_history = cfg.model.drift_model.len_field_history
    except Exception:
        len_field_history = cfg.model.denoise_model.len_field_history

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(
        torch.load(f"checkpoints/{project}/{checkpoint_name}/model.pth")
    )
    model.eval()
    model.to("cuda")
    return model, cfg, len_field_history


def _prepare_inputs(
    test_dataset: Any,
    test_sample_index: int,
    preprocesser: Any,
    len_field_history: int,
) -> dict[str, torch.Tensor | None]:
    """Pull one trajectory out of the dataset and run it through the preprocessor."""

    sample = test_dataset[test_sample_index]
    trajectory = sample["x"].unsqueeze(0)
    field_cond = sample["field_cond"].unsqueeze(0) if "field_cond" in sample else None
    pars_cond = sample["pars_cond"].unsqueeze(0) if "pars_cond" in sample else None

    init_data = preprocesser.transform(
        base=trajectory[..., len_field_history - 1],
        field_history=trajectory[..., 0:len_field_history],
        is_batch=True,
    )
    init_data["field_cond"] = preprocesser.transform(
        field_cond=field_cond,
        is_batch=True,
        is_trajectory=True,
    )["field_cond"]
    init_data["pars_cond"] = preprocesser.transform(
        pars_cond=pars_cond,
        is_batch=True,
        is_trajectory=True,
    )["pars_cond"]

    transformed_traj = preprocesser.transform(
        base=trajectory,
        is_batch=True,
        is_trajectory=True,
    )["base"]

    return {
        "init_data": init_data,
        "trajectory": transformed_traj,
    }


def _make_observations(
    trajectory: torch.Tensor,
    obs_operator: Any,
    num_physical_steps: int,
    variance: float,
    seed: int,
) -> torch.Tensor:
    """Build noisy observations from the ground-truth trajectory.

    Seeded so every method on the same (case, test_id) sees identical observations.
    """

    gen = torch.Generator(device="cpu").manual_seed(seed)
    observations = torch.zeros(1, obs_operator.num_obs, num_physical_steps)
    sigma = torch.sqrt(torch.tensor(variance))
    for i in range(num_physical_steps):
        observations[:, :, i] = obs_operator(trajectory[:, :, :, :, i].to("cuda")).cpu()
        observations[:, :, i] += (
            torch.randn(observations[:, :, i].shape, generator=gen) * sigma
        )
    return observations


def _build_diffusion_term(method_cfg: DictConfig, model: Any):
    """Construct the diffusion_term callable. Only meaningful for SI."""
    if method_cfg.diffusion_term is None:
        return None
    multiplier = float(method_cfg.diffusion_term.multiplier)
    return lambda t: multiplier * model.interpolation.gamma(t)


def _instantiate_posterior(
    cfg: DictConfig, model: Any, obs_operator: Any
) -> tuple[Any, Any]:
    """Build likelihood + posterior models from the composed config."""

    likelihood_model = hydra.utils.instantiate(
        cfg.method.likelihood_model,
        obs_operator=obs_operator,
        model=model,
        variance=cfg.case.variance,
        ensemble_size=cfg.case.likelihood_ensemble_size,
    )

    diffusion_term = _build_diffusion_term(cfg.method, model)

    posterior_model = hydra.utils.instantiate(
        cfg.method.posterior_model,
        model=model,
        likelihood_model=likelihood_model,
        diffusion_term=diffusion_term,
    )
    return likelihood_model, posterior_model


def _run_one_test_id(
    cfg: DictConfig,
    test_sample_index: int,
    model: Any,
    checkpoint_name: str,
    preprocesser: Any,
    test_dataset: Any,
    len_field_history: int,
    out_dir: Path,
) -> None:
    """Run posterior + prior sampling for a single test trajectory and save it."""

    prepped = _prepare_inputs(
        test_dataset=test_dataset,
        test_sample_index=test_sample_index,
        preprocesser=preprocesser,
        len_field_history=len_field_history,
    )
    init_data = prepped["init_data"]
    trajectory = prepped["trajectory"]

    obs_operator = hydra.utils.instantiate(
        cfg.case.obs_operator,
        data_size=init_data["base"][0].shape,
    )
    logger.info(
        f"obs operator: num_obs={obs_operator.num_obs} "
        f"({obs_operator.num_obs / obs_operator.num_dofs * 100:.2f}% observed)"
    )

    likelihood_model, posterior_model = _instantiate_posterior(
        cfg, model, obs_operator
    )

    obs_seed = 1000 * test_sample_index + 7
    observations = _make_observations(
        trajectory=trajectory,
        obs_operator=obs_operator,
        num_physical_steps=cfg.case.num_physical_steps,
        variance=cfg.case.variance,
        seed=obs_seed,
    )

    stepper = _STEPPERS[cfg.method.stepper]
    common_input = {
        "base": (
            init_data["base"]
            if isinstance(model, FollmerStochasticInterpolant)
            else None
        ),
        "field_history": init_data["field_history"],
        "field_cond": init_data["field_cond"],
        "pars_cond": init_data["pars_cond"],
        "stepper": stepper,
        "num_physical_steps": cfg.case.num_physical_steps,
    }

    mp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if cfg.case.mixed_precision
        else contextlib.nullcontext()
    )

    with mp_ctx:
        logger.info(f"Sampling posterior (test_id={test_sample_index})...")
        posterior_trajectory = posterior_model.sample_trajectory(
            **common_input,
            ensemble_size=cfg.case.ensemble_size,
            num_steps=cfg.case.num_steps,
            observations=observations[:, :, len_field_history:],
        )

        logger.info(f"Sampling prior (test_id={test_sample_index})...")
        prior_trajectory = model.sample_trajectory(
            **common_input,
            num_steps=50,
        )

    posterior_trajectory = preprocesser.inverse_transform(
        base=posterior_trajectory, is_batch=True, is_trajectory=True
    )["base"]
    prior_trajectory = preprocesser.inverse_transform(
        base=prior_trajectory, is_batch=True, is_trajectory=True
    )["base"]
    true_trajectory = preprocesser.inverse_transform(
        base=trajectory.cpu(), is_batch=True, is_trajectory=True
    )["base"]

    payload = {
        "posterior_trajectory": posterior_trajectory.cpu(),
        "prior_trajectory": prior_trajectory.cpu(),
        "true_trajectory": true_trajectory.cpu(),
        "observations": observations.cpu(),
        "obs_indices_on_grid": obs_operator.obs_indices_on_grid.cpu(),
        "obs_indices_c_h_w": obs_operator.obs_indices_c_h_w,
        "meta": {
            "case": cfg.case.name,
            "method": cfg.method.name,
            "checkpoint_name": checkpoint_name,
            "test_sample_index": int(test_sample_index),
            "num_physical_steps": int(cfg.case.num_physical_steps),
            "num_steps": int(cfg.case.num_steps),
            "ensemble_size": int(cfg.case.ensemble_size),
            "len_field_history": int(len_field_history),
            "variance": float(cfg.case.variance),
            "obs_seed": int(obs_seed),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample_{test_sample_index}_steps_{int(cfg.case.num_steps)}.pt"
    torch.save(payload, out_path)
    logger.info(f"Saved {out_path}")


@hydra.main(  # type: ignore[misc]
    config_path="../configs",
    config_name="benchmark",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info(f"Case  : {cfg.case.name}")
    logger.info(f"Method: {cfg.method.name}")

    test_indices = cfg.case.test_sample_indices
    if isinstance(test_indices, int):
        test_indices = [test_indices]
    else:
        test_indices = list(test_indices)
    logger.info(f"Test sample indices: {test_indices}")

    torch.manual_seed(42)

    checkpoint_name = _resolve_checkpoint_name(cfg)
    logger.info(f"Checkpoint: {cfg.case.project}/{checkpoint_name}")

    model, train_cfg, len_field_history = _load_pretrained(
        cfg.case.project, checkpoint_name
    )
    preprocesser = hydra.utils.instantiate(train_cfg.preprocesser)
    test_dataset = hydra.utils.instantiate(train_cfg.test_data)

    out_dir = Path(cfg.results_root) / cfg.case.name / cfg.method.name
    out_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, out_dir / f"config_steps_{int(cfg.case.num_steps)}.yaml")

    for test_id in test_indices:
        _run_one_test_id(
            cfg=cfg,
            test_sample_index=int(test_id),
            model=model,
            checkpoint_name=checkpoint_name,
            preprocesser=preprocesser,
            test_dataset=test_dataset,
            len_field_history=len_field_history,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
