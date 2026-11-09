"""Generative baselines on Lorenz 96: FlowDAS, SDA, D-Flow (SGLD) and FIG.

Nothing here implements a baseline. Each posterior is built by the Navier--Stokes
``build_posterior`` (``cases/navier_stokes/_ns_pipeline.py``), so the construction
is the NS one by design; this module only supplies its inputs and runs the result
through the same ``assimilate`` loop as the observation-interpolant cells.

Inputs to ``build_posterior``:

* ``method_cfg``: ``configs/method/<file>.yaml`` with every knob the NS branch
  reads replaced by the explicit scalar of ``methods.<cell>`` in
  ``configs/case/lorenz96.yaml``. Every knob must be set there (no
  ``default:`` fallback), unknown keys are refused, and no schedule table
  reaches ``build_posterior`` (:func:`method_config`).
* ``prior``: a ``LoadedPrior`` holding only the network the method runs on:
  FlowDAS the SI prior; D-Flow and FIG the FM prior; SDA the diffusion prior
  ``DenoiseDiffusionModel.from_flow_matching(fm)`` (NS ``diffusion_from_fm``).
* operator and variance: ``observation_model(problem, stats)`` only, whose
  ``to_model`` also maps the observations (``assimilate(obs_to_model=...)``).
* ``num_steps`` = the cell's M (50). For D-Flow it is the SGLD chain length
  (NS maps M to ``num_optim_steps``); its differentiable rollout keeps
  ``ode_steps`` midpoint steps. ``burn`` is passed as NS does but
  ``DFlowPosterior`` never reads it (it returns the chain endpoint), so
  ``burn > num_steps`` changes nothing.

``build_posterior`` lets environment variables override the YAML (FLOWDAS_ZETA,
SDA_GAMMA, FIG_K, ...). These cells refuse to run while any of them is set
(:func:`check_no_env_overrides`).

Batching. The sequences run as one batch, as ``run_generative`` does, for the
methods whose update is per member (FlowDAS, SDA, D-Flow). FIG differentiates
the GLOBAL residual norm ``||y_t - H x||`` over the whole batch (faithful to the
official sampler; NS runs one trajectory, i.e. the ``E`` members, per batch), so
batching sequences would couple them: FIG runs one sequence per batch, each with
its own seed ``derive_seed(SEED_LABEL, "fig", split, seed, index)``.

NFE (:data:`NFE_MEANING`) is forward calls of the method's network per
assimilation step, for one evaluation of all members (a sequential batch over
sequences is divided back out). Backward passes are not counted.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from results_schema import Method

METHOD_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "method"

#: cell -> tidy method label.
BASELINES = {
    "flowdas": Method.FLOWDAS,
    "sda": Method.SDA,
    "dflow": Method.D_FLOW_SGLD,
    "fig": Method.GUIDED_FM_FIG,
}
#: cell -> method config file whose structure (stepper, targets) is reused.
METHOD_FILES = {
    "flowdas": "flowdas.yaml",
    "sda": "sda.yaml",
    "dflow": "dflow_sgld.yaml",
    "fig": "guided_fm_fig.yaml",
}
#: cell -> network the NS branch runs it on.
NETWORK = {"flowdas": "si", "sda": "diffusion", "dflow": "fm", "fig": "fm"}

# Knobs the NS branch of build_posterior reads from ``likelihood_model``, with
# their type; ``None`` allowed only where NS treats null as "off".
_FLOAT, _INT, _OPT_FLOAT, _OPT_INT = "float", "int", "float|null", "int|null"
KNOBS = {
    "flowdas": {
        "guidance_scale": _FLOAT,  # zeta on the raw score
        "num_mc_samples": _INT,  # J
        "max_grad_norm": _OPT_FLOAT,  # null: uncapped
        "mc_chunk": _OPT_INT,  # null: all J at once (memory only)
    },
    "sda": {"gamma_sda": _FLOAT},
    "dflow": {
        "step_size": _FLOAT,  # eta
        "noise_scale": _FLOAT,  # s
        "lambda_reg": _FLOAT,  # lambda
        "precond_decay": _FLOAT,  # omega
        "precond_eps": _FLOAT,  # delta
        "ode_steps": _INT,  # midpoint steps of the differentiable rollout
        "burn": _INT,  # stored by DFlowPosterior but never read: no effect
    },
    "fig": {
        "guidance_steps": _INT,  # k
        "guidance_scale": _FLOAT,  # c
        "interpolant_noise": _FLOAT,  # w
    },
}
#: Variables ``build_posterior`` reads ahead of the YAML.
ENV_OVERRIDES = (
    "FLOWDAS_ZETA",
    "FLOWDAS_MAX_GRAD_NORM",
    "FLOWDAS_MC_CHUNK",
    "SDA_GAMMA",
    "FIG_K",
    "FIG_C",
    "DFLOW_NSTEPS",
    "DFLOW_STEP_SIZE",
    "DFLOW_NOISE_SCALE",
    "DFLOW_LAMBDA",
)
#: Also refused: any variable with these prefixes (a future override).
ENV_PREFIXES = ("FLOWDAS_", "SDA_", "FIG_", "DFLOW_")
#: Cells whose update couples the members of one batch.
POOLS_ACROSS_BATCH = ("fig",)
#: What ``nfe`` counts at M steps (forward calls of the method's network per
#: assimilation step, for all members at once; backward passes not counted).
NFE_MEANING = {
    "flowdas": "1 (first step) + (M-1) x [prior drift + 1 Heun drift of the x1 "
    "predictor]; the J Monte-Carlo samples are not network calls (one batched "
    "pass of the obs operator over J*B)",
    "sda": "1 (first step score) + (M-1) x [reverse-SDE drift + Tweedie score]",
    "dflow": "M SGLD steps x (2 ode_steps forwards + 2 ode_steps recomputed by "
    "gradient checkpointing) + 2 ode_steps for the final rollout",
    "fig": "M velocity calls; the k corrector steps touch no network (they apply "
    "h, or its jvp at the denoised state for a nonlinear h)",
}
#: cell -> (exception type, message fragment) that reports a diverged sampler.
DIVERGENCE_ERRORS = {"sda": (FloatingPointError, "has diverged")}


def check_no_env_overrides(environ=None) -> None:
    """Refuse to run while an environment override of build_posterior is set."""
    environ = os.environ if environ is None else environ
    found = sorted(
        key
        for key in environ
        if key in ENV_OVERRIDES or key.startswith(ENV_PREFIXES)
    )
    if found:
        raise RuntimeError(
            f"Environment overrides {found} are set; they would change a baseline "
            "cell behind its configuration. Unset them (use `cell --set` with a "
            "--tag to change a knob)."
        )


def _checked(name: str, key: str, value, kind: str):
    nullable = kind.endswith("|null")
    if value is None:
        if nullable:
            return None
        raise ValueError(f"{name}.{key} must be set (null would fall back to a default).")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}.{key} must be a number, got {value!r}.")
    if kind.startswith("int"):
        if int(value) != value:
            raise ValueError(f"{name}.{key} must be an integer, got {value!r}.")
        return int(value)
    if not math.isfinite(float(value)):
        raise ValueError(f"{name}.{key} must be finite, got {value!r}.")
    return float(value)


def cell_settings(name: str, settings: dict) -> tuple[int, dict]:
    """``(num_steps, knobs)`` of a cell; every knob explicit, unknown keys refused."""
    expected = set(KNOBS[name]) | {"num_steps"}
    unknown = set(settings) - expected
    missing = expected - set(settings)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown settings {sorted(unknown)}, missing {sorted(missing)}; "
            f"the block needs exactly {sorted(expected)}."
        )
    knobs = {
        key: _checked(name, key, settings[key], kind) for key, kind in KNOBS[name].items()
    }
    steps = _checked(name, "num_steps", settings["num_steps"], _INT)
    if steps < 1:
        raise ValueError(f"{name}.num_steps must be positive, got {steps}.")
    if name == "flowdas" and knobs["num_mc_samples"] < 1:
        # NS reads ``num_mc_samples or E``: 0 would silently become J = E.
        raise ValueError("flowdas.num_mc_samples must be positive.")
    if name == "fig" and knobs["guidance_steps"] < 1:
        raise ValueError("fig.guidance_steps must be positive.")
    if name == "dflow" and knobs["ode_steps"] < 1:
        raise ValueError("dflow.ode_steps must be positive.")
    return steps, knobs


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def method_config(name: str, knobs: dict):
    """The method YAML with the knobs of the NS branch replaced by ``knobs``.

    Refuses a YAML that has a schedule table (or any other mapping) left in
    ``likelihood_model`` after the replacement, so build_posterior cannot
    resolve a value from another case's table.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(METHOD_CONFIGS / METHOD_FILES[name]))
    block = dict(cfg.get("likelihood_model") or {})
    block.update(knobs)
    cfg["likelihood_model"] = block
    leftover = []
    for section, node in cfg.items():
        if isinstance(node, dict):
            leftover += [f"{section}.{k}" for k, v in node.items() if not _is_scalar(v)]
        elif not _is_scalar(node):
            leftover.append(section)
    if leftover:
        raise ValueError(
            f"{name}: {sorted(leftover)} are not scalars after applying the L96 "
            "block; add them to KNOBS so the case sets them."
        )
    return OmegaConf.create(cfg)


def build_baseline(
    name: str, problem, stats, members: int, settings: dict, device: str, checkpoints
):
    """``(network, posterior, stepper, observed, num_steps)`` for one baseline cell.

    ``checkpoints`` is the case's ``checkpoints`` block (``si_run``, ``fm_run``).
    """
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import LoadedPrior, build_posterior
    from scisi.models.diffusion_model import DenoiseDiffusionModel

    check_no_env_overrides()
    steps, knobs = cell_settings(name, settings)
    method_cfg = method_config(name, knobs)
    si = fm = diffusion = None
    if NETWORK[name] == "si":
        si = generative.load_prior(checkpoints.si_run, device)
    else:
        fm = generative.load_prior(checkpoints.fm_run, device)
        if NETWORK[name] == "diffusion":
            diffusion = DenoiseDiffusionModel.from_flow_matching(fm)
    prior = LoadedPrior(
        si_model=si,
        fm_model=fm,
        preprocesser=None,
        test_dataset=None,
        len_field_history=1,
        train_cfg=OmegaConf.create({}),
        checkpoint_name="lorenz96",
        has_trained_weights=True,
        diffusion_model=diffusion,
    )
    observed = generative.observation_model(problem, stats)
    model, posterior, stepper = build_posterior(
        method_name=BASELINES[name].value,
        method_cfg=method_cfg,
        prior=prior,
        obs_operator=observed.operator,
        variance=observed.variance,
        likelihood_ensemble_size=members,
        case_key="lorenz96",
        scenario_key="lorenz96",
        num_steps=steps,
    )
    return model, posterior, stepper, observed, steps


def run_baseline(cfg, name, problem, sequences, split, seed, members, settings):
    """Run a baseline cell; returns the driver's ``CellOutput``."""
    from cases.lorenz96 import driver
    import cases.lorenz96._l96_generative as generative
    from cases.lorenz96._l96_system import resolve_device
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter
    from common.seeding import derive_seed

    stats = generative.Standardisation.from_data(driver.DATA_PATH, problem)
    device = resolve_device(cfg.evaluation.device)
    model, posterior, stepper, observed, steps = build_baseline(
        name, problem, stats, members, settings, device, cfg.checkpoints
    )
    initial = torch.stack(
        [
            driver.initial_ensemble(problem, seq, members, split, index, seed)
            for index, seq in enumerate(sequences)
        ]
    )
    observations = torch.stack([seq.observations for seq in sequences])
    if name in POOLS_ACROSS_BATCH:
        batches = [
            (slice(i, i + 1), derive_seed(driver.SEED_LABEL, name, split, seed, i))
            for i in range(len(sequences))
        ]
    else:
        batches = [(slice(None), derive_seed(driver.SEED_LABEL, name, split, seed))]
    # SDA's network is the FM velocity net (attach_nfe_counter falls back to .model).
    counter = attach_nfe_counter(model)
    counter.reset()
    start = time.time()
    nfe = None
    try:
        parts = [
            generative.assimilate(
                posterior,
                stepper,
                stats,
                initial[part],
                observations[part],
                steps,
                batch_seed,
                obs_to_model=observed.to_model,
            )
            for part, batch_seed in batches
        ]
        ensembles = torch.cat(parts)
        nfe = counter.count / observations.shape[1] / len(batches)
    except DIVERGENCE_ERRORS.get(name, (_Never, ""))[0] as error:
        # The sampler reports its own divergence (SDA's Gamma not PD / not finite,
        # nonlinear h only). As for DM-SDE in run_generative: the whole cell is
        # recorded as diverged, any other error crashes, and nfe stays None.
        if DIVERGENCE_ERRORS[name][1] not in str(error):
            raise
        print(f"{name} diverged: {error}", flush=True)
        shape = (len(sequences), observations.shape[1], members, problem.dim_state)
        ensembles = torch.full(shape, float("nan"), dtype=torch.float64)
    seconds = time.time() - start
    return driver.CellOutput(ensembles, seconds, nfe, members, steps)


class _Never(Exception):
    """Placeholder: a cell without a divergence error catches nothing."""
