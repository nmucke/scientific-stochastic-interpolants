"""The six Navier--Stokes-built cells on Lorenz 96: FlowDAS, SDA, D-Flow (SGLD), FIG,
SURGE (FlowDAS) and SURGE (SDA); and TMPD (:func:`run_tmpd`, built in
``_l96_generative``, not by NS, linear sweep only).

Nothing here implements a method. Each posterior is built by the Navier--Stokes
``build_posterior`` (``cases/navier_stokes/_ns_pipeline.py``), so the construction
is the NS one by design; this module only supplies its inputs (:func:`ns_posterior`,
used by :func:`run_baseline` here and by ``_l96_smc.run_surge`` for the two SURGE
cells) and runs the result through the same ``assimilate`` loop as the
observation-interpolant cells.

Inputs to ``build_posterior`` (one table, :data:`NS_CELLS`):

* ``method_cfg``: ``configs/method/<file>.yaml`` with every knob of the cell's
  blocks replaced by the explicit scalar of ``methods.<cell>`` in
  ``configs/case/lorenz96.yaml``. Every knob must be set there (no ``default:``
  fallback), unknown keys are refused (:func:`ns_settings`), and no table or other
  mapping may reach ``build_posterior`` (:func:`ns_method_config`). Scalars of the
  method YAML that the case does not set (D-Flow's ``burn``, which
  ``DFlowPosterior`` never reads; FlowDAS' ``posterior_model.resample``) pass
  through unchanged.
* ``prior``: a ``LoadedPrior`` holding only the network the method runs on:
  FlowDAS and SURGE (FlowDAS) the SI prior; D-Flow and FIG the FM prior; SDA and
  SURGE (SDA) the diffusion prior ``DenoiseDiffusionModel.from_flow_matching(fm)``
  (NS ``diffusion_from_fm``).
* operator and variance: ``observation_model(problem, stats)`` only, whose
  ``to_model`` also maps the observations (``assimilate(obs_to_model=...)``).
* ``num_steps`` = the cell's M. For D-Flow it is the SGLD chain length (NS maps M
  to ``num_optim_steps``); its differentiable rollout keeps ``ode_steps`` midpoint
  steps.

``build_posterior`` lets environment variables override the YAML (FLOWDAS_ZETA,
SDA_GAMMA, FIG_K, ...). These cells refuse to run while any variable with one of
the :data:`ENV_PREFIXES` is set (:func:`check_no_env_overrides`).

Batching. FlowDAS and D-Flow update each member on its own, so the sequences run as
one batch, as ``run_generative`` does, with the seed
``derive_seed(SEED_LABEL, name, split, seed)``. FIG differentiates the GLOBAL
residual norm ``||y_t - H x||`` over the whole batch (faithful to the official
sampler; NS runs one trajectory, i.e. the ``E`` members, per batch), so batching
sequences would couple them; SDA reports a divergence (Gamma not PD / not finite,
nonlinear h) for its whole batch, so in one batch a single diverged member would
NaN every sequence. Both run one sequence per batch, each with its own seed
``derive_seed(SEED_LABEL, name, split, seed, index)``, so a divergence NaNs only
its own sequence. (The DM-SDE cells keep their behaviour: their non-finite-drift
assertion records the whole cell as diverged, see ``driver.run_generative``.)

NFE is forward calls of the method's network per assimilation step, for one
evaluation of all members (a sequential batch over sequences is divided back out);
backward passes are not counted. At M steps:

* FlowDAS: 1 (first step) + (M-1) x [prior drift + 1 Heun drift of the x1
  predictor]; the J Monte-Carlo samples are not network calls (one batched pass of
  the obs operator over J*B).
* SDA: 1 (first step score) + (M-1) x [reverse-SDE drift + Tweedie score].
* D-Flow: M SGLD steps x (2 ode_steps forwards + 2 ode_steps recomputed by gradient
  checkpointing) + 2 ode_steps for the final rollout.
* FIG: M velocity calls; the k corrector steps touch no network (they apply h, or
  its jvp at the denoised state for a nonlinear h).
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

# Knob kinds; ``|null`` only where NS treats null as "off". Integers must be >= 1.
FLOAT, INT, OPT_FLOAT, OPT_INT = "float", "int", "float|null", "int|null"
#: cell -> (method YAML, tidy method = build_posterior's method_name, network,
#: {YAML block: {knob: kind}}). ``num_steps`` is every cell's M on top of these.
NS_CELLS: dict[str, tuple[str, Method, str, dict[str, dict[str, str]]]] = {
    "flowdas": (
        "flowdas.yaml",
        Method.FLOWDAS,
        "si",
        {
            "likelihood_model": {
                "guidance_scale": FLOAT,  # zeta on the raw score
                "num_mc_samples": INT,  # J (NS reads `J or E`: 0 is refused)
                "max_grad_norm": OPT_FLOAT,  # null: uncapped
                "mc_chunk": OPT_INT,  # null: all J at once (memory only)
            }
        },
    ),
    "sda": ("sda.yaml", Method.SDA, "diffusion", {"likelihood_model": {"gamma_sda": FLOAT}}),
    "dflow": (
        "dflow_sgld.yaml",
        Method.D_FLOW_SGLD,
        "fm",
        {
            "likelihood_model": {
                "step_size": FLOAT,  # eta
                "noise_scale": FLOAT,  # s
                "lambda_reg": FLOAT,  # lambda
                "precond_decay": FLOAT,  # omega
                "precond_eps": FLOAT,  # delta
                "ode_steps": INT,  # midpoint steps of the differentiable rollout
            }
        },
    ),
    "fig": (
        "guided_fm_fig.yaml",
        Method.GUIDED_FM_FIG,
        "fm",
        {
            "likelihood_model": {
                "guidance_steps": INT,  # k
                "guidance_scale": FLOAT,  # c
                "interpolant_noise": FLOAT,  # w
            }
        },
    ),
    "surge_flowdas": (
        "surge_flowdas.yaml",
        Method.SURGE_FLOWDAS,
        "si",
        {
            "likelihood_model": {
                "guidance_scale": FLOAT,  # FlowDAS zeta
                "num_mc_samples": INT,  # J
                "max_grad_norm": OPT_FLOAT,  # null: no cap
            },
            "posterior_model": {"ess_threshold": FLOAT},
        },
    ),
    "surge_sda": (
        "surge_sda.yaml",
        Method.SURGE_SDA,
        "diffusion",
        {"likelihood_model": {"gamma_sda": FLOAT}, "posterior_model": {"ess_threshold": FLOAT}},
    ),
}
#: Cells run by :func:`run_baseline` (the two SURGE cells run in ``_l96_smc``).
BASELINE_CELLS = ("flowdas", "sda", "dflow", "fig")
#: Cells run one sequence per batch (see the module docstring).
PER_SEQUENCE = ("fig", "sda")
#: Fixed, untuned settings of the method YAML (not knobs of the L96 block): the
#: SI-prior SURGE cell uses the exact Follmer-drift denoiser for its reward
#: (``SurgeStochasticInterpolantPosterior(si_denoiser="follmer")``; NS keeps the
#: library default ``velocity``).
FIXED = {"surge_flowdas": {"posterior_model": {"si_denoiser": "follmer"}}}
#: ``build_posterior`` reads environment overrides with these prefixes.
ENV_PREFIXES = ("FLOWDAS_", "SDA_", "FIG_", "DFLOW_")


def check_no_env_overrides(environ=None) -> None:
    """Refuse to run while an environment override of build_posterior is set."""
    environ = os.environ if environ is None else environ
    found = sorted(key for key in environ if key.startswith(ENV_PREFIXES))
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
        if int(value) != value or value < 1:
            raise ValueError(f"{name}.{key} must be a positive integer, got {value!r}.")
        return int(value)
    if not math.isfinite(float(value)):
        raise ValueError(f"{name}.{key} must be finite, got {value!r}.")
    return float(value)


def ns_settings(name: str, settings: dict) -> tuple[int, dict[str, dict]]:
    """``(num_steps, {block: {knob: value}})`` of an NS-built cell.

    Every knob of :data:`NS_CELLS` and ``num_steps`` must be in ``settings``, with
    a value of its kind; any other key is refused.
    """
    blocks = NS_CELLS[name][3]
    expected = {"num_steps", *(key for knobs in blocks.values() for key in knobs)}
    unknown, missing = set(settings) - expected, expected - set(settings)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown settings {sorted(unknown)}, missing {sorted(missing)}; "
            f"the block needs exactly {sorted(expected)}."
        )
    steps = _checked(name, "num_steps", settings["num_steps"], INT)
    knobs = {
        block: {key: _checked(name, key, settings[key], kind) for key, kind in keys.items()}
        for block, keys in blocks.items()
    }
    return steps, knobs


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def ns_method_config(name: str, knobs: dict[str, dict]):
    """The method YAML with each block's knobs replaced by ``knobs``.

    Refuses a YAML that has a schedule table (or any other mapping) left anywhere
    after the replacement, so build_posterior cannot resolve a value from another
    case's table.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(METHOD_CONFIGS / NS_CELLS[name][0]))
    for block, values in (*knobs.items(), *FIXED.get(name, {}).items()):
        cfg[block] = {**(cfg.get(block) or {}), **values}
    leftover = []
    for section, node in cfg.items():
        if isinstance(node, dict):
            leftover += [f"{section}.{k}" for k, v in node.items() if not _is_scalar(v)]
        elif not _is_scalar(node):
            leftover.append(section)
    if leftover:
        raise ValueError(
            f"{name}: {sorted(leftover)} are not scalars after applying the L96 "
            "block; add them to NS_CELLS so the case sets them."
        )
    return OmegaConf.create(cfg)


def ns_prior(network: str, checkpoints, device: str):
    """A ``LoadedPrior`` holding only ``network``: ``si``, ``fm``, or ``diffusion``
    (with the FM prior whose velocity network it evaluates)."""
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import LoadedPrior
    from scisi.models.diffusion_model import DenoiseDiffusionModel

    si = fm = diffusion = None
    if network == "si":
        si = generative.load_prior(checkpoints.si_run, device)
    else:
        fm = generative.load_prior(checkpoints.fm_run, device)
        if network == "diffusion":
            diffusion = DenoiseDiffusionModel.from_flow_matching(fm).eval().to(device)
    return LoadedPrior(
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


def ns_posterior(cfg, name: str, problem, stats, members: int, settings: dict, device: str):
    """``(network, posterior, stepper, observed, num_steps)`` of an NS-built cell."""
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import build_posterior

    check_no_env_overrides()
    steps, knobs = ns_settings(name, settings)
    _, method, network, _ = NS_CELLS[name]
    observed = generative.observation_model(problem, stats)
    model, posterior, stepper = build_posterior(
        method_name=method.value,
        method_cfg=ns_method_config(name, knobs),
        prior=ns_prior(network, cfg.checkpoints, device),
        obs_operator=observed.operator,
        variance=observed.variance,
        likelihood_ensemble_size=members,
        case_key="lorenz96",
        scenario_key="lorenz96",
        num_steps=steps,
    )
    return model, posterior, stepper, observed, steps


def run_baseline(cfg, name, problem, sequences, split, seed, members, settings):
    """Run FlowDAS, SDA, D-Flow or FIG; returns the driver's ``CellOutput``."""
    from cases.lorenz96 import driver
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter
    from common.seeding import derive_seed

    stats, device, initial, observations = driver.sampler_inputs(
        cfg, problem, sequences, split, seed, members
    )
    model, posterior, stepper, observed, steps = ns_posterior(
        cfg, name, problem, stats, members, settings, device
    )
    if name in PER_SEQUENCE:
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
    parts, diverged = [], False
    for part, batch_seed in batches:
        try:
            parts.append(
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
            )
        except FloatingPointError as error:
            # SDA reports its own divergence (Gamma not PD / not finite, nonlinear
            # h only): that sequence is recorded as diverged, any other error
            # crashes, and nfe stays None.
            if name != "sda" or "has diverged" not in str(error):
                raise
            print(f"{name} diverged: {error}", flush=True)
            parts.append(driver.diverged_ensembles(problem, observations[part], members))
            diverged = True
    ensembles = torch.cat(parts)
    nfe = None if diverged else counter.count / observations.shape[1] / len(batches)
    return driver.CellOutput(ensembles, time.time() - start, nfe, members, steps)


def run_tmpd(cfg, name, problem, sequences, split, seed, members, settings):
    """TMPD (boys_tweedie_2024) on the FM prior; returns the driver's ``CellOutput``.

    Not an NS construction: the posterior is ``_l96_generative.build_tmpd_posterior``
    (the FM network read as a diffusion model with the DM-SDE cell's
    endpoint-vanishing ``g``), with the cell's ``SamplerSettings``. Linear ``h`` only
    (``driver.run_method`` refuses the arctan sweep). TMPD updates each member on its
    own, so the sequences run as one batch with the seed
    ``derive_seed(SEED_LABEL, "tmpd", split, seed)``, as ``run_generative`` does.
    ``nfe`` counts the network forwards only; the reverse-mode VJPs per assimilation
    step (``d_y`` per guided step for full / diagonal covariance, 2 for row_sum) are
    returned as ``vjp`` and written as the ``vjp`` metric.
    """
    from cases.lorenz96 import driver
    import cases.lorenz96._l96_generative as generative
    from cases.navier_stokes._ns_pipeline import attach_nfe_counter
    from common.seeding import derive_seed

    settings = generative.SamplerSettings(**settings)
    stats, device, initial, observations = driver.sampler_inputs(
        cfg, problem, sequences, split, seed, members
    )
    model = generative.load_prior(cfg.checkpoints.fm_run, device)
    posterior, stepper = generative.build_posterior(
        Method.TMPD, model, problem, stats, settings
    )
    counter = attach_nfe_counter(model)
    counter.reset()
    start = time.time()
    ensembles = generative.assimilate(
        posterior,
        stepper,
        stats,
        initial,
        observations,
        settings.num_steps,
        derive_seed(driver.SEED_LABEL, name, split, seed),
        obs_to_model=generative.observation_model(problem, stats).to_model,
    )
    steps = observations.shape[1]
    nfe = counter.count / steps
    vjp = posterior.likelihood_model.vjp_count / steps
    return driver.CellOutput(
        ensembles, time.time() - start, nfe, members, settings.num_steps, vjp=vjp
    )
