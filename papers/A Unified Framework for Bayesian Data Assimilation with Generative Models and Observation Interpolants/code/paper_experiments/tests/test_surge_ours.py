"""SURGE on our samplers: ``SURGE (SI-SDE)`` and ``SURGE (DM-SDE)`` (NS case).

CPU tests on tiny random-weight priors (no data or trained weights). They check
that the SURGE proposal step IS our guided SDE step (SURGE only adds the SMC
reweighting), that the builder wires the right classes with the Ours likelihood
tables, that the configs inherit the Ours tables, and the variant tagging.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_forecast_correction_ns_urban import G, L, _prior, _truth  # noqa: E402
from cases.navier_stokes import driver as ns_driver  # noqa: E402
from cases.navier_stokes._ns_pipeline import (  # noqa: E402
    attach_nfe_counter,
    build_posterior,
    run_assimilation,
)
from results_schema import Method  # noqa: E402
from scisi.likelihood_models.observation_operators import (  # noqa: E402
    LinearObservationOperator,
)
from scisi.posterior_models.flow_matching_posterior import FlowMatchingPosterior  # noqa: E402
from scisi.posterior_models.stochastic_interpolant_posterior import (  # noqa: E402
    StochasticInterpolantPosterior,
)
from scisi.posterior_models.surge_posterior import (  # noqa: E402
    SurgeLiftedFlowMatchingPosterior,
    SurgeStochasticInterpolantPosterior,
)

torch.set_num_threads(2)
EXP = Path(__file__).resolve().parents[1]
PAIRS = {"SURGE (SI-SDE)": "Ours (SI-SDE)", "SURGE (DM-SDE)": "Ours (DM-SDE)"}
MODES = ("dps_jacobian_free", "inflated_shared")


def _cfgs() -> dict:
    """Method configs exactly as the NS driver loads them (with ``inherit``)."""
    drv = ns_driver.NavierStokesRunner.__new__(ns_driver.NavierStokesRunner)
    return {m.value: c for m, c in drv._method_cfgs().items()}


def _op() -> LinearObservationOperator:
    return LinearObservationOperator(type="random", data_size=(1, G, G), percent_obs=0.1, seed=3)


def _build(name: str, mode: str, prior, op):  # type: ignore[no-untyped-def]
    return build_posterior(
        method_name=name, method_cfg=_cfgs()[name], prior=prior, obs_operator=op,
        variance=0.0025, likelihood_ensemble_size=4, likelihood_mode=mode,
        case_key="navier_stokes", scenario_key="sparse_5", num_steps=100,
        jacobian_refresh_every=10 if mode == "inflated_shared" else None,
    )


def test_configs_inherit_ours_tables() -> None:
    cfgs = _cfgs()
    for surge, ours in PAIRS.items():
        c, o = cfgs[surge], cfgs[ours]
        assert c.name == surge and "inherit" not in c
        assert OmegaConf.to_container(c.likelihood_model) == OmegaConf.to_container(o.likelihood_model)
        assert c.stepper == o.stepper == "sde"
        assert float(c.posterior_model.ess_threshold) == 0.5


@pytest.mark.parametrize("mode", MODES)
def test_builder_classes_and_shared_likelihood(mode: str) -> None:
    prior, op = _prior(1), _op()
    for surge, ours in PAIRS.items():
        _, ps, _ = _build(surge, mode, prior, op)
        _, po, _ = _build(ours, mode, prior, op)
        want = SurgeStochasticInterpolantPosterior if "SI" in surge else SurgeLiftedFlowMatchingPosterior
        assert type(ps) is want
        lk_s, lk_o = ps.likelihood_model, po.likelihood_model
        assert type(lk_s) is type(lk_o)
        for attr in ("likelihood_mode", "jacobian_refresh_every", "jacobian_damping", "model_class"):
            assert getattr(lk_s, attr, None) == getattr(lk_o, attr, None), attr
        assert ps.model is po.model and ps.variance == pytest.approx(0.0025)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("surge", list(PAIRS))
def test_proposal_step_equals_ours_step(surge: str, mode: str) -> None:
    """One SURGE ``_one_step`` (before SMC weights) == one Ours guided step."""
    prior, op = _prior(1), _op()
    _, ps, _ = _build(surge, mode, prior, op)
    _, po, _ = _build(PAIRS[surge], mode, prior, op)
    E = 3
    g = torch.Generator().manual_seed(1)
    x = torch.randn(E, 1, G, G, generator=g)
    fh = torch.randn(E, 1, G, G, L, generator=g)
    y = op(torch.randn(1, 1, G, G, generator=g)).expand(E, -1).contiguous()
    t = torch.full((1, 1), 0.37)  # the sampler passes one pseudo-time for the batch
    dt = torch.tensor(0.05)
    outs = []
    for post in (po, ps):
        if isinstance(post, (SurgeStochasticInterpolantPosterior, SurgeLiftedFlowMatchingPosterior)):
            post._pre_step(x, y, t)
        torch.manual_seed(7)
        with torch.no_grad():
            outs.append(post._one_step(base=x.clone(), t=t, field_history=fh, field_cond=None,
                                       pars_cond=None, observations=y, dt=dt))
    assert torch.allclose(outs[0], outs[1], atol=1e-5, rtol=1e-5), (outs[0] - outs[1]).abs().max()
    assert torch.isfinite(ps._incr_chunks[-1]).all()


def test_dm_first_node_unguided() -> None:
    """At tau = 0 the DM-SDE combo applies no guidance, like FlowMatchingPosterior."""
    prior, op = _prior(1), _op()
    _, ps, _ = _build("SURGE (DM-SDE)", "dps_jacobian_free", prior, op)
    x = torch.randn(2, 1, G, G)
    grad, _ = ps._compute_guidance(x, torch.zeros(1, 1), op(x), torch.randn(2, 1, G, G, L),
                                   None, None, torch.tensor(0.1), torch.zeros_like(x))
    assert torch.count_nonzero(grad) == 0


@pytest.mark.parametrize("surge", list(PAIRS))
def test_rollout_runs(surge: str) -> None:
    prior, op = _prior(1), _op()
    model, post, stepper = _build(surge, "dps_jacobian_free", prior, op)
    T = L + 2
    truth = _truth(op, 1, T)
    nfe = attach_nfe_counter(model)
    torch.manual_seed(0)
    res = run_assimilation(
        posterior=post, model=model, truth_obs=truth, ensemble_size=4, num_steps=4,
        num_physical_steps=T, stepper=stepper, nfe_counter=nfe,
        gaussian_base=surge != "SURGE (SI-SDE)",
    )
    assert res.posterior_trajectory.shape == (4, 1, G, G, T)
    assert torch.isfinite(res.posterior_trajectory).all()


def test_dm_requires_diffusion() -> None:
    prior, op = _prior(1), _op()
    with pytest.raises(ValueError):
        SurgeLiftedFlowMatchingPosterior(model=prior.fm_model, obs_operator=op, diffusion_term=None)


def test_registry_and_variants() -> None:
    for m in (Method.SURGE_SI_SDE, Method.SURGE_DM_SDE):
        assert m in ns_driver.WIRED_METHODS and m in ns_driver.VARIANT_METHODS
        assert m not in ns_driver.NS_METHODS  # default lineup unchanged
        assert (EXP / "configs" / "method" / f"{ns_driver.METHOD_CONFIG_NAME[m]}.yaml").exists()
    assert ns_driver.OURS_METHODS == frozenset(
        (Method.OURS_SI_SDE, Method.OURS_DM_SDE, Method.OURS_FM_ODE)
    )
