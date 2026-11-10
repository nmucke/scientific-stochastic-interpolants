"""SURGE / SMC cells of the Lorenz 96 case and their ESS records (``_l96_smc.py``).

That every SMC cell runs and writes its ESS records (and is deterministic) is
test_lorenz96_case.test_every_cell_runs_..."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cases.lorenz96._l96_generative as generative  # noqa: E402
import cases.lorenz96._l96_smc as smc_mod  # noqa: E402
from cases.lorenz96 import driver  # noqa: E402
from cases.lorenz96._l96_smc import (  # noqa: E402
    SMC_ESS,
    SMC_ESS_MIN,
    SMC_KEYS,
    SMC_RESAMPLES,
    SURGE_CELLS,
    SMCRecorder,
)
from cases.lorenz96.driver import initial_ensemble, load_sequences  # noqa: E402
from results_schema import Method  # noqa: E402
from scisi.particle_filter.bootstrap import BootstrapParticleFilter  # noqa: E402
from scisi.posterior_models.surge_posterior import SurgePosteriorBase  # noqa: E402
from test_lorenz96_case import tiny_case  # noqa: E402,F401  (fixture)


def test_cells_and_labels():
    assert {name: driver.CELLS[name] for name in SURGE_CELLS} == {
        "surge_si_sde": (Method.OURS_SI_SDE, "surge"),
        "surge_dm_sde": (Method.OURS_DM_SDE, "surge"),
        "surge_unguided_si": (Method.OURS_SI_SDE, "surge_unguided"),
        "surge_unguided_dm": (Method.OURS_DM_SDE, "surge_unguided"),
        "surge_flowdas": (Method.SURGE_FLOWDAS, None),
        "surge_sda": (Method.SURGE_SDA, None),
    }
    for name in SURGE_CELLS:
        assert name not in driver.GENERATIVE
        assert name in driver.TUNABLE


def test_config_blocks_are_complete_at_fixed_m():
    for obs in ("linear", "arctan"):
        cfg = driver.case_config(obs)
        for name in SURGE_CELLS:
            settings = driver.method_settings(cfg, name)
            assert settings["num_steps"] == driver.FIXED_NUM_STEPS
            assert "ess_threshold" in settings
        # Ours inside SURGE runs the plain cell's sampler block (inherited).
        for name, plain in smc_mod.SURGE_OURS.items():
            own = dict(cfg.methods[name])
            assert set(own) == {"ess_threshold"}
            assert driver.method_settings(cfg, name) == {
                **driver.method_settings(cfg, plain), **own
            }


def test_a_cell_without_smc_writes_no_smc_keys(tiny_case):
    records = driver.evaluate_cell("enkf", 0, tiny_case)
    assert not {r.metric for r in records} & set(SMC_KEYS)
    out = driver.run_method(
        tiny_case, "enkf", driver.problem_of(tiny_case),
        load_sequences(driver.problem_of(tiny_case), "test", 2), "test", 0,
    )
    assert out.smc is None


@pytest.mark.parametrize("name", sorted(SURGE_CELLS))
def test_surge_cells_record_smc_and_depend_on_the_seed(tiny_case, name):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    a = driver.run_method(tiny_case, name, problem, sequences, "test", 0)
    assert a.ensembles.shape == (2, problem.num_steps, 6, problem.dim_state)
    assert len(a.smc) == 2 and all(len(e[SMC_ESS]) == problem.num_steps for e in a.smc)
    c = driver.run_method(tiny_case, name, problem, sequences, "test", 1)
    assert not torch.equal(a.ensembles, c.ensembles)


def test_one_sequence_per_surge_call_is_enforced():
    # The recorder refuses a call holding more members than one ensemble.
    posterior = SimpleNamespace(_maybe_resample=lambda b, h, force=False: (b, h))
    SMCRecorder(posterior, 6)
    with pytest.raises(ValueError, match="one sequence"):
        posterior._maybe_resample(torch.zeros(12, 1, 1, 16), None)


class ToyPosterior:
    """The resampling interface of SurgePosteriorBase, with weights set by hand."""

    def __init__(self, threshold):
        self.ess_threshold = threshold
        self._log_w = None
        self.calls = []

    _normalized_weights = SurgePosteriorBase._normalized_weights

    def _maybe_resample(self, base, field_history, force=False):
        self.calls.append(force)
        return SurgePosteriorBase._maybe_resample(self, base, field_history, force)


def test_ess_recorder_matches_a_hand_computation():
    post = ToyPosterior(threshold=0.4)
    recorder = SMCRecorder(post, 4)
    base = torch.arange(4.0).reshape(4, 1)

    def hand(w):
        w = torch.tensor(w, dtype=torch.float64)
        return float(w.sum() ** 2 / (w**2).sum()) / 4

    torch.manual_seed(0)
    # Window 1: pseudo-step weights (1,1,1,1) -> ESS/N 1; then (1,1,1,5) -> 4/7,
    # not below 0.4: no resample; final forced.
    post._log_w = torch.zeros(4)
    post._maybe_resample(base, base)
    post._log_w = torch.log(torch.tensor([1.0, 1.0, 1.0, 5.0]))
    post._maybe_resample(base, base)
    post._maybe_resample(base, base, force=True)
    # Window 2: (1,0,0,0) collapses -> resample (counted), weights reset to
    # uniform; then (1,2,3,4); final forced.
    post._log_w = torch.log(torch.tensor([1.0, 1e-30, 1e-30, 1e-30]))
    post._maybe_resample(base, base)
    assert torch.equal(post._log_w, torch.zeros(4))  # the library resampled
    post._log_w = torch.log(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    post._maybe_resample(base, base)
    post._maybe_resample(base, base, force=True)
    out = recorder.take(2)
    assert out[SMC_ESS] == pytest.approx([hand([1, 1, 1, 5]), hand([1, 2, 3, 4])])
    assert hand([1, 1, 1, 5]) == pytest.approx(4 / 7)
    assert out[SMC_ESS_MIN] == pytest.approx([4 / 7, 0.25])
    assert out[SMC_RESAMPLES] == [0.0, 1.0]
    assert post.calls == [False, False, True, False, False, True]


def test_pf_matched_ensembles_are_unchanged_and_gain_ess(tiny_case):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    out = driver.run_pf_matched(tiny_case, "pf_matched", problem, sequences, "test", 0, 6, {})
    pf = BootstrapParticleFilter(
        problem.obs_matrix, problem.obs_variance, "optimal", num_saved_samples=6
    )
    for index, seq in enumerate(sequences):
        gen = torch.Generator().manual_seed(
            driver.derive_seed(driver.SEED_LABEL, "pf-matched", "test", index, 0)
        )
        x0 = initial_ensemble(problem, seq, 6, "test", index, 0)
        run = pf.run(problem.model, x0.T.contiguous(), seq.observations, gen)
        assert torch.equal(out.ensembles[index], run.samples)
        assert out.smc[index][SMC_ESS] == (run.ess / 6).tolist()
        assert out.smc[index][SMC_ESS_MIN] == out.smc[index][SMC_ESS]
        assert out.smc[index][SMC_RESAMPLES] == [0.0] * problem.num_steps


@pytest.mark.parametrize(
    ("prior", "mode"),
    [("si_sde", "dps_jacobian_free"), ("si_sde", "inflated"), ("dm_sde", "dps_jacobian_free")],
)
def test_surge_with_weights_ignored_moves_members_as_the_plain_sampler(
    tiny_case, prior, mode, monkeypatch
):
    """Mirror of the multistep test: SURGE's proposal is the wrapped sampler."""
    monkeypatch.setattr(
        SurgePosteriorBase,
        "_maybe_resample",
        lambda self, base, field_history, force=False: (base, field_history),
    )
    problem = driver.problem_of(tiny_case)
    seq = load_sequences(problem, "test", 1)[0]
    stats = generative.Standardisation(2.2, 4.0)
    run = tiny_case.checkpoints.si_run if prior == "si_sde" else tiny_case.checkpoints.fm_run
    model = generative.load_prior(run)
    settings = smc_mod.SurgeSettings(num_steps=10, likelihood_mode=mode)
    sampler = generative.SamplerSettings(num_steps=10, likelihood_mode=mode)
    method = Method.OURS_SI_SDE if prior == "si_sde" else Method.OURS_DM_SDE
    plain, plain_stepper = generative.build_posterior(method, model, problem, stats, sampler)
    surge, surge_stepper = smc_mod.build_surge_ours(prior, model, problem, stats, settings)
    x0 = initial_ensemble(problem, seq, 6, "test", 0, 0)[None]
    to_model = generative.observation_model(problem, stats).to_model
    args = (stats, x0, seq.observations[None], 10, 7)
    a = generative.assimilate(plain, plain_stepper, *args, obs_to_model=to_model)
    b = generative.assimilate(surge, surge_stepper, *args, obs_to_model=to_model)
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_unknown_settings_and_ns_overrides_are_refused(tiny_case, monkeypatch):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 1)
    bad = {**driver.method_settings(tiny_case, "surge_si_sde"), "diffusion_scale": 1.0}
    with pytest.raises(ValueError, match="unknown"):
        driver.run_method(tiny_case, "surge_si_sde", problem, sequences, "test", 0, bad)
    bad = {**driver.method_settings(tiny_case, "surge_unguided_dm"), "likelihood_mode": "x"}
    with pytest.raises(ValueError, match="unknown"):
        driver.run_method(tiny_case, "surge_unguided_dm", problem, sequences, "test", 0, bad)
    monkeypatch.setenv("FLOWDAS_ZETA", "0.1")
    with pytest.raises(RuntimeError, match="FLOWDAS_ZETA"):
        driver.run_method(tiny_case, "surge_sda", problem, sequences, "test", 0)


def test_surge_baselines_use_the_configured_scalars(tiny_case, monkeypatch):
    import cases.lorenz96._l96_baselines as baselines

    seen = {}
    real = baselines.ns_posterior

    def spy(cfg, name, *args):
        out = real(cfg, name, *args)
        seen[name] = out[1]
        return out

    monkeypatch.setattr(baselines, "ns_posterior", spy)
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 1)
    flowdas = {**driver.method_settings(tiny_case, "surge_flowdas"),
               "guidance_scale": 0.37, "ess_threshold": 0.3, "max_grad_norm": 5.0}
    driver.run_method(tiny_case, "surge_flowdas", problem, sequences, "test", 0, flowdas)
    sda = {**driver.method_settings(tiny_case, "surge_sda"), "gamma_sda": 0.02,
           "ess_threshold": 0.7}
    driver.run_method(tiny_case, "surge_sda", problem, sequences, "test", 0, sda)
    lik = seen["surge_flowdas"].likelihood_model
    assert seen["surge_flowdas"].ess_threshold == 0.3
    assert lik.guidance_scale == pytest.approx(0.37) and lik.num_mc_samples == 4
    assert lik.max_grad_norm == 5.0
    assert seen["surge_sda"].ess_threshold == 0.7
    assert seen["surge_sda"].likelihood_model.gamma_sda == pytest.approx(0.02)
    # Operator and variance: the case's observation model.
    observed = generative.observation_model(problem, generative.Standardisation(2.2, 4.0))
    for posterior in seen.values():
        assert posterior.variance == pytest.approx(observed.variance)
        assert torch.equal(posterior.obs_operator.obs_matrix, observed.operator.obs_matrix)
