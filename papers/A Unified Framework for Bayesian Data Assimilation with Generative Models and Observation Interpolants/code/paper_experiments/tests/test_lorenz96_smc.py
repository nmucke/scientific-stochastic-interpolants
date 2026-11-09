"""SURGE / SMC cells of the Lorenz 96 case and their ESS records (``_l96_smc.py``)."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from common.paths import REPO_ROOT  # noqa: E402
from results_schema import Method, Metric  # noqa: E402
from scisi.particle_filter.bootstrap import BootstrapParticleFilter  # noqa: E402
from scisi.posterior_models.surge_posterior import SurgePosteriorBase  # noqa: E402

SMC_CELLS = sorted(SURGE_CELLS) + ["pf_matched", "pf_reference"]
CONFIG_DIR = driver.CASE_CONFIG.parents[1] / "method"


@pytest.fixture
def tiny_case(tmp_path, monkeypatch):
    """Two short test sequences, E=6, untrained networks, M=3 (as test_lorenz96_case)."""
    cfg = driver.case_config()
    cfg.problem.num_steps = 2
    cfg.evaluation.test_sequences = 2
    cfg.evaluation.ensemble_size = 6
    cfg.evaluation.device = "cpu"
    cfg.evaluation.reference_particles = 256
    for name in SURGE_CELLS:
        cfg.methods[name].num_steps = 3
    cfg.methods.surge_flowdas.num_mc_samples = 4
    for name in driver.GENERATIVE:
        cfg.methods[name].num_steps = 3
    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    problem = driver.problem_of(cfg)
    for index, seq in enumerate(load_sequences(problem, "test", 2)):
        for replicate in range(2):
            driver._reference_task((problem, seq, "test", index, replicate, 256))

    config_dir = REPO_ROOT / "config"

    def untrained(run, device="cpu"):
        name = "stochastic_interpolant" if run == cfg.checkpoints.si_run else "flow_matching"
        model_cfg = OmegaConf.load(config_dir / f"lorenz96_{name}.yaml").model
        with torch.random.fork_rng():
            torch.manual_seed(0)
            return hydra.utils.instantiate(model_cfg).eval()

    monkeypatch.setattr(generative, "load_prior", untrained)
    monkeypatch.setattr(
        generative.Standardisation,
        "from_data",
        classmethod(lambda cls, path, problem: cls(2.2, 4.0)),
    )
    return cfg


def test_cells_and_labels():
    assert SURGE_CELLS == {
        "surge_si_sde": (Method.OURS_SI_SDE, "surge"),
        "surge_dm_sde": (Method.OURS_DM_SDE, "surge"),
        "surge_unguided_si": (Method.OURS_SI_SDE, "surge_unguided"),
        "surge_unguided_dm": (Method.OURS_DM_SDE, "surge_unguided"),
        "surge_flowdas": (Method.SURGE_FLOWDAS, None),
        "surge_sda": (Method.SURGE_SDA, None),
    }
    for name, label in SURGE_CELLS.items():
        assert driver.CELLS[name] == label
        assert name not in driver.GENERATIVE
        assert name in driver.TUNABLE
    assert len(set(driver.CELLS.values())) == len(driver.CELLS)


def test_config_blocks_are_complete_at_fixed_m():
    cfg = driver.case_config()
    for name in SURGE_CELLS:
        settings = driver.method_settings(cfg, name)
        assert settings["num_steps"] == driver.FIXED_NUM_STEPS
        assert "ess_threshold" in settings
    for name in smc_mod.SURGE_BASELINES:
        method_cfg = smc_mod.baseline_method_config(
            name, driver.method_settings(cfg, name), CONFIG_DIR
        )
        for block in ("likelihood_model", "posterior_model"):
            for key, value in method_cfg[block].items():
                assert key == "_target_" or not isinstance(value, (dict, list)) and not (
                    OmegaConf.is_config(value)
                ), (name, key)


@pytest.mark.parametrize("name", SMC_CELLS)
def test_smc_cell_runs_and_writes_ess_records(tiny_case, name):
    records = driver.evaluate_cell(name, 0, tiny_case, allow_other_m=True)
    metrics = {record.metric for record in records}
    assert {
        Metric.MEAN_ERROR_VS_PF.value,
        Metric.SLICED_W2.value,
        Metric.RMSE.value,
        Metric.CRPS.value,
        driver.DIVERGED_STEPS,
    } <= metrics
    assert set(SMC_KEYS) <= metrics
    if name != "pf_reference":
        assert {driver.INNOVATION_LAG1, driver.INNOVATION_MSQ} <= metrics
    if name in SURGE_CELLS:
        nfe = next(r.value for r in records if r.metric == Metric.NFE.value)
        # At least the prior at x_k and the reward at x_{k+1} per pseudo-step.
        assert nfe >= 2 * (tiny_case.methods[name].num_steps - 1)
    for record in records:
        if record.metric in SMC_KEYS:
            assert math.isfinite(record.value)
            if record.metric != SMC_RESAMPLES:
                assert 0.0 < record.value <= 1.0 + 1e-9
    per_sequence = json.loads(
        driver.cell_file("per_sequence", name, 0, tiny_case).with_suffix(".json").read_text()
    )
    for key in SMC_KEYS:
        assert len(per_sequence[key]) == 2
    with open(driver.cell_file("per_step", name, 0, tiny_case), encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["metric"] in SMC_KEYS]
    assert {(r["metric"], r["step"]) for r in rows} == {
        (key, str(step)) for key in SMC_KEYS for step in range(2)
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
def test_surge_cells_are_deterministic(tiny_case, name):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    a = driver.run_method(tiny_case, name, problem, sequences, "test", 0)
    b = driver.run_method(tiny_case, name, problem, sequences, "test", 0)
    assert torch.equal(a.ensembles, b.ensembles)
    assert a.smc == b.smc and a.nfe == b.nfe
    assert a.ensembles.shape == (2, problem.num_steps, 6, problem.dim_state)
    c = driver.run_method(tiny_case, name, problem, sequences, "test", 1)
    assert not torch.equal(a.ensembles, c.ensembles)


def test_one_sequence_per_surge_call_is_enforced(tiny_case):
    with pytest.raises(ValueError, match="one sequence"):
        smc_mod.assimilate_one(
            None, None, None, torch.zeros(2, 6, 16), torch.zeros(2, 3, 4), 3, 0, None
        )
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
    out = driver.run_pf_matched(problem, sequences, "test", 0, 6)
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
    settings = smc_mod.SurgeOursSettings(num_steps=10, likelihood_mode=mode)
    sampler = generative.SamplerSettings(num_steps=10, likelihood_mode=mode)
    method = Method.OURS_SI_SDE if prior == "si_sde" else Method.OURS_DM_SDE
    plain, plain_stepper = generative.build_posterior(method, model, problem, stats, sampler)
    surge, surge_stepper = smc_mod.build_surge_ours(prior, model, problem, stats, settings)
    x0 = initial_ensemble(problem, seq, 6, "test", 0, 0)[None]
    to_model = generative.observation_model(problem, stats).to_model
    args = (stats, x0, seq.observations[None], 10, 7)
    a = generative.assimilate(plain, plain_stepper, *args, obs_to_model=to_model)
    b = smc_mod.assimilate_one(surge, surge_stepper, *args, to_model)
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
    flowdas = driver.method_settings(tiny_case, "surge_flowdas")
    with pytest.raises(ValueError, match="missing"):
        smc_mod.baseline_method_config(
            "surge_flowdas", {k: v for k, v in flowdas.items() if k != "max_grad_norm"},
            CONFIG_DIR,
        )
    with pytest.raises(ValueError, match="unknown"):
        smc_mod.baseline_method_config("surge_sda", {**flowdas}, CONFIG_DIR)
    with pytest.raises(ValueError, match="scalar"):
        smc_mod.baseline_method_config(
            "surge_flowdas", {**flowdas, "guidance_scale": {"default": 1.0}}, CONFIG_DIR
        )
    monkeypatch.setenv("FLOWDAS_ZETA", "0.1")
    with pytest.raises(RuntimeError, match="FLOWDAS_ZETA"):
        driver.run_method(tiny_case, "surge_sda", problem, sequences, "test", 0)


def test_surge_baselines_use_the_configured_scalars(tiny_case, monkeypatch):
    seen = {}
    real = smc_mod.build_surge_baseline

    def spy(name, settings, *args):
        model, posterior, stepper = real(name, settings, *args)
        seen[name] = posterior
        return model, posterior, stepper

    monkeypatch.setattr(smc_mod, "build_surge_baseline", spy)
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
