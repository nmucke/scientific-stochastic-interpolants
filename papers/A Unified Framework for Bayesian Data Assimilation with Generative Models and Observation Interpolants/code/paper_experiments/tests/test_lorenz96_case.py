"""Plumbing of the Lorenz 96 case: shared inputs, split separation and scoring."""

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
from cases.lorenz96 import driver  # noqa: E402
from cases.lorenz96._l96_generative import (  # noqa: E402
    Standardisation,
    assimilate,
    build_obs_operator,
)
from cases.lorenz96._l96_system import Problem, load_problem  # noqa: E402
from cases.lorenz96.driver import (  # noqa: E402
    initial_ensemble,
    load_sequences,
    score,
    score_ensemble,
)
from common.paths import REPO_ROOT  # noqa: E402
from results_schema import Metric, ResultsWriter  # noqa: E402


@pytest.fixture
def problem() -> Problem:
    return Problem(num_steps=4)


def test_case_config_matches_problem_defaults():
    assert load_problem() == Problem()


def test_sequences_are_reproducible_and_splits_differ(problem):
    a = load_sequences(problem, "test", 2)
    b = load_sequences(problem, "test", 2)
    other = load_sequences(problem, "validation", 2)
    assert torch.equal(a[0].observations, b[0].observations)
    assert not torch.equal(a[0].truth, a[1].truth)
    assert not torch.equal(a[0].truth, other[0].truth)


def test_initial_ensemble_is_shared_and_varies_with_sequence_and_seed(problem):
    first, second = load_sequences(problem, "test", 2)
    a = initial_ensemble(problem, first, 8, "test", 0, 1)
    assert torch.equal(a, initial_ensemble(problem, first, 8, "test", 0, 1))
    assert not torch.equal(a, initial_ensemble(problem, first, 8, "test", 0, 2))
    assert not torch.equal(a, initial_ensemble(problem, second, 8, "test", 1, 1))


def test_obs_operator_matches_problem_matrix(problem):
    operator = build_obs_operator(problem)
    x = torch.randn(3, 1, 1, problem.dim_state)
    expected = x.reshape(3, -1) @ problem.obs_matrix.float().T
    torch.testing.assert_close(operator(x), expected)
    y = torch.randn(3, operator.num_obs)
    torch.testing.assert_close(
        operator.transpose(y).reshape(3, -1), y @ problem.obs_matrix.float()
    )


def reference_from(samples):
    return dict(mean=samples.mean(1), variance=samples.var(1), samples=samples)


def test_score_of_the_reference_itself_vanishes():
    gen = torch.Generator().manual_seed(0)
    samples = torch.randn(3, 512, 4, generator=gen, dtype=torch.float64)
    out = score_ensemble(samples, reference_from(samples), torch.zeros(3, 4))
    assert max(out[Metric.MEAN_ERROR_VS_PF.value]) < 1e-12
    assert max(out[Metric.VAR_ERROR_VS_PF.value]) < 1e-12
    assert max(out[Metric.SLICED_W2.value]) < 1e-9


def test_score_units():
    """A one-std shift scores 1; doubling the variance scores log 2."""
    gen = torch.Generator().manual_seed(1)
    samples = torch.randn(2, 256, 3, generator=gen, dtype=torch.float64)
    ref = reference_from(samples)
    shifted = score(
        ref["mean"] + ref["variance"].sqrt(),
        2 * ref["variance"],
        samples,
        ref,
        ref["mean"],
    )
    assert shifted[Metric.MEAN_ERROR_VS_PF.value] == pytest.approx([1.0, 1.0])
    assert shifted[Metric.VAR_ERROR_VS_PF.value] == pytest.approx(
        [torch.log(torch.tensor(2.0)).item()] * 2
    )


def test_diverged_member_is_dropped_at_its_step_and_counted():
    gen = torch.Generator().manual_seed(2)
    samples = torch.randn(3, 32, 3, generator=gen, dtype=torch.float64)
    bad = samples.clone()
    bad[1, 0, 0] = float("nan")
    bad[2, 1, 2] = 1e6
    out = score_ensemble(bad, reference_from(samples), torch.zeros(3, 3))
    assert out[Metric.DIVERGED_FRACTION.value] == pytest.approx([0.0, 1 / 32, 1 / 32])
    for metric in (Metric.SLICED_W2, Metric.CRPS, Metric.RMSE, Metric.SPREAD):
        assert all(math.isfinite(v) for v in out[metric.value])


def test_fully_diverged_step_is_nan_only_at_that_step():
    samples = torch.randn(2, 16, 3, dtype=torch.float64)
    bad = samples.clone()
    bad[1] = float("nan")
    out = score_ensemble(bad, reference_from(samples), torch.zeros(2, 3))
    assert out[Metric.DIVERGED_FRACTION.value] == [0.0, 1.0]
    assert math.isfinite(out[Metric.SLICED_W2.value][0])
    assert math.isnan(out[Metric.SLICED_W2.value][1])


def test_finite_mean_ignores_diverged_entries():
    assert driver.finite_mean([1.0, float("nan"), 3.0]) == 2.0
    assert math.isnan(driver.finite_mean([float("nan")]))


class RecordingPosterior:
    """Stands in for a sampler and encodes where each value came from."""

    def __init__(self, mode="inflated"):
        self.likelihood_model = SimpleNamespace(likelihood_mode=mode)

    def sample_trajectory(self, base, field_history, observations, **kwargs):
        self.observations = observations
        self.kwargs = kwargs
        steps = kwargs["num_physical_steps"]
        batch, _, _, d = base.shape
        out = torch.zeros(batch, 1, 1, d, steps)
        member = torch.arange(batch, dtype=torch.float32).reshape(-1, 1, 1, 1)
        site = torch.arange(d, dtype=torch.float32).reshape(1, 1, 1, -1)
        for t in range(steps):
            out[..., t] = 1000 * t + 10 * member + 0.01 * site
        return out


def test_assimilate_layout_keeps_sequence_member_step_and_site_apart():
    stats = Standardisation(0.0, 1.0)
    S, E, T, d, d_y = 2, 3, 4, 5, 2
    initial = torch.zeros(S, E, d)
    observations = torch.arange(S * T * d_y, dtype=torch.float32).reshape(S, T, d_y)
    posterior = RecordingPosterior()
    out = assimilate(posterior, None, stats, initial, observations, 7, 0)
    assert out.shape == (S, T, E, d)
    for s, t, e, i in [(0, 0, 0, 0), (1, 3, 2, 4), (1, 0, 1, 2), (0, 2, 2, 1)]:
        expected = 1000 * (t + 1) + 10 * (s * E + e) + 0.01 * i
        assert out[s, t, e, i].item() == pytest.approx(expected, abs=1e-2)
    # Member (s, e) sees the observation row of sequence s at every step.
    for s in range(S):
        for e in range(E):
            torch.testing.assert_close(
                posterior.observations[s * E + e], observations[s].T
            )
    assert posterior.kwargs["batch_size"] == S * E


def test_assimilate_refuses_a_mode_that_mixes_sequences():
    posterior = RecordingPosterior("inflated_shared")
    with pytest.raises(ValueError, match="mixing"):
        assimilate(
            posterior,
            None,
            Standardisation(0.0, 1.0),
            torch.zeros(2, 3, 5),
            torch.zeros(2, 4, 2),
            7,
            0,
        )


@pytest.fixture
def tiny_case(tmp_path, monkeypatch):
    """A complete small case on disk, with untrained priors."""
    cfg = driver.case_config()
    cfg.problem.num_steps = 3
    cfg.evaluation.test_sequences = 2
    cfg.evaluation.ensemble_size = 6
    cfg.evaluation.device = "cpu"
    cfg.evaluation.enkf_large_size = 12
    cfg.evaluation.reference_particles = 256
    for name in driver.GENERATIVE:
        cfg.methods[name].num_steps = 3
        cfg.methods[name].likelihood_mode = "dps_jacobian_free"
    for name in driver.PRIOR_ENKF:
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
        # Fixed weights, so that two runs of a cell use the same network.
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


@pytest.mark.parametrize("name", sorted(driver.CELLS))
def test_every_cell_runs_and_writes_tidy_records(tiny_case, name):
    records = driver.evaluate_cell(name, 0, tiny_case)
    metrics = {record.metric for record in records}
    assert {Metric.MEAN_ERROR_VS_PF.value, Metric.RMSE.value} <= metrics
    assert all(record.seed == 0 for record in records)
    if name in driver.GENERATIVE:
        nfe = next(r.value for r in records if r.metric == Metric.NFE.value)
        assert nfe >= tiny_case.methods[name].num_steps
    if name != "pf_reference":
        assert {driver.INNOVATION_LAG1, driver.INNOVATION_MSQ} <= metrics
    assert driver.cell_path("metrics", name, 0).name.endswith("__traj0__.csv")
    assert driver.cell_path("per_step", name, 0).exists()


def test_reference_for_another_system_is_refused(tiny_case):
    with pytest.raises(ValueError, match="Stale"):
        driver.load_reference("test", 0, 0, Problem(num_steps=4))


def test_runner_uses_case_seeds_and_rejects_foreign_scenario_and_method():
    cfg = OmegaConf.create(
        dict(
            case=OmegaConf.to_container(driver.case_config()),
            scenario=dict(name="lorenz96"),
            method=dict(name="Ours (SI-SDE)"),
        )
    )
    runner = driver.Lorenz96Runner(cfg)
    assert runner.seeds == tuple(cfg.case.evaluation.seeds)
    cfg.method.name = "SDA"
    with pytest.raises(ValueError, match="not part"):
        runner.methods()
    cfg.scenario.name = "sparse_5"
    with pytest.raises(ValueError, match="scenario=lorenz96"):
        driver.Lorenz96Runner(cfg)


def test_cells_flow_through_aggregation_and_figures(tiny_case, tmp_path, monkeypatch):
    import aggregate_lorenz96
    import make_lorenz96_figures

    root = tmp_path / "root"
    root.mkdir()
    (root / "lorenz96").symlink_to(tmp_path)
    monkeypatch.setattr(aggregate_lorenz96, "RESULTS", root)
    monkeypatch.setattr(make_lorenz96_figures, "RESULTS", root)
    for seed in (0, 1):
        path = driver.cell_path("metrics", "enkf", seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with ResultsWriter(path) as writer:
            writer.extend(driver.evaluate_cell("enkf", seed, tiny_case))
    # A diverged sequence must not stop the reduction.
    sequence_file = driver.cell_path("per_sequence", "enkf", 0).with_suffix(".json")
    entries = json.loads(sequence_file.read_text())
    entries[Metric.RMSE.value][0] = float("nan")
    sequence_file.write_text(json.dumps(entries))

    monkeypatch.setattr(sys, "argv", ["aggregate_lorenz96.py"])
    aggregate_lorenz96.main()
    aggregated = tmp_path / "aggregated"
    with open(aggregated / "all.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rmse = next(r for r in rows if r["metric"] == Metric.RMSE.value)
    assert rmse["method"] == "EnKF" and int(rmse["n_traj"]) == 2
    assert (aggregated / "per_step.csv").exists()
    assert (aggregated / "sequence_spread.csv").exists()

    figures = tmp_path / "figures"
    figures.mkdir()
    make_lorenz96_figures.apply_style()
    make_lorenz96_figures.plot_metric("rmse", "RMSE", figures)
    make_lorenz96_figures.plot_ess(figures)
    assert (figures / "lorenz96_rmse_vs_step.pdf").exists()
    assert (figures / "lorenz96_ess.pdf").exists()


# --------------------------------------------------------------------------- #
# Forecast-residual correction and the trained-prior EnKF baseline
# --------------------------------------------------------------------------- #


def test_corrected_cells_start_from_their_plain_settings():
    cfg = driver.case_config()
    for fc, plain in driver.CORRECTED.items():
        settings = driver.method_settings(cfg, fc)
        own = driver.method_settings(cfg, plain)
        assert {k: v for k, v in settings.items() if k != "correction"} == own
        assert "correction" in settings
        assert "correction" not in own


def test_overrides_reach_nested_keys_and_keep_strings():
    base = {"num_steps": 25, "correction": {"gain_scale": 1.0}}
    out = driver.apply_overrides(
        base,
        [
            "correction.forecast_num_steps=5",
            "correction.localisation=null",
            "likelihood_mode=dps_jacobian_free",
        ],
    )
    assert out["correction"] == {
        "gain_scale": 1.0,
        "forecast_num_steps": 5,
        "localisation": None,
    }
    assert out["likelihood_mode"] == "dps_jacobian_free"
    assert base["correction"] == {"gain_scale": 1.0}  # not mutated


def test_exact_residual_bank_has_the_transition_covariance():
    bank = generative.exact_residual_bank(16, 0.3).reshape(16, -1).double()
    torch.testing.assert_close(bank.T @ bank / 16, 0.3 * torch.eye(16, dtype=torch.float64))


def test_lag1_autocorrelation_limits():
    persistent = torch.ones(10, 4)
    alternating = torch.tensor([(-1.0) ** t for t in range(10)]).reshape(-1, 1).repeat(1, 4)
    assert driver.lag1_autocorrelation(persistent) == pytest.approx(1.0)
    assert driver.lag1_autocorrelation(alternating) == pytest.approx(-1.0)
    white = torch.randn(4000, 4, generator=torch.Generator().manual_seed(0))
    assert abs(driver.lag1_autocorrelation(white)) < 0.05


def test_lag1_autocorrelation_pairs_only_adjacent_finite_steps():
    # +1, -1 alternate, so every adjacent pair gives -1. Dropping the NaN row first
    # would pair the steps either side of it (+1, +1 here) and give > -1.
    z = torch.tensor([1.0, -1.0, 1.0, float("nan"), 1.0, -1.0, 1.0]).reshape(-1, 1)
    assert driver.lag1_autocorrelation(z) == pytest.approx(-1.0)
    # One non-finite site makes the whole step unusable.
    z2 = z.repeat(1, 3)
    z2[3] = torch.tensor([1.0, float("inf"), 1.0])
    assert driver.lag1_autocorrelation(z2) == pytest.approx(-1.0)
    # Not mean-removed: a constant bias counts as correlation.
    biased = 3.0 + torch.randn(4000, 2, generator=torch.Generator().manual_seed(1)) * 0.1
    assert driver.lag1_autocorrelation(biased) > 0.99
    assert math.isnan(driver.lag1_autocorrelation(torch.tensor([[1.0], [float("nan")], [1.0]])))


def test_normalised_innovations_refuse_several_noise_stages():
    problem = Problem(num_steps=4, noise_stages=2)
    members = torch.randn(8, problem.dim_state, dtype=torch.float64)
    y = torch.zeros(problem.obs_indices.numel(), dtype=torch.float64)
    with pytest.raises(ValueError, match="noise_stages=1"):
        driver.normalised_innovations(problem, members, y)


def test_tagged_cells_are_kept_apart_from_the_default_cell(tiny_case, tmp_path):
    with pytest.raises(ValueError, match="needs --tag"):
        driver.evaluate_cell("enkf", 0, tiny_case, overrides=["inflation=1.05"])
    with pytest.raises(ValueError, match="takes no settings"):
        driver.evaluate_cell("oracle", 0, tiny_case, overrides=["x=1"], tag="x")
    for bad in ("a__b", "traj3", "a b", "_a"):
        with pytest.raises(ValueError, match="Bad tag"):
            driver.check_tag("enkf", bad, None)
    records = driver.evaluate_cell(
        "enkf", 0, tiny_case, overrides=["inflation=1.05"], tag="infl1.05"
    )
    assert {r.variant for r in records} == {"matched_infl1.05"}
    assert driver.tagged_variant("forecast_correction", "fs5") == "forecast_correction_fs5"
    assert driver.tagged_variant(None, "fs5") == "fs5"
    assert driver.tagged_variant("matched", "") == "matched"
    tagged = driver.cell_path("per_step", "enkf", 0, "infl1.05")
    assert tagged.name == "lorenz96__enkf-infl1.05__traj0__.csv" and tagged.exists()
    assert driver.cell_path("per_sequence", "enkf", 0, "infl1.05").with_suffix(".json").exists()
    assert (tmp_path / "ensembles" / "enkf-infl1.05__seed0.pt").exists()
    # The default cell's outputs are untouched.
    assert not driver.cell_path("per_step", "enkf", 0).exists()
    assert not (tmp_path / "ensembles" / "enkf__seed0.pt").exists()


def test_tuning_csv_header_is_merged_not_misaligned(tmp_path):
    path = tmp_path / "tune.csv"
    assert driver.merge_tuning_header(path, ["a", "b"]) == ["a", "b"]
    path.write_text("a,b\n1,2\n")
    merged = driver.merge_tuning_header(path, ["a", "b", "c"])
    assert merged == ["a", "b", "c"]
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"a": "1", "b": "2", "c": ""}]
    # A narrower rerun keeps the wider stored header.
    assert driver.merge_tuning_header(path, ["a", "b"]) == ["a", "b", "c"]


def test_normalised_innovations_use_the_true_forecast_mean(problem):
    gen = torch.Generator().manual_seed(3)
    members = torch.randn(32, problem.dim_state, generator=gen, dtype=torch.float64)
    from scisi.dynamics.lorenz96 import integrate

    flow = integrate(members, problem.model.steps_per_stage, problem.dt, problem.forcing)
    y = (flow @ problem.obs_matrix.T).mean(0)
    z = driver.normalised_innovations(problem, members, y)
    torch.testing.assert_close(z, torch.zeros_like(z))


@pytest.mark.parametrize("name", sorted(driver.CORRECTED))
def test_disabled_correction_is_bit_identical_to_the_plain_cell(tiny_case, name):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    plain = driver.CORRECTED[name]
    settings = driver.method_settings(tiny_case, plain)
    a = driver.run_method(tiny_case, plain, problem, sequences, "test", 0, settings)
    off = dict(settings, correction={"enabled": False, "localisation": 4})
    b = driver.run_method(tiny_case, name, problem, sequences, "test", 0, off)
    assert torch.equal(a.ensembles, b.ensembles)
    assert a.nfe == b.nfe


@pytest.mark.parametrize("name", ["si_sde_fc", "fm_ode_fc"])
def test_enabled_correction_costs_two_forecasts_of_its_own_step_count(tiny_case, name):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    plain = driver.CORRECTED[name]
    settings = driver.method_settings(tiny_case, plain)
    a = driver.run_method(tiny_case, plain, problem, sequences, "test", 0, settings)
    on = dict(settings, correction={"forecast_num_steps": 2, "localisation": 4})
    b = driver.run_method(tiny_case, name, problem, sequences, "test", 0, on)
    assert not torch.equal(a.ensembles, b.ensembles)
    assert torch.isfinite(b.ensembles).all()
    # Online Q_hat: two unguided forecasts per member and step.
    assert b.nfe - a.nfe == pytest.approx(2 * 2)
    # The guided draw of the first step is the plain cell's own (same seed stream);
    # sequences are corrected independently, so changing one sequence's
    # observations leaves the other's members unchanged.
    mod = list(sequences)
    mod[1] = driver.Sequence(
        mod[1].prior_mean, mod[1].truth, mod[1].observations + 5.0
    )
    c = driver.run_method(tiny_case, name, problem, mod, "test", 0, on)
    assert torch.equal(b.ensembles[0, 0], c.ensembles[0, 0])
    assert not torch.equal(b.ensembles[1, 0], c.ensembles[1, 0])


def test_oracle_fc_without_gain_is_the_exact_kernel_spread(tiny_case):
    """gain_scale=0 leaves the exact one-step draws: their mean matches the oracle's."""
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 1)
    big = 4000
    zero = {"correction": {"gain_scale": 0.0, "localisation": None}}
    a = driver.run_oracle_fc(problem, sequences, "test", 0, big, zero).ensembles
    b = driver.run_oracle(problem, sequences, "test", 0, big).ensembles
    # Same initial members, independent noise: first-step moments agree.
    torch.testing.assert_close(a[0, 0].mean(0), b[0, 0].mean(0), atol=0.15, rtol=0)
    torch.testing.assert_close(a[0, 0].var(0), b[0, 0].var(0), atol=0.15, rtol=0.1)


def test_prior_enkf_uses_the_enkf_update_on_prior_forecasts(tiny_case, monkeypatch):
    """With an identity 'prior' the cell is the EnKF on a persistence forecast."""
    from cases.lorenz96._l96_generative import assimilate_prior_enkf
    from scisi.particle_filter.enkf import EnsembleKalmanFilter

    problem = driver.problem_of(tiny_case)
    import scisi.posterior_models.forecast_correction as fc

    monkeypatch.setattr(
        fc, "draw_unguided_forecast", lambda posterior, history, **kw: history[..., -1]
    )
    stats = Standardisation(2.0, 3.0)
    S, E, T = 2, 8, 3
    initial = torch.randn(S, E, problem.dim_state, dtype=torch.float64)
    obs = torch.randn(S, T, problem.obs_indices.numel(), dtype=torch.float64)
    enkf = EnsembleKalmanFilter(problem.obs_matrix, problem.obs_variance)
    posterior = SimpleNamespace(device="cpu")
    out = assimilate_prior_enkf(
        posterior, None, stats, initial, obs, 1, 0, enkf,
        [torch.Generator().manual_seed(s) for s in range(S)],
    )
    for s in range(S):
        gen = torch.Generator().manual_seed(s)
        x = stats.forward(initial[s].float()).double() * 3.0 + 2.0  # float32 round trip
        for t in range(T):
            x = enkf.analyse(x, obs[s, t], gen)
            torch.testing.assert_close(out[s, t], x, atol=1e-5, rtol=1e-5)
            x = stats.inverse(stats.forward(x).float().double())
