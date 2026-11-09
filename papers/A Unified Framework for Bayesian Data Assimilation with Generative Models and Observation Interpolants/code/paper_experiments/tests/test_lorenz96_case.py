"""Plumbing of the Lorenz 96 case: shared inputs, split separation and scoring."""

from __future__ import annotations

import csv
import json
import math
import statistics
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
    monkeypatch.setattr(driver, "RESULTS", tmp_path)
    problem = driver.problem_of(cfg)
    for index, seq in enumerate(load_sequences(problem, "test", 2)):
        for replicate in range(2):
            driver._reference_task((problem, seq, "test", index, replicate, 256))

    config_dir = REPO_ROOT / "config"

    def untrained(run, device="cpu"):
        name = "stochastic_interpolant" if run == cfg.checkpoints.si_run else "flow_matching"
        model_cfg = OmegaConf.load(config_dir / f"lorenz96_{name}.yaml").model
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
    for name, seed in (("enkf", 0), ("enkf", 1), ("pf_reference", 0)):
        path = driver.cell_path("metrics", name, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with ResultsWriter(path) as writer:
            writer.extend(driver.evaluate_cell(name, seed, tiny_case))
    # A diverged sequence must not stop the reduction.
    sequence_file = driver.cell_path("per_sequence", "enkf", 0).with_suffix(".json")
    entries = json.loads(sequence_file.read_text())
    entries[Metric.RMSE.value][0] = float("nan")
    sequence_file.write_text(json.dumps(entries))
    other = json.loads(
        driver.cell_path("per_sequence", "enkf", 1).with_suffix(".json").read_text()
    )

    monkeypatch.setattr(sys, "argv", ["aggregate_lorenz96.py"])
    aggregate_lorenz96.main()
    aggregated = tmp_path / "aggregated"
    with open(aggregated / "all.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rmse = next(
        r for r in rows if r["metric"] == Metric.RMSE.value and r["method"] == "EnKF"
    )
    assert int(rmse["n_traj"]) == 2 and float(rmse["std"]) > 0
    # One seed: the std is not measured, so NaN rather than the library's 0.0.
    reference = [r for r in rows if r["variant"] == "reference"]
    assert reference and all(math.isnan(float(r["std"])) for r in reference)
    assert any(r["metric"] == driver.DIVERGED_STEPS for r in reference)
    with open(aggregated / "per_step.csv", encoding="utf-8") as handle:
        per_step = list(csv.DictReader(handle))
    assert all(
        math.isnan(float(r["std"])) for r in per_step if r["variant"] == "reference"
    )
    with open(aggregated / "sequence_spread.csv", encoding="utf-8") as handle:
        spread = {(r["cell"], r["metric"]): r for r in csv.DictReader(handle)}
    # Sequence 0 keeps only seed 1, whose value was not replaced by NaN.
    rmse_0 = other[Metric.RMSE.value][0]
    rmse_1 = (entries[Metric.RMSE.value][1] + other[Metric.RMSE.value][1]) / 2
    row = spread[("enkf", Metric.RMSE.value)]
    assert int(row["n"]) == 2
    assert float(row["std"]) == pytest.approx(abs(rmse_0 - rmse_1) / 2**0.5)
    crps = [(a + b) / 2 for a, b in zip(entries["crps"], other["crps"])]
    assert float(spread[("enkf", "crps")]["std"]) == pytest.approx(
        statistics.stdev(crps)
    )

    figures = tmp_path / "figures"
    figures.mkdir()
    make_lorenz96_figures.apply_style()
    make_lorenz96_figures.plot_metric("rmse", "RMSE", figures)
    make_lorenz96_figures.plot_ess(figures)
    assert (figures / "lorenz96_rmse_vs_step.pdf").exists()
    assert (figures / "lorenz96_ess.pdf").exists()
    # A label that matches no SERIES entry must not give an empty figure.
    monkeypatch.setattr(make_lorenz96_figures, "SERIES", (("EnKF (renamed)", "", "x", "k", "-"),))
    with pytest.raises(ValueError, match="No SERIES entry"):
        make_lorenz96_figures.plot_metric("rmse", "RMSE", figures)


def test_enkf_rejects_unknown_settings(problem):
    sequences = load_sequences(problem, "test", 1)
    with pytest.raises(TypeError, match="inflaton"):
        driver.run_enkf(problem, sequences, "test", 0, 8, {"inflaton": 1.1})


@pytest.mark.parametrize("name", sorted(driver.GENERATIVE))
def test_inflated_likelihood_runs_through_assimilate(tiny_case, name, monkeypatch):
    built = []
    real_build = generative.build_posterior

    def spy(method, model, problem, stats, settings):
        built.append(settings.likelihood_mode)
        return real_build(method, model, problem, stats, settings)

    monkeypatch.setattr(generative, "build_posterior", spy)
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    settings = {**driver.method_settings(tiny_case, name), "likelihood_mode": "inflated"}
    out = driver.run_method(tiny_case, name, problem, sequences, "test", 0, settings)
    assert built == ["inflated"]
    assert out.ensembles.shape == (2, problem.num_steps, 6, problem.dim_state)
    assert torch.isfinite(out.ensembles).all()
    assert out.nfe >= settings["num_steps"]


def test_only_the_dm_sde_drift_assertion_counts_as_diverged(tiny_case, monkeypatch):
    def failing(message):
        def assimilate(*args, **kwargs):
            raise AssertionError(message)

        return assimilate

    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    drift = "FM-SDE prior drift is non-finite; use an endpoint-vanishing g_tau"
    monkeypatch.setattr(generative, "assimilate", failing(drift))
    out = driver.run_method(tiny_case, "dm_sde", problem, sequences, "test", 0)
    assert torch.isnan(out.ensembles).all() and out.nfe is None
    records = driver.evaluate_cell("dm_sde", 0, tiny_case)
    diverged = next(r.value for r in records if r.metric == driver.DIVERGED_STEPS)
    assert diverged == 2 * problem.num_steps
    assert Metric.NFE.value not in {r.metric for r in records}
    with pytest.raises(AssertionError):  # the same message from another sampler
        driver.run_method(tiny_case, "si_sde", problem, sequences, "test", 0)
    monkeypatch.setattr(generative, "assimilate", failing("shapes differ"))
    with pytest.raises(AssertionError, match="shapes differ"):
        driver.run_method(tiny_case, "dm_sde", problem, sequences, "test", 0)


def test_count_diverged_steps_counts_non_finite_entries():
    scores = [
        {"a": [1.0, float("nan"), 2.0], "b": [0.0, 0.0, float("inf")]},
        {"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]},
    ]
    assert driver.count_diverged_steps(scores) == 2


@pytest.fixture
def tuning_case(tiny_case, monkeypatch):
    tiny_case.evaluation.validation_sequences = 2
    problem = driver.problem_of(tiny_case)
    for index, seq in enumerate(load_sequences(problem, "validation", 2)):
        driver._reference_task((problem, seq, "validation", index, 0, 256))
    monkeypatch.setattr(driver, "case_config", lambda: tiny_case)
    return tiny_case


def test_tune_reads_validation_only_and_a_diverged_setting_is_ineligible(
    tuning_case, tmp_path, monkeypatch
):
    splits = []
    real_sequences, real_reference = driver.load_sequences, driver.load_reference
    real_run = driver.run_method

    def sequences_spy(problem, split, count):
        splits.append(split)
        return real_sequences(problem, split, count)

    def reference_spy(split, *args):
        splits.append(split)
        return real_reference(split, *args)

    def run_spy(cfg, name, problem, sequences, split, seed, settings=None):
        splits.append(split)
        out = real_run(cfg, name, problem, sequences, split, seed, settings)
        if settings["inflation"] == 2.0:
            out.ensembles[:, 1:] = float("nan")  # every member lost after step 0
        return out

    monkeypatch.setattr(driver, "load_sequences", sequences_spy)
    monkeypatch.setattr(driver, "load_reference", reference_spy)
    monkeypatch.setattr(driver, "run_method", run_spy)
    args = SimpleNamespace(
        method="enkf", grid=json.dumps({"inflation": [1.0, 2.0]}), threads=1
    )
    driver.cmd_tune(args)
    assert splits and set(splits) == {"validation"}

    path = tmp_path / "tuning" / "enkf__inflation.csv"
    with open(path, encoding="utf-8") as handle:
        good, bad = list(csv.DictReader(handle))
    assert json.loads(good["base"]) == driver.method_settings(tuning_case, "enkf")
    assert int(good["diverged_steps"]) == 0
    assert math.isfinite(float(good[Metric.SLICED_W2.value]))
    assert int(bad["diverged_steps"]) == 2 * (tuning_case.problem.num_steps - 1)
    for metric in driver.TUNING_SCORES:
        assert math.isinf(float(bad[metric.value]))
    # Ranking by the selection metric can never pick the diverged setting.
    best = min((good, bad), key=lambda row: float(row[Metric.SLICED_W2.value]))
    assert best is good

    driver.cmd_tune(args)  # same columns: appends
    with open(path, encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    path.write_text("inflation,seconds\n1.0,0.1\n")  # an older layout
    with pytest.raises(ValueError, match="has columns"):
        driver.cmd_tune(args)


def test_reference_refuses_other_replicate_counts(tiny_case, monkeypatch):
    tiny_case.evaluation.reference_replicates = 3
    monkeypatch.setattr(driver, "case_config", lambda: tiny_case)
    with pytest.raises(ValueError, match="reference_replicates must be 2"):
        driver.cmd_reference(SimpleNamespace(split="test", workers=1))
