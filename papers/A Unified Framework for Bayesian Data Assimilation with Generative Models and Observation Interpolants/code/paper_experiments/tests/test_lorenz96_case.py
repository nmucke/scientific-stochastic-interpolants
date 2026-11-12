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
from lorenz96_lineup import ROW_BY_CELL, ROWS, SMC_METRICS  # noqa: E402
from results_schema import Method, Metric, ResultsWriter  # noqa: E402


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


TINY_STEPS = 3  # pseudo-steps of every block with a num_steps (baselines, SURGE too)


def make_tiny(obs: str, tmp_path, monkeypatch):
    """A complete small ``obs`` case on disk, with untrained priors: 2 test sequences
    of 3 steps, E = 6, every sampler at a few pseudo-steps."""
    cfg = driver.case_config(obs)
    cfg.problem.num_steps = 3
    cfg.evaluation.test_sequences = 2
    cfg.evaluation.ensemble_size = 6
    cfg.evaluation.device = "cpu"
    cfg.evaluation.enkf_large_size = 12
    cfg.evaluation.reference_particles = 256
    cfg.oracle_arctan.proposals = 64
    for name, block in cfg.methods.items():
        if "num_steps" in block:
            block.num_steps = TINY_STEPS
        if "likelihood_mode" in block:
            block.likelihood_mode = "dps_jacobian_free"
        if "num_mc_samples" in block:
            block.num_mc_samples = 4
    cfg.methods.dflow.ode_steps = 2
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


@pytest.fixture
def tiny_case(tmp_path, monkeypatch):
    return make_tiny("linear", tmp_path, monkeypatch)


BASE_METRICS = {
    Metric.MEAN_ERROR_VS_PF.value,
    Metric.VAR_ERROR_VS_PF.value,
    Metric.SLICED_W2.value,
    driver.KL_KNN,
    driver.ENERGY_DISTANCE,
    Metric.RMSE.value,
    Metric.SPREAD.value,
    Metric.CRPS.value,
    Metric.DIVERGED_FRACTION.value,
    driver.DIVERGED_STEPS,
}


def expected_metrics(name: str, obs: str) -> set[str]:
    row = ROW_BY_CELL[name]
    out = set(BASE_METRICS)
    if name == "pf_reference":  # scored from its stored moments: no members to drop
        out.discard(Metric.DIVERGED_FRACTION.value)
    else:
        out |= {driver.INNOVATION_LAG1, driver.INNOVATION_MSQ}
    if name != "pf_subsample":
        out.add(Metric.SECONDS.value)
    if row.generative:
        out.add(Metric.NFE.value)
    if row.smc:
        out |= set(SMC_METRICS)
    if obs == "arctan" and name in ("oracle", "oracle_fc"):
        out |= {driver.ORACLE_ESS_MEAN, driver.ORACLE_ESS_MIN}
    return out


@pytest.mark.parametrize("obs", ["linear", "arctan"])
@pytest.mark.parametrize("name", sorted(driver.CELLS))
def test_every_cell_runs_writes_its_metrics_and_is_deterministic(name, obs, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = make_tiny(obs, tmp_path, monkeypatch)
    records = driver.evaluate_cell(name, 0, cfg, allow_other_m=True, write_metrics=True)
    assert {r.metric for r in records} == expected_metrics(name, obs)
    method, variant = driver.CELLS[name]
    assert all(r.method == method.value and r.variant == variant for r in records)
    assert {r.scenario for r in records} == {driver.SCENARIOS[obs].value}
    assert all(r.seed == 0 and r.E for r in records)
    assert all(r.M == (TINY_STEPS if ROW_BY_CELL[name].generative else None) for r in records)
    assert all(math.isfinite(r.value) for r in records if r.metric == Metric.RMSE.value)
    values = {r.metric: r.value for r in records}
    if Metric.NFE.value in values:
        assert values[Metric.NFE.value] >= TINY_STEPS - 1
    for key in SMC_METRICS:
        if key in values:
            assert math.isfinite(values[key])
            if key != SMC_METRICS[-1]:
                assert 0.0 < values[key] <= 1.0 + 1e-9
    if driver.ORACLE_ESS_MIN in values:
        assert 1.0 <= values[driver.ORACLE_ESS_MIN] <= values[driver.ORACLE_ESS_MEAN] <= 64
    for kind in ("metrics", "per_step"):
        path = driver.cell_file(kind, name, 0, cfg)
        assert path.exists() and path.parent == tmp_path / obs / kind
    per_sequence = json.loads(driver.cell_file("per_sequence", name, 0, cfg).with_suffix(".json").read_text())
    assert all(len(v) == 2 for v in per_sequence.values())
    if ROW_BY_CELL[name].smc:
        with open(driver.cell_file("per_step", name, 0, cfg), encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r["metric"] in SMC_METRICS]
        assert {(r["metric"], r["step"]) for r in rows} == {
            (key, str(step)) for key in SMC_METRICS for step in range(3)
        }
    if name == "pf_reference":
        return
    # Deterministic: a second run gives the stored ensembles bit for bit.
    stored = torch.load(
        tmp_path / obs / "ensembles" / f"{driver.cell_label(name, '', records[0].M)}__seed0.pt"
    )
    problem = driver.problem_of(cfg)
    again = driver.run_method(cfg, name, problem, load_sequences(problem, "test", 2), "test", 0)
    assert torch.equal(again.ensembles.float(), stored)


def test_cells_and_lineup_agree():
    """driver.CELLS is lorenz96_lineup.ROWS; M-independent cells have no num_steps."""
    assert list(driver.CELLS) == [row.cell for row in ROWS]
    assert len(set(driver.CELLS.values())) == len(driver.CELLS)
    assert set(driver.RUNNERS) == set(driver.CELLS) - {"pf_reference"}
    for obs in ("linear", "arctan"):
        cfg = driver.case_config(obs)
        for row in ROWS:
            assert driver.CELLS[row.cell] == (Method(row.method), row.variant or None)
            settings = driver.method_settings(cfg, row.cell)
            assert ("num_steps" in settings) == row.generative, row.cell


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
    (root / "lorenz96").symlink_to(tmp_path)  # driver.RESULTS: holds the linear sweep root
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
    aggregated = tmp_path / "linear" / "aggregated"
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
        driver.run_enkf(None, "enkf", problem, sequences, "test", 0, 8, {"inflaton": 1.1})


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
    records = driver.evaluate_cell("dm_sde", 0, tiny_case, allow_other_m=True)
    diverged = next(r.value for r in records if r.metric == driver.DIVERGED_STEPS)
    assert diverged == 2 * problem.num_steps
    assert Metric.NFE.value not in {r.metric for r in records}
    with pytest.raises(AssertionError):  # the same message from another sampler
        driver.run_method(tiny_case, "si_sde", problem, sequences, "test", 0)
    monkeypatch.setattr(generative, "assimilate", failing("shapes differ"))
    with pytest.raises(AssertionError, match="shapes differ"):
        driver.run_method(tiny_case, "dm_sde", problem, sequences, "test", 0)


@pytest.mark.parametrize("name", ["si_sde", "dm_sde_fc"])
def test_inflated_shared_runs_one_sequence_per_call(name, tiny_case, monkeypatch):
    """inflated_shared cannot batch sequences (assimilate refuses it): the cell runs
    each sequence alone, with its own seed and a fresh posterior / correction; the
    default mode still makes one batched call with the cell seed."""
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    real = generative.assimilate
    calls = []

    def spy(posterior, stepper, stats, initial, observations, num_steps, seed, **kwargs):
        calls.append((initial.shape[0], seed, posterior, kwargs.get("forecast_correction")))
        return real(posterior, stepper, stats, initial, observations, num_steps, seed, **kwargs)

    monkeypatch.setattr(generative, "assimilate", spy)
    settings = driver.with_num_steps(name, driver.method_settings(tiny_case, name), 4)
    shared = driver.apply_overrides(settings, ["likelihood_mode=inflated_shared"])
    out = driver.run_method(tiny_case, name, problem, sequences, "test", 0, shared)
    assert out.ensembles.shape == (2, problem.num_steps, 6, problem.dim_state)
    assert torch.isfinite(out.ensembles).all() and out.nfe is not None
    assert [c[0] for c in calls] == [1, 1]
    assert calls[0][1] != calls[1][1]  # one seed per sequence
    assert calls[0][2] is not calls[1][2]  # a fresh posterior per sequence
    if name in driver.CORRECTED:
        assert calls[0][3] is not None and calls[0][3] is not calls[1][3]
    # nfe is per step for one batch of members, not summed over the sequences.
    assert settings["num_steps"] <= out.nfe < 2 * problem.num_steps * settings["num_steps"]
    calls.clear()
    plain = driver.apply_overrides(settings, ["likelihood_mode=inflated"])
    driver.run_method(tiny_case, name, problem, sequences, "test", 0, plain)
    base = driver.CORRECTED.get(name, name)
    assert [c[0] for c in calls] == [2]
    assert calls[0][1] == driver.derive_seed(driver.SEED_LABEL, base, "test", 0)


def test_count_diverged_steps_counts_non_finite_core_scores_only():
    nan = float("nan")
    scores = [
        {"a": [1.0, nan, 2.0], "b": [0.0, 0.0, float("inf")], driver.KL_KNN: [nan, 1.0, 1.0]},
        {"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0], driver.KL_KNN: [nan, nan, 1.0],
         driver.ENERGY_DISTANCE: [1.0, nan, 1.0]},
    ]
    # Steps 1 and 2 of sequence 0; a NaN of a diagnostic distance does not count.
    assert driver.count_diverged_steps(scores) == 2


@pytest.fixture
def tuning_case(tiny_case, monkeypatch):
    tiny_case.evaluation.validation_sequences = 2
    problem = driver.problem_of(tiny_case)
    for index, seq in enumerate(load_sequences(problem, "validation", 2)):
        driver._reference_task((problem, seq, "validation", index, 0, 256))
    monkeypatch.setattr(driver, "case_config", lambda obs="linear": tiny_case)
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

    path = tmp_path / "linear" / "tuning" / "enkf__inflation.csv"
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
    monkeypatch.setattr(driver, "case_config", lambda obs="linear": tiny_case)
    with pytest.raises(ValueError, match="reference_replicates must be 2"):
        driver.cmd_reference(SimpleNamespace(split="test", workers=1))


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
    assert (tmp_path / "linear" / "ensembles" / "enkf-infl1.05__seed0.pt").exists()
    # The default cell's outputs are untouched.
    assert not driver.cell_path("per_step", "enkf", 0).exists()
    assert not (tmp_path / "linear" / "ensembles" / "enkf__seed0.pt").exists()


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
    a = driver.run_oracle_fc(tiny_case, "oracle_fc", problem, sequences, "test", 0, big, zero).ensembles
    b = driver.run_oracle(tiny_case, "oracle", problem, sequences, "test", 0, big, {}).ensembles
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


def test_correction_block_maps_exact_to_the_fixed_true_q_and_refuses_unknown_keys(problem):
    cfg = generative.correction_config(problem, {"q_source": "exact", "localisation": 4}, 8, 0.3)
    assert cfg.q_source == "fixed" and cfg.q_variance == 0.3 and cfg.group_size == 8
    assert cfg.enabled and cfg.localisation is not None
    online = generative.correction_config(problem, {}, 8, 0.3)
    assert online.q_source == "online" and online.q_variance is None
    with pytest.raises(ValueError, match="Unknown correction settings"):
        generative.correction_config(problem, {"forecast_mode": "prior"}, 8, 0.3)
    with pytest.raises(ValueError, match="q_source"):
        generative.correction_config(problem, {"q_source": "bank"}, 8, 0.3)


@pytest.mark.parametrize("radius", [None, 4])
def test_exact_q_equals_a_bank_with_the_transition_covariance(problem, radius):
    """``q_source: exact`` (library ``fixed``) gives the K_q of a residual bank whose
    sample covariance is exactly ``q I`` (``sqrt(q d) e_j``), with or without the
    taper: H is a selection, so the taper is 1 at every observed site."""
    from scisi.posterior_models.forecast_correction import ForecastCorrection

    d, q, members = problem.dim_state, 0.3, 8
    operator = build_obs_operator(problem)
    m = problem.obs_indices.numel()
    gen = torch.Generator().manual_seed(0)
    x, f = (torch.randn(members, 1, 1, d, generator=gen) for _ in range(2))
    y = torch.randn(1, m, generator=gen)
    e = torch.randn(members, m, generator=gen)
    block = {"q_source": "exact", "localisation": radius, "gain_scale": 0.5}
    fixed = ForecastCorrection(generative.correction_config(problem, block, members, q))
    bank_cfg = generative.correction_config(problem, block, members, q)
    bank_cfg.q_source, bank_cfg.q_variance = "bank", None
    bank_cfg.residual_bank = (torch.eye(d) * (q * d) ** 0.5).reshape(d, 1, 1, d)
    bank = ForecastCorrection(bank_cfg)
    a = fixed.correct(x, f, y, operator, problem.obs_variance, perturbations=e)
    b = bank.correct(x, f, y, operator, problem.obs_variance, perturbations=e)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    assert not torch.equal(a, x)


def test_a_plain_sampler_refuses_a_correction_block(tiny_case):
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 1)
    settings = {**driver.method_settings(tiny_case, "si_sde"), "correction": {}}
    with pytest.raises(TypeError, match="correction"):
        driver.run_method(tiny_case, "si_sde", problem, sequences, "test", 0, settings)


def test_the_dm_sde_fc_drift_assertion_counts_as_diverged(tiny_case, monkeypatch):
    def assimilate(*args, **kwargs):
        raise AssertionError("FM-SDE prior drift is non-finite; use an endpoint-vanishing g_tau")

    monkeypatch.setattr(generative, "assimilate", assimilate)
    problem = driver.problem_of(tiny_case)
    sequences = load_sequences(problem, "test", 2)
    out = driver.run_method(tiny_case, "dm_sde_fc", problem, sequences, "test", 0)
    assert torch.isnan(out.ensembles).all() and out.nfe is None
    with pytest.raises(AssertionError):
        driver.run_method(tiny_case, "si_sde_fc", problem, sequences, "test", 0)


def test_tune_a_corrected_cell_over_dotted_keys(tuning_case, tmp_path, monkeypatch):
    seen = []
    real_run = driver.run_method

    def run_spy(cfg, name, problem, sequences, split, seed, settings=None):
        seen.append(settings)
        return real_run(cfg, name, problem, sequences, split, seed, settings)

    monkeypatch.setattr(driver, "run_method", run_spy)
    args = SimpleNamespace(
        method="fm_ode_fc",
        grid=json.dumps({"correction.gain_scale": [0.0, 0.5]}),
        set=["correction.forecast_num_steps=2"],
        tag="t",
        threads=1,
        allow_other_m=True,  # the tiny case runs a few pseudo-steps, not M = 50
    )
    driver.cmd_tune(args)
    assert [s["correction"]["gain_scale"] for s in seen] == [0.0, 0.5]
    assert all(s["correction"]["forecast_num_steps"] == 2 for s in seen)
    assert all(s["num_steps"] == tuning_case.methods.fm_ode.num_steps for s in seen)
    path = tmp_path / "linear" / "tuning" / "fm_ode_fc__correction.gain_scale__t.csv"
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert json.loads(rows[0]["base"])["correction"]["forecast_num_steps"] == 2
    for row in rows:
        assert math.isfinite(float(row[driver.INNOVATION_LAG1]))
        # The sampler's M plus two forecasts of 2 steps, except at gain 0, which
        # still draws the forecasts.
        assert float(row[Metric.NFE.value]) == tuning_case.methods.fm_ode.num_steps + 4
    # The plain cell writes no innovation columns.
    plain = SimpleNamespace(
        method="enkf", grid=json.dumps({"inflation": [1.0]}), threads=1
    )
    driver.cmd_tune(plain)
    with open(tmp_path / "linear" / "tuning" / "enkf__inflation.csv", encoding="utf-8") as handle:
        assert driver.INNOVATION_LAG1 not in next(csv.reader(handle))
    with pytest.raises(ValueError, match="nothing to tune"):
        driver.cmd_tune(SimpleNamespace(method="oracle", grid="{}", threads=1))


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_sequence_spread_keeps_metrics_every_seed_has(tmp_path, capsys, order):
    import aggregate_lorenz96

    old = {Metric.RMSE.value: [1.0, 2.0]}  # a review-4 file: no innovation keys
    new = {Metric.RMSE.value: [3.0, 5.0], driver.INNOVATION_LAG1: [0.1, 0.2]}
    for seed, entries in zip(order, (old, new)):
        (tmp_path / f"lorenz96__enkf__traj{seed}__.json").write_text(json.dumps(entries))
    rows = aggregate_lorenz96.sequence_spread(tmp_path)
    assert [r["metric"] for r in rows] == [Metric.RMSE.value]
    assert rows[0]["std"] == pytest.approx(statistics.stdev([2.0, 3.5]))
    assert driver.INNOVATION_LAG1 in capsys.readouterr().out


def test_innovations_are_nan_with_a_warning_for_several_noise_stages(caplog):
    problem = Problem(num_steps=3, noise_stages=2)
    sequences = load_sequences(problem, "test", 2)
    ensembles = torch.randn(2, 3, 6, problem.dim_state, dtype=torch.float64)
    with caplog.at_level("WARNING"):
        out = driver.innovation_output(ensembles, sequences, "test", 0, problem)
    assert len(out) == 2
    assert all(math.isnan(v) for entry in out for v in entry.values())
    assert "noise_stages" in caplog.text


def test_innovations_without_an_own_initial_ensemble_leave_out_the_first_step(problem):
    gen = torch.Generator().manual_seed(0)
    ensemble = torch.randn(4, 16, problem.dim_state, generator=gen, dtype=torch.float64)
    observations = torch.randn(4, problem.obs_indices.numel(), generator=gen)
    full = driver.innovation_scores(problem, ensemble[0], ensemble[1:], observations[1:])
    skipped = driver.innovation_scores(problem, None, ensemble, observations)
    assert skipped == pytest.approx(full)
    assert driver.NO_INITIAL_ENSEMBLE == ("pf_subsample",)


def test_a_tag_cannot_reuse_an_untagged_cell_label():
    with pytest.raises(ValueError, match="untagged cell"):
        driver.check_tag("si_sde", driver.FC_VARIANT, None)
    driver.check_tag("si_sde", "fs5", None)
    driver.check_tag("si_sde_fc", "off", None)


def test_a_disabled_correction_cell_needs_a_tag(tiny_case):
    tiny_case.methods.si_sde_fc.correction.enabled = False
    with pytest.raises(ValueError, match="enabled=false"):
        driver.evaluate_cell("si_sde_fc", 0, tiny_case, allow_other_m=True)
    off = {"correction": {"enabled": False}}
    driver.check_correction_label("si_sde_fc", off, "off")
    driver.check_correction_label("si_sde", off, "")
    with pytest.raises(ValueError, match="enabled=false"):
        driver.check_correction_label("oracle_fc", off, "")
