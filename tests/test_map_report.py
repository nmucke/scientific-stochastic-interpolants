"""Reports must preserve matched cases and uncertainty rather than pool runs."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "map_report",
    Path(__file__).resolve().parents[1]
    / "research/ito_maps/experiments/report_results.py",
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def evaluation(name="FlowMapModel", split="validation"):
    rows = []
    for steps in [1, 4]:
        for trajectory, origin, value in [(160, 5, 1.0), (160, 25, 3.0), (161, 5, 8.0)]:
            rows.append(
                {
                    "trajectory": trajectory,
                    "origin": origin,
                    "lead": 1,
                    "num_steps": steps,
                    "ensemble_size": 8,
                    **{metric: value for metric in report.METRICS},
                }
            )
    return {
        "split": split,
        "ensemble_size": 8,
        "forecast_horizon": 1,
        "coordinate_system": "dataset normalized coordinates",
        "rows": rows,
        "model_class": "test." + name,
        "source_path": "/tmp/" + name + "/evaluation.json",
        "source_sha256": "dummy",
        "parameters": None,
    }


def test_report_bootstraps_trajectories_after_averaging_origins():
    rows, metadata = report.summarize([evaluation()], n_bootstrap=1000)
    assert rows[0]["crps"] == 5.0  # Equal weights for trajectory means 2 and 8.
    assert rows[0]["crps_ci95_lower"] == 2.0
    assert rows[0]["crps_ci95_upper"] == 8.0
    assert rows[0]["num_forecasts"] == 3
    assert rows[0]["num_trajectories"] == 2
    assert rows[0]["parameters"] is None
    assert metadata["trajectory_ids"] == [160, 161]
    assert rows[0]["crps_ci95_lower"] == rows[1]["crps_ci95_lower"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("split", "test"),
        ("ensemble_size", 16),
        ("forecast_horizon", 5),
        ("coordinate_system", "original"),
    ],
)
def test_report_rejects_incompatible_evaluations(field, value):
    first, second = evaluation(), evaluation("FollmerItoMapModel")
    second[field] = value
    with pytest.raises(ValueError, match="Incompatible"):
        report.summarize([first, second])


def test_report_rejects_case_mismatch_instead_of_silent_intersection():
    first, second = evaluation(), evaluation("FollmerItoMapModel")
    second["rows"][0]["trajectory"] = 162
    with pytest.raises(ValueError, match="Different trajectory/origin"):
        report.summarize([first, second])


def test_explicit_split_selects_without_relabeling():
    with pytest.raises(ValueError, match="Expected split"):
        report.summarize([evaluation()], split="test")


def test_report_retains_optional_metrics_without_inventing_older_values(
    tmp_path, monkeypatch
):
    import csv
    import json

    first, second = evaluation(), evaluation("FollmerItoMapModel")
    for row in second["rows"]:
        row.update(rms_spread=2.0, rms_spread_skill=0.8, rank_extreme_fraction=0.2)
    paths = []
    for index, data in enumerate((first, second)):
        path = tmp_path / f"evaluation_{index}.json"
        path.write_text(json.dumps(data))
        paths.append(path)
    monkeypatch.setattr(report, "plot_quality", lambda *args: None)
    output = tmp_path / "report"
    rows, _ = report.create_report(paths, output, n_bootstrap=100)
    assert "rms_spread" not in rows[0]
    assert rows[-1]["rms_spread"] == 2.0
    assert rows[-1]["rank_extreme_fraction_ci95_lower"] == pytest.approx(0.2)
    with (output / "summary.csv").open() as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["rms_spread"] == ""
    assert written[-1]["rms_spread"] == "2.0"
    assert "--" in (output / "rms_calibration_table.tex").read_text()


def test_student_mean_teacher_comparison_separates_seed_and_data_uncertainty(
    tmp_path, monkeypatch
):
    import copy
    import json
    import sys

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/ito_maps/experiments")
    )
    import training_seed_report

    teacher = evaluation("FlowMatchingModel", split="test")
    teacher["rows"] = [
        dict(row, num_steps=64) for row in teacher["rows"] if row["num_steps"] == 1
    ]
    students = []
    for seed, offset in enumerate([1.0, 2.0, 3.0]):
        student = copy.deepcopy(teacher)
        student.update(
            model_class="test.FlowMapModel",
            training_seed=seed,
            source_path=f"/tmp/student{seed}.json",
        )
        student["rows"] = [
            dict(
                row,
                num_steps=1,
                **{metric: row[metric] + offset for metric in report.METRICS},
            )
            for row in student["rows"]
        ]
        students.append(student)
    result = training_seed_report.generate(
        students + [teacher], tmp_path, steps=[1], replicates=1000
    )
    comparison = next(
        row for row in result["fixed_teacher_comparisons"] if row["metric"] == "crps"
    )
    assert comparison["training_seeds"] == [0, 1, 2]
    assert comparison["student_mean"] == 7.0
    assert comparison["student_seed_sd"] == 1.0
    assert comparison["student_seed_min"] == 6.0
    assert comparison["student_seed_max"] == 8.0
    assert comparison["reference_mean"] == 5.0
    # Matched shifts cancel trajectory variation; seed variation remains separate.
    assert comparison["difference_student_mean_minus_reference"] == 2.0
    assert comparison["difference_data_cluster_ci95_lower"] == 2.0
    assert comparison["difference_data_cluster_ci95_upper"] == 2.0
    assert comparison["relative_change_percent"] == 40.0
    assert json.loads((tmp_path / "fixed_teacher_comparison.json").read_text())[
        "comparisons"
    ]


def test_student_seed_comparison_rejects_mismatched_teacher_cases(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "research/ito_maps/experiments")
    )
    import training_seed_report

    student = evaluation(split="test")
    student["training_seed"] = 0
    teacher = evaluation("FlowMatchingModel", split="test")
    teacher["rows"] = [
        dict(row, num_steps=64) for row in teacher["rows"] if row["num_steps"] == 1
    ]
    teacher["rows"][0]["origin"] = 99
    with pytest.raises(ValueError, match="Different trajectory/origin"):
        training_seed_report.generate(
            [student, teacher], tmp_path, steps=[1], replicates=10
        )
