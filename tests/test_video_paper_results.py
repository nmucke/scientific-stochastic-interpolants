"""CPU checks for complete-result gating and fixed visual selection in the paper."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


paper = module(ROOT / "research/ito_maps/paper/build_video_results.py", "video_paper")
fixtures = module(ROOT / "tests/test_video_report.py", "video_report_fixtures")


def setup_report(
    tmp_path, families=None, datasets=("kth", "clevrer"), incomplete=False
):
    families = paper.FAMILY_ORDER if families is None else families
    report_root, video_root = tmp_path / "report", tmp_path / "video"
    (video_root / "results").mkdir(parents=True)
    for dataset in datasets:
        destination = report_root / dataset
        destination.mkdir(parents=True)
        entries, summary, evaluations = [], [], []
        for family in families:
            steps = (
                (1,)
                if family == "persistence"
                else ((1, 8) if "Map" in family else (16, 64))
            )
            item = fixtures.fixture(family, steps)
            item["dataset"] = dataset
            directory = tmp_path / "evaluations" / dataset / family
            directory.mkdir(parents=True)
            path = directory / "evaluation.json"
            path.write_text(json.dumps(item))
            entries.append({"path": str(path), "sha256": paper.sha256(path)})
            evaluations.append(item)
            summary.extend(paper.REPORT.summarize_evaluation(item))
            budget = paper.VISUAL_BUDGETS[family]
            np.savez(
                directory / f"case_000_steps_{budget}.npz",
                truth=np.zeros((4, 1, 8, 8), dtype=np.float32),
                predictions=np.stack([np.zeros((2, 1, 8, 8)), np.ones((2, 1, 8, 8))]),
                history_length=2,
            )
            (directory / f"case_000_steps_{budget}.json").write_text(
                json.dumps({"clip_id": "clip_3"})
            )
            (directory / "cases.json").write_text(
                json.dumps([{"clip_id": "clip_3"}, {"clip_id": "clip_7"}])
            )
        protocol = {key: evaluations[0][key] for key in paper.REPORT.MATCH_FIELDS}
        (destination / "report_manifest.json").write_text(
            json.dumps(
                {
                    "protocol": protocol,
                    "allow_incomplete": incomplete,
                    "inputs": entries,
                }
            )
        )
        paper.REPORT.write_csv(destination / "summary.csv", summary)
        Image.new("RGB", (20, 20), "white").save(destination / "fvd_vs_cost.png")
        frozen = video_root / f"results/{dataset}_protocol.json"
        frozen.write_text(json.dumps({"dataset": dataset}))
        (video_root / f"results/{dataset}_audit.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "protocol_sha256": paper.sha256(frozen),
                    "evaluation_artifacts": {
                        Path(entry["path"]).parent.name: entry["sha256"]
                        for entry in entries
                    },
                }
            )
        )
    return report_root, video_root


def test_full_build_uses_report_values_and_predeclared_example(tmp_path):
    reports, video = setup_report(tmp_path)
    output, snippet = tmp_path / "assets", tmp_path / "results.tex"
    result = paper.build_results(reports, output, snippet, video_root=video)
    assert result["status"] == "complete_audited_comparison"
    assert result["predeclared_visual_selection"]["sha256"] == paper.sha256(
        paper.VISUAL_SELECTION
    )
    assert "2 real and 4 generated" in snippet.read_text()
    assert "90\\%" in snippet.read_text()
    for dataset in ("kth", "clevrer"):
        qualitative = result["datasets"][dataset]["qualitative"]
        assert qualitative["member"] == 0
        assert qualitative["clip_id"] == "clip_3"
        assert {
            item["model_class"]: item["steps"] for item in qualitative["inputs"]
        } == paper.VISUAL_BUDGETS
        assert paper.sha256(
            output / f"{dataset}_unrounded_summary.csv"
        ) == paper.sha256(reports / dataset / "summary.csv")
        assert paper.sha256(output / f"{dataset}_fvd_vs_cost.png") == paper.sha256(
            reports / dataset / "fvd_vs_cost.png"
        )
        text = (output / f"{dataset}_quality_table.tex").read_text()
        assert "1.5000" in text  # reporter-computed mean CRPS
        assert "Persistence" in text and "128" in text  # solver64 x two leads
        assert (output / f"{dataset}_fixed_forecast.pdf").exists()


@pytest.mark.parametrize(
    "problem",
    [
        "missing_dataset",
        "development_report",
        "missing_persistence",
        "stale_audit",
        "changed_evaluation",
    ],
)
def test_incomplete_or_unaudited_reports_rejected_before_writes(tmp_path, problem):
    families = paper.FAMILY_ORDER[:-1] if problem == "missing_persistence" else None
    datasets = ("kth",) if problem == "missing_dataset" else ("kth", "clevrer")
    reports, video = setup_report(
        tmp_path, families, datasets, problem == "development_report"
    )
    if problem == "stale_audit":
        (video / "results/kth_protocol.json").write_text("{}")
    if problem == "changed_evaluation":
        path = tmp_path / "evaluations/kth/FlowMapModel/evaluation.json"
        path.write_text(path.read_text() + " ")
    output, snippet = tmp_path / "assets", tmp_path / "results.tex"
    with pytest.raises(ValueError):
        paper.build_results(reports, output, snippet, video_root=video)
    assert not output.exists() and not snippet.exists()


def test_development_must_not_use_either_canonical_destination(tmp_path):
    for output, snippet in [
        (paper.DEFAULT_OUTPUT, tmp_path / "partial.tex"),
        (tmp_path / "assets", paper.DEFAULT_SNIPPET),
    ]:
        with pytest.raises(ValueError, match="noncanonical"):
            paper.build_results(tmp_path, output, snippet, allow_incomplete=True)


def test_fixed_case_and_truth_must_match(tmp_path):
    reports, video = setup_report(
        tmp_path,
        families=["FlowMapModel", "FollmerItoMapModel"],
        datasets=("kth",),
        incomplete=True,
    )
    report = paper.load_dataset_report(reports, "kth", True, video)
    output = tmp_path / "assets"
    output.mkdir()
    path = tmp_path / "evaluations/kth/FlowMapModel/case_000_steps_1.npz"
    np.savez(
        path,
        truth=np.ones((4, 1, 8, 8)),
        predictions=np.zeros((2, 2, 1, 8, 8)),
        history_length=2,
    )
    with pytest.raises(ValueError, match="same observed case"):
        paper.fixed_example("kth", report, output, True)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({"clip_id": "clip_7"}))
    with pytest.raises(ValueError, match="first selected case"):
        paper.fixed_example("kth", report, output, True)
