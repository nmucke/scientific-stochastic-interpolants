"""Do not turn autonomous boundedness into fictitious forecast accuracy."""

import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "research/ito_maps/experiments/report_long_horizon.py"
_spec = importlib.util.spec_from_file_location("long_horizon_report", _path)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def write_fixture(path, model="FlowMapModel", failed=False):
    path.mkdir()
    protocol = dict(trajectories=[180], origin=5, history_length=5, ensemble_size=8,
                    horizon=4, seed=0, data_sha256="data", truth_leads=2,
                    initial_history_sha256="history", truth_sha256="truth", snapshot_dt=None,
                    inverse_affine={"mean": 0, "std": 3.09969},
                    model_class=model, num_steps=1 if model == "FlowMapModel" else 8)
    rows = []
    for lead in range(1, 4 if failed else 5):
        finite = not (failed and lead == 3)
        rows.append(dict(trajectory=180, origin=5, lead=lead, ensemble_size=8,
                         num_steps=protocol["num_steps"],
                         enstrophy_member_mean=4.0 if finite else None,
                         enstrophy_member_q05=3.0 if finite else None,
                         enstrophy_member_q95=5.0 if finite else None,
                         enstrophy_of_mean=3.0 if finite else None,
                         rms_spread=1.0 if finite else None,
                         rmse=1.0 if lead <= 2 else None,
                         spread_skill_ratio=1.06 if lead <= 2 else None,
                         truth_enstrophy=4.0 if lead <= 2 else None,
                         finite_member_fraction=1.0 if finite else .875,
                         max_abs_vorticity=10 if finite else None,
                         seconds=.1, cumulative_seconds=.1 * lead,
                         nfe=protocol["num_steps"], cumulative_nfe=protocol["num_steps"] * lead))
    result = dict(status="nonfinite" if failed else "complete", rows=rows, completed_leads=len(rows))
    (path / "protocol.json").write_text(json.dumps(protocol))
    (path / "diagnostics.json").write_text(json.dumps(result))
    return path


def test_no_skill_beyond_available_truth(tmp_path):
    path = write_fixture(tmp_path / "run")
    result = json.loads((path / "diagnostics.json").read_text())
    result["rows"][2]["rmse"] = 0.5
    (path / "diagnostics.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="beyond available truth"):
        report.load_runs([path])


def test_matched_initial_condition_and_data_required(tmp_path):
    first = write_fixture(tmp_path / "one")
    second = write_fixture(tmp_path / "two", "FollmerItoMapModel")
    protocol = json.loads((second / "protocol.json").read_text())
    protocol["initial_history_sha256"] = "different"
    (second / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Unmatched"):
        report.load_runs([first, second])


def test_failure_preserved_in_summary(tmp_path):
    path = write_fixture(tmp_path / "run", failed=True)
    summary = report.summarize_run(report.load_runs([path])[0])
    assert summary["first_nonfinite_lead"] == 3
    assert summary["last_all_finite_lead"] == 2
    assert summary["recorded_last_lead"] == 3
    assert summary["requested_horizon"] == 4
    assert summary["last_finite_enstrophy"] == 4


def test_build_report_with_archive_context(tmp_path):
    paths = [write_fixture(tmp_path / "one"),
             write_fixture(tmp_path / "two", "FollmerItoMapModel", failed=True)]
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(dict(enstrophy_q05=3, enstrophy_q95=6, n_fields=2000)))
    output = tmp_path / "report"
    summary = report.build_report(paths, output, reference)
    assert len(summary) == 2
    assert (output / "autonomous_enstrophy_spread.png").exists()
    assert (output / "truth_supported_spread_skill.png").exists()
    text = (output / "README.md").read_text()
    assert "first 2 forecast leads" in text
    assert "not a reference continuation" in text
    assert "nonfinite" in text


def test_spatial_spectra_reject_incorrect_parseval_normalization(tmp_path):
    entries = [dict(model_class="FlowMapModel", num_steps=1, ensemble_size=2,
                    lead=lead, spectrum_members=[[.1, .2, .3], [.2, .3, .4]],
                    spectrum_mean=[.15, .25, .35], spectrum_q05=[.1, .2, .3],
                    spectrum_q95=[.2, .3, .4], enstrophy_members=[.6, .9])
               for lead in (1, 4096)]
    spectra = dict(k=[0, 1, 2], axis_nyquist=2, models=entries, truth=None)
    files = report.plot_spatial_spectra(spectra, tmp_path)
    assert files == ["first_step_enstrophy_spectrum.png", "last_step_enstrophy_spectrum.png"]
    assert all((tmp_path / filename).exists() for filename in files)
    entries[0]["enstrophy_members"][0] = 1.2
    with pytest.raises(ValueError, match="Parseval"):
        report.plot_spatial_spectra(spectra, tmp_path)
