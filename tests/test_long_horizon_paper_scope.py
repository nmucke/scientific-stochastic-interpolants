"""The manuscript must not turn incomplete or unaudited runs into final results."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "research/ito_maps/paper/build_long_horizon_figures.py"
_spec = importlib.util.spec_from_file_location("long_paper", _path)
paper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper)


def write_inputs(root, complete=False):
    root.mkdir(exist_ok=True)
    families = list(paper.EXPECTED_BUDGETS) if complete else list(paper.EXPECTED_BUDGETS)[2:]
    audit, spectra = {"status": "passed", "runs": []}, {"comparison_complete": complete, "models": []}
    for family in families:
        steps = paper.EXPECTED_BUDGETS[family]
        directory = root / family
        directory.mkdir(exist_ok=True)
        protocol = dict(trajectories=[180], origin=5, history_length=5, ensemble_size=8,
                        horizon=4096, seed=0, data_sha256="data", truth_leads=95,
                        initial_history_sha256="history", truth_sha256="truth", snapshot_dt=.5,
                        inverse_affine={"mean": 0, "std": 3}, model_class=family, num_steps=steps)
        rows = [dict(trajectory=180, origin=5, lead=lead, ensemble_size=8,
                     num_steps=steps, finite_member_fraction=1) for lead in range(1, 4097)]
        (directory / "protocol.json").write_text(json.dumps(protocol))
        (directory / "diagnostics.json").write_text(json.dumps(dict(status="complete", rows=rows)))
        hashes = {f"{name}_sha256": hashlib.sha256((directory / f"{name}.json").read_bytes()).hexdigest()
                  for name in ("protocol", "diagnostics")}
        audit["runs"].append(dict(model_class=family, run_directory=family, **hashes))
        spectra["models"].extend(dict(model_class=family, lead=lead, trajectory=180,
                                      ensemble_size=8, num_steps=steps, provenance=hashes)
                                  for lead in (1, 4096))
    audit_name = "independent_all_model_audit.json" if complete else "independent_completed_map_audit.json"
    spectra_name = "spatial_enstrophy_spectra_all_models.json" if complete else "spatial_enstrophy_spectra_maps_preview.json"
    (root / audit_name).write_text(json.dumps(audit))
    (root / spectra_name).write_text(json.dumps(spectra))
    (root / "reference_enstrophy.json").write_text("{}")


def test_partial_final_pair_retains_explicit_preview(tmp_path):
    write_inputs(tmp_path)
    (tmp_path / "independent_all_model_audit.json").write_text('{"status":"running"}')
    runs, *_, complete = paper.load_verified_inputs(tmp_path, tmp_path)
    assert not complete and len(runs) == 2


def test_both_final_artifacts_switch_to_all_four(tmp_path):
    write_inputs(tmp_path)
    write_inputs(tmp_path, complete=True)
    runs, *_, complete = paper.load_verified_inputs(tmp_path, tmp_path)
    assert complete and {run["family"] for run in runs} == set(paper.EXPECTED_BUDGETS)


def test_bad_final_audit_does_not_silently_fall_back(tmp_path):
    write_inputs(tmp_path)
    write_inputs(tmp_path, complete=True)
    path = tmp_path / "independent_all_model_audit.json"
    audit = json.loads(path.read_text())
    audit["status"] = "failed"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="has not passed"):
        paper.load_verified_inputs(tmp_path, tmp_path)


def test_spectra_must_match_audited_artifact_hashes(tmp_path):
    write_inputs(tmp_path, complete=True)
    path = tmp_path / "spatial_enstrophy_spectra_all_models.json"
    spectra = json.loads(path.read_text())
    spectra["models"][0]["provenance"]["diagnostics_sha256"] = "another run"
    path.write_text(json.dumps(spectra))
    with pytest.raises(ValueError, match="provenance differs"):
        paper.load_verified_inputs(tmp_path, tmp_path)
