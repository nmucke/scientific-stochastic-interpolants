"""Completion gates and artifact preservation for the CPU finalization command."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'finalize_long_horizon', Path(__file__).parents[1]
    / 'research/ito_maps/experiments/finalize_long_horizon.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def complete_metadata(root):
    for name, (model, budget) in module.FAMILIES.items():
        directory = root/name
        (directory/'raw_states').mkdir(parents=True)
        protocol = dict(model_class=model,num_steps=budget,horizon=4096,
            ensemble_size=8,trajectories=[180],origin=5,history_length=5,
            feedback_clipping=False,initial_history_sha256='same',truth_sha256='same',
            data_sha256='same',inverse_affine={'std':3},snapshot_dt=.5,seed=7,truth_leads=95)
        result = dict(status='complete',completed_leads=4096,requested_horizon=4096,
            first_failure=None,rows=[dict(lead=i,finite_member_fraction=1.,rmse=None)
                                    for i in range(1,4097)])
        (directory/'protocol.json').write_text(json.dumps(protocol))
        (directory/'diagnostics.json').write_text(json.dumps(result))
        for filename in ('restart.pt','raw_states/lead_000001.npy','raw_states/lead_004096.npy'):
            (directory/filename).touch()
    return root/'flow_matching'/'diagnostics.json'


def test_complete_preflight_and_shortened_run_rejection(tmp_path):
    path = complete_metadata(tmp_path)
    assert len(module.preflight(tmp_path))==4
    value = json.loads(path.read_text())
    value['completed_leads']=1024
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='requires complete 4096-lead'):
        module.preflight(tmp_path)


def test_nonfinite_run_and_row_rejected(tmp_path):
    path=complete_metadata(tmp_path)
    value=json.loads(path.read_text())
    value['status']='nonfinite'
    value['first_failure']={'lead':4096}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='nonfinite'):
        module.preflight(tmp_path)
    value['status']='complete';value['first_failure']=None
    value['rows'][-1]['finite_member_fraction']=.875
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='nonfinite member at lead 4096'):
        module.preflight(tmp_path)


def test_existing_audit_semantic_match_preserved_mismatch_refused(tmp_path):
    staged,destination=tmp_path/'new.json',tmp_path/'existing.json'
    destination.write_text('{"value": 1}\n')
    original=destination.read_bytes()
    staged.write_text('{\n"value":1\n}')
    module.preserve_json(staged,destination)
    assert destination.read_bytes()==original
    staged.write_text('{"value":2}')
    with pytest.raises(ValueError,match='differs'):
        module.preserve_json(staged,destination)
    assert destination.read_bytes()==original


def test_report_recomputation_ignores_only_pdf_dates(tmp_path):
    staged,destination=tmp_path/'new',tmp_path/'existing'
    staged.mkdir();destination.mkdir()
    (staged/'figure.pdf').write_bytes(b'%PDF /CreationDate (D:20261007010000Z) scientific plot A')
    (destination/'figure.pdf').write_bytes(b'%PDF /CreationDate (D:20261007000000Z) scientific plot A')
    original=(destination/'figure.pdf').read_bytes()
    module.preserve_report(staged,destination)
    assert (destination/'figure.pdf').read_bytes()==original
    (staged/'figure.pdf').write_bytes(b'%PDF /CreationDate (D:20261007010000Z) scientific plot B')
    with pytest.raises(ValueError,match='differs'):
        module.preserve_report(staged,destination)
    assert (destination/'figure.pdf').read_bytes()==original
