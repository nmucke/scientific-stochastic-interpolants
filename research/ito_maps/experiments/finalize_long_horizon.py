#!/usr/bin/env python3
"""Finalize the four completed NS rollouts with existing CPU audit/report tools.

Run from the repository root. Incomplete/nonfinite runs fail before any outputs
are created. Existing products are independently recomputed and compared, never
replaced. The final_report/ directory preserves any legacy report/ unchanged.
No training, model inference, CUDA operations, or paper build occur.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

FAMILIES = {
    'flow_map': ('FlowMapModel', 1),
    'follmer_ito_map': ('FollmerItoMapModel', 8),
    'follmer_stochastic_interpolant': ('FollmerStochasticInterpolant', 64),
    'flow_matching': ('FlowMatchingModel', 64),
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preflight(root):
    """Require the prespecified complete finite experiment before processing."""
    directories = []
    common = None
    for name, (model, budget) in FAMILIES.items():
        directory = root/name
        try:
            p, d = read(directory/'protocol.json'), read(directory/'diagnostics.json')
        except (OSError, ValueError) as error:
            raise ValueError(f'{name}: missing or unreadable completed-run metadata: {error}') from error
        requirements = dict(model_class=model, num_steps=budget, horizon=4096,
                            ensemble_size=8, trajectories=[180], origin=5,
                            history_length=5, feedback_clipping=False)
        for key, value in requirements.items():
            if p.get(key) != value:
                raise ValueError(f'{name}: protocol {key}={p.get(key)!r}; expected {value!r}')
        if d.get('status') != 'complete' or d.get('completed_leads') != 4096 or d.get('requested_horizon') != 4096:
            raise ValueError(f'{name}: finalization requires complete 4096-lead run; '
                             f"status={d.get('status')!r}, completed={d.get('completed_leads')!r}, "
                             f"first_failure={d.get('first_failure')!r}")
        if d.get('first_failure') is not None:
            raise ValueError(f'{name}: run records a nonfinite failure: {d["first_failure"]}')
        rows = d.get('rows', [])
        if len(rows) != 4096 or [r.get('lead') for r in rows] != list(range(1,4097)):
            raise ValueError(f'{name}: incomplete, duplicate, or unordered diagnostic leads')
        for row in rows:
            if row.get('finite_member_fraction') != 1.0:
                raise ValueError(f'{name}: nonfinite member at lead {row["lead"]}')
            for value in row.values():
                values = value if isinstance(value, list) else [value]
                if any(isinstance(v, (int,float)) and not math.isfinite(v) for v in values):
                    raise ValueError(f'{name}: nonfinite diagnostic at lead {row["lead"]}')
        for filename in ('restart.pt','raw_states/lead_000001.npy','raw_states/lead_004096.npy'):
            if not (directory/filename).is_file():
                raise ValueError(f'{name}: missing required final artifact {filename}')
        signature = {key:p[key] for key in ('initial_history_sha256','truth_sha256',
                    'data_sha256','inverse_affine','snapshot_dt','seed','truth_leads')}
        if common is not None and signature != common:
            raise ValueError(f'{name}: initial state, truth, normalization, seed, or time protocol mismatch')
        common = signature
        directories.append(directory)
    return directories


def preserve_json(staged, destination):
    if destination.exists():
        if read(staged) != read(destination):
            raise ValueError(f'Existing artifact differs from independent recomputation; preserved: {destination}')
        print(f'Verified existing {destination}', flush=True)
    else:
        with destination.open('xb') as stream:
            stream.write(staged.read_bytes())
        print(f'Created {destination}', flush=True)


def report_bytes(path):
    value = path.read_bytes()
    if path.suffix == '.pdf':
        # Matplotlib adds run-time metadata; scientific contents must still match.
        value = re.sub(rb'/(?:CreationDate|ModDate)\s*\([^)]*\)', b'', value)
    return value


def preserve_report(staged, destination):
    if not destination.exists():
        shutil.copytree(staged, destination)
        return
    expected = {p.relative_to(staged) for p in staged.rglob('*') if p.is_file()}
    existing = {p.relative_to(destination) for p in destination.rglob('*') if p.is_file()}
    if expected != existing:
        raise ValueError(f'Existing report has a different file inventory; preserved: {destination}')
    for relative in sorted(expected):
        if report_bytes(staged/relative) != report_bytes(destination/relative):
            raise ValueError(f'Existing report differs from recomputation; preserved: {destination/relative}')
    print(f'Verified existing report (PDF timestamps excluded): {destination}', flush=True)


@contextmanager
def exclusive_lock(path):
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another long-horizon finalization is already running') from error
        yield


def finalize(root, archive):
    directories = preflight(root)
    scripts = Path(__file__).resolve().parent
    audit = root/'independent_all_model_audit.json'
    spectra = root/'spatial_enstrophy_spectra_all_models.json'
    report = root/'final_report'
    final = root/'finalization_manifest.json'
    reference = root/'reference_enstrophy.json'
    if not reference.is_file() or not archive.is_file():
        raise ValueError('Archive and archive-context reference must exist')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    def run(script, *arguments):
        subprocess.run([sys.executable, str(scripts/script), *map(str,arguments)],
                       env=environment, check=True)
    with exclusive_lock(root/'.finalize.lock'), tempfile.TemporaryDirectory(prefix='.finalize-',dir=root) as temp:
        stage = Path(temp)
        if final.exists():
            for item in read(final)['artifacts']:
                if not Path(item['path']).is_file() or sha(item['path']) != item['sha256']:
                    raise ValueError(f'Previously finalized artifact changed or missing: {item["path"]}')
        run('audit_long_horizon_units.py','--run-dirs',*directories,'--archive',archive,'--output',stage/'audit.json')
        preserve_json(stage/'audit.json', audit)
        run('spatial_enstrophy_spectra.py','--run-dirs',*directories,'--archive',archive,
            '--leads',1,4096,'--trajectory',180,'--output',stage/'spectra.json')
        if not read(stage/'spectra.json').get('comparison_complete'):
            raise ValueError('Spatial spectrum comparison is not complete')
        preserve_json(stage/'spectra.json', spectra)
        run('report_long_horizon.py','--inputs',*directories,'--reference',reference,
            '--spectra',spectra,'--output',stage/'report')
        preserve_report(stage/'report', report)
        paths = [audit,spectra,reference,*sorted(p for p in report.rglob('*') if p.is_file())]
        payload = dict(schema_version=1,status='complete',cpu_only=True,
            protocol='Four models; trajectory180, origin5, horizon4096, E8; fixed budgets1/8/64/64',
            artifact_comparison='JSON semantic equality; report byte equality except PDF creation/modification timestamps',
            artifacts=[dict(path=str(p),sha256=sha(p)) for p in paths])
        (stage/'final.json').write_text(json.dumps(payload,indent=2)+'\n')
        preserve_json(stage/'final.json',final)
    print(f'All four NS runs audited and reported: {report}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('research/ito_maps/results/long_horizon'))
    parser.add_argument('--archive',type=Path,default=Path('data/stochastic_navier_stokes/data.npz'))
    parser.add_argument('--check-only',action='store_true',help='Validate completed finite run metadata without writing artifacts')
    args = parser.parse_args()
    try:
        if args.check_only:
            preflight(args.root)
            print('All four runs pass completed-run preflight.')
        else:
            finalize(args.root,args.archive)
    except (ValueError,OSError,subprocess.CalledProcessError) as error:
        parser.exit(1,f'Finalization refused: {error}\n')


if __name__ == '__main__':
    main()
