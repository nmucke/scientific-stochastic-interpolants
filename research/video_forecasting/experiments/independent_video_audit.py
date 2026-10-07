"""CPU-only independent arithmetic review of saved pixel-video evaluation artifacts.

Does not import production metric/dataset/report code. Reads frozen artifacts;
only writes the separate audit JSON specified by --output. No detector replay.
"""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ssim(a, b):
    # Independent Wang SSIM formula, population covariance, valid 11x11 window.
    def filt(x):
        return gaussian_filter(x, sigma=(0, 1.5, 1.5), truncate=3.5, mode='reflect')
    ma, mb = filt(a), filt(b)
    va, vb = filt(a*a)-ma*ma, filt(b*b)-mb*mb
    cov = filt(a*b)-ma*mb
    value = ((2*ma*mb+.01**2)*(2*cov+.03**2)
             / ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)))
    return float(value[:, 5:-5, 5:-5].mean())


def metrics(pred, target):
    x, y = (pred.astype(np.float64)+1)/2, (target.astype(np.float64)+1)/2
    e = len(x)
    pairs = [x[i]-x[j] for i in range(e) for j in range(i)]
    crps = np.abs(x-y).mean()-sum(np.abs(p).mean() for p in pairs)/(e*(e-1))
    energy = np.sqrt(np.mean((x-y)**2, axis=(1,2,3))).mean()
    energy -= sum(np.sqrt(np.mean(p*p)) for p in pairs)/(e*(e-1))
    mean = x.mean(0)
    rmse = np.sqrt(np.mean((mean-y)**2))
    spread = np.sqrt(x.var(0, ddof=1).mean())
    lo, hi = np.quantile(x, [.05,.95], axis=0)
    result = dict(crps=float(crps), energy_score=float(energy), rmse=float(rmse),
                  rms_spread=float(spread), spread_skill_ratio=float(spread*np.sqrt((e+1)/e)/rmse),
                  coverage_90=float(((y>=lo)&(y<=hi)).mean()),
                  out_of_range_fraction=float(((x<0)|(x>1)).mean()))
    clipped, truth, clipped_mean = np.clip(x,0,1), np.clip(y,0,1), np.clip(mean,0,1)
    psnrs = -10*np.log10(np.maximum(((clipped-truth)**2).mean((1,2,3)),1e-10))
    ssims = [ssim(member, truth) for member in clipped]
    result.update(psnr_sample_mean=float(psnrs.mean()), psnr_oracle_best=float(psnrs.max()),
                  psnr_ensemble_mean=float(-10*np.log10(max(((clipped_mean-truth)**2).mean(),1e-10))),
                  ssim_sample_mean=float(np.mean(ssims)), ssim_oracle_best=float(max(ssims)),
                  ssim_ensemble_mean=ssim(clipped_mean, truth))
    clipped_crps = np.abs(clipped-truth).mean()-sum(
        np.abs(clipped[i]-clipped[j]).mean() for i in range(e) for j in range(i))/(e*(e-1))
    return result, abs(float(crps-clipped_crps))


def fvd_eigen(real, generated):
    """Independent dense covariance eigen route, versus production low-rank SVD."""
    a, b = real.astype(np.float64), generated.astype(np.float64)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    eigen, q = np.linalg.eigh(ca)
    # The empirical covariance has at most n-1 nonzero eigenvalues. Removing
    # nullspace roundoff avoids its amplification by the square root.
    keep = eigen > eigen.max()*np.finfo(float).eps*len(eigen)
    eigen, q = eigen[keep], q[:, keep]
    middle = np.sqrt(eigen[:,None])*(q.T@cb@q)*np.sqrt(eigen[None,:])
    cross = np.sqrt(np.maximum(np.linalg.eigvalsh((middle+middle.T)/2),0)).sum()
    value = ((a.mean(0)-b.mean(0))**2).sum()+np.trace(ca)+np.trace(cb)-2*cross
    return float(max(value,0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('kth','clevrer'), default='kth')
    parser.add_argument('--results', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path('research/video_forecasting/results')
    args.results = args.results or root/'test'/args.dataset
    args.protocol = args.protocol or root/f'{args.dataset}_protocol.json'
    args.report = args.report or root/f'{args.dataset}_completed_report'/args.dataset/'summary.csv'
    args.output = args.output or root/f'{args.dataset}_independent_audit.json'
    protocol = read(args.protocol)
    assert protocol['dataset']==args.dataset
    history, horizon = protocol['history_length'], protocol['forecast_horizon']
    count, members = protocol['num_clips'], protocol['ensemble_size']
    total = history+horizon
    manifest_path = Path(protocol['manifest']['path'])
    assert sha(manifest_path)==protocol['manifest']['sha256']
    manifest = read(manifest_path)
    eligible = [c for c in manifest['clips'] if c['split']=='test' and c['num_frames']>=total]
    selected = np.random.default_rng(protocol['seed']).choice(len(eligible),count,replace=False).tolist()
    assert selected==protocol['selected_indices']
    cases = [eligible[i] for i in selected]
    assert cases==protocol['cases']
    subjects = None
    if args.dataset=='kth':
        assert {c['person'] for c in cases}==set(range(21,26))
        subjects = {s:{c['person'] for c in manifest['clips'] if c['split']==s}
                    for s in ('train','validation','test')}
        assert not (subjects['train']&subjects['test'] or subjects['validation']&subjects['test'])
    sources = {s:{c['source_member'] for c in manifest['clips'] if c['split']==s}
               for s in ('train','validation','test')}
    assert not (sources['train']&sources['test'] or sources['validation']&sources['test'])
    source_count = len({c['source_member'] for c in cases})
    channels, resolution = manifest['channels'], protocol['resolution']
    expected_runs={(name,step) for name,config in protocol['models'].items() for step in config['num_steps']}
    expected_runs.update(('persistence',step) for step in protocol['persistence_steps'])
    observed_runs=set()
    for case in cases:
        assert sha(manifest_path.parent/case['path'])==case['sha256']
    report = list(csv.DictReader(args.report.open()))
    first_real = None
    findings = []
    global_errors = defaultdict(float)
    for model_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        ev = read(model_dir/'evaluation.json')
        assert ev['status']=='complete' and ev['selected_indices']==selected
        assert read(model_dir/'cases.json')==cases
        assert ev['history_length']==history and ev['forecast_horizon']==horizon and ev['ensemble_size']==members
        assert ev['feedback_clipping'] is False
        rows = {(r['num_steps'],r['clip_id'],r['lead']):r for r in ev['rows']}
        assert len(rows)==len(ev['rows'])
        for fvd_row in ev['fvd']:
            steps = fvd_row['num_steps']
            observed_runs.add((model_dir.name,steps))
            errors = defaultdict(float)
            inspected = 0
            raw_min,raw_max = float('inf'),float('-inf')
            clipping_difference = 0.
            raw_hashes = {}
            for case_number,case in enumerate(cases[:3]):
                prefix = model_dir/f'case_{case_number:03d}_steps_{steps}'
                rawpath = prefix.with_suffix('.npz')
                raw_hashes[str(rawpath)] = sha(rawpath)
                with np.load(rawpath) as raw:
                    truth,pred = raw['truth'],raw['predictions']
                    assert int(raw['history_length'])==history
                assert truth.shape==(total,channels,resolution,resolution)
                assert pred.shape==(members,horizon,channels,resolution,resolution)
                source = np.load(manifest_path.parent/case['path'],mmap_mode='r')[:total].transpose(0,3,1,2)
                assert np.array_equal(np.rint((truth.astype(float)+1)*127.5).astype('uint8'),source)
                assert np.max(np.abs(truth-(source.astype(float)/127.5-1)))<1.2e-7
                raw_min,raw_max=min(raw_min,float(pred.min())),max(raw_max,float(pred.max()))
                for lead in range(1,horizon+1):
                    calculated, difference = metrics(pred[:,lead-1],truth[history+lead-1])
                    recorded = rows[steps,case['clip_id'],lead]
                    for key,value in calculated.items():
                        error=abs(value-recorded[key])
                        errors[key]=max(errors[key],error)
                        assert error<1e-10,(model_dir,steps,case_number,lead,key,error)
                    clipping_difference=max(clipping_difference,difference)
                    inspected+=1
                # Reconstruct every displayed pixel: first two fixed members.
                lead_indices=sorted(set([0,horizon//4,horizon//2,3*horizon//4,horizon-1]))
                sheets=[]
                for sequence in [truth[history:],pred[0],pred[1]]:
                    pictures=[]
                    for lead in lead_indices:
                        pixels=((np.clip(sequence[lead],-1,1)+1)*np.float32(127.5)).astype('uint8')
                        if channels==1: pixels=np.repeat(pixels,3,axis=0)
                        pictures.append(pixels.transpose(1,2,0))
                    sheets.append(np.concatenate(pictures,axis=1))
                assert np.array_equal(np.asarray(Image.open(prefix.with_suffix('.png'))),np.concatenate(sheets,axis=0))
                assert read(prefix.with_suffix('.json'))['clip_id']==case['clip_id']
            featurepath=model_dir/f'features_steps_{steps}.npz'
            with np.load(featurepath) as f:
                real,generated=f['real'],f['generated']
            assert real.shape==(count,400) and generated.shape==(count,members,400)
            if first_real is None: first_real=real.copy()
            assert np.array_equal(real,first_real)
            independent_pooled=fvd_eigen(real,generated.reshape(count*members,400))
            independent_members=[fvd_eigen(real,generated[:,i]) for i in range(members)]
            fvd_error=max(abs(independent_pooled-fvd_row['fvd_pooled']),
                          max(abs(a-b) for a,b in zip(independent_members,fvd_row['fvd_member_values'])))
            assert fvd_error<1e-5,fvd_error
            csvrow=next(r for r in report if r['model_class']==ev['model_class'] and int(r['num_steps'])==steps)
            report_error=0.
            for key in ('fvd_pooled','fvd_member_mean','fvd_member_std'):
                report_error=max(report_error,abs(float(csvrow[key])-fvd_row[key]))
            # Recompute source-video-equal-weight aggregation directly from rows.
            score_keys=list(errors)+['lpips_sample_mean','lpips_oracle_best','seconds','nfe']
            for key in score_keys:
                bylead=[]
                for lead in range(1,horizon+1):
                    bysource=defaultdict(list)
                    for r in ev['rows']:
                        if r['num_steps']==steps and r['lead']==lead:
                            bysource[r['source_video']].append(r[key])
                    assert len(bysource)==source_count
                    bylead.append(float(np.mean([np.mean(v) for v in bysource.values()])))
                if key in ('seconds','nfe'):
                    checks={f'rollout_{key}':sum(bylead)}
                else:
                    checks={f'mean_{key}':np.mean(bylead),f'last_{key}':bylead[-1]}
                for column,value in checks.items():
                    report_error=max(report_error,abs(float(csvrow[column])-value))
            assert report_error<1e-10,report_error
            for key,value in errors.items(): global_errors[key]=max(global_errors[key],value)
            findings.append(dict(model=ev['model_class'],steps=steps,rows_recomputed=inspected,
                raw_prediction_range=[raw_min,raw_max],max_unclipped_vs_clipped_crps_difference=clipping_difference,
                metric_max_absolute_errors=dict(errors),fvd_max_absolute_error=fvd_error,
                independent_fvd_pooled=independent_pooled,independent_fvd_members=independent_members,
                report_max_absolute_error=report_error,features_sha256=sha(featurepath),
                evaluation_sha256=sha(model_dir/'evaluation.json'),raw_saved_case_sha256=raw_hashes))
            print(f'{model_dir.name} steps={steps}: {inspected} rows; FVD error {fvd_error:.3g}',flush=True)
    assert observed_runs==expected_runs,(observed_runs,expected_runs)
    result=dict(dataset=args.dataset,status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
        audit_method='Independent NumPy proper scores/PSNR, SciPy Gaussian SSIM, dense covariance-eigen FVD; no production metric imports',
        protocol_sha256=sha(args.protocol),manifest_sha256=sha(manifest_path),report_sha256=sha(args.report),
        eligible_test_clips=len(eligible),selected_test_clips=len(cases),selected_source_videos=source_count,
        heldout_subjects=sorted(subjects['test']) if subjects else None,future_only_indices=list(range(history,total)),
        saved_cases_checked=[c['clip_id'] for c in cases[:3]],
        metric_max_absolute_errors=dict(global_errors),runs=findings,
        reconstruction=f'Saved truth roundtrips byte-exactly to source frames; all {3*len(findings)} contact sheets match first frozen cases, first two members pixel-exactly.',
        fvd=f'All {len(findings)} feature files have identical real features ({count}x400), generated {count}x{members}x400; pooled {count*members} vs{count} real, each member{count} vs{count} real. FVD includes observed{history}-frame prefix and{horizon}future frames.',
        limitations=[f'Raw forecast arithmetic is checked for the three saved cases ({3*horizon*len(findings)} future rows), not unsaved{count-3}cases.',
                     'LPIPS and I3D neural features are not re-extracted; saved features and detector/config provenance were checked by the existing strict audit.',
                     'Bootstrap confidence intervals are not independently recomputed here; report point estimates are checked from every row.'],
        issues_found=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
