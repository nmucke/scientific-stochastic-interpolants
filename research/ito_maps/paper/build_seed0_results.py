"""Generate a clearly labeled seed-zero test table from completed artifacts."""
import hashlib
import json
from pathlib import Path

paper = Path(__file__).resolve().parent
root = paper.parent / 'results/test_one_step'
specifications = [
    ('Forecasting SI', 'stochastic_interpolant', (16, 64, 128)),
    ('Flow matching', 'flow_matching', (64, 128)),
    (r'It\^o map', 'ito_rollout_seed0', (1, 8)),
    ('Flow map', 'flow_rollout_seed0', (1, 8)),
]
expected = {(t,o) for t in range(180,200) for o in (5,45,85)}
lines = [r'\begin{table}[t]',
    r'\caption{Seed-zero one-interval test measurements: 20 held-out trajectories, three origins each, and 16 ensemble members. Student checkpoints were selected before testing. Values average the 60 contexts; paired trajectory intervals appear in the text. Additional training seeds remain pending.}',
    r'\label{tab:seed0-test}',r'\centering',r'\begin{tabular}{lrrr}',r'\toprule',
    r'Model & NFE & CRPS $\downarrow$ & Seconds $\downarrow$ \\',r'\midrule']
provenance=[]
calibration = [r'\begin{table}[t]',
    r'\caption{Seed-zero held-out uncertainty diagnostics, averaged over the same 60 contexts as Table~\ref{tab:seed0-test}. RMS spread uses unbiased sample variance; coverage and extreme ranks are percentages. At this finite ensemble size, nominal 90\% coverage is not a calibration target.}',
    r'\label{tab:seed0-calibration}',r'\centering',r'\setlength{\tabcolsep}{4.5pt}',r'\begin{tabular}{lrrrr}',r'\toprule',
    r'Model & NFE & Spread & Coverage & Extremes \\',r'\midrule']
for label,name,budgets in specifications:
    path=root/name/'evaluation.json';d=json.loads(path.read_text())
    assert d['split']=='test' and d['ensemble_size']==16 and d['forecast_horizon']==1
    for budget in budgets:
        rows=[r for r in d['rows'] if r['num_steps']==budget and r['lead']==1]
        assert len(rows)==60 and {(r['trajectory'],r['origin']) for r in rows}==expected
        row=next(r for r in d['summary'] if r['num_steps']==budget and r['lead']==1)
        lines.append(f"{label} & {row['nfe']:.0f} & {row['crps']:.5f} & {row['seconds']:.3f} "+r'\\')
        calibration.append(f"{label} & {row['nfe']:.0f} & {row['rms_spread']:.4f} & {100*row['coverage90']:.2f} & {100*row['rank_extreme_fraction']:.2f} "+r'\\')
    provenance.append({'source':str(path.relative_to(paper.parent)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'checkpoint':d.get('checkpoint'), 'checkpoint_sha256':d.get('checkpoint_sha256')})
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
calibration += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
(paper/'generated').mkdir(exist_ok=True)
(paper/'generated/test_seed0_table.tex').write_text('\n'.join(lines))
(paper/'generated/test_seed0_calibration.tex').write_text('\n'.join(calibration))
(paper/'generated/test_seed0_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Wrote seed-zero test table from complete matched sources.')
