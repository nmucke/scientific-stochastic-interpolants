"""Create a manuscript-only table from existing, matched validation artifacts."""
import hashlib
import json
from pathlib import Path

paper = Path(__file__).resolve().parent
research = paper.parent
specifications = [
    ('SI teacher', 'validation_baselines/stochastic_interpolant', (16, 64)),
    ('Flow-matching teacher', 'validation_baselines/flow_matching', (16, 64)),
    (r'It\^o map, PSD', 'validation_psd/ito_psd_seed0', (1, 8)),
    ('Flow map, PSD', 'validation_psd/flow_psd_seed0', (1, 8)),
    (r'It\^o map, refined $K=4$', 'validation_rollout/ito_rollout_seed0', (1, 8)),
    ('Flow map, refined', 'validation_rollout/flow_rollout_seed0', (1, 8)),
    (r'It\^o map, refined $K=16$', 'validation_rollout/ito_k16_rollout_seed0', (1, 8)),
]
(paper / 'generated').mkdir(exist_ok=True)
lines = [
    r'\begin{table*}[t]',
    r'\caption{Preliminary validation only: four trajectories (160--163), three origins (5, 45, 85), eight ensemble members, one physical forecast interval, and one student training seed. CRPS is in normalized coordinates. Brackets show trajectory-bootstrap 95\% intervals; these do not describe training-seed uncertainty. Latency is seconds per ensemble forecast on one RTX 3090. Refined maps use 2,000 PSD updates followed by 500 simulation-based teacher-rollout updates. These development cases were used for methodological selection and are not held-out test evidence.}',
    r'\label{tab:validation-development}',
    r'\centering',
    r'\begin{tabular}{lrll}',
    r'\toprule',
    r'Model & NFE & CRPS $\downarrow$ [95\% interval] & Latency (s) $\downarrow$ \\',
    r'\midrule',
]
provenance = []
expected_cases = {(t, o) for t in range(160, 164) for o in (5, 45, 85)}
for label, directory, budgets in specifications:
    path = research / 'results' / directory / 'evaluation.json'
    data = json.loads(path.read_text())
    assert data['split'] == 'validation'
    assert data['ensemble_size'] == 8 and data['forecast_horizon'] == 1
    for steps in budgets:
        rows = [r for r in data['rows'] if r['num_steps'] == steps]
        assert {(r['trajectory'], r['origin']) for r in rows} == expected_cases
        row = next(r for r in data['summary'] if r['num_steps'] == steps and r['lead'] == 1)
        low, high = row['crps_ci95']
        lines.append(f"{label} & {row['nfe']:.0f} & {row['crps']:.4f} [{low:.4f}, {high:.4f}] & {row['seconds']:.3f} " + r'\\')
    provenance.append({'source': str(path.relative_to(research)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'checkpoint': data.get('checkpoint'), 'checkpoint_sha256': data.get('checkpoint_sha256')})
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
(paper/'generated/validation_table.tex').write_text('\n'.join(lines))
(paper/'generated/validation_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')
print('Wrote matched validation-only table and source hashes.')
