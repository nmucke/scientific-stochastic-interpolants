"""Render manuscript figures and claims from the frozen seed-zero report.

This script never samples forecasts or chooses checkpoints.  CSV inputs and
their raw-source hashes are retained by the experiment report generator.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / 'results'
REPORT = RESULTS / 'final_report'
FIGURES = PAPER / 'figures'
GENERATED = PAPER / 'generated'
FIGURES.mkdir(exist_ok=True)
GENERATED.mkdir(exist_ok=True)
sources = []


def read(path, kind='json'):
    sources.append({'source': str(path.relative_to(PAPER.parent)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if kind == 'csv':
        with path.open() as handle:
            return list(csv.DictReader(handle))
    return json.loads(path.read_text())


one = read(REPORT / 'one_step/summary.csv', 'csv')
rollout = read(REPORT / 'rollout/all_leads.csv', 'csv')
one_pairs = read(REPORT / 'one_step/paired_crps.json')
roll_pairs = read(REPORT / 'rollout/paired_crps.json')
assert all(r['split'] == 'test' and int(r['ensemble_size']) == 16 for r in one + rollout)
assert all(int(r['num_trajectories']) == 20 for r in one + rollout)

models = ['Föllmer Itô map', 'Flow map', 'Föllmer SI', 'Flow matching']
labels = ['Itô map', 'Flow map', 'Forecasting SI', 'Flow matching']
colors = ['#2166ac', '#b2182b', '#439466', '#8a64a2']
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.6), layout='constrained')
for model, label, color in zip(models, labels, colors):
    rows = sorted([r for r in one if r['model'] == model], key=lambda r: int(r['num_steps']))
    assert len(rows) == (4 if model in models[:2] else 5)
    for axis, metric in zip(axes, ['nfe', 'seconds']):
        x = np.array([float(r[metric]) for r in rows])
        y = np.array([float(r['crps']) for r in rows])
        low = np.array([float(r['crps_ci95_lower']) for r in rows])
        high = np.array([float(r['crps_ci95_upper']) for r in rows])
        axis.plot(x, y, '-o', color=color, label=label, markersize=3, linewidth=1.2)
        axis.fill_between(x, low, high, color=color, alpha=.10, linewidth=0)
        axis.set_xscale('log')
        axis.grid(alpha=.2)
        axis.set_ylabel('Fair CRPS ↓')
axes[0].set_xlabel('Neural evaluations per transition')
axes[1].set_xlabel('Seconds per 16-member forecast')
axes[0].legend(frameon=False, loc='upper right')
for ext in ['pdf', 'png']:
    fig.savefig(FIGURES / f'seed0_quality_cost.{ext}', dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.5), layout='constrained')
for model, label, color in zip(models, labels, colors):
    budgets = [1, 8] if model in models[:2] else [64]
    for budget in budgets:
        rows = sorted([r for r in rollout if r['model'] == model and int(r['num_steps']) == budget],
                      key=lambda r: int(r['lead']))
        assert [int(r['lead']) for r in rows] == list(range(1, 11))
        x = np.array([int(r['lead']) for r in rows])
        for axis, metric, scale in zip(axes, ['crps', 'coverage90', 'cumulative_seconds'], [1, 100, 1]):
            y = np.array([float(r[metric])*scale for r in rows])
            low = np.array([float(r[metric+'_ci95_lower'])*scale for r in rows])
            high = np.array([float(r[metric+'_ci95_upper'])*scale for r in rows])
            axis.plot(x, y, '--' if budget == 8 else '-', color=color,
                      label=f'{label}, {budget}', linewidth=1.25)
            axis.fill_between(x, low, high, color=color, alpha=.07, linewidth=0)
            axis.set_xlabel('Physical lead (snapshots)')
            axis.set_xticks([1, 5, 10])
            axis.grid(alpha=.2)
axes[0].set_ylabel('Fair CRPS ↓')
axes[1].set_ylabel('Empirical coverage (%)')
axes[2].set_ylabel('Cumulative seconds ↓')
axes[2].set_yscale('log')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3,
           frameon=False, columnspacing=1.4)
for ext in ['pdf', 'png']:
    fig.savefig(FIGURES / f'seed0_rollout.{ext}', dpi=200)
plt.close(fig)

def pair(rows, candidate, cn, reference, rn, lead):
    return next(r for r in rows if r['candidate_label'] == candidate and
                r['candidate_steps'] == cn and r['reference_label'] == reference and
                r['reference_steps'] == rn and r['lead'] == lead)

macros = ['% Generated from frozen seed-zero sources; never combine with training-seed intervals.']
for key, data in [
        ('ItoOneSI', pair(one_pairs, models[0], 1, models[2], 16, 1)),
        ('ItoEightSI', pair(one_pairs, models[0], 8, models[2], 64, 1)),
        ('FlowOneFM', pair(one_pairs, models[1], 1, models[3], 64, 1)),
        ('ItoEightRoll', pair(roll_pairs, models[0], 8, models[2], 64, 10)),
        ('FlowOneRoll', pair(roll_pairs, models[1], 1, models[3], 64, 10))]:
    for suffix, value, precision in [
            ('Difference', data['difference_candidate_minus_reference'], 4),
            ('Lower', data['difference_ci95'][0], 4),
            ('Upper', data['difference_ci95'][1], 4),
            ('Percent', data['relative_change_percent'], 2),
            ('Speed', data['latency_speedup_reference_over_candidate'], 2)]:
        # Rollout paired-comparison seconds are per-transition; use cumulative
        # measured costs below for the rollout speed macros.
        if suffix == 'Speed' and 'Roll' in key:
            c = next(r for r in rollout if r['model'] == data['candidate_label'] and
                     int(r['num_steps']) == data['candidate_steps'] and int(r['lead']) == 10)
            r = next(r for r in rollout if r['model'] == data['reference_label'] and
                     int(r['num_steps']) == data['reference_steps'] and int(r['lead']) == 10)
            value = float(r['cumulative_seconds']) / float(c['cumulative_seconds'])
        macros.append('\\newcommand{\\sz'+key+suffix+'}{'+f'{value:.{precision}f}'+'}')

training = [r'\begin{table}[t]',
            r'\caption{Recorded seed-zero student training-and-validation time on one RTX 3090. Stage one has 2,000 updates; stage two has 500. Times include teacher evaluations and validation, but exclude model/data setup and original teacher pretraining.}',
            r'\label{tab:training-cost}', r'\centering',r'\begin{tabular}{lrrr}',r'\toprule',
            r'Model & Stage 1 (s) & Stage 2 (s) & Total (min) \\',r'\midrule']
for family, label in [('ito', r'It\^o map'), ('flow', 'Flow map')]:
    times = []
    for stage in ['psd', 'rollout']:
        history = read(RESULTS / f'training/{family}_{stage}_seed0/history.json')
        assert history[-1]['step'] == (2000 if stage == 'psd' else 500)
        times.append(history[-1]['elapsed_seconds'])
    training.append(f'{label} & {times[0]:.1f} & {times[1]:.1f} & {sum(times)/60:.2f} '+r'\\')
training += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
(GENERATED / 'seed0_training_cost.tex').write_text('\n'.join(training))
(GENERATED / 'seed0_claims.tex').write_text('\n'.join(macros)+'\n')
(GENERATED / 'study_figure_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
print('Built seed-zero figures, claim macros, and recorded training costs.')
