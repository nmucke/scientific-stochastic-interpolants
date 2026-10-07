"""Prepare final manuscript tables only after all frozen student seeds exist.

Unlike seed-zero tables, these outputs summarize student-training variability.
The strict completeness check deliberately fails while confirmation is pending.
No manuscript input is switched automatically and no forecasts are sampled.
"""
import csv
import hashlib
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER.parent / 'results'
REPORT = RESULTS / 'final_report'
OUTPUT = PAPER / 'generated'
MODELS = {'Föllmer Itô map': (r'It\^o map', 'ito', 'Ito'),
          'Flow map': ('Flow map', 'flow', 'Flow')}
sources = []


def read(path, csv_file=False):
    sources.append({'source': str(path.relative_to(PAPER.parent)),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if csv_file:
        with path.open() as handle:
            return list(csv.DictReader(handle))
    return json.loads(path.read_text())


reports = {}
per_seed = {}
comparisons = {}
for horizon, folder in [(1, 'one_step'), (10, 'rollout')]:
    directory = REPORT / folder / 'student_training_seeds'
    report = read(directory / 'training_seed_report.json')
    assert report['forecast_horizon'] == horizon and report['split'] == 'test'
    for model in MODELS:
        entries = [r for r in report['sources'] if r['model'] == model]
        assert sorted(r['training_seed'] for r in entries) == [0, 1, 2], (
            f'Confirmation incomplete: {folder}/{model} needs seeds 0, 1, 2')
        assert len({r['fixed_teacher_checkpoint'] for r in entries}) == 1
    rows = report['across_seed']
    expected = {(model, budget, lead) for model in MODELS
                for budget in (1, 8) for lead in range(1, horizon+1)}
    assert len(rows) == len(expected)
    assert {(r['model'], r['num_steps'], r['lead']) for r in rows} == expected
    for row in rows:
        assert row['num_training_seeds'] == 3 and sorted(row['training_seeds']) == [0, 1, 2]
        assert row['num_test_trajectories'] == 20 and row['crps_student_seed_sd'] is not None
    reports[horizon] = rows
    per_seed[horizon] = read(directory / 'per_student_seed.csv', csv_file=True)
    assert all(int(r['ensemble_size']) == 16 and int(r['num_test_trajectories'] if
               'num_test_trajectories' in r else r['num_trajectories']) == 20
               for r in per_seed[horizon])
    comparisons[horizon] = read(directory / 'fixed_teacher_comparison.json')['comparisons']
    assert all(sorted(r['training_seeds']) == [0, 1, 2] and
               r['num_training_seeds'] == 3 for r in comparisons[horizon])

teacher_rows = {1: read(REPORT / 'one_step/summary.csv', csv_file=True),
                10: read(REPORT / 'rollout/all_leads.csv', csv_file=True)}


def aggregate(horizon, model, budget):
    return next(r for r in reports[horizon] if r['model'] == model and
                r['num_steps'] == budget and r['lead'] == horizon)


def individual(horizon, model, budget, seed):
    return next(r for r in per_seed[horizon] if r['model'] == model and
                int(r['num_steps']) == budget and int(r['lead']) == horizon and
                int(r['training_seed']) == seed)


table = [r'\begin{table}[t]',
         r'\caption{Confirmation over three independently trained students with fixed teachers. CRPS is the mean $\pm$ sample standard deviation across training seeds 0, 1, and 2. One-interval evaluation uses 60 contexts; physical lead 10 uses 20 contexts. This standard deviation is distinct from a trajectory-bootstrap confidence interval.}',
         r'\label{tab:replication}',r'\centering',r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{lrrr}',r'\toprule',
         r'Model & Steps & One interval & Physical lead 10 \\',r'\midrule']
detail = [r'\begin{table*}[t]',
          r'\caption{Individual student-training replications under the frozen protocol. The evaluation sampling seed and test cases are shared; each student repeats both training stages with its own training seed. Coverage is the finite-ensemble empirical interval coverage, not a population 90\% criterion.}',
          r'\label{tab:individual-seeds}',r'\centering',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
          r'Model & Steps & Training seed & One-interval CRPS & Lead-10 CRPS & Lead-10 RMS spread & Lead-10 coverage (\%) \\',r'\midrule']
macros = ['% Three trained students, fixed teacher checkpoints; point estimates only.']
paired = [r'\begin{table*}[t]',
          r'\caption{Three-student mean CRPS differences from each fixed 64-step teacher. Intervals resample the same 20 test trajectories after averaging student scores; they condition on these trained students and do not include teacher-training uncertainty. The final column reports reference-to-student measured latency ratios, using cumulative costs for physical lead 10.}',
          r'\label{tab:replication-paired}',r'\centering',r'\begin{tabular}{lrrrrr}',r'\toprule',
          r'Model & Steps & Physical lead & CRPS difference [95\% interval] & Change (\%) & Speedup \\',r'\midrule']
for model, (label, family, key) in MODELS.items():
    for budget, word in [(1, 'One'), (8, 'Eight')]:
        first, last = aggregate(1, model, budget), aggregate(10, model, budget)
        table.append(f"{label} & {budget} & ${first['crps']:.4f} \\pm {first['crps_student_seed_sd']:.4f}$ & ${last['crps']:.4f} \\pm {last['crps_student_seed_sd']:.4f}$ "+r'\\')
        for horizon, row, name in [(1, first, 'Interval'), (10, last, 'Roll')]:
            for metric, suffix, scale, places in [
                    ('crps', 'CRPS', 1, 5), ('crps_student_seed_sd', 'SD', 1, 5),
                    ('coverage90', 'Coverage', 100, 2), ('rms_spread_skill', 'Ratio', 1, 3)]:
                macros.append('\\newcommand{\\rep'+key+word+name+suffix+'}{'+f'{row[metric]*scale:.{places}f}'+'}')
            comparison = next(r for r in comparisons[horizon] if r['model'] == model and
                              r['num_steps'] == budget and r['lead'] == horizon and r['metric'] == 'crps')
            assert comparison['reference_steps'] == 64
            for metric, suffix, places in [
                    ('difference_student_mean_minus_reference', 'Difference', 5),
                    ('difference_data_cluster_ci95_lower', 'Lower', 5),
                    ('difference_data_cluster_ci95_upper', 'Upper', 5),
                    ('relative_change_percent', 'Percent', 2)]:
                macros.append('\\newcommand{\\rep'+key+word+name+suffix+'}{'+f'{comparison[metric]:.{places}f}'+'}')
            reference = next(r for r in teacher_rows[horizon] if
                             r['model'] == comparison['reference_model'] and
                             int(r['num_steps']) == 64 and int(r['lead']) == horizon)
            student_cost = sum(r['seconds'] for r in reports[horizon] if
                               r['model'] == model and r['num_steps'] == budget)
            reference_cost = float(reference['seconds' if horizon == 1 else 'cumulative_seconds'])
            macros.append('\\newcommand{\\rep'+key+word+name+'Speed}{'+f'{reference_cost/student_cost:.2f}'+'}')
            paired.append(f"{label} & {budget} & {horizon} & {comparison['difference_student_mean_minus_reference']:.5f} [{comparison['difference_data_cluster_ci95_lower']:.5f}, {comparison['difference_data_cluster_ci95_upper']:.5f}] & {comparison['relative_change_percent']:.2f} & {reference_cost/student_cost:.2f}"+r'\\')
        for seed in (0, 1, 2):
            a, b = individual(1, model, budget, seed), individual(10, model, budget, seed)
            detail.append(f"{label} & {budget} & {seed} & {float(a['crps']):.5f} & {float(b['crps']):.5f} & {float(b['rms_spread']):.4f} & {100*float(b['coverage90']):.2f} "+r'\\')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
detail += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
paired += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
ito_one = aggregate(1, 'Föllmer Itô map', 1)
si_sixteen = next(r for r in teacher_rows[1] if r['model'] == 'Föllmer SI' and int(r['num_steps']) == 16)
macros += [
    r'\newcommand{\repItoOneModerateGain}{'+f"{100*(1-ito_one['crps']/float(si_sixteen['crps'])):.2f}"+'}',
    r'\newcommand{\repItoOneModerateSpeed}{'+f"{float(si_sixteen['seconds'])/ito_one['seconds']:.2f}"+'}']

cost = [r'\begin{table}[t]',r'\caption{Recorded training-and-validation time for each complete two-stage student replication. These times exclude model/data initialization and original teacher pretraining.}',
        r'\label{tab:replication-time}',r'\centering',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Model & Seed & Stage 1 (s) & Stage 2 (s) \\',r'\midrule']
for model, (label, family, _) in MODELS.items():
    for seed in (0, 1, 2):
        times = []
        for stage, updates in [('psd', 2000), ('rollout', 500)]:
            history = read(RESULTS / f'training/{family}_{stage}_seed{seed}/history.json')
            assert history[-1]['step'] == updates
            times.append(history[-1]['elapsed_seconds'])
        cost.append(f'{label} & {seed} & {times[0]:.1f} & {times[1]:.1f} '+r'\\')
cost += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']

# Write only after every completeness and training-history check has passed.
OUTPUT.mkdir(exist_ok=True)
for filename, lines in [('replication_table.tex', table),
                        ('replication_individuals.tex', detail),
                        ('replication_claims.tex', macros),
                        ('replication_paired.tex', paired),
                        ('replication_training_cost.tex', cost)]:
    (OUTPUT / filename).write_text('\n'.join(lines).rstrip()+'\n')
(OUTPUT / 'replication_sources.json').write_text(json.dumps(sources, indent=2)+'\n')
print('Built complete three-student replication tables and point-estimate macros.')
