"""Render the recorded fixed-history NS compression diagnostic for the paper."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

paper = Path(__file__).resolve().parent
source = paper.parent / 'results/ns_path_compression/results.json'
data = json.loads(source.read_text())
assert data['split'] == 'validation'
rows = data['results']
fig, ax = plt.subplots(figsize=(3.3, 2.35), layout='constrained')
for index, row in enumerate(rows):
    values = row['per_group_within_variance']
    offsets = np.linspace(-.10, .10, len(values))
    ax.scatter(index + offsets, values, color='#9a9a9a', s=12, alpha=.8,
               label='Coarse-path groups' if index == 0 else None, zorder=2)
ax.errorbar(range(len(rows)), [r['within_coarse_path_variance'] for r in rows],
            yerr=[r['within_variance_standard_error_across_groups'] for r in rows],
            color='#0072B2', fmt='o-', lw=1.2, markersize=4, capsize=3,
            label='Mean ± 1 standard error', zorder=3)
ax.set_xticks(range(len(rows)), [str(r['num_path_features']) for r in rows])
ax.set_yscale('log')
ax.set_xlabel('Brownian features K')
ax.set_ylabel('Omitted endpoint variance')
ax.grid(axis='y', which='major', alpha=.25)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc='lower left')
ax.tick_params(labelsize=8)
(paper / 'figures').mkdir(exist_ok=True)
fig.savefig(paper / 'figures/ns_compression.pdf', metadata={'Title': 'Fixed-history Brownian compression diagnostic'})
fig.savefig(paper / 'figures/ns_compression.png', dpi=200)
plt.close(fig)
(paper/'generated/compression_source.json').write_text(json.dumps({
    'source':str(source.relative_to(paper.parent)),
    'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'checkpoint_sha256':data['checkpoint_sha256'],
    'figure':'figures/ns_compression.pdf',
    'error_bars':'One standard error across 8 coarse Brownian groups; not across pixels or histories',
}, indent=2)+'\n')
print('Rendered fixed-history compression figure from recorded results.')
