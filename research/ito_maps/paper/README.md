# Building the manuscript from a checkout

From this directory:

```sh
make all PYTHON=/path/to/the/project/python
```

Use the project Python environment (NumPy, Matplotlib, Pillow) and a TeX
installation with `latexmk`/`pdflatex`. The official ICML style files are retained
under `template/`. The default Python path uses the repository's `.venv` when
available. Datasets, model checkpoints, metric weights and forecast NPZ arrays
are not needed for this manuscript build.

Most figures and tables regenerate from tracked JSON/CSV reports. The video
setup uses `research/video_forecasting/notes/data_summary.json`, whose dataset
manifest identity must match the frozen protocol. If local dataset manifests
exist, their hashes and split counts are additionally verified. No raw dataset
manifest is required when building from a checkout.

The audited `figures/video_results/` bundle is deliberately retained in Git:
18 compact TeX/CSV/PNG/PDF/provenance files, about 1.6 MB. It includes the fixed
main and appendix frame sequences, FVD/cost figures, formatted tables, unrounded
summary copies and `paper_results_manifest.json`. Narrow `.gitignore` exceptions
apply only to these file types in this exact directory. Other generated figures,
raw NPZ arrays, checkpoints and datasets remain ignored. `make all` requires the
bundle but never reruns its data-dependent figure generator.

To regenerate that bundle after a new audited evaluation, use the explicit
`make video-results-artifacts` target. This requires both complete strict video
reports and their saved raw fixed-case NPZ arrays. It preserves the predeclared
case/member/budget selections. Review the regenerated assets and stage the whole
bundle together with its snippets/provenance:

```sh
git add research/ito_maps/paper/figures/video_results \
        research/ito_maps/paper/generated/video_results.tex \
        research/ito_maps/paper/generated/video_main_results.tex
```

Run that `git add` from the repository root. No force-add is needed. Normal
compilation and full scientific regeneration are separate operations: the
retained images support a standalone checkout build without repeating inference.

An isolated `git checkout-index` copy plus the pending source changes and exact
retained bundle passed `make all` on 2026-10-07 with no dataset/checkpoint/NPZ
files present. All 18 retained files remained byte-identical, and extracted PDF
text matched the reviewed working-tree PDF exactly. See
`../reviews/checkout_paper_build.json` for source and asset hashes.
