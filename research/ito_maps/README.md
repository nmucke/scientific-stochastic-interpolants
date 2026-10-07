# Itô maps for probabilistic forecasting

Working branch: `research/ito-map-forecasting`.
Started: 2026-10-07. Lead: root research agent.

## Objective and research discipline
Adapt conditional Föllmer forecasting (arXiv:2403.13724) to Brownian-path-conditioned Itô maps (arXiv:2606.11156), with deterministic flow-map matching (arXiv:2406.07507) as a comparator. Reuse the repository architecture, datasets, preprocessing, and training conventions. Test whether comparable forecast distributions can be obtained with fewer network evaluations and lower latency. This is a hypothesis; negative or inconclusive results must also be reported.

Research and a first manuscript draft precede implementation. Keep exact-vs-discretized constructions distinct, preserve independent trajectory splits, select checkpoints on validation only, and never invent results. Existing teacher checkpoint training histories are not fully available and should be disclosed.

## Layout
- `notes/`: mathematical derivations, design decisions, literature and ICML requirements.
- `sources/`: downloaded primary papers and extracted source material.
- `paper/`: LaTeX manuscript, bibliography, official template.
- `experiments/`: reproducible experiment scripts and manifests.
- `results/`: actual measurements and generated figures.

## Confirmed environment
- Python 3.13.9, PyTorch 2.9.1+cu128; project `.venv` works.
- One RTX 3090 (24 GiB); use a single GPU worker at a time.
- Data: `data/stochastic_navier_stokes/data.npz`, key `state`, float32 `(200,100,128,128)`.
- Existing split: train trajectories 0–159, validation 160–179, test 180–199.
- History length 5; normalized field scale 3.09969 from existing configs.
- Existing teachers: `checkpoints/stochastic_navier_stokes/{stochastic_interpolant,stochastic_interpolant_small,flow_matching}/`.
- `pdflatex`, `latexmk`, and `pdftotext` available.

## Active responsibilities
- root: orchestration, experiment protocol, integration, tests and experiments.
- methodology: Itô-map paper, conditional derivation and correctness.
- flow_research: flow-map paper and repository integration design.
- paper: current ICML instructions/template and manuscript.

## Status
Main literature review and first manuscript draft were completed before implementation. The manuscript now builds with the official ICML 2026 template; next-cycle requirements remain to be rechecked. Conditional Itô/flow map classes, Brownian bridge features, trainer, entry points, configs, and tests are implemented. Initial student GPU training and validation-only development comparisons are in progress. No held-out-test superiority claim has been established.

## Handover reading order
1. `notes/methodology.md`: conditional derivation, path construction, and scope limits.
2. `notes/flow_maps_and_code_design.md`: deterministic map connection and integration design.
3. `notes/code_equation_audit.md`: current code-to-math mapping, compressed PSD caveat, scoring/timing boundaries.
4. `notes/experiment_protocol.md`: trajectory splits, baseline and validation/test discipline.
5. `notes/navier_stokes_provenance.md`: what is known and still unverified about physical data generation.
6. `notes/broader_literature.md`: nearest prior work and defensible novelty boundary.
7. `paper/guidelines.md`, then `paper/main.tex`: venue requirements and current manuscript.
8. `experiments/train_psd_pilots.sh`, run-specific configs/checkpoints, and saved results: inspect actual executed commands and completed updates before reporting anything.

Analytical compression diagnostic: `experiments/brownian_compression.py` with recorded outputs in `experiments/results/brownian_compression/`. It checks finite-feature conditional projection for a linear SDE; its analytic and Monte Carlo values are not Navier–Stokes forecast results. Development baseline measurements live under `results/validation_baselines/`; smoke outputs under `results/smoke/` prove execution only. The root agent owns the live experiment sequence and final checkpoint selection.
