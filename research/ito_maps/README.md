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
Initial literature and repository inspection in progress. No implementation or measured results yet.
