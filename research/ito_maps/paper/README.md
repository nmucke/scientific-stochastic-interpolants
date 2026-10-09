# Building the manuscript

From the repository root:

```sh
make -C research/ito_maps/paper
```

This runs `latexmk` with `TEXINPUTS=.:template:` and `BSTINPUTS=.:template:` so
the unmodified official ICML 2026 style files under `template/` are found. Do not
run bare `latexmk` without these variables; it corrupts the build cache. To
recover, run from this directory:

```sh
TEXINPUTS=.:template: BSTINPUTS=.:template: latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Current state

- Scope: stochastically forced Navier--Stokes forecasting only.
- Method, theory, experimental protocol, metric definitions, related work and
  the appendices on training modes and differences from the original Itô-map
  construction are complete.
- All Navier--Stokes results are pending. Every result-dependent statement,
  table and figure is a red `\pending{...}` placeholder (macros `\pending` and
  `\pendingbox` in `macros.tex`). Floats keep their captions and
  labels, and the placeholder lists the planned rows/columns or panels, so the
  new results can be dropped in without restructuring. `notes/results_plan.md` lists
  every placeholder location.
- The build needs no datasets, checkpoints, results or Python: there are no
  generated tables or figures at present.

## Layout

```
main.tex            preamble, title block, abstract; \input's everything below in order
macros.tex          theorem environments, notation, \pending / \pendingbox placeholders
references.bib
sections/           main text, one file per section
  introduction  setup (incl. Algorithm 1)  ito_maps  related_work
  experiments (NS protocol + main results table)  limitations  impact_statement
appendix/           one file per appendix section, in compiled order
  implementation (incl. ideal identities, compute)  training_modes
  ito_maps_differences  ns_experimental_details (\input's ns_results)
  replications  development_ablation  metrics  seed0_uncertainty
  qualitative_forecast  compression_diagnostic  linear_sde_compression
  long_horizon  draft_status
template/           unmodified official ICML 2026 kit
notes/              planning notes, not compiled: guidelines, results_plan, reviewer_risks
```

## Overleaf

`make -C research/ito_maps/paper overleaf` writes `overleaf.zip` (git-ignored):
the sources plus the used ICML style files flattened into the root, test-compiled
without `TEXINPUTS`. Upload it via *New Project → Upload Project*.

## Adding results later

Write a `build_<name>.py` that reads the new results and writes into
`generated/` or `figures/`, add it to `ARTIFACTS` with a rule in the
`Makefile`, and replace the corresponding `\pendingbox{...}` by the
`\input`/`\includegraphics`.

## Archive

Initial-stage results were moved to `../archive/results/` and are not used. The
manuscript assets derived from them, and the dropped second application, were
moved to `../archive/paper/`: the former `build_*.py` scripts, `generated/`,
`figures/`, the corresponding `.tex` sources, the old long-horizon integration
plan, and `pre_placeholder_tex/` with the pre-placeholder versions of the current
sources. `archive/` is ignored by Git.
