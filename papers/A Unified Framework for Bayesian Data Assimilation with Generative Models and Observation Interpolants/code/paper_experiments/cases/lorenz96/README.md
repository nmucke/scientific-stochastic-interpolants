# Case 4 — Stochastic Lorenz 96

Learned SI and FM priors with observation-interpolant posterior sampling, against
EnKF and a converged particle filter that serves as the reference posterior.

## System

Lorenz 96 on a ring of 16 sites, F = 8, RK4 with dt = 0.01. Between observations
(0.2 time units) the state moves by the deterministic flow plus additive
N(0, I) noise, so the transition kernel is exactly N(Phi(x), I). Every fourth site is
observed with variance 4. All settings live in `configs/case/lorenz96.yaml`.

## Protocol

`pf_gate.py` decides whether the particle filter can serve as ground truth;
`simulate_data.py` and `train.py` build the priors; `driver.py` computes references,
tunes on validation sequences and scores every method on the same held-out sequences;
`aggregate_lorenz96.py` and `make_lorenz96_figures.py` reduce and plot.
`run_lorenz96.sh` (`tune`, `apply`, `final`) runs all of it, per sweep.

The current status, the evidence and the exact commands are in `GPU_HANDOFF.md`.

## Runbook: two observation sweeps

Two sweeps share every script, selected by the second argument / `--obs`: `linear`
(tidy scenario `lorenz96`, settings in `methods:`) and `arctan` (`lorenz96_arctan`,
`methods_arctan:`). Each has its own root `results/lorenz96/<obs>/` (reference/,
tuning/, metrics/, per_step/, per_sequence/, ensembles/, aggregated/, logs/), its own
figures `figures/lorenz96/<obs>/` and tables `generated/tab_lorenz96_<obs>{,_vs_M}.tex`.
A file of one sweep found in the other's root is refused by the aggregation.

One script, three human-gated stages per sweep:

```bash
cd "$PE"
./run_lorenz96.sh tune  linear   # validation references, stage-1 tuning, selection,
                                 # stage-2 tuning, selection, then STOP
# review results/lorenz96/linear/tuning/selected_stage{1,2}.md
./run_lorenz96.sh apply linear   # writes the picks into methods: of configs/case/lorenz96.yaml
# approve the compute
./run_lorenz96.sh final linear   # test references, prior quality, the grid, aggregation,
                                 # figures, tables
# the same three commands with `arctan` (picks go into methods_arctan:)
```

Logs: `results/lorenz96/<obs>/logs/<stage>_<time>.log`, exit status in `<log>.exit`.
Environment: `PY`, `THREADS` (tune), `SEEDS="0 1 2"`, `M_VALUES` (default
`lorenz96_lineup.M_VALUES`), `FORCE=1` (rerun done cells), `ONLY="cell ..."` (tune).

- `tune`: the validation-only `driver.py tune` jobs of the `GRIDS` table at the top
  of `run_lorenz96.sh`. M is fixed at 50 and never tuned (the driver refuses another
  M); the picks are reused at every M. Stage-2 rows name a parent cell: its stage-1
  picks are passed as `--set`, and `{"times":[...]}` values are factors on the
  parent's pick. A job whose CSV (`tuning/<cell>__<keys>__s<k>.csv`) has every grid
  point is skipped; a partial CSV is moved aside and the job reruns.
  `select_lorenz96_tuning.py --obs <obs> --stage <k>` picks the lowest validation
  `sliced_w2` (rows with `diverged_steps > 0` ineligible) and writes
  `tuning/selected_stage<k>.yaml` (per cell `tuned`, `set`, `full`) and `.md` (every
  grid, row counts, pick, runner-up gap).
- `apply`: `select_lorenz96_tuning.py --obs <obs> apply` merges each pick's `tuned`
  keys into the cell's own one-line block of `methods:` / `methods_arctan:` (nested
  keys too: a `correction.gain_scale` pick keeps the block's `localisation` and
  `forecast_num_steps`). Only those lines change; comments and every other line are
  kept, and the result is re-read and checked before it is written. Keys a stage-2
  cell received from its parent by `--set` are not copied: the parent's own pick is
  written to the parent, which the `_fc` and SURGE-around-Ours cells inherit.
- `final`: test references, `prior-quality` (skipped when `prior_quality.json`
  exists), the generative cells (`python lorenz96_lineup.py cells generative`) at
  every M (`cell --num-steps M`, files `lorenz96__<cell>__M<M>__traj<s>__.csv`; M=50
  first, then 25, 100, 250) and the M-independent cells (`cells fixed`) once per
  seed with the M=50 block; `pf_reference` once; then `aggregate_lorenz96.py`,
  `make_lorenz96_figures.py` and `make_lorenz96_tables.py`, each with `--obs`. Done
  (cell, M, seed) files are skipped; a failed cell does not stop the others, but
  stops the script before `pf_reference` and the aggregation. Cells run one at a
  time, because concurrent cells inflate each other's `seconds`.
- Lineup labels, table groups, figure styles, the M values and the shared metric
  names: `lorenz96_lineup.py` (`ROWS`; `driver.CELLS` is derived from it). The main
  table is at M=50; `tab_lorenz96_<obs>_vs_M.tex` and the `lorenz96_<metric>_vs_M`
  figures show the generative cells against M.
- Provenance (2026-11-10, `GPU_HANDOFF.md` section 15): the SURGE cells on the SI
  prior (`surge_si_sde`, `surge_unguided_si`, `surge_flowdas`) use the exact
  Follmer-drift reward denoiser (`si_denoiser="follmer"`) under unchanged variant
  names; `sda` runs one sequence per batch; `kl_knn` uses the unique PF draws and
  does not count toward `diverged_steps`. Results of these cells, and every arctan
  `diverged_steps` / tuning eligibility, from before that date must be rerun.
