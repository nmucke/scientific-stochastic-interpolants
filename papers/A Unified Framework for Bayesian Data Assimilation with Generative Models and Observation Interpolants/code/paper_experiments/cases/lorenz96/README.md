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
`run_lorenz96_grid.sh` runs the final comparison.

The current status, the evidence and the exact commands are in `GPU_HANDOFF.md`.

## Runbook: two observation sweeps

Two sweeps share every script, selected by `OBS` / `--obs`: `linear` (tidy scenario
`lorenz96`, settings in `methods:`) and `arctan` (`lorenz96_arctan`,
`methods_arctan:`). Each has its own root `results/lorenz96/<obs>/` (reference/,
tuning/, metrics/, per_step/, per_sequence/, ensembles/, aggregated/, logs/), its own
figures `figures/lorenz96/<obs>/` and tables `generated/tab_lorenz96_<obs>{,_vs_M}.tex`.
A file of one sweep found in the other's root is refused by the aggregation.

```bash
cd "$PE"
OBS=linear STAGE=tune  ./run_lorenz96_sweep.sh   # tuning + selection, then STOP
# review results/lorenz96/linear/tuning/selected_stage{1,2}.md, paste the picks into
# methods: (methods_arctan: for arctan) of configs/case/lorenz96.yaml, approve compute
OBS=linear STAGE=final ./run_lorenz96_sweep.sh   # = run_lorenz96_grid.sh
```

Logs: `results/lorenz96/<obs>/logs/<stage>_<time>.log`, exit status in `<log>.exit`.

- `tune_lorenz96.sh` (`OBS`, `GRID=1|2`, `ONLY="cell ..."`): validation-only
  `driver.py tune` jobs. **All grids are the `GRIDS` table at its top.** M is fixed
  at 50 and never tuned (a grid naming `num_steps` is refused); the picks are reused
  at every M. Stage 2 rows name a parent cell: its stage-1 picks are passed as
  `--set`, and `{"times":[...]}` values are factors on the parent's pick. Jobs whose
  CSV (`tuning/<cell>__<keys>__s<k>.csv`) has every grid point are skipped; a partial
  CSV is moved aside and the job reruns.
- `select_lorenz96_tuning.py --obs <obs> --stage <k>`: lowest validation `sliced_w2`,
  rows with `diverged_steps > 0` ineligible; writes `tuning/selected_stage<k>.yaml`
  (per cell `tuned`, `set`, `full`) and `.md` (every grid, row counts, pick,
  runner-up gap). It never edits the config.
- `run_lorenz96_grid.sh` (`OBS`, `SEEDS="0 1 2"`, `M_VALUES="25 50 100 250"`,
  `JOBS=1`, `FORCE=0`): test references, prior quality, the generative cells
  (`GENERATIVE_CELLS`) at every M (`cell --num-steps M`, files
  `lorenz96__<cell>__M<M>__traj<s>__.csv`) and the M-independent cells
  (`M_INDEPENDENT_CELLS`) once per seed, all cells at M=50 first; `pf_reference`
  once; then `aggregate_lorenz96.py`, `make_lorenz96_figures.py` and
  `make_lorenz96_tables.py`, each with `--obs`. Done (cell, M, seed) files are
  skipped; a failed cell stops the script before aggregation.
- Lineup labels, table groups and figure styles: `lorenz96_lineup.py` (`ROWS`).
  The main table is at M=50; `tab_lorenz96_<obs>_vs_M.tex` and the
  `lorenz96_<metric>_vs_M` figures show the generative cells against M.
