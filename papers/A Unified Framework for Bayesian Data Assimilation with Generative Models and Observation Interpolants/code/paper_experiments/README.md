# Paper experiments

The active experiment tree is
`papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments/`.
Its results and generated figures live here. Run the commands below from the
repository root; quote `$EXPERIMENT_DIR` because the manuscript directory contains
spaces:

```bash
EXPERIMENT_DIR="$PWD/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
PY="$PWD/.venv/bin/python"
"$PY" "$EXPERIMENT_DIR/run.py" --help
```

Shell launchers locate the repository-root `.venv` automatically. Python entry
points work from the repository root, this experiment directory, or another
working directory when invoked with the correct interpreter and script path.
Checkpoint data and mask paths are anchored to the repository; default results,
figures, and Hydra logs are anchored to this experiment directory. Explicit
relative Python output overrides are resolved against the launch directory.
Grid scripts run from the repository root, so relative `ROOT` overrides are
repository-relative; use an absolute `ROOT` for an output directory elsewhere.

This folder defines **all experiments** for *"A General Observation-Interpolant
Method for Data Assimilation with Flow-Based Generative Models."* It is the home
of the canonical results schema and the LaTeX table emitter; every case driver
conforms to the schema here, and `sections/results.tex` is filled from the
snippets emitted here.

For historical grid and run notes, see `results/README.md` and `RUN_STATUS.md`.
The current grid scripts are the source of truth for run defaults.

---

## Folder layout

```
paper_experiments/
  README.md                  # this file
  results_schema.py          # canonical tidy record + writer/loader + enums
  make_tables.py             # tidy results -> LaTeX snippets for results.tex
  run.py                     # hydra entrypoint: run a case driver -> tidy file
  configs/                   # mirrors paper/configs conventions
    benchmark.yaml           #   defaults: case + method + scenario
    case/                    #   analytical, navier_stokes, urban
    method/                  #   si_sde, dm_sde, fm_ode + baselines
    scenario/                #   superres_32/16, sparse_5/1p5
  cases/
    analytical/              # Case 1 driver + README (closed-form posterior)
    navier_stokes/           # Case 2 driver + README (learned prior, main bench)
    urban/                   # Case 3 driver + README (uDALES, multi-variable)
  common/
    seeding.py               # fixed seed list + per-(scenario,test,seed) seeds
    aggregation.py           # mean +/- std over seeds (reproducibility Sec 9)
    runner.py                # ExperimentRunner base class (the stable seam)
  generated/                 # emitted .tex snippets (the paper \input's these)
  results/                   # tidy .csv/.jsonl results files (gitignore-able)
```

---

## The three cases

| Case | Folder | What it tests | Key tables / figures |
|---|---|---|---|
| 1. Analytical linear–Gaussian | `cases/analytical` | correctness vs the closed-form posterior, no training | `tab:analytical_results`, `fig:analytical_panels` |
| 2. Stochastic Navier–Stokes | `cases/navier_stokes` | learned prior, high-dim chaotic, the main benchmark | `tab:ns_accuracy`, `tab:ns_calibration_cost`, `tab:ablation`, `fig:ns_trajectories`, `fig:ns_diagnostics` |
| 3. Urban airflow (uDALES) | `cases/urban` | multi-variable applied realism, solid obstacles; **generative-only, sparse-only, no KL/energy** | `tab:urban_accuracy`, `tab:urban_calibration_cost`, `fig:urban_fields` |

Each case folder has its own `README.md` with the deliverables and TODO seams.

**Trained priors (Case 2, GPU machine).** The Navier–Stokes case loads two trained priors —
SI at `checkpoints/stochastic_navier_stokes/stochastic_interpolant_small/` and FM at
`checkpoints/stochastic_navier_stokes/flow_matching/` (each with `model.pth` + `config.yaml`),
named by the `checkpoints.si_run` / `checkpoints.fm_run` keys in
`configs/case/navier_stokes.yaml`. On the laptop (no `model.pth`) the driver runs with random
weights and a loud warning — smoke-scale only. See `RUN_STATUS.md` for the full-scale run
and the case YAMLs for checkpoint and likelihood settings.

## Methods (canonical labels — `results_schema.Method`)

Our three samplers (one shared unified loop, differing only in `g_tau`, `w_tau`,
the source, and whether a Brownian increment is added):

- **Ours (SI-SDE)** — `si_sde.yaml`
- **Ours (FM-ODE)** — `fm_ode.yaml`
- **Ours (DM-SDE)** — `dm_sde.yaml` (shown in the paper as **DM-SDE**, a
  diffusion-model-style SDE on the FM prior)

Baselines, grouped by (prior, sampler) (generative ones share the trained prior):

- **SI prior + SDE:** FlowDAS
- **Flow-matching prior + ODE:** Guided FM (FIG), Guided FM (OT-ODE), D-Flow SGLD
- **Diffusion-model prior + SDE:** SDA, SURGE — both use the DM prior built from
  the FM model (`DenoiseDiffusionModel.from_flow_matching`)
- **Classical (Navier–Stokes ONLY — true solver):** EnKF (E=1000 non-localized =
  ground-truth posterior / KL reference; E=64 localized = baseline), LETKF,
  particle filter, ensemble score filter

**DROPPED from the paper:** the legacy "Guided FM" (one-step DPS-on-flow) and
"Guided diffusion" (DPS). Their `Method` enum entries are kept for back-compat but
are removed from the run registries and `make_tables`.

The exact strings are the `Method` enum *values*; `make_tables.py` keys every
row off them. Methods are cited in prose + an appendix "Method descriptions"
section (no per-row `\cite`).

## Scenarios (canonical labels — `results_schema.Scenario`)

- `sparse 5%` and `sparse 1.5625%` — sparse sensors (the NS + urban scenarios)
- `32^2->128^2` and `16^2->128^2` — super-resolution (block-average operator);
  defined in code but **urban uses sparse only**
- `analytical` — Case 1's single joint scenario

## Metrics (canonical keys — `results_schema.Metric`)

`rmse`, `rmse_velocity`, `rmse_temperature`, `energy_spec_rmse` (point accuracy);
`crps`, `spread_skill` (calibration — report `|1 - spread/skill|`); `kl_points`,
`sliced_w2` (distributional fidelity); `nfe`, `seconds` (cost). Definitions:
spec Section 3.

---

## The tidy results schema (`results_schema.py`)

Every case driver emits rows with **exactly** the spec columns (Section 8):

```
case, method, scenario, metric, value, std, E, M, seed, NFE, seconds
```

One metric value per row (long format). `value` is the mean over seeds when
`seed == -1` (aggregated); `std` is the across-seed standard deviation. `E`
(ensemble size) and `M` (pseudo-time steps) are the sampler settings; `NFE` and
`seconds` are cost (also expressible as their own metric rows). Writer/loader
support `.csv` and `.jsonl`, append-only, and round-trip.

## Results → LaTeX flow

```
case driver (cases/*/driver.py)         # per-seed ResultRecord rows
        │   ExperimentRunner.run()
        ▼
common/aggregation.aggregate_over_seeds  # mean +/- std over the fixed seed list
        ▼
results/<case>_results.csv               # tidy file (the spec's "one file per case")
        │   make_tables.py
        ▼
generated/tab_*.tex                       # one snippet per labelled table
        │   \input
        ▼
manuscript/sections/results.tex
```

`make_tables.py` maps a tidy `(method, scenario, metric)` triple to a specific
LaTeX cell. The mapping (which columns each table has) is declarative in
`make_tables.TABLE_SPECS`:

| Tidy rows | results.tex label | Columns |
|---|---|---|
| `kl_points`, `sliced_w2` @ `analytical` | `tab:analytical_results` | KL, Sliced-W2 |
| NS `rmse` / `energy_spec_rmse` / `kl_points` × {`32^2->128^2`,`5%`} | `tab:ns_accuracy` | 6 cells |
| NS `crps` / `spread_skill` × scen + `nfe`/`seconds` | `tab:ns_calibration_cost` | 6 cells |
| urban `rmse_velocity`/`rmse_temperature`/`kl_points` × scen | `tab:urban_accuracy` | 6 cells |
| urban `crps`/`spread_skill` × scen + cost | `tab:urban_calibration_cost` | 6 cells |
| NS ablation tags (`ablation:*`) on DM-SDE | `tab:ablation` | RMSE, CRPS, Spread–skill |

Each emitted snippet is the `tabular` **body** (the data rows plus the
`\midrule` that separates "ours" from the baselines), so it drops straight into
the matching `\begin{tabular}` in `results.tex` between the header `\toprule`
block and the closing `\bottomrule`. Figures (`fig:*`) are emitted by separate
plotting code (a per-case TODO) and inserted by replacing each `\figbox{...}`
with `\includegraphics`, per spec Section 8.

---

### Current per-case pipeline (reduced grid)

Each case is **run → aggregate → make figures**, all reading/writing
`results/<case>/`:

```bash
# After setting EXPERIMENT_DIR and PY above, from the repository root:
bash "$EXPERIMENT_DIR/run_analytical_grid.sh"
"$PY" "$EXPERIMENT_DIR/aggregate_analytical.py"
"$PY" "$EXPERIMENT_DIR/make_analytical_figures.py"

bash "$EXPERIMENT_DIR/run_ns_grid.sh"
"$PY" "$EXPERIMENT_DIR/aggregate_ns.py"
"$PY" "$EXPERIMENT_DIR/make_ns_figures.py"

bash "$EXPERIMENT_DIR/run_urban_grid.sh"
"$PY" "$EXPERIMENT_DIR/aggregate_urban.py"
"$PY" "$EXPERIMENT_DIR/make_urban_figures.py"

# Inspect GPU timing commands without launching runs:
"$PY" "$EXPERIMENT_DIR/measure_timing.py" --dry-run --limit 1
"$PY" "$EXPERIMENT_DIR/rerun_timing.py" --dry-run --limit 1
```

`aggregate_<case>.py` reduces the per-cell files ACROSS trajectories (trajectory
keyed by the `traj<N>` filename token): `all.csv` holds the scalar metrics (mean ±
std over trajectories, time already averaged in-run) and, for NS/urban,
`per_step.csv` holds the per-assimilation-step curves (mean ± std over trajectories
at each step). `make_tables.py` consumes `all.csv` via `--results`. Shared reduction
lives in `common/aggregate_lib.py`; the three per-case scripts are thin wrappers
(they replaced the retired `aggregate_grid.py` / `aggregate_multitraj.py`).

`make_<case>_figures.py` produces, from those aggregates + the traj1 states:

* **metric-vs-M** (`<case>_<metric>_vs_M`) — from `all.csv`;
* **metric-vs-step** (`<case>_<metric>_vs_step`, NS/urban) — from `per_step.csv`;
* **state field maps** (`<case>_states_<scenario>`, traj1) — truth / posterior mean
  / |error| / spread at the final step, read straight from `states/traj1/*.npz`.

Generated figures (`.pdf` and `.png`) are written under this experiment tree's
`figures/<case>/`. They are not copied into a manuscript figure directory
automatically. Use the figure scripts' `--out` option to write an additional
explicit destination when needed.

> **Trajectory note:** `run_{ns,urban}_grid.sh` pass `+test_index=$N` so trajectory
> `N` uses test sample `N` (`test_sample_indices=[1..5]`); without it every "traj"
> would rerun the same sample and the trajectory averaging would be a no-op.

Use `"$PY" "$EXPERIMENT_DIR/run.py"` for a direct case run.

---

## Binding author decisions (GAP_ANALYSIS Section 6)

These constrain the case drivers and are baked into the configs here:

- **SI schedule: quadratic-β (`β=t²`) is kept; the paper is not changed.** Every
  schedule-derived quantity (`a_tau`, `A_tau`, source moments, `G_tau`) must be
  implemented **generally in `α, β, γ` and their derivatives**, never hard-coded
  to rectified flow.
- **Multiplicative gain `G_tau` was DROPPED** (it didn't improve accuracy — see
  `DESIGN_NOTES.md` §4). Accuracy comes from inflating the covariance, not the
  gain. The ablation is now a **covariance axis** (`inflated` / `inflated_shared`
  vs isotropic Jacobian-free); `dps_full`/`_apply_gain` are kept off-by-default.
- **uDALES data is author-provided** (`.nc` + `mask.npz`); no in-repo CFD
  generator. Case 3 (urban) is generative-only, sparse-only, no KL/energy.
- **Final baseline lineup (2026-06-29):** FlowDAS, Guided FM (FIG), Guided FM
  (OT-ODE), D-Flow SGLD, SDA, SURGE + the classical filters (NS only). The legacy
  Guided FM / Guided diffusion (DPS) baselines are dropped from the paper.

## TODO seams (where the rebuilt `src/scisi` plugs in)

- `common/runner.py::ExperimentRunner.evaluate` — the stable contract; case
  drivers implement it.
- `cases/analytical/driver.py` — GAP E4; analytic SI drift + three samplers; KL /
  sliced-W2 via `analytical_utils`.
- `cases/navier_stokes/driver.py` — GAP E1 (block-average obs op), E4 (3
  samplers), E7/E9/E10/E11 (metrics), E12 (ablation knobs).
- `cases/urban/driver.py` — GAP E2 (author data + solid-cell masking).
- Method configs reference `src/scisi` `_target_`s that are mid-rebuild
  (FM `.score` is GAP L1; DM-SDE posterior is GAP P3; several baseline targets do
  not exist yet — marked `TODO(E5)`).
