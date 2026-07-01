# `paper_experiments/` — experiment definitions for the paper sync

This folder defines **all experiments** for *"A General Observation-Interpolant
Method for Data Assimilation with Flow-Based Generative Models."* It is the home
of the canonical results schema and the LaTeX table emitter; every case driver
conforms to the schema here, and `sections/results.tex` is filled from the
snippets emitted here.

Source of truth for *what* to run: `results/README.md` (the reduced grid), with
`RUN_STATUS.md` for status and `DESIGN_NOTES.md` for design rationale. The original
pre-restructure experiment spec and gap analysis are archived at
`../archive/paper_new/{EXPERIMENTS_IMPLEMENTATION_SPEC,GAP_ANALYSIS}.md`.

> **Status (2026-06-29).** Fully implemented. The schema, seeding, aggregation,
> and the table emitter (`make_tables.py`) work; all three case drivers
> (`cases/*/driver.py`) are implemented with all methods wired. **Analytical case
> is DONE** (real numbers, all 11 methods, 3 figures). **NS and urban headline
> runs are PENDING the GPU** — their manuscript table cells are `\tbd`. See
> `RUN_STATUS.md` (status) and `results/README.md` (the reduced-grid layout) for
> what is done vs pending.

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
(and `DESIGN_NOTES.md` for the checkpoint/likelihood-mode/cost details).

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

## How to reproduce a table

```bash
# 0. (prove the pipeline now — no real numbers needed)
python paper_experiments/make_tables.py --demo
#    -> writes generated/demo_results.csv and every generated/tab_*.tex

# 1. run a case for all its methods/scenarios over the fixed seed list
#    (works once the case driver is implemented — see status note above)
python paper_experiments/run.py --multirun \
    case=navier_stokes method=si_sde,dm_sde,fm_ode,flowdas scenario=superres_32,sparse_5
#    -> results/navier_stokes_results.csv  (aggregated mean +/- std over seeds)

# 2. emit the LaTeX snippets
python paper_experiments/make_tables.py \
    --results results/navier_stokes_results.csv \
    --out paper_experiments/generated
#    -> generated/tab_ns_accuracy.tex, tab_ns_calibration_cost.tex, tab_ablation.tex

# 3. in results.tex, \input the snippet inside the matching tabular.
```

Run `python paper_experiments/run.py` with `.venv/bin/python` or `uv run python`.

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
