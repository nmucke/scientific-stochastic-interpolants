# Case 3 — Urban airflow CFD over a building array (uDALES)

Applied, multi-variable realism (spec Section 6): incompressible airflow + heat
transport over an array of buildings (bluff bodies). The state is **4 channels —
`(u, v, w, thl)`**: the three velocity components plus the (potential)
temperature `thl` (~295 K). Boundary layers, wakes, and recirculation behind
obstacles — anisotropic, spatially heterogeneous statistics.

> **Channel-count note:** the trained UNet has `in_channels = out_channels = 4`,
> and `scisi.data.datasets.UDalesDataset` stacks `[u, v, w, thl]` (NOT 3
> channels). The dataset's `num_channels = 5` attribute is a hardcoded leftover —
> only 4 channels are actually loaded, so the driver derives the channel count
> from the loaded test trajectory (`sample["x"].shape[0] == 4`), not that
> attribute. Accuracy is reported per-variable: **velocity RMSE** over `(u, v, w)`
> and **temperature RMSE** over `thl`.

## Data (author-provided — archive/PROJECT_HANDOFF.md §B.4)

The uDALES runs are **author-provided** under `data/udales/` (`sim_*.nc`,
~179 sims × 250 timesteps each, 128×128) + `data/udales/mask.npz` (solid-cell
mask, `mask==1` fluid / `mask==0` building interior). The active data + mask paths
come from each checkpoint's baked-in `test_data` dataset config; the
`data_path` / `mask_path` keys in `configs/case/urban.yaml` are informational.
Normalisation is the per-channel mean/std `preprocesser` stored in the checkpoint
`config.yaml`. **Solid-cell masking** is applied consistently: the model receives
the mask as `field_cond`; the sparse obs operator places sensors only in fluid
cells; and every metric excludes solid cells.

## Observation scenarios

**Sparse `5%` and sparse `1.5625%` ONLY** (random observation in fluid cells).
**NO super-resolution.** Sparse sensors are restricted to **fluid** cells (no
sensor inside a building). Observe all 4 channels.

## Deliverables

- **`tab:urban_accuracy`** — per-variable RMSE (velocity, temperature) ×
  {sparse 5%, sparse 1.5625%}. **NO KL** (no ground-truth posterior) and **NO
  energy/enstrophy metrics**. → `generated/tab_urban_accuracy.tex`.
- **`tab:urban_calibration_cost`** — CRPS, `|1-spread/skill|`, NFE, s/step.
  → `generated/tab_urban_calibration_cost.tex`.
- **`fig:urban_fields`** — geometry + truth/prior/posterior for velocity and
  temperature; mark building footprints and sensor locations. *(Figure: TODO.)*

Calibration is assessed by **spread–skill + split CRPS** (observed/unobserved),
scored against the ground-truth STATE. **Generative-only** (no conventional
baselines — urban has no true forward solver here). Method lineup = the NS
generative set: our three (SI-SDE, DM-SDE, FM-ODE) + FlowDAS, Guided FM (FIG),
Guided FM (OT-ODE), D-Flow SGLD, SDA, SURGE. `URBAN_METHODS` excludes EnKF /
LETKF / PF / EnSF and the dropped legacy Guided FM / Guided diffusion.

**Status (2026-06-29): PREPARED, NOT yet run** — the headline runs are pending the
GPU (all `\tbd` cells in the urban manuscript tables come from these).

## Configuration (`configs/case/urban.yaml`)

- `checkpoints.si_run: stochastic_interpolant_big_gamma1`
- `checkpoints.fm_run: flow_matching_big`
- `checkpoints.diffusion_from_fm: true` (DM prior built from the FM model)
- `test_sample_indices: [1,2,3,4,5]`
- Data: `data/udales/*.nc` + `data/udales/mask.npz` (test split = sims 170–178).
  Models under `checkpoints/udales/`.

## Tidy rows this case emits

`case=urban`; `scenario ∈ {sparse 5%, sparse 1.5625%}`; `metric ∈ {rmse_velocity,
rmse_temperature, crps, crps_observed, crps_unobserved, spread_skill}` plus
`nfe`/`seconds`.

## Implementation

- `driver.py::UrbanRunner` — generative-only method/scenario grid; mirrors
  `NavierStokesRunner`. Delegates to `_urban_pipeline`.
- `_urban_pipeline.py` — reuses the NS loader / `build_posterior` /
  `run_assimilation` / `prepare_truth_and_obs` seams; adds the fluid keep-mask,
  the fluid-restricted sparse sensor draw, and the urban `compute_metrics`
  (per-variable RMSE + split CRPS + spread–skill, no KL).

## Open TODOs

- Per-variable physical-space observation noise `sigma` (currently a single
  scalar `R = sigma^2 I` in normalised space; see `urban.yaml::variance`).
- `fig:urban_fields` generation.

## Forecast-residual correction (optional, default OFF)

Shared switches, variant tags, diagnostics, cost and run order:
[`cases/navier_stokes/README.md`](../navier_stokes/README.md#forecast-residual-correction-optional-default-off).
Urban specifics:

* **Only assimilated steps are corrected** (`assimilate_every=3` in the grid: 49
  of 145 steps); the free-running steps call the prior directly.
* **Localisation**: Gaspari-Cohn on the 128^2 grid, non-periodic by default
  (`periodic: [false, false]`; the slice edges are not continuous in the data),
  the four channels at a site at distance 0, optional cross-variable taper
  `channel_taper_offdiag` (1 = keep all cross-covariances). The taper is zero on
  solid cells, so the gain never moves state inside buildings. Distances are
  straight lines and do not go around buildings. Default radius 8 m, **untuned**:
  tune on validation rows (test rows 0 and 7 are scored by nothing; 6 is the
  tuning row, 8 the foreign initial condition).
* **Constraint** (`post_update`, `_urban_constraints.IncrementProjection`). The
  state is a horizontal slice of a 3D LES, so it is NOT horizontally
  divergence-free (du/dx + dv/dy = -dw/dz; on sim_170 the RMS horizontal
  divergence is about as large as the derivative terms). The projection
  therefore acts on the INCREMENT x'' - x': in physical units it is replaced by
  its orthogonal projection onto {zero in solid cells, zero discrete horizontal
  divergence} (masked Poisson problem, batched CG). Then x_out has exactly the
  guided sample's divergence and the obstacle cells keep the guided values;
  w and thl are only masked. u varies along W and v along H (checked against
  the walls in the data). Stencil `central` (default) or `forward`.
  Switches: `project_divergence`, `mask_obstacles`, `u_axis`, `v_axis`,
  `divergence_stencil`, `cg_tol`, `cg_max_iter`.
* **Caveat.** "No added horizontal divergence" is stronger than the physics
  requires: a physical increment can change du/dx + dv/dy together with dw/dz.
  On random smooth test increments the projection removed 65-70% of the u, v
  increment. Compare `project_divergence: true` and `false` on validation data.
  The diagnostic `projection_change_rel` shows how much it removes.
* **Diagnostics** (`diagnostics: true`), per corrected step: `div_rms_guided`, `div_rms_corrected_raw`,
  `div_rms_out` (horizontal divergence of the states, 1/s, interior fluid cells),
  `increment_div_rms_raw` / `_out`, `solid_increment_rms_raw` / `_out`,
  `projection_change_rel`, `cg_iterations`, `cg_relative_residual`, and the Q share.
* Example: `FC=measure TRAJ=7 SAVE_TRAJ=7 GRPS=ours_jacfree bash paper_experiments/run_urban_grid.sh`, then
  `FC=on FC_SEED=0 TRAJ=7 SAVE_TRAJ=7 GRPS=ours_jacfree bash paper_experiments/run_urban_grid.sh`.
