# Case 2 — Stochastic incompressible Navier–Stokes

The **main benchmark** (spec Section 5): learned SI/FM priors, high-dimensional,
chaotic, with realistic observation operators.

## System

Vorticity form on the torus `[0,2π]^2`:

```
dω + (u·∇)ω dt = ν Δω dt − α ω dt + ε dξ,   u = ∇⊥ψ,  −Δψ = ω
```

`dξ` = temporally white forcing on selected Fourier modes — the stochastic
forcing makes the transition density `p(ω^n|ω^{n-1})` a genuine distribution.
PDE constants (`ν, α, ε`, forced band, `dt`, assimilation interval, trajectory
counts) are **TODO** in `configs/case/navier_stokes.yaml` (confirm with
codebase/user).

## Protocol

Train SI (SI-SDE) and FM (DM-SDE, FM-ODE) priors with the **same** architecture,
data, and schedules. Assimilate `y^1..y^N` autoregressively at ensemble size `E`,
steps `M`. All methods run on **identical truth + observations + masks** (shared
seeds via `common/seeding.py`). `R = 0.05^2 I`.

## Deliverables

- **`tab:ns_accuracy`** — vorticity RMSE, energy-spectrum RMSE, KL-at-points ×
  {`32^2->128^2`, `5%`}. → `generated/tab_ns_accuracy.tex`.
- **`tab:ns_calibration_cost`** — CRPS, `|1-spread/skill|`, NFE, s/step × same.
  → `generated/tab_ns_calibration_cost.tex`.
- **`tab:ablation`** — covariance axis (per-member `inflated` vs shared
  `inflated_shared` vs isotropic Jacobian-free); `g_tau` low/med/high (incl.
  `g=0`=FM-ODE); `M ∈ {10,50,100}`; `E ∈ {16,64,256}`. On the DM-SDE sampler.
  → `generated/tab_ablation.tex`.
- **Appendix table** — `1.5625%` column, same format (super-res dropped; sparse
  only — see Methods note).
- **`fig:ns_trajectories`**, **`fig:ns_diagnostics`** — figure code exists; run
  with `+save_figures=true`.

**Method lineup (as of 2026-06-29).** Generative: our three (SI-SDE, DM-SDE,
FM-ODE) + FlowDAS, Guided FM (FIG), Guided FM (OT-ODE), D-Flow SGLD, SDA, SURGE.
Classical (true-solver, NS only): EnKF, LETKF, particle filter, ensemble score
filter. The **E=1000 non-localized EnKF is the ground-truth posterior / KL
reference**; E=64 localized EnKF is a baseline. The legacy "Guided FM" (one-step
DPS-on-flow) and "Guided diffusion" (DPS) are **DROPPED** from the paper — their
`Method` enum entries are kept for back-compat but removed from the run
registries (`NS_METHODS`/`WIRED_METHODS`, `make_tables`).

## Tidy rows this case emits

`case=navier_stokes`; `scenario ∈ {32^2->128^2, 16^2->128^2, sparse 5%, sparse
1.5625%}`; `metric ∈ {rmse, energy_spec_rmse, kl_points, crps, spread_skill}`
plus `nfe`/`seconds`. Ablation rows use `method=Ours (DM-SDE)` and the
`ablation:*` tags in the `scenario` column (consumed by
`make_tables.render_ablation_body`).

## Status (2026-06-29 — all methods wired; full headline runs PENDING the GPU)

`driver.py` + `_ns_pipeline.py` + `_ns_figures.py` implement the full
assimilation + evaluation pipeline, and all 10 generative + 4 classical methods
are wired. **The headline 5-trajectory grid has NOT been re-run with the fixed
methods** — the NS manuscript table cells are `\tbd` pending the GPU. The old NS
baseline numbers are STALE (FlowDAS / SDA changed from the bug fixes; legacy
Guided FM / Guided diffusion removed; D-Flow SGLD and SURGE are new and their NS
hyperparameters are not yet locked).

- **Obs operators (E1)** — block-average super-res (`super_res`) and seeded
  sparse masks (`random`, mask via `common.seeding.mask_seed`).
- **Three samplers (E4)** — SI-SDE / DM-SDE / FM-ODE, sharing the trained prior.
  DM-SDE is shown in the paper as "DM-SDE" — a diffusion-model-style SDE on
  the FM prior. The FM model is a **dedicated trained FM checkpoint**
  (`flow_matching` / `flow_matching_big`), not the SI drift reused.
- **Metrics (E7/E9/E10/E11/E13)** — ensemble-mean RMSE, log-spectrum energy
  RMSE, unbiased CRPS, spread-skill `|1-ratio|` (guarded for `E<2`), KL-at-points
  (observed vs unobserved) measured against a **large-E reference ensemble** drawn
  once per (scenario, seed) by the headline SI-SDE sampler on the same
  truth+obs+mask (size = `reference_ensemble_size`), NFE (drift-net forward
  counter) + wall-clock (`StepTimer`).
- **Ablation (E12)** — `run_ablation()` (entrypoint; `run.py +ablation=true`)
  drives `evaluate_ablation`, sweeping the **covariance axis** (`inflated_shared`
  vs isotropic Jacobian-free — the multiplicative gain `dps_full` is dropped from
  the paper, kept off-by-default in code), g_tau (incl. `g=0` == FM-ODE), M and E.
  Each sweep point is emitted with a per-point tag `ablation:<axis>:<value>` AND
  the actual swept `E`/`M`, so across-seed aggregation keeps points distinct.
- **Figures** — `_ns_figures` writes `fig:ns_trajectories` + `fig:ns_diagnostics`;
  `make_method_figures.py` / `make_conventional_figures.py` write per-method and
  per-scenario comparison panels.

### Diffusion prior is built from the FM model
The SDA and SURGE baselines (and the DM-SDE sampler's diffusion sibling) use a
**diffusion prior constructed from the trained FM model** via
`DenoiseDiffusionModel.from_flow_matching` (velocity mode), because the separately
trained diffusion checkpoint is weak. This is controlled by
`case.checkpoints.diffusion_from_fm: true` in `navier_stokes.yaml`.

### Pointing at real weights (GPU box)
`configs/case/navier_stokes.yaml`:
- `checkpoints.si_run` — real SI run name (dir holds `model.pth`).
- `checkpoints.fm_run` — real FM run name (a dedicated trained FM checkpoint).
- `checkpoints.diffusion_from_fm: true` — build the DM prior from the FM model.
- `require_weights: true` — hard-fail if the dir / `model.pth` is missing (no
  silent random-weights fallback). Default `false` keeps the random-init smoke
  path with a LOUD warning.

### Bugs fixed while wiring
- `FlowMatchingModel.score` and `InterpolantGaussianLikelihood.score` did not
  rank-expand the pseudo-time `t` before the grid-space schedule products
  (`beta(t)*x` etc.), so every FM/SI field run crashed on shape mismatch. Both
  now expand `t` to the state rank for grid math while keeping `[B,1]` for the
  drift/score network and observation-space terms.

### Remaining for a full-scale GPU run (PENDING, 2026-06-29)
- The headline 5-trajectory runs (E=64, M=50; E=1000 only for the ground-truth
  EnKF) with the FIXED methods — these fill all `\tbd` cells in the NS table.
- D-Flow SGLD and SURGE NS hyperparameter sweeps (never completed; D-Flow pSGLD
  is ~17 min/config from the gradient-checkpointing recompute) — their NS configs
  are not yet locked.
- The step-count benchmark (M=100 / 500) is still deferred.
- Locked so far: FIG (k=1, c=80, w=0); OT-ODE (σ_y²=0, γ=4), from a traj1 /
  sparse-5% sweep vs the E=1000 EnKF posterior.

## Forecast-residual correction (optional, default OFF)

Background and derivation: [`FORECAST_CORRECTION.md`](../../FORECAST_CORRECTION.md).
Library: `scisi.posterior_models.forecast_correction`. Case wiring (shared with
urban): `common/forecast_correction.py`; NS-only pieces (zero-mean increment,
enstrophy spectra): `cases/navier_stokes/_ns_correction.py`; grid switch:
`common/fc_grid.sh`; tests: `tests/test_forecast_correction_ns_urban.py`. This
section covers both cases; the urban README adds only the urban-specific parts.

With `case.forecast_correction.enabled=false` (the default) nothing is built
and every run, its `variant` and its file names are bit-for-bit what they were
(tested). It only applies to the methods in `forecast_correction.methods`
(default: the three Ours samplers), so baselines in the same run stay plain.
SURGE posteriors refuse it (library).

### Switches (`configs/case/{navier_stokes,urban}.yaml`, block `forecast_correction`)

The YAML block is the only place defaults live.

| key | default | meaning |
| --- | --- | --- |
| `enabled` | `false` | master switch |
| `forecast_num_steps` | `null` (= M) | SDE/ODE steps of the unguided forecasts f_i (and of the online-Q forecast f'_i) |
| `forecast_stepper` | `null` (= sampler's) | `sde` / `ode` (an ODE stepper is refused for SDE forecasts) |
| `localisation_radius` | `8.0` | Gaspari-Cohn half-width in grid cells (support 16); `null` = global; or `{default: 8, sparse_1p5: 12}` |
| `gain_scale` | `1.0` | multiplier on K_ens - K_q (note 4.4); `0` = measure only |
| `update_cap` | `null` | optional Mahalanobis RMS cap per observation |
| `q_source` | `online` | `online` (2nd forecast per member) or `bank` (`residual_bank`: a `torch.save`d tensor from `build_residual_bank`) |
| `forecast_seed` | `null` | int: forecasts on a forked RNG, seeded per (case, scenario, test_index, seed, forecast_seed); the guided stream is that of the plain run |
| `diagnostics` | `false` | per-step enstrophy spectra (NS) |
| `zero_mean_increment` (NS) | `true` | remove the domain mean of the increment (see below) |
| `periodic`, `mask_obstacles` (urban) | `[false, false]`, `true` | see the urban README |

**One run** (`run.py`; any key of the block as `case.forecast_correction.<key>=...`):

```bash
python run.py case=navier_stokes '+ns_methods=["Ours (SI-SDE)"]' '+ns_scenarios=["sparse 5%"]' \
    num_steps=50 likelihood_mode=dps_jacobian_free +test_index=2 \
    case.forecast_correction.enabled=true case.forecast_correction.forecast_num_steps=10
```

**Grid scripts** (`run_ns_grid.sh`, `run_urban_grid.sh`; Ours groups only):
`FC=on FC_OVERRIDES="forecast_num_steps=10 gain_scale=0.5" bash run_ns_grid.sh`.
`FC_OVERRIDES` is a space-separated list of `<key>=<value>` for the block (Hydra
syntax, no spaces inside a value, e.g. `localisation_radius={default:8,superres_16:12}`);
`FC_DIAG=false` turns the per-step diagnostics off. `FC=off` (default) leaves the
commands exactly as they were; skip/resume works as before (one file per setting).

**Variant tag.** A corrected cell's rows get `variant=<base>_fc<steps|M>_<hash8>`
(e.g. `jacfree_fc10_2204f643`) and its files the suffix `__fc<steps|M>_<hash8>`;
`hash8` is the first 8 hex of the sha1 of every block key except `enabled`,
`methods`, `diagnostics` and `diagnostics_root`, so two settings that change the
numbers never share a tag. The grid takes the tag from
`python -m common.forecast_correction tag <case> <overrides...>`, the same function
the drivers use. The diagnostics JSON's `settings` records the full block.

### Geometry

* **NS**: doubly periodic Gaspari-Cohn (chordal distance) on the 128^2 grid. The
  observation sites are read off H (`H c / H 1`): sensor cells for the sparse
  masks, block centres for the 4x4 / 8x8 block averages.
* **Zero mean.** A localised gain is not a combination of whole anomaly
  fields, so the raw update can shift the domain-mean vorticity (contrary to the
  remark in note 4.5). `zero_mean_increment` projects the increment onto zero
  mean; the diagnostics log the shift before and after.
* Default radius 8 follows the scratch result (localisation at 8 sites kept
  85-103% of member enstrophy; global 42-50%). **It is not tuned. Tune radius and
  `gain_scale` on validation trajectories only** (NS test rows 0-9, which the grid
  does not score; e.g. `TRAJ="2 3" SAVE_TRAJ=2`).

### Diagnostics ("measure before fixing", note 5 stage 3)

Every corrected cell writes one JSON to `results/<case>/forecast_correction/`
(`diagnostics_root`) and logs a one-line summary:

* `lag1_autocorrelation`: lag-1 autocorrelation of the normalised forecast
  innovations `ybar - H fbar` over assimilation steps (near 0 = no per-member
  bias); `innovation_msq` per step (about 1 when calibrated); `cap_fraction`,
  `update_rms`, `spread_rms`, `q_rms`.
* `q_share_state` = `(q_rms / spread_rms)^2` = tr Q / tr P_f per step (library
  diagnostics). Near 1 means Q dominates and the correction has little to add.
* NS with `diagnostics: true`: `enstrophy_before` / `enstrophy_after` /
  `enstrophy_ratio_bands` per band (edges 0.5, 2, 4, 8, 16, 32, 64, 91 in integer
  wavenumbers), member min/median/max of the total ratio, increment domain mean
  before/after.

**Measuring without correcting**: `FC_OVERRIDES="gain_scale=0 forecast_seed=0"`.
The update is zero and the forecasts come from a forked RNG, so the trajectory,
and hence every metric, equals the plain run's (tested bit-for-bit on CPU); only
NFE/seconds include the forecasts, and the rows carry their own tag.

### Cost per assimilated step (network passes per member)

* Online Q (default): `2 x forecast_num_steps` extra prior passes, no JVPs. Bank: `1 x forecast_num_steps`.
* For comparison, the plain jacfree sampler costs about M passes per generated
  step; shared mode adds JVPs. So at `forecast_num_steps = M` online roughly triples jacfree's NFE on NS
  (every step assimilated); urban assimilates 49 of 145 steps, so the extra is
  about `2 n_f x 49/145`, about +0.7 M per generated step. NFE in the tidy rows includes the forecasts.
* Linear algebra (CPU, 2 threads, E=64, measured with the earlier wiring): NS 32^2
  (m=1024) 0.25 s per step after a 2.5 s first call (taper build); sparse 5% 0.15 s;
  urban sparse 1.5625% (m=1024) about 1 s. GPU timings are not measured.

### Order of GPU runs (note section 5)

1. Unit checks: `pytest tests/test_forecast_correction_lg.py tests/test_forecast_correction_ns_urban.py` and the library tests.
2. Lorenz 96 with trained priors (`cases/lorenz96/`), including unguided prior + localised EnKF.
3. **Measure before fixing** on NS and urban: `FC=on FC_OVERRIDES="gain_scale=0 forecast_seed=0"`, jacfree
   group, M=50, E=64, one or two validation trajectories, all scenarios. If `lag1_autocorrelation` is near 0 and
   `q_share_state` near 1, the correction is not warranted there; report that and stop.
4. **Correction**, first settings: `FC=on FC_OVERRIDES="forecast_seed=0"` (forecast steps = M, radius 8, gain 1,
   no cap, online Q, diagnostics on), jacfree group, same validation trajectories. Check that the enstrophy band
   ratios stay near 1 (not the 42-50% of a global update) and lag-1 moves towards 0. Then tune on validation:
   `localisation_radius` in {4, 8, 16}, `gain_scale` in {0.5, 1} (choose for lag-1 near 0), then
   `forecast_num_steps` in {M, 20, 10} (check the forecast spread against the M-step one before trusting few
   steps), optionally `update_cap=3` if `update_rms` shows outliers, then optionally `q_source=bank`.
5. Paper grid with the chosen settings (both Ours groups), next to the plain cells.

**Not wired:** the "unguided prior + EnKF update" alternative (note sections 2 and 5) is not available for NS or urban.
It must be added as a baseline before any claim that the correction beats an EnKF on these cases.
