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

Train SI (SI-SDE) and FM (FM-SDE, FM-ODE) priors with the **same** architecture,
data, and schedules. Assimilate `y^1..y^N` autoregressively at ensemble size `E`,
steps `M`. All methods run on **identical truth + observations + masks** (shared
seeds via `common/seeding.py`). `R = 0.05^2 I`.

## Deliverables

- **`tab:ns_accuracy`** — vorticity RMSE, energy-spectrum RMSE, KL-at-points ×
  {`32^2->128^2`, `5%`}. → `generated/tab_ns_accuracy.tex`.
- **`tab:ns_calibration_cost`** — CRPS, `|1-spread/skill|`, NFE, s/step × same.
  → `generated/tab_ns_calibration_cost.tex`.
- **`tab:ablation`** — full / Jacobian-free / no `G_tau`; `g_tau` low/med/high
  (incl. `g=0`=FM-ODE); `M ∈ {10,50,100}`; `E ∈ {16,64,256}`. On the FM-SDE
  sampler. → `generated/tab_ablation.tex`.
- **Appendix table** — `16^2->128^2` and `1.5625%` columns, same format.
- **`fig:ns_trajectories`**, **`fig:ns_diagnostics`** — figure code is a TODO.

Methods: our three + FlowDAS, Guided FM, Guided diffusion, SDA, ensemble score
filter, EnKF, particle filter.

## Tidy rows this case emits

`case=navier_stokes`; `scenario ∈ {32^2->128^2, 16^2->128^2, sparse 5%, sparse
1.5625%}`; `metric ∈ {rmse, energy_spec_rmse, kl_points, crps, spread_skill}`
plus `nfe`/`seconds`. Ablation rows use `method=Ours (FM-SDE)` and the
`ablation:*` tags in the `scenario` column (consumed by
`make_tables.render_ablation_body`).

## Status (wiring landed)

`driver.py` + `_ns_pipeline.py` + `_ns_figures.py` now implement the full
assimilation + evaluation pipeline:

- **Obs operators (E1)** — block-average super-res (`super_res`) and seeded
  sparse masks (`random`, mask via `common.seeding.mask_seed`).
- **Three samplers (E4)** — SI-SDE / FM-SDE / FM-ODE, sharing the trained prior;
  FlowDAS wired. FM prior reuses the SI checkpoint's UNet drift + a rectified-flow
  interpolation so `FlowMatchingModel.score` (L1) is defined; the FM correctness
  asserts live in `FlowMatchingPosterior` (no `Phi_0^obs` reweighting; bounded
  drift at `τ→1`).
- **Metrics (E7/E9/E10/E11/E13)** — ensemble-mean RMSE, log-spectrum energy
  RMSE, unbiased CRPS, spread-skill `|1-ratio|` (guarded for `E<2`), KL-at-points
  (observed vs unobserved) measured against a **large-E reference ensemble** drawn
  once per (scenario, seed) by the headline SI-SDE sampler on the same
  truth+obs+mask (size = `reference_ensemble_size`), NFE (drift-net forward
  counter) + wall-clock (`StepTimer`).
- **Ablation (E12)** — `run_ablation()` (entrypoint; `run.py ablation=true`) drives
  `evaluate_ablation`, which sweeps gain (likelihood_mode: `dps_full`/full,
  `dps_jacobian_free`/JF, `inflated`/G=I), g_tau (incl. `g=0` == FM-ODE), M and E.
  Each sweep point is emitted with a per-point tag `ablation:<axis>:<value>` AND
  the actual swept `E`/`M`, so across-seed aggregation (keyed on E/M) keeps points
  distinct; the representative point per axis also fills the canonical
  `make_tables` row (`ablation:steps_sweep`, etc.).
- **Figures** — `_ns_figures` writes `fig:ns_trajectories` + `fig:ns_diagnostics`.

The other baselines (Guided FM/diffusion, SDA, ensemble score filter, EnKF,
particle filter) emit `--` TODO rows (Phase 4).

### Pointing at real weights (GPU box)
`configs/case/navier_stokes.yaml`:
- `checkpoints.si_run` — real SI run name (dir holds `model.pth`).
- `checkpoints.fm_run` — real FM run name, or `null` to reuse the SI
  architecture + SI drift weights (no dedicated FM checkpoint exists yet).
- `require_weights: true` — hard-fail if the dir / `model.pth` is missing (no
  silent random-weights fallback). Default `false` keeps the random-init smoke
  path with a LOUD warning.

### Bugs fixed while wiring
- `FlowMatchingModel.score` and `InterpolantGaussianLikelihood.score` did not
  rank-expand the pseudo-time `t` before the grid-space schedule products
  (`beta(t)*x` etc.), so every FM/SI field run crashed on shape mismatch. Both
  now expand `t` to the state rank for grid math while keeping `[B,1]` for the
  drift/score network and observation-space terms.

### Remaining for a full-scale GPU run
- Set `checkpoints.si_run` (+ `fm_run` once trained) to real run names holding
  `model.pth` and `require_weights: true`.
- Run on GPU with `likelihood_mode=inflated` (the full-Sigma_s path does one
  network JVP per observation column per pseudo-time step, ≈ `N_y` extra forwards
  — far too slow on CPU at 128² where `N_y` is 819–1024; cheap on GPU), E=64,
  M=50, full seed list, all test ids.
- `reference_ensemble_size: 1024` for KL-at-points (kept small in the smoke).
- Implement the Phase-4 baselines.
