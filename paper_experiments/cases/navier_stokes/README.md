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
- **`tab:ablation`** — covariance axis (per-member `inflated` vs shared
  `inflated_shared` vs isotropic Jacobian-free); `g_tau` low/med/high (incl.
  `g=0`=FM-ODE); `M ∈ {10,50,100}`; `E ∈ {16,64,256}`. On the FM-SDE sampler.
  → `generated/tab_ablation.tex`.
- **Appendix table** — `1.5625%` column, same format (super-res dropped; sparse
  only — see Methods note).
- **`fig:ns_trajectories`**, **`fig:ns_diagnostics`** — figure code exists; run
  with `+save_figures=true`.

**Method lineup (as of 2026-06-29).** Generative: our three (SI-SDE, FM-SDE,
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
plus `nfe`/`seconds`. Ablation rows use `method=Ours (FM-SDE)` and the
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
- **Three samplers (E4)** — SI-SDE / FM-SDE / FM-ODE, sharing the trained prior.
  FM-SDE is shown in the paper as "FM-SDE (DM)" — a diffusion-model-style SDE on
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
The SDA and SURGE baselines (and the FM-SDE sampler's diffusion sibling) use a
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
