# Improvement ideas — handoff (2026-07-03)

Context: the 2026-07-02 speed refactor of `src/scisi/{likelihood_models,posterior_models}`
was audited and is **output-preserving** (analytical 5-seed A/B bitwise; fp64 `score()`
parity on the NS UNet: jacfree diff 0, shared ≤7.5e-15; NS jacfree trajectory A/B ≤1%).
Do NOT revert it. The observed "performs worse" has three real causes, each with work
items below. Evidence artifacts: `/export/scratch1/ntm/tmp_ab_test/` (old worktree at
4950b27, A/B CSVs, `parity_test.py`, `diagnose_shared.py`). Background: memory notes
`refactor_regression_audit`, `shared_mode_perf`, `likelihood_simplification`.

NOTE: `InterpolantGaussianLikelihood` was further simplified on 2026-07-03 (uncommitted:
`dps_full` + deprecated kwargs removed, bitwise-verified) — read the current file before
editing; line numbers in older notes are stale.

---

## P0 — Stabilize `inflated_shared` at NS scale (blocks the paper's "shared" rows)

**Symptom.** SI-SDE shared diverges (grid: rmse ~1e7 at 16²→128²/M=100/E=64; small cell
E=8/NP=12/M=50 → rmse 672, per-step rmse ×4 per DA window from step ~2). DM-SDE/FM-ODE
shared don't diverge but are spiky-unstable and WORSE than jacfree (0.84 / 1.41 vs
0.32 / 0.31 at M=100). Pre-existing (old code diverges the same way, ×2/step; the correct
dropout fix removed inference-noise that partially masked it). NOT a linear-algebra bug:
schedule coefficients are correct (`c_iso, c_jac ≥ 0` for the linear-γ schedule) and
`w_τ = a_τ + ½g²` reduces exactly to `g0²(1−τ)`.

**Mechanism** (see `diagnose_shared.py`): the assembled
`Σ̄ = β²R + c_iso·HHᵀ + c_jac·HJHᵀ` (J = UNet drift Jacobian at the ensemble mean) stays
PD at healthy states but is SHRUNK below the isotropic (jacfree) value by the Jacobian
term → guidance ‖S̄‖ ~ 3–13 against a state scale of ~1 → overshoot → error compounds
through the autoregressive window feedback. The learned J is also non-symmetric, so
HJHᵀ contributes non-normal amplification the theory (symmetric Σ_s) doesn't have.

**Ideas, cheapest first** (each is a few lines in `gaussian_likelihood.py`, shared branch):

1. **Symmetrize + eigenvalue-floor Σ̄.** Use `Σ̄_sym = ½(Σ̄+Σ̄ᵀ)`, `eigh`, clamp
   eigenvalues to `≥ β²R + c_iso·λ_min(HHᵀ)` (i.e., never below the jacfree-equivalent
   floor), reconstruct, solve. N_y ≤ 1024 → negligible cost vs the JVPs. This alone may
   fix DM/FM spikes; may not be enough for SI (the front factor also applies J).
2. **Jacobian damping knob.** `Σ_s = c_iso·I + λ·c_jac·J`, `λ ∈ [0,1]` config-exposed
   (λ=0 ≡ jacfree, λ=1 = current). Sweep λ ∈ {0.1, 0.3, 0.5, 1.0} on the small diverging
   cell. Prior art: this is exactly how SDA's `gamma_sda` tames its HHᵀ term.
3. **Residual-bounded step cap.** Cap `‖w_τ·S̄‖` per member so one Euler step moves at
   most the observation-space residual (FlowDAS `max_grad_norm` style, but relative to
   `‖ŷ−μ̄‖` rather than a fixed constant, so it stays schedule-consistent and inactive
   in the healthy regime).
4. **Symmetrize HJHᵀ** before assembly (`½(HJHᵀ + (HJHᵀ)ᵀ)`) — matches the theory
   (Σ_s symmetric) and kills the non-normal part. Combine with (1).
5. **Isotropic front factor, inflated solve.** Use full Σ_s only inside Σ̄ (the solve)
   but the jacfree front factor `H` (PiGDM-style inflation without the second J
   application). Halves the J amplification path; one-line change; the analytical
   evidence suggests damped front factors help (the deleted bespoke sampler's damped
   front factor was the best performer).
6. **Per-member J** at reduced cadence (`jacobian_refresh_every` per member) — only if
   1–5 fail; cost blows up.

**Verify with:** (expect rmse ≤ ~0.35, no growth trend; then scale to E=64/NP=20)

```bash
REF=paper_experiments/results/navier_stokes/reference/traj1/gt
.venv/bin/python -u paper_experiments/run.py case=navier_stokes seeds=[0] \
  ensemble_size=8 case.num_physical_steps=12 case.require_weights=true \
  case.device=cuda +test_index=1 num_steps=50 likelihood_mode=inflated_shared \
  '+ns_methods=["Ours (SI-SDE)"]' '+ns_scenarios=["16^2->128^2"]' \
  "+kl_reference_states=$REF" +save_per_step=true \
  +per_step_file=/tmp_ab/si_ps.csv results_file=/tmp_ab/si.csv
```

Jacfree reference numbers at this exact cell (must not regress): SI 0.276, DM 0.342,
FM 0.348. Once stable, retest `jacobian_refresh_every=5` (4× speedup) for stability
under the lagged Jacobian.

## P1 — Analytical case: close (or accept) the 2× KL gap vs the deleted bespoke samplers

`65f82dd` deleted `cases/analytical/samplers.py` and rerouted Ours rows through
`src/scisi`. Five-seed kl_points: SI jacfree 0.069→0.124, DM 0.084→0.167, FM 0.103→0.247,
FM-shared 0.0013→0.0042. Two mechanisms, both in the src integrators (not bugs, choices):

1. **FM/DM time grid.** Bespoke evaluated drift at RIGHT endpoints τ=(i+1)·dt including
   the τ→1 contraction; src is left-endpoint Euler (last eval at (M−1)/M). Options:
   evaluate FM drift at the interval midpoint or right endpoint, or wire `heun_step`
   (already in `scisi/sampling/sde_solvers.py`) for FM-ODE. Cheap experiment on the
   analytical case (CPU, ~1 min): if KL recovers to ~0.10, adopt for NS too.
2. **jacfree front factor.** Bespoke used the exact denoiser slope `βc/(β²c+σ²_τ) < 1`
   (damped) in ALL modes; src is faithful to the paper corollary (front = H). Exact slope
   only exists closed-form → either document the gap or fold the damping insight into
   P0-idea-5.

Also fix lineup drift in `cases/analytical/driver.py` (`ANALYTICAL_METHODS`):
`SURGE_FLOWDAS`/`SURGE_SDA` were commented out in 9d4122c (reduced-lineup spec says IN),
and `GUIDED_FM_FIG` is still in but diverges there (kl ~3e10, untuned analytical cells;
spec says dropped). Decide and make config, spec (`results/README.md`), and driver agree.

## P2 — Grid / results hygiene

- **Don't burn GPU-hours on diverging shared cells.** `run_ns_grid.sh` is currently set
  to the full grid incl. `ours_shared`; either drop that group until P0 lands or add an
  early-abort in `run_assimilation` (ensemble rmse > threshold ⇒ mark cell diverged,
  stop). The early-abort is generally useful for baselines too.
- **KL reference horizon mismatch.** References at `results/navier_stokes/reference/`
  are NP=15 while the grid runs NP=20 → `kl_points` rows are NaN/partial. Rerun
  `run_ns_reference.sh` at NP=20.
- **Always pass `+kl_reference_states=...`** to ad-hoc runs: without it the NS driver
  silently builds an E=1000 EnKF reference per cell (a slow second 7-window pass).
- **Baseline hparams were tuned against the buggy (noisy) prior.** The dropout fix
  changes every NS/urban prior rollout; FlowDAS ζ, FIG (k,c), D-Flow settings were tuned
  before it. Spot-check one cell per baseline before trusting the tuned constants.

## P3 — Smaller items

- **Regression guard for future refactors:** commit a tiny seeded CPU test (analytical
  case, E=64, M=20, all modes) asserting bitwise equality against a checked-in
  `baseline.pt` (harness pattern exists: `lik_harness.py` in the 2026-07-03 session
  scratchpad; `compare_run.py` pattern in memory `perf-pass-2026-07`).
- `src/scisi/posterior_models/archive.py` is import-broken at HEAD (starts mid-file, no
  imports; nothing imports it) — delete it or restore its header.
- Analytical shared mode is ~50× slower per call than the bespoke closed form (JVP
  machinery on a 2-D model). Harmless for correctness; if the timing column matters for
  the paper's cost table, special-case or footnote it.
- `run.py` logs `Scenario: <hydra default>` (superres_32) regardless of the cells
  actually run — cosmetic but misleading in logs; log the resolved `ns_scenarios` instead.
- Cleanup when done: `git worktree remove /export/scratch1/ntm/tmp_ab_test/old_wt`
  (then remove the CSV/scripts dir). Note `/tmp` on this box is ~full; keep run logs on
  `/export/scratch1`.
