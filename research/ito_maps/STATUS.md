# Status

## Improvement work (HANDOVER.md), updated 2026-11-23

Spec: `notes/ideas/HANDOVER.md` (branch `research/ito-map-improvement-ideas`). Protocol: [PROTOCOL.md](PROTOCOL.md). Nothing below has been reviewed or merged into `dev`.

| Unit | Branch / worktree (`/scratch/ntmucke/scientific-stochastic-interpolants/`) | State |
|---|---|---|
| Baseline maps (Itô PSD local K=4, flow PSD; seed 0; dropout 0) | `research/itomap-baselines` / `wt_itomap_baselines` | smoke 1445467 passed (all phases + resume); V100 jobs Itô 1445729, flow 1445730 **held**: shared `.venv` torch 2.15+cu132 has no sm_70 kernels; waiting for a cu126 venv |
| Phase 0 diagnostics D1, D5–D7 (+ WP7 budget class) | `research/itomap-phase0` / `wt_itomap_phase0` (head 445a38b) | D1 **gate passed** (SI-128 anomaly/white 0.004–87, peak shell 7). Budget closes on archive. SI-128 long run (4 traj × E=8, 500 leads): enstrophy 4.21 vs data 4.76 [4.65, 4.88], budget residual −17% of injection (leads 96–500), lag-1 0.66 vs 0.61, shell-64 power 0.52× data: D7 fires for SI. Suites before/after pass (208/224). D2–D4 wait for baselines. Open: residual std 1.42× prediction |
| WP1 + WP2 core (NoiseCovariance, variance clock, matched β, invariant) | `research/itomap-core` / `wt_itomap_core` (head 8b609d5) | suites: dev snapshot 205 pass / 3 fail (KTH, snapshot lacked `scripts/encode_kth.py`), after 249 pass. Smoke 1445698 failed on `seed` override, fixed (`++seed`), resubmitted 1448820 (MIG). Long WP1 runs need V100 + cu126 venv |

- Data on DelftBlue: `data.npz` only; no 256² companion (does not exist). No earlier Itô/flow-map checkpoints or archived results here, hence the baseline retraining.
- Large outputs: `/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/`; job logs in `slurm_itomap/logs/`.
- Waiting for review: nothing yet.

## Earlier status (2026-10-08)

Updated 2026-10-08. Scope: stochastic Navier–Stokes only. The video-forecasting extension (KTH/CLEVRER) has been dropped from this folder and from the paper. The initial-stage results are archived and will not be used in the paper.

## Done

- **Method and theory.** Conditional Itô maps for the forecasting SDE `dX = b dt + ε(1−t) dW`, with `X_0 = c(h)`. Variance-clock Brownian bridges give exact shared-path coupling. Diagonal plus composition (PSD) objectives, with the coupling and compression propositions. Written up in the manuscript (repo `Few-Step-Probabilistic-Forecasting-with-Conditional-It-o-Maps`, cloned into `manuscripts/`).
- **Implementation** in `src/scisi`. Itô-map and flow-map models, `MapTrainer` with two modes (PSD, with an optional teacher diagonal, and `teacher_rollout`), and Hydra train/test entry points. Teacher initialization, EMA, resume and a mask guard are included.
- **Paper (method side).**
  - New appendices: teacher distillation vs. progressive self-distillation (manuscript `appendix/training_modes.tex`) and differences from Pan et al. (manuscript `appendix/ito_maps_differences.tex`).
  - Every NS result is a red `\pending{}` placeholder; `notes/paper_planning/results_plan.md` lists them.
  - The build needs no data or results.
- **Knowledge consolidated** into [notes/key_findings.md](notes/key_findings.md). Originals are in `archive/notes/`.

## Initial stage (archived, superseded)

A two-stage pipeline was run for the Itô and flow maps with three student seeds: teacher-anchored PSD for 2,000 updates, then teacher-rollout refinement for 500 updates. Held-out tests and a long-horizon run followed. Qualitative lessons are in §5 of `notes/key_findings.md`:
- The flow map was the strongest fast baseline.
- The Itô maps were underdispersed.
- Refinement mattered.
- K16 features did not beat K4.

Raw outputs are in `archive/results/` and the old handover is in `archive/STATUS_initial_stage.md`. Old checkpoints remain under `checkpoints/stochastic_navier_stokes/{ito,flow}_{psd,rollout}_seed*`.

## In progress / to decide

- The working-tree `config/follmer_ito_map_stochastic_navier_stokes.yaml` sets `teacher: null`. It is a teacher-free PSD run: random initialization, data diagonal `R_t`, and EMA self-consistency validation. A map training job was running at the time of this cleanup.
- Define the new NS experiment protocol before running anything:
  - splits and origins;
  - ensemble size;
  - budgets;
  - a preregistered noninferiority margin;
  - which training variant(s) to report (teacher-free PSD, teacher-anchored PSD, plus rollout refinement).
- Then update the hard-coded defaults flagged **old** in [experiments/README.md](experiments/README.md). Run `experiments/audit_ns_time_reference.py` first if long-horizon runs are planned.

## Next steps (from notes/key_findings.md §6)

1. **Exact-diffusion solver control.** Run SI with exact integrated noise to separate the map's benefit from the noise quadrature.
2. **Diagnose the underdispersion**: PSD bias, the `1/(t−s)²` weighting, the EMA, K. Consider spread-aware fine-tuning.
3. **Define the hypothesized advantage** of the stochastic map over the deterministic map, and test it.
4. **NS provenance.** Recover the viscosity, forcing and drag, or regenerate a documented dataset with conditional reference ensembles.
5. **Teachers.** Retrain SI/FM teachers with matched widths and log their cost.
6. **Code fixes before relaunch**: EarlyStopping, AMP `unscale_`, `pars_cond` shapes, worker RNG on resume.
