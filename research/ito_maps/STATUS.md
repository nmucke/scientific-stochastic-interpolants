# Status

## Improvement work (HANDOVER.md), updated 2026-11-23

Spec: `notes/ideas/HANDOVER.md` (branch `research/ito-map-improvement-ideas`). Protocol: [PROTOCOL.md](PROTOCOL.md). Nothing below has been reviewed or merged into `dev`.

| Unit | Branch / worktree (`/scratch/ntmucke/scientific-stochastic-interpolants/`) | State |
|---|---|---|
| 64×64 dataset + config (development data from 2026-11-25) | `feature/ns-64-dataset` / `wt_ns64` (eae7a18, pushed) | built: `data/stochastic_navier_stokes_64/`, 7 tests pass; UNet [16,32,64], 4.33M params; MIG smoke ~4 it/s for maps (128: ~1.1) |
| Baseline maps on 64 (Itô PSD local K=4, flow PSD; seeds 0–2; dropout 0; patience 20) | `research/itomap-baselines` / `wt_itomap_baselines` (0c7e216) | **trained** 2026-11-26/27: 70/70 epochs each, ~16.5 h on V100; best val Itô 0.098/0.106/0.102, flow 0.144/0.145/0.140, still improving at the end. Validation eval (maps 1/2/4/8, white SI 16/64/128) = 1496188 |
| Phase 0 diagnostics on 64 | `research/itomap-phase0` / `wt_itomap_phase0` (93e42ee) | stage 1 done (1473403). Agent reporting stage 1 and submitting stage 2 (SI white) and stage 3 (six maps) |
| WP1 step 1 on 64 (regressor, spectrum, SI white_t2 / coloured_t / coloured_matched, 150 epochs) | `research/itomap-core` / `wt_itomap_core` (f01975a) | **done, falsifier fired** (seed 0): K=4 omitted variance white 2.6%, coloured_t 17.9%, matched 22.8%; few-step Euler still underdispersed (spread ratio 0.45–0.68 at N=2–8); coloured N=128 teachers worse than white (CRPS +0.02–0.03, spread/skill 0.75 vs 0.98). Report `results/wp1_stochastic_navier_stokes_64/wp1_step1_report_seed0.json`. **Waiting for review** |
| 128×128 runs | — | cancelled 2026-11-25 (user); partial checkpoints kept, not used |

- Data on DelftBlue: `data.npz` only; no 256² companion (does not exist). No earlier Itô/flow-map checkpoints or archived results here, hence the baseline retraining.
- Large outputs: `/scratch/ntmucke/scientific-stochastic-interpolants/results_itomap/`; job logs in `slurm_itomap/logs/`.
- Waiting for review: WP1 step 1 (falsifier fired; decision on controls pending).

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
