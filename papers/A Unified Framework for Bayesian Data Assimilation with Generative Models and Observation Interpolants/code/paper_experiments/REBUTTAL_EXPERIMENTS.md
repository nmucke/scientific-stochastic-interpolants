# ICLR 2027 rebuttal: every experiment that was run

Last updated 2026-11-12. This is the single entry point for the rebuttal runs made on
DelftBlue (TU Delft cluster) by two working sessions: a Navier–Stokes/uDALES session and a
Lorenz 96 session. Their code was unified on branch `ICLR-2027-review` (merge `46021c1`,
PR #58). The code and launchers for every experiment are on this branch. The **result files
are not in git** (tens of GB): see "Results" at the end for where they are and how to copy
them.

`PE` = `papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and
Observation Interpolants/code/paper_experiments` (the path contains spaces). Paths starting
with `/scratch/ntmucke/...` are on DelftBlue. `SSI` = `/scratch/ntmucke/scientific-stochastic-interpolants`.

Related documents on this branch:
- `PE/LORENZ96_DELFTBLUE_HANDOVER.md`, `PE/LORENZ96_HANDOVER_2026-11-12.md`: Lorenz 96 setup,
  decisions, job history. These are more detailed than the Lorenz 96 section below.
- `PE/FORECAST_CORRECTION.md`: the forecast-residual correction (FC) method.
- `PE/README.md`, `PE/cases/*/README.md`: the cases, configs and result layout.
- On DelftBlue, `SSI/paper_experiments/results/INDEX.md`: per-folder index of the result tree,
  kept up to date by `tools/sync_all_results.sh`.

## Contents
1. Navier–Stokes, wrong initial condition, long horizon (main grid)
2. NS extension: SURGE on our samplers, and the forecast correction (r=8, r=20)
3. NS: generative prior + EnKF (does the EnKF win because of the true PDE?)
4. NS: EnKF with localisation r=20 on super-resolution
5. NS: checks (old vs new code equivalence, smoke tests, benchmarks)
6. uDALES (urban) with the forecast correction
7. Lorenz 96 (linear and arctan observations)
8. Which code ran which experiment
9. Tables, figures and the manuscript
10. Results: where they are, how to copy them, what is pending
11. Caveats, known issues and open decisions

---

## 1. Navier–Stokes, wrong IC, long horizon: main grid (complete)

**Purpose.** Reviewer request: show recovery from a poor initial condition over a longer
horizon (the original paper starts from the true state and assimilates 15 steps).

**Settings** (identical in every NS rebuttal experiment unless stated):
- Truth/observations from held-out test trajectories 10–14 (5 trajectories), seed 0.
- Foreign initial condition: the 5-frame history and x0 come from test trajectory 15
  (`+ic_index=15`). Logged as "Foreign initial condition: history + x0 from test trajectory 15";
  saved as `ic_index` in the states npz.
- `case.num_physical_steps=40`: 5 history + **35 assimilation steps**.
- E=64 members, **M=100** sampler steps only (no M sweep), `+skip_kl_reference=true` (KL is not
  used for NS), divergence guard `+divergence_rmse_threshold=10.0` (a trajectory whose RMSE
  exceeds 10 is stopped, NaN-padded and excluded from means; counted as "div k/n").
- Scenarios: super-resolution 16²→128², 32²→128²; sparse sensors 5% and 1.5625%.
- Saved ensembles for trajectory 11 only (`states/traj11`).
- Hardware: Tesla V100S 32 GB (`gpu-v100`), env `.venv-cu126` (torch 2.14.1+cu126; CUDA 13
  builds do not support V100).

**Methods** (one method per Slurm task, 280 cells = 14 method-variants × 4 scenarios × 5 traj.):
- Ours SI-SDE / DM-SDE / FM-ODE × {`jacfree` (Jacobian-free covariance), `shared_jac10`
  (ensemble-shared inflated covariance, Jacobian refreshed every k=10 sampler steps)}.
- FlowDAS, SURGE (FlowDAS), SDA, SURGE (SDA), D-Flow SGLD, Guided FM (FIG).
- EnKF (true solver, E=64, inflation 1.0, **Gaspari–Cohn r=20 on sparse, global on
  super-resolution**: the paper's setting, `run_ns_grid.sh`), particle filter.
- No k-ablation (user decision). The E=1000 EnKF reference was not run.

**Launcher:** `PE/slurm_wrong_ic/` (`submit.sh`, `make_manifest.py`, `run_task.sh`).
**Slurm:** per (trajectory × scenario) arrays 1265369–1265385, plus requeue array 1275092
(87 cells, run from the frozen copy, see §8). An earlier launch (all cells) failed on V100
with CUDA-13 torch and was cancelled/relaunched.
**Results:** `results_wrong_ic_np40/navier_stokes/` (copy: `results/navier_stokes_wrong_ic_np40/main/`).
**Status:** complete, 280/280. 10 trajectory-cells diverged: EnKF 16² (5/5, global EnKF,
no inflation) and Ours SI-SDE shared (32²: 2/5, sparse 5%: 1/5, sparse 1.5625%: 2/5).

**Headline findings** (mean over 5 trajectories, see the tables for numbers):
- Super-resolution: our samplers recover from the wrong IC and beat all baselines; FM-ODE
  Jac-free is best on RMSE/CRPS; on 32² our late-window RMSE (~0.065) matches the true-IC run.
- Sparse: no generative method recovers (RMSE ~0.85–1.0); the localised E=64 EnKF is clearly
  best (RMSE 0.34 at 5%, 0.77 at 1.5625%).
- SURGE baselines and the particle filter collapse the ensemble (spread–skill ≈ 1).
- SI-SDE with the shared likelihood is the fragile combination (5 of 20 diverged).

## 2. NS extension: SURGE on our samplers, forecast correction (FC)

Same settings as §1. **Launcher:** `PE/slurm_wrong_ic_ext/` (`submit_ext.sh`, `make_manifest_ext.py`,
`run_task.sh`). **Results:** `results_wrong_ic_np40_ext/navier_stokes/` (copy: `.../ext/`).
Variant tags in the file names identify the configuration.

| Experiment | Cells | Tag / variant | Slurm | Status |
|---|---|---|---|---|
| SURGE on Ours: SURGE (SI-SDE), SURGE (DM-SDE) × {jacfree, shared_jac10}; ESS threshold 0.5 (not tuned; user: ESS tuning is pointless at E=64); no SURGE for FM-ODE (g=0) | 80 | `surge_ours_<cov>` | 1281559 | complete |
| FC, localisation radius 8 (defaults), Ours × {jacfree, shared_jac10} | 120 | `<cov>_fc25_70a3134d` | 1281559 | complete |
| FC, localisation radius 20 (matches the EnKF's r=20) | 120 | `<cov>_fc25_8797d2ae` | 1305160 (1301219 cancelled and requeued from frozen code, §8) | **queued** |

**FC settings** (both radii): online Q̂ (a second unguided forecast per member), 25 unguided
forecast steps (`forecast_num_steps=25`, user choice), gain scale 1, no update cap,
diagnostics on. All untuned defaults (user decision: run with defaults).

**Findings:** SURGE on Ours collapses the ensemble (spread–skill ≈ 1) and is worse than plain
by 0.05–0.15 RMSE. FC r=8 rarely helps: clear gains only for shared SI/DM at 32² (fewer
divergences), clear losses at sparse 5% with the shared likelihood (~+0.09 RMSE).

## 3. NS: generative prior + EnKF

**Purpose:** test whether the EnKF does well because it propagates with the true PDE or because
of the Kalman update. Each member gets an UNGUIDED forecast from our trained prior (SI-SDE,
DM-SDE via the lifted FM SDE, FM-ODE; M=100 steps, per-member 5-frame history), then the
standard EnKF analysis. Everything else is identical to the EnKF baseline: initial ensemble
(`init_perturb_std=0.05`), perturbed observations and random keys, R, H, localisation (r=20 sparse,
none super-res), inflation 1.0, metrics.

**Code:** methods "SI-SDE prior + EnKF", "DM-SDE prior + EnKF", "FM-ODE prior + EnKF"
(commit `ac4713b`; `PE/cases/navier_stokes/enkf_baseline.py` `run_prior_enkf_baseline`,
`PE/cases/navier_stokes/driver.py` `_run_prior_enkf`, `PE/tests/test_ns_prior_enkf.py`).
Tested: with the true solver plugged in, the new loop equals `run_enkf_baseline` bit for bit.
Independently reviewed before launch.
**Launcher:** `PE/slurm_prior_enkf/` (scripts pinned to the isolated copy they ran from, §8).
**Slurm:** 1302963 (60 tasks = 3 priors × 4 scenarios × 5 traj.). **Results:**
`results_wrong_ic_np40_prior_enkf/navier_stokes/`. **Status: queued.**
**Smoke** (3 steps, traj 11): prior + EnKF tracks the true-solver EnKF (sparse 5% RMSE 0.66 vs
0.62; 32² 1.31–1.36 vs 1.46). Outputs: `SSI/paper_experiments/smoke_prior_enkf/`.
**Caveats:** the divergence guard applies to these methods but not to the original EnKF code
path; on super-resolution neither EnKF variant is localised, so the sparse scenarios are the
informative test.

## 4. NS: EnKF with localisation r=20 on super-resolution

**Purpose:** the paper's EnKF is global on super-resolution (and diverges on 16² from the wrong
IC). This runs the true-solver EnKF with Gaspari–Cohn r=20 on 16² and 32² as well.
**Code fix needed:** the localised EnKF had no observation coordinates for the block-average
super-resolution operator (crashed). Commit `948fa40` places each observation at the |H|-weighted
block centroid (16²: 7, 23, …, 247 on the 256² grid).
**Launcher:** `PE/slurm_enkf_loc20/` (`run_cpu.sh` handles CPU or GPU via `DEVICE`; GPU sets JAX
matmul precision to highest; `array_task.sh` maps task → scenario × trajectory).
**Slurm:** 1305653 (10 tasks, V100, 12 h). Smoke 1304035 (MIG, passed: 2-step RMSE 16² 0.85,
32² 0.33 vs ~1.44 global). **Results:** `SSI/paper_experiments/enkf_loc20_superres/production/`.
**Status: queued.** In tables this row is "EnKF (r=20 everywhere)"; its sparse cells equal the
EnKF row (same configuration).

## 5. NS: checks, smoke tests, benchmarks

Scripts in `PE/slurm_bench/` (outputs on DelftBlue in `SSI/slurm_bench/`, copy in
`results/_smoke_and_bench/slurm_bench/`):
- `equiv/` (`equiv.sh`, `equiv2.sh`, `compare.py`): old code (frozen) vs new code (after pulling
  PR #57) vs old code again, 2 assimilation steps, all methods. Result: every method that is
  deterministic run-to-run (DM-SDE, FM-ODE, FIG, EnKF, PF) is bit-identical old vs new; the others
  differ old-vs-new by the same amount as old-vs-old. Conclusion: the pulled changes do not alter
  NS results.
- `surge_ours/smoke.sh`: SURGE-on-Ours smoke (showed the ensemble collapse).
- `fc/bench.sh`: FC cost/memory benchmark.
- `mig_pair.sh`: runs two commands on two MIG slices in one job (MIG QoS allows 1 job per user).
- `gpucheck`, `jaxcheck`, `bench.sbatch`, `v100/check.sh`, `hook/sitecustomize.py`: environment checks.

## 6. uDALES (urban) with the forecast correction

**Settings** (reconstructed from the manuscript's `tab:urban_accuracy_M50` run log; urban.yaml
defaults do NOT match and everything is passed explicitly): Ours SI/DM/FM-ODE × shared Jacobian
(`inflated_shared`, k=10, λ=0.95 from the method YAMLs) × {sparse 1.5625%, 0.78125%} × test
trajectories 1–5 = **30 cells**; foreign IC from trajectory 8 (`+ic_index=8`); 150 physical
steps; DA every 3 steps from step 5; E=64; M=50; FC with `forecast_num_steps=25`, online Q̂,
radius 8, gain 1 (tag `fc25_780902ac`). Plain (no-FC) comparison rows are the manuscript's
original August runs (RTX 3090), not rerun.
**Launcher:** `PE/slurm_udales_fc/`. **Slurm:** 1300217 (tasks 0–19), 1305161 (tasks 20–29,
requeued from frozen code, §8). **Results:** `results_udales_fc/urban/` (copy `results/urban_fc/`).
**Status:** 18/30 done at writing, rest running/queued.
**Finding so far:** all 10 finished SI-SDE cells DIVERGE with FC (guard trips at steps 16–25;
plain SI-SDE on the same cells is stable at RMSE 0.55–0.70). DM-SDE/FM-ODE pending. Likely
cause: untuned FC (radius 8 straight-line ignores buildings, gain 1, no cap). The urban FC
diagnostics have no per-step energy check (proposed patch not applied).

## 7. Lorenz 96

Run by the Lorenz 96 session; full details in `PE/LORENZ96_HANDOVER_2026-11-12.md` and
`PE/LORENZ96_DELFTBLUE_HANDOVER.md` (decisions, tuning protocol, every job id). Summary:
- Two sweeps: `linear` observations (A100 MIG) and `arctan` observations (V100); seed 0,
  12 test sequences, E=64, M ∈ {25, 50, 100, 250}; full baseline lineup; PF ground truth.
- Shared-Jacobian likelihood cells (refresh every pseudo-step in L96, not every 10th), with
  and without FC; corrected trained cells use the online Q only.
- Launch scripts: `PE/slurm_lorenz96/`; driver `PE/cases/lorenz96/`; `PE/run_lorenz96.sh`.
- Status at writing: linear plain cells complete; arctan 74/78 (4 `si_sde_fc` cells rerunning
  with online Q, job 1305012, then final aggregation 1302733); shared cells running (1304946
  linear, 1304947 arctan). Then rerun `run_lorenz96.sh final <obs>` and `make_lorenz96_appendix.py`.
- Results on DelftBlue: `/scratch/ntmucke/lorenz96_delftblue/results/lorenz96` (copy in the shared
  tree `results/lorenz96/`), figures `.../figures/lorenz96`, tables in the L96 worktree
  `generated/tab_lorenz96*.tex`.
- The L96 manuscript branch `lorenz96-appendix-results` (local only) is to be REMADE (user decision).

## 8. Which code ran which experiment

| Experiment | Code that ran it | Same as on this branch? |
|---|---|---|
| NS main grid, first 193 cells | live checkout at `6cfa977` + uncommitted NS edits | yes: those edits are in `ac8aa01`; equivalence checked (§5) |
| NS main grid, 87 requeued cells (1275092) | frozen copy `SSI/code_ns_wrong_ic` (same code) | yes |
| SURGE-on-Ours + FC r=8 (1281559) | live `e06ab51` + uncommitted (= `ac8aa01`) | yes |
| FC r=20 (1305160), uDALES tasks 20–29 (1305161) | frozen worktree `SSI/code_rebuttal_ac8aa01` (commit `ac8aa01`) | yes |
| uDALES tasks 0–19 (1300217) | live `e06ab51` + uncommitted (= `ac8aa01`) | yes |
| prior + EnKF (1302963) | isolated copy `SSI/code_ns_prior_enkf` | yes: committed as `ac4713b`, byte-identical |
| EnKF r=20 super-res (1305653) | `SSI/code_ns_wrong_ic` + obs_coords fallback | yes: committed as `948fa40` |
| Lorenz 96 (all) | worktree `/home/ntmucke/ssi-lorenz96`, branch `review/lorenz96-delftblue` @ `64dc61c` | yes: merged in `46021c1` |

Every results root has a `run_config.txt` (commit, dirty-file count, settings, job ids) and a
`code_<timestamp>.diff` snapshot of the uncommitted diff at submission.

## 9. Tables, figures and the manuscript

- NS: `PE/refresh_ns_appendix.sh` regenerates everything (main-grid aggregate and tables,
  FC table, the all-runs aggregate `aggregated_all/` + `tab:ns_all_wrong_ic`, and the per-step
  figures). Pending cells are detected automatically (expected vs finished files) and shown as
  "pending"/"partial". Key scripts: `aggregate_ns.py` (`--fc-root`, `--fc-only`), `aggregate_ns_all.py`
  (constant `ENKF_LOC20_ROOT` → `enkf_loc20_superres/production`), `make_ns_figures.py`
  (`--fc-per-step`, `--fc-lineup full`, `--appendix-all`), `common/latex_tables.py` (bold = best,
  underline = second best, ±std, ties at 2 decimals in unscaled units).
- Hand-typed manuscript tables (original true-IC `tab:ns_accuracy`, `tab:ablation`, urban table)
  were edited with `PE/tools/underline_second_best.py` and `PE/tools/add_std_to_static_tables.py`
  (std taken from the original aggregates, each cell verified against the printed value).
- Manuscript repo `iclr_da_interpolant_manuscript`, branch **`ns-wrong-ic-40`** @ `8b20bd1`
  (pushed; 1 commit ahead of `main`, merges cleanly): setup prose (35 steps, poor IC),
  `tab:ns_accuracy_fc` (FC r=8 + baselines, below `tab:ns_accuracy`), appendix
  `tab:ns_all_wrong_ic` + per-step figures (FC, all-ours, all-baselines), underline/±std in all
  results tables. The original `tab:ns_accuracy` (true-IC results) is kept; the paragraph above
  it still describes the true-IC results (to be decided). Build: `make` (TeX Live on DelftBlue at
  `/scratch/ntmucke/texlive/2026`; latexmk needs a full Perl, e.g. `pixi global install perl`).

## 10. Results: where, how to copy, what is pending

**Shared tree on DelftBlue:** `SSI/paper_experiments/results/` (122 GB total). It holds the original
paper results (`analytical/`, `navier_stokes/` 33 GB, `urban*/` ~69 GB, `timing/`) and copies of all
rebuttal results: `lorenz96/` (1.4 GB), `navier_stokes_wrong_ic_np40/{main,ext,prior_enkf,enkf_loc20_superres}`
(14 GB), `urban_fc/` (3.2 GB), `_smoke_and_bench/` (2.6 GB); plus `../figures/lorenz96/` and
`../generated/tab_lorenz96*.tex`. `results/INDEX.md` there lists every folder with its source,
status and job ids. Originals stay at their source paths (jobs write there).
Backup of the NS session's trees: `/scratch/ntmucke/backups/rebuttal_ns_session_20261112/`.

**When all jobs have finished** (`squeue -u ntmucke` empty):
1. Lorenz 96: `run_lorenz96.sh final linear` and `final arctan`, then `make_lorenz96_appendix.py`
   (in the L96 worktree; see its handover).
2. Copy everything into the shared tree: `bash SSI/paper_experiments/sync_all_results.sh`
   (also in `PE/tools/`; copy-only, never deletes, logs counts and checksums).
3. NS tables/figures: `bash PE/refresh_ns_appendix.sh` (fills the pending rows/curves).
4. uDALES: aggregate `results/urban_fc/urban` (no aggregation script written yet for the FC cells).

**Copying to another computer** (rebuttal results only, ~21 GB; add the original case folders if
needed; drop `--exclude` lines to include saved ensembles):

    rsync -av --exclude='.jax_cache' --exclude='.tmp' \
      delftblue:/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results/{INDEX.md,README.md,lorenz96,navier_stokes_wrong_ic_np40,urban_fc,_smoke_and_bench} \
      <local>/paper_experiments/results/
    rsync -av delftblue:/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/{figures/lorenz96,generated} <local>/paper_experiments/

The NS/uDALES folders are results ROOTS: pass e.g. `results/navier_stokes_wrong_ic_np40/main` as
`--results-root` to the aggregation/figure scripts. Manifests and logs inside contain the original
DelftBlue paths (provenance only).

## 11. Caveats, known issues, open decisions

- **SURGE on Ours is defined differently per case (open, to revisit):** NS uses methods "SURGE
  (SI-SDE)/(DM-SDE)" with the `velocity` SI denoiser and `SurgeLiftedFlowMatchingPosterior`; Lorenz 96
  and the analytical multistep case use variant `surge` with the `follmer` denoiser and
  `OursSurgeDiffusionPosterior`.
- **FC is untuned everywhere** (radius 8 or 20, gain 1, 25 forecast steps, no cap). On uDALES it
  destabilises SI-SDE.
- **Ablation table** (`tab:ablation`) holds the original true-IC M=50 runs; no k-ablation was run from
  the wrong IC.
- **Timing/cost:** the rebuttal grids were timed under GPU contention; cost columns use the original
  clean timings, which do not depend on the IC.
- **Hardware:** NS EnKF results on A100 differ ~2% from V100 on unlocalised 32² (TF32); keep
  production on V100 (r=20 EnKF on GPU uses highest matmul precision).
- **Divergence handling:** diverged trajectories are dropped whole and counted; the original EnKF code
  path ignores the divergence guard (EnKF 16² ran to NaN), our methods and prior+EnKF honour it.
- **E=1000 EnKF reference** was not run for the wrong-IC setting (row shown as "--" or dropped).
- **Test pin:** `PE/tests/test_analytical_metrics.py` compares to rel=1e-12 (archived values differ by
  1 ulp across BLAS builds).
- **Manuscript text** above `tab:ns_accuracy` and some appendix sentences describe the true-IC
  results; to be rewritten once the final table choice is made.
