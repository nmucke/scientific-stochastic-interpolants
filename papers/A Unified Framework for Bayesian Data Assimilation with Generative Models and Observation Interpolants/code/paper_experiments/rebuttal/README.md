# ICLR 2027 rebuttal: index of all experiments (branch `review/rebuttal-runs`, wrapped up 2026-11-23)

Everything that was run for the ICLR 2027 rebuttal: what each experiment is, where its results are, how to rerun or analyse it, and what is still open. The branch was merged into `ICLR-2027-review`, tagged `rebuttal-runs-final`, and locked on GitHub (see "Resuming" below).

## Where everything is

| What | Where |
|---|---|
| **This index + docs** (git) | `$PE/rebuttal/`: this README; `docs/` (REBUTTAL_EXPERIMENTS.md = cluster campaign log §1–15, Lorenz 96 handovers, L96 x² and \|x\| handovers, \|x\| 5-sequence summary table); `run_scripts/` (snapshot of every local run script and queue, same tree as the results folder); `manuscript_generators/` (snapshot of the scripts that make the manuscript tables and figures) |
| **Local results** (not in git, squamish) | `$PE/results/rebuttal_iclr2027/` (~2.7 GB): `runs/` (all local run roots, logs, queues; was `results/_runs`), `lorenz96/{linear,arctan,square,abs}` (PF references, gate, tuning, 3-sequence runs), `analytical_multistep/`, `figures_l96/`, `navier_stokes_physsteps45_sparse5_cancelled/`, `stale_lorenz96_from_worktree/`. The old paths (`results/_runs`, `results/lorenz96`, `results/analytical_multistep`, `results/figures`, `results/navier_stokes_physsteps45_sparse5`, `results/_stale`) are symlinks into this folder, so every script, driver and doc path still works. |
| **Cluster results** (DelftBlue, mirrored) | `$PE/results/rebuttal_iclr2027/cluster_mirror` → `/export/scratch2/ntm/delftblue_results/results` (148 GB; `INDEX.md` there). Originals on DelftBlue `/scratch/ntmucke/...` |
| **Manuscript generators (live)** | `$PE/results/rebuttal_iclr2027/manuscript_generators` → `/export/scratch2/ntm/delftblue_results/results_tables` (scripts + generated .tex/figures; git snapshot of the scripts in `rebuttal/manuscript_generators/`) |
| **Manuscript** | `$PE/results/rebuttal_iclr2027/manuscript` → `$REPO/manuscripts/iclr_da_interpolant_manuscript` (own git repo, Overleaf; review version = `main_review.tex` + `sections_review/`) |

`REPO=/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants`, `PE="$REPO/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"` (quote it: spaces). Below, `R` = `$PE/results/_runs` = `$PE/results/rebuttal_iclr2027/runs`, `MIR` = the cluster mirror, `RT` = the live manuscript generators, `MS` = the manuscript repo.

Doc abbreviations used below: RE = `rebuttal/docs/REBUTTAL_EXPERIMENTS.md` (cluster campaign log, §1–15); L96H = `rebuttal/docs/LORENZ96_HANDOVER_2026-11-12.md` (+ `LORENZ96_DELFTBLUE_HANDOVER.md`); SQH = `rebuttal/docs/L96_SQUARE_HANDOVER.md`; ABSH = `rebuttal/docs/L96_ABS_HANDOVER.md` (copy of the one in MS); STATUS = `R/STATUS.md` (copy: `run_scripts/STATUS.md`); TMPD = `$PE/TMPD.md`; FC = `$PE/FORECAST_CORRECTION.md` (method docs, left in place); DB = DelftBlue `/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments`. "memory" = notes from the agent sessions, summarised here.

## Resuming

1. **Code:** the branch is locked (read-only on GitHub). Continue from the tag or from `ICLR-2027-review`:
   `git switch -c review/rebuttal-runs-2 rebuttal-runs-final` (or branch off `ICLR-2027-review`). To reopen the branch itself, unlock it:
   `gh api -X DELETE repos/nmucke/scientific-stochastic-interpolants/branches/review/rebuttal-runs/protection`.
2. **Environment (squamish):** `source "$PE/results/_runs/env.sh"` (repo `.venv`, `PYTHONPATH=src`). NS EnKF on CPU needs `.venv-jaxcfd` + `ENKF_JAX_PLATFORM=cpu`. Job wrapper: `bash "$R/run_job.sh" <name> '<cmd>'` (writes `R/<name>_<ts>.log` + `.exit`). Run long jobs in detached tmux (`rb_*` sessions).
3. **GPU:** the RTX 3090 on squamish gave repeated CUDA launch failures (11-10, 11-11, 11-18): reboot / check it before GPU runs. All late L96 runs were on CPU: `env CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu OMP_NUM_THREADS=4 taskset -c a-b python ...` (never `env VARS && cmd`).
4. **New Lorenz 96 runs** (any `--obs linear|arctan|square|abs`): `cases/lorenz96/driver.py` (`reference`, `cell --defer-scoring`, `cell --rescore`), or `cases/lorenz96/quick_cell.py --root DIR --sequences N --ensemble-size E -- <driver args>` for a separate tree. Queue template: `run_scripts/abs5/queue/` (`jobs.txt` = `name|args`, `lane.sh` 4-core lanes pulling from a shared queue, `run_all.sh`, `summary.py`).
5. **Analysis / figures:** L96 x²/\|x\|: `make_l96_square_figures.py --obs {square,abs}`, `make_l96_square_step_figures.py`; manuscript tables/figures: `RT/make_*.py` (section 4).

## 1. Experiments by case


### 1a. Multistep linear-Gaussian ("Exp 1"; analytical, K=20 cycles, N=1e5, seeds 0–4)

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| Exp 1 main grid: 80 cells (`paper`, `partial`, FC variants `paper_fc*`, SURGE, EnKF) at M=50 | squamish CPU | done 2026-11-09 | `$PE/results/analytical_multistep/{metrics,per_step,aggregated}`; copy in `MIR/analytical_multistep/` (165 metric files incl. the rows below) | `R/exp1.sh` → `$PE/run_analytical_multistep_grid.sh`; driver `cases/analytical/multistep.py` | STATUS "Finished" | yes (`tab_ms_l96`, `appendix_analytical_case.tex`) |
| SURGE-cell rerun after the SI-denoiser fix (10 `{paper,partial}_surge` cells) | squamish CPU | done 2026-11-10 | same root; old cells in `analytical_multistep/_stale_surge_velocity_denoiser/` | `R/exp1_surge_rerun.sh` (via `R/after_fixer.sh`) | STATUS | yes |
| M-sweep M ∈ {10, 25, 100, 250} (base + SURGE + FC shared and Jac-free) | squamish CPU | done 2026-11-11 | same root, `aggregated/msweep.csv`; figures `$PE/figures/analytical_multistep/msweep_*` | `R/exp1_msweep.sh <M…>`; then `R/msweep_figures_when_done.sh` → `make_analytical_multistep_msweep_figures.py` | STATUS | yes (KL-vs-M figure) |
| TMPD-only cells `paper_tmpd__M{10,25,50,100,250}` (25 files) | squamish CPU | done 2026-11-18 | same root; copied to MIR | `R/exp1_tmpd.sh <M…>` (`LINEUPS=tmpd`, `tmpd_only`; commit `7fd70cd`) | TMPD; memory | yes (MS commit a5881ba per memory; TMPD equals Ours DM-SDE shared) |
| Timing cell | squamish | done | log only | `exp1_timecell_*.log` | – | – |

### 1b. Lorenz 96, linear and arctan observations (seed 0, 12 test sequences, E=64, M ∈ {25, 50, 100, 250})

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| Linear PF references (N=4e6, 2 replicates) + gate (gate PASSED earlier, reused) | squamish CPU | done 2026-11-09 | `$PE/results/lorenz96/linear/{reference,gate}`; `MIR/lorenz96/linear/reference` | `R/l96_refs.sh`, `R/build_pf_terms.py` | STATUS, `R/gate_summary.md` | yes (reference) |
| Arctan gate + PF references | squamish CPU | done 2026-11-10 (PASSED) | `$PE/results/lorenz96/arctan/{gate,reference}` | `R/arctan_gate_refs.sh` | STATUS, `R/gate_summary.md` | yes |
| KL feasibility study (kNN-KL / energy distance vs PF) | squamish | done 2026-11-09 | log | `R/kl_feasibility.py` | STATUS | method choice only |
| Linear tuning (validation set, 20 steps) | squamish GPU | done 2026-11-10; picks committed `52930eb` | `$PE/results/lorenz96/linear/tuning` | `run_lorenz96.sh tune linear` | STATUS | indirectly |
| Linear final grid, local attempt | squamish GPU | **abandoned**: GPU faults on 11-10 and 11-11; handed over to DelftBlue | 52 metric files kept locally as a cross-check, `linear/_local_rtx3090_seed0/` in MIR; CPU-fallback cells in `linear/_cpu_fallback_after_gpu_fault_20261110_1811/` | `R/final_linear_then_tune_arctan.sh` | STATUS "Incident" | no (cross-check only) |
| Arctan tuning on CPU, local | squamish CPU | **abandoned** 2026-11-11 | `$PE/results/lorenz96/arctan/_stale_cpu_tuning_20261111/` | `run_lorenz96.sh tune arctan` | STATUS | no |
| Linear grid, 78 cells (full lineup) | DelftBlue A100 MIG | done (job 1300170 final) | `MIR/lorenz96/linear/` (DB: `/scratch/ntmucke/lorenz96_delftblue/results/lorenz96`) | `$PE/slurm_lorenz96/cell.sbatch` + `joblist_linear.txt`; `run_lorenz96.sh final linear` | L96H | yes (`tab_ms_l96` at M=250, appendix) |
| Arctan tuning + grid, 78 cells | DelftBlue V100 | done (`si_sde_fc` rerun with online Q in job 1305012, final in 1302733); old exact-Q cells in `arctan/_stale_exactQ/` | `MIR/lorenz96/arctan/` | `slurm_lorenz96/{tune_packed,cell,final}.sbatch` | L96H | yes |
| Shared-Jacobian cells (6 cells × 4 M per sweep) + shared tuning | DelftBlue (linear: MIG 1304946; arctan: V100 1304947; tuning 1303718) | done (status taken from the RE §13 export, "all jobs finished"; I did not count the files) | `MIR/lorenz96/{linear,arctan}/*-shared__M*` | `slurm_lorenz96/shared_{tune,cell}.sbatch` | L96H | yes |
| TMPD, linear only, M ∈ {25, 50, 100, 250} | DelftBlue MIG (job 1309774) | done 2026-11-13 | `MIR/lorenz96/linear/*tmpd*`, `MIR/lorenz96/_tmpd_source/` (symlinks) | `$PE/slurm_tmpd/l96/` | RE §12 | yes |
| Seeds 1–2 | – | **not run** (user decision pending) | – | – | L96H | no |
| "linear_long": PF feasibility for longer obs. interval / obs. variance | squamish CPU | **abandoned** 2026-11-18 (user stopped it) | patch + config `R/linear_long_abandoned/`; logs `R/rb_l96long_sweep*.log` (scripts were in a scratchpad and are lost) | – | memory | no |
| Reference non-Gaussianity diagnostics (linear, square) | squamish | done (figures) | `$PE/results/figures/l96_nongauss/{linear,square}` | `$PE/make_l96_reference_nongaussianity.py` | memory | no (unverified) |

### 1c. Lorenz 96, h(x) = x² (`--obs square`; bimodality test; settings copied from the arctan picks, untuned)

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| PF reference, `square_mixture` proposal, N=4e6, sequences 0–4 × 2 replicates | squamish CPU | done; **fails the gate** (replicates disagree on P(x>0) by up to 0.31) | `$PE/results/lorenz96/square/reference/test/` (only seq0 rep0 has `.pfterms.pt`) | `driver.py --obs square reference --split test --workers 10` (log `R/rb_sq_ref_*.log`) | SQH | no |
| Quick run, seq 0, E=64: EnKF; SI-SDE shared and shared+corr. at M=50/100/250; SI-SDE Jac-free at M=100 | squamish CPU | done, scored | `R/sq1_quick/lorenz96/square/{ensembles,metrics}` | `R/sq1_quick/cmd_*.sh`, `run_cell.py` (`quick_cell.py`, `--defer-scoring` then `--rescore`) | SQH, `R/square_deferred_scoring.md` | no |
| Same at E=256, M=100 (EnKF, SI shared, +corr., Jac-free) | squamish CPU | done | `R/sq1_quick_E256/` | `R/sq1_quick_E256/cmd_*.sh` | SQH | no |
| EnKF on 12 and 5 sequences | squamish CPU | done, unused | `$PE/results/lorenz96/square/{_12seq_enkf,_5seq_enkf}`, `square/ensembles/enkf__seed0.pt` | `sq5_enkf_*`, `sq3_enkf_*` | SQH | no |
| 3- and 5-sequence SI-SDE attempts (`sq3_si*`, `sq5_si*`), GPU attempts (`sq_si`, `sq_si_gpu_blocking`) | squamish | **abandoned**: GPU fault, or stopped (no `.exit`, or exit 1) | logs in R | – | SQH | no |
| Figures/tables | squamish | done (quick1) | `RT/figures/l96_square/quick1*/` (the folder also holds an arctan application of the script: `metrics_arctan.*`) | `RT/make_l96_square_figures.py`, `$PE/make_l96_square_step_figures.py` | SQH | no |

### 1d. Lorenz 96, h(x) = |x| (`--obs abs`; exact optimal-proposal PF `abs_optimal`; code committed in `53ab0a4`)

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| PF reference, sequences 0–4 × 2 replicates, N=4e6 (replicates agree to about 0.01 in P(x>0)) | squamish CPU | done | `$PE/results/lorenz96/abs/reference/test/` | `driver.py --obs abs reference --split test --workers 6` (`R/rb_abs_ref_*.log`) | ABSH, `R/abs_operator_ready.md` | no |
| **Main:** all L96 methods except TMPD, 5 sequences, E=64, M ∈ {25, 50, 100, 250}, 93 cells | squamish CPU | done 2026-11-19 08:54, 0 failures | `R/abs5/lorenz96/abs/{ensembles,metrics,…}`; table `R/abs5/queue/summary.{md,csv}` | `R/abs5/queue/{jobs.txt,lane.sh,run_all.sh,summary.py}`; config `evaluation_abs: {test_sequences: 5}` | ABSH §0 | no ("none of it is in the manuscript yet") |
| Seq 0, E=64: SI/DM/FM × {Jac-free, Jac-free+corr., shared, shared+corr.} × 4 M (overnight: 36 cells + SI cells) | squamish CPU | done (superseded by abs5) | `R/abs1_quick/{lorenz96,overnight/summary.md}`; SI at M=250 retried after exit 1 (`_failed_gpu_M250/`) | `R/abs1_quick/{chain_main.sh,cmd_*.sh,overnight/*}` | ABSH §1 | no |
| Seq 0, E=256, M=100, SI-SDE variants + EnKF | squamish CPU | done | `R/abs1_quick_E256/` | `cmd_*.sh` | ABSH §2 | no |
| 3 sequences, M=100, SI-SDE + EnKF | squamish CPU | done | `$PE/results/lorenz96/abs/{metrics,ensembles}` | logs `R/abs3_*` | ABSH §3 | no |
| Figures (SI-SDE only, 1- and 3-sequence runs) | squamish | done; none for abs5 | `$PE/results/figures/l96_abs/{quick1_M*,quick1_E256_M100,seq3_M100}` | `make_l96_square_figures.py --obs abs` | ABSH | no |

### 1e. Navier–Stokes, poor (wrong) IC (ic_index=15, NP=40 → 35 DA steps, E=64, M=100, traj 10–14, 4 scenarios, guard 10, KL skipped, states traj 11)

| Experiment | Ran on | Status | Results (MIR; DB original) | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| Main grid, 280 cells (Ours × {jacfree, shared_jac10}, FlowDAS, SDA, both SURGE, D-Flow, FIG, EnKF, PF) | DelftBlue V100 (1265369–85, requeue 1275092) | done 280/280; 10 cells diverged | `MIR/navier_stokes_wrong_ic_np40/main/` (DB `results_wrong_ic_np40/`) | `$PE/slurm_wrong_ic/{submit.sh,make_manifest.py,run_task.sh}` | RE §1 | super-res: yes (`tab_ns_accuracy` poor-IC columns, `tab_ns_wic_superres`, appendix figures). Sparse: **no**, as far as I can tell (appendix covers only super-res) |
| SURGE on Ours (SI/DM-SDE), 80 cells | DelftBlue V100 (1281559) | done | `MIR/.../ext/` (`surge_ours_<cov>`) | `$PE/slurm_wrong_ic_ext/` | RE §2 | yes ("+ SURGE" rows) |
| Forecast correction r=8 (`fc25_70a3134d`), 120 cells | DelftBlue V100 (1281559) | done | `MIR/.../ext/` | `slurm_wrong_ic_ext/` | RE §2, FC | no (the MS uses r=20) |
| Forecast correction r=20 (`fc25_8797d2ae`), 120 cells | DelftBlue V100 (1305160) | done (per the 11-16 export) | `MIR/.../ext/` | `slurm_wrong_ic_ext/` (frozen `code_rebuttal_ac8aa01`) | RE §2, §13 | yes ("+ corr. (r=20)") |
| Prior (SI/DM/FM) + EnKF, 60 cells (no localisation on super-res) | DelftBlue V100 (1302963) | done (60 metric files) | `MIR/.../prior_enkf/` | `$PE/slurm_prior_enkf/` (commit `ac4713b`) | RE §3 | yes |
| True-solver EnKF, Gaspari–Cohn r=20 on super-res, 10 cells | DelftBlue V100 (1305653) | done (10 metric files) | `MIR/.../enkf_loc20_superres/production/` | `$PE/slurm_enkf_loc20/` (fix `948fa40`) | RE §4 | yes ("EnKF (r=20)") |
| TMPD (row_sum), 20 cells | DelftBlue MIG (1311468) | done; sparse 5% NaN from step 1 | `MIR/.../tmpd/` | `$PE/slurm_tmpd/ns_wrong_ic/submit.sh` | RE §12, TMPD | yes (super-res) |
| **EnKF E=1000, r=20 reference**, 20 cells | DelftBlue V100 (1355752/3) | **cancelled before start** (2026-11-18) | (DB `results_wrong_ic_np40_enkf_r20_e1000/`, empty) | `$PE/slurm_enkf_e1000/` (on `origin/ICLR-2027-review`, `3cb5ef1`) | RE §15 (origin) | no (row "--") |
| Local E=1000 global EnKF reference (super-res, traj 10–14) | squamish GPU | **abandoned**: first cell FAIL; second run killed; no metrics | `R/ns_wic_ref/` | `R/ns_wic_ref/run.sh` (needs `.venv-jaxcfd`) | memory | no |
| Local reruns of the 14 diverged poor-IC SI-SDE cells (main, r=8, r=20, SURGE) | squamish GPU | **abandoned**: every cell FAIL on a CUDA/cuDNN fault | `R/wic_si_div/` (`cells.txt`, `logs/_failed_cuda_fault/`) | `R/wic_si_div/run.sh` | memory | no |
| Rerun of the diverged 32² SI-SDE shared + corr. r=20 cell, traj 11, other seeds | squamish GPU | partial: seed 1 done (35 steps); seeds 2, 3, 0 stopped; 1-step smoke with seed 0 did not diverge | `R/si_corr_seeds/{smoke,full}` | `R/si_corr_seeds/{run.sh,overnight.sh}` | `R/si_corr_seeds/README.md` | yes: non-NaN field panel (`RT/make_ns_wic_fields.py` reads it) |
| Shared-Jacobian k-ablation from the poor IC | – | **not run** (user decision) | – | – | RE §1, §11 | no (`tab:ablation` = original August runs) |

### 1f. Navier–Stokes, true IC (manuscript setting: NP=20 → 15 DA steps, M=100, E=64, traj 10–14, no KL)

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| FC r=8, Ours × shared_jac10, 60 cells | DelftBlue V100/MIG (1334067 + rebalancing arrays) | done 2026-11-17 | `MIR/navier_stokes_true_ic/ext/navier_stokes/` (`*ours_shared_jac10__fc25_70a3134d*`) | `$PE/slurm_trueic_fc/` (`RADIUS=8 bash submit.sh`), on origin | RE §14 (origin) | no (the MS chose r=20) |
| FC r=20, Ours × shared_jac10, 60 cells | DelftBlue V100 (1334068) | done | same folder (`*shared_jac10__fc25_8797d2ae*`) | `RADIUS=20 bash submit.sh` | RE §14 | yes (`tab_ns_accuracy`, `tab_ns_accuracy_sparse` "+ corr. (r=20)") |
| FC r=20, Ours × Jac-free, 60 cells | DelftBlue V100/MIG (1350434/5, 1351039/40, 1351360) | done 2026-11-18 | same folder (`*ours_jacfree__fc25_8797d2ae*`) | `slurm_trueic_fc/` (`VARIANT=jacfree`) | RE §14c | yes (tables, curves, spectra, fields) |
| TMPD (row_sum), 20 cells | DelftBlue MIG (1340840) | done; sparse diverges (5/5 at 5%) | `MIR/navier_stokes_true_ic/tmpd/` | `slurm_tmpd/ns_trueic/submit.sh` (`NO_KL=1`) | RE §14b | yes |
| SI/FM prior + EnKF, r=20 on all scenarios, 40 cells | DelftBlue MIG (1336446) | done | `MIR/navier_stokes_true_ic/prior_enkf/` | `$PE/slurm_trueic_prior_enkf/submit.sh` | RE §14b | yes |
| FC smokes (1 cell, NP=7) | DelftBlue MIG (1334053/4) | done | `MIR/_smoke_and_bench/navier_stokes_true_ic_fc_smoke/` | – | RE §14 | no |
| 16², traj 11, re-run of Ours Jac-free + SURGE(FlowDAS) + FIG with states saved (for figures) | squamish GPU | done 2026-11-16 | `R/tic16_m100/{metrics,per_step,states}` | `R/tic16_m100/{run.sh,queue.sh}` | memory; `RT/make_ns_wic_fields.py` | yes (field and spectra panels) |
| True-solver EnKF E=64, r=20, super-res, 10 cells | squamish CPU | **cancelled** by the user 2026-11-18 17:35, at step 8–11 of 19; no metrics (only a 2-step smoke) | `R/tic_enkf_loc20/` | `R/tic_enkf_loc20/cell.sh` | memory | no |

### 1g. Other NS items

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| NP=45 long-horizon sweep, sparse 5%, M ∈ {25, 50, 100, 250}, traj 10 only (started on `ICLR-2027-review` **before** this branch) | squamish GPU | **cancelled** 2026-11-08; 10 partial metric files | `$PE/results/navier_stokes_physsteps45_sparse5/` | `run_ns_grid.sh` NP=45 | memory `ns_np45_sweep.md` | no (replaced by the NP=40 poor-IC grid) |
| "sparse k=1 / M=100" | – | **unverified**: I found no rebuttal run of this. The only k=1 runs are the original August ablation (`MIR/navier_stokes/ablation/*shared_jac1*`, M=50) | – | – | – | `tab:ablation` (original) |
| Code-equivalence checks, smokes, benchmarks | DelftBlue | done | `MIR/_smoke_and_bench/{slurm_bench,smoke_prior_enkf,tmpd_smoke}` | `$PE/slurm_bench/` | RE §5 | no |

### 1h. Urban (uDALES): foreign IC 8, NP=150, DA every 3 steps, E=64, M=50, sparse 1.5625% and 0.78125%, traj 1–5

| Experiment | Ran on | Status | Results | Entry point | Doc | In MS |
|---|---|---|---|---|---|---|
| FC r=8 (`fc25_780902ac`), Ours × shared_jac10, 30 cells | DelftBlue V100 (1300217, 1305161) | done (30 metric files); all diverged (MS: "diverged in every urban run") | `MIR/urban_fc/urban/` | `$PE/slurm_udales_fc/` | RE §6 | yes (row "Ours + corr.: div") |
| TMPD (row_sum), 10 cells merged from 4×16-member parts | DelftBlue MIG (smoke 1311490, production 1311493, merge 1333047) | done; all 10 diverged | `MIR/urban_tmpd/urban/` (parts in `parts/`) | `$PE/slurm_tmpd/udales_split/{submit.sh,merge.sh}` | RE §12–13 | yes (row "TMPD: div") |
| Plain (no-FC) comparison rows | – | not rerun (original August RTX 3090 runs) | `MIR/urban*/` | – | RE §6 | yes (original) |

---

## 2. Map of `R` (`$PE/results/_runs`)

Directories:

| Entry | Belongs to |
|---|---|
| `_aborted/` | 2026-11-10 duplicate L96 linear tune launches (aborted) |
| `abs1_quick/`, `abs1_quick_E256/` | L96 abs: seq 0 at E=64 (incl. `overnight/`) and at E=256 (1d) |
| `abs5/` | L96 abs main 5-sequence sweep, 93 cells (1d) |
| `identity/` | bitwise identity snapshots of L96 linear ensembles (refactor checks) |
| `linear_long_abandoned/` | abandoned L96 long-interval case (patch + yaml) |
| `ns_wic_ref/` | abandoned local poor-IC NS E=1000 EnKF reference (1e) |
| `si_corr_seeds/` | poor-IC 32² SI-SDE shared + corr. r=20 rerun with another seed (1e) |
| `sq1_quick/`, `sq1_quick_E256/` | L96 square: seq 0 at E=64 and at E=256 (1c) |
| `sq_si_gpu_blocking/` | square GPU attempt with `CUDA_LAUNCH_BLOCKING` (failed) |
| `tic16_m100/` | true-IC NS 16² traj 11 rerun with states (1f) |
| `tic_enkf_loc20/` | cancelled true-IC NS EnKF r=20 on CPU (1f) |
| `wic_si_div/` | abandoned local reruns of diverged poor-IC SI cells (1e) |

Scripts and notes:

| Entry | Belongs to |
|---|---|
| `env.sh`, `run_job.sh` | environment, job wrapper (writes `<name>_<ts>.log` + `.exit`) |
| `STATUS.md` | live log for multistep and L96 (squamish) up to 2026-11-16 |
| `exp1.sh`, `exp1_surge_rerun.sh`, `exp1_msweep.sh`, `exp1_tmpd.sh`, `msweep_figures_when_done.sh`, `test_analytical_noarchive.sh` | multistep (1a) |
| `l96_refs.sh`, `arctan_gate_refs.sh`, `build_pf_terms.py`, `kl_feasibility.py`, `time_final_cells.py`, `final_linear_then_tune_arctan.sh` | L96 linear/arctan (1b) |
| `after_fixer.sh`, `post_merge_checks.sh`, `linear_identity_snapshot.py`, `snap_before.sh` | code-merge gates (tests + identity) |
| `copy_snellius_results.sh` | MIR import 2026-11-16 |
| `gate_summary.md` | linear + arctan gate table |
| `square_deferred_scoring.md` | square `--defer-scoring` / `--rescore` how-to |
| `abs_operator_ready.md` | abs wiring and commands |

Log families (`*.log` + `.exit`; no `.exit` means killed or still running when stopped):

| Family | Belongs to / outcome |
|---|---|
| `test_root_*`, `test_pe_*`, `post_merge_checks_*`, `after_fixer_*`, `linear_snapshot_*` | test and identity gates, 11-09/11-10, exit 0 |
| `exp1_*`, `msweep_figures_*` | 1a, all exit 0 (`exp1_tmpd_a_…114743` superseded by `…114906`) |
| `l96_prior_quality_*`, `l96_refs_*`, `arctan_gate_refs_*`, `kl_feasibility_*`, `build_pf_terms_*`, `time_final_M250_*`, `tune_linear_*` | 1b, exit 0 (`tune_linear_…093925` without exit = superseded) |
| `tune_linear_surge_flowdas_*` | 1b, widened SURGE(FlowDAS) grid, exit 0 |
| `final_linear_seed0_*` (11-10, 11-11) | 1b local final; no exit; stopped after GPU faults |
| `tune_arctan_cpu_*` | 1b; stopped 11-11 |
| `copy_snellius_results_*` | MIR import, exit 0 |
| `ns_wic_ref_*` | 1e, abandoned |
| `wic_si_div_*` | 1e, every cell FAIL (GPU fault) |
| `sq_*`, `sq1_*`, `sq3_*`, `sq5_*`, `sqE256_*`, `rb_sq_ref_*` | 1c. `sq1_*`/`sqE256_*`/`sq*_enkf` exit 0; `sq3_si*`, `sq5_si*`, `sq_si*` failed or were killed; `rb_sq_ref` finished all 5 sequences (no `.exit` file) |
| `abs1_*`, `abs3_*`, `absE256_*`, `abs_overnight_*`, `rb_abs_ref_*` | 1d. All exit 0 except `abs1_si_{fc,shared}_M250_…194756` (exit 1; rerun …201729 OK) and 2 `absE256_…194639` launches without exit (relaunched …194915 OK) |
| `rb_l96long_sweep*` | abandoned linear_long PF feasibility |

---

## 3. Open items and known issues

- **Poor-IC NS EnKF E=1000 reference: never run.**
  - DelftBlue jobs 1355752/3 were cancelled before they started.
  - The local attempt `R/ns_wic_ref` failed.
  - The MS shows "--" for this row.
- **Diverged poor-IC SI-SDE (shared_jac10) cells.** 5 of 20 main cells diverged, plus several r=8/r=20 FC cells and one SURGE cell (14 listed in `R/wic_si_div/cells.txt`).
  - The local reruns were not possible because of GPU faults.
  - The 32² traj 11 cell is knife-edge: seed 0 did not diverge on the 3090 in a 1-step smoke. The MS field panel uses the seed 1 rerun.
- **Global EnKF diverged** on poor-IC 16² (5/5). The original EnKF code path ignores the divergence guard.
- **Last-step guard caveat** (RE §12): if the guard trips on the last step, the cell is not padded with NaN, so `aggregate_ns._is_diverged` misses it.
  - The uDALES merge handles this; the NS cells were not checked (**unverified**).
- **squamish RTX 3090 GPU faults** (`unspecified launch failure`, illegal memory access, cuDNN EXECUTION_FAILED) on 11-10, 11-11 and 11-18. As a result:
  - the L96 grid was moved to DelftBlue;
  - square and abs ran on CPU only;
  - NS local runs were abandoned.
  - The machine needs a reboot or hardware check.
- **L96 square PF reference fails the gate.** Replicates disagree in P(x>0) by up to 0.31 and in log-Z by up to 4.9 (likely path degeneracy). The user accepted it for now; fix with larger N or MCMC moves. Square results cover only seq 0 and SI-SDE.
- **L96 |x|:**
  - Every SURGE variant is extremely poor (sliced W2 113–242% worse than the EnKF), and D-Flow SGLD degrades with M. Both are suspicious and unchecked.
  - Nothing was tuned on |x|.
  - There are no figures for abs5.
  - The legend reads "±y/c" and should read "±y".
  - The config value `evaluation_abs.test_sequences` was changed for abs5 (5). `evaluation_square` = 3.
- **L96 shared cells refresh the Jacobian every pseudo-step (k=1)**, while the field cases use k=10. The MS states this.
- **L96 seeds 1–2 not run.**
- **FC is untuned everywhere** (r=8/20, gain 1, 25 forecast steps, no cap).
  - On uDALES, every FC and every TMPD cell diverged.
  - On true-IC NS, FC worsens every cell (r=20 worse than r=8 in 9/12 on RMSE).
- **SURGE-on-Ours is defined differently per case**: NS uses the `velocity` denoiser; L96 and multistep use `follmer` (RE §11, open).
- **Not run (user decisions):** poor-IC k-ablation, NS TMPD with FC or SURGE, TMPD on nonlinear observations, and uDALES aggregation scripts for the FC cells ("no aggregation script written yet", RE §10; the MS now reports "div").
- **Not in the manuscript:** NS poor-IC sparse results and the L96 square/abs experiments.

## 4. `RT` generator scripts (`make_*.py`)

| Script | Purpose (from its docstring) |
|---|---|
| `make_results_tables.py` | per-case LaTeX tables (multistep, L96, NS, uDALES) from the MIR per-trajectory files, mean ± std, diverged excluded; `--ours-only` variant → `results_tables{,_ours}_body.tex` |
| `make_ns_table1.py` | new `tab:ns_accuracy` (true + poor IC super-res, + corr. r=20, TMPD, prior+EnKF) and the NS timing table |
| `make_ns_wrong_ic_step_figures.py` | poor-IC NS metric vs step (super-res 16²/32²) |
| `make_ns_true_ic_step_figures.py` | true-IC NS metric vs step, 4 scenarios, same design as the poor-IC figures |
| `make_ns_true_ic_spectra.py` | true-IC enstrophy spectra with the new runs (needs the repo `.venv`; uses `R/tic16_m100`) |
| `make_ns_wic_fields.py` | NS field panels (truth, obs, mean, std, \|err\|) for poor and true IC (uses `R/si_corr_seeds`, `R/tic16_m100`) |
| `make_ns_wic32_tex.py` | LaTeX wrappers for the appendix field figures (`ns_wic_*`, `ns_tic_*`) |
| `make_vs_M_figures.py` | metric vs M for multistep and L96 linear/arctan (`figures/vs_M_{ours,baselines}_{kl,w2}`) |
| `make_l96_square_figures.py` | L96 square/abs metrics table + final-step bimodal marginals (PF vs EnKF vs SI-SDE) |

`RT` also holds the generated `.tex` files: `ns_table1.tex`, `ns_table_sparse.tex`, `ns_timing.tex`, `manuscript_tab_ms_l96.tex`, `manuscript_tab_ns_wic_superres.tex`, and the `ns_{wic,tic}_*_figure.tex` files. Its figures are in `figures/{ns_tic,ns_tic_spectra,ns_wic,ns_wic_fields,l96_square}/`.
