# Rebuttal runs — STATUS

Branch `review/rebuttal-runs` (from `ICLR-2027-review` @ e06ab51). Clone:
`/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants`.
Env: `source "$PE/results/_runs/env.sh"` (sets REPO, PE, venv, PYTHONPATH, PY, MPLBACKEND).
Job wrapper: `bash "$PE/results/_runs/run_job.sh" <name> '<cmd>'` → `_runs/<name>_<ts>.log` + `.exit`.

## Environment (2026-11-09)
- Python 3.13.9, torch 2.9.1+cu128 (CUDA 12.8, available), RTX 3090 24 GB, Ryzen 9 3950X 16C/32T, 125 GB RAM.
- `uv sync` run 2026-11-09 (re-resolved; no committed lockfile).
- Earlier work on this machine (clone `/export/scratch1/ntm/postdoc/scisi-lorenz96`, branch `review/lorenz96` @ b8f2049):
  torch 2.15 crashed during training; training was done on torch 2.9.0+cu128. Logs: that clone's `logs/`.

## Lorenz 96 artefacts on disk
- Moved (user request 2026-11-09) from `scisi-lorenz96` into this clone:
  - `data/lorenz96/data.npz` (2026-11-08 16:55)
  - `checkpoints/lorenz96/{si,fm}/` — 40 epochs each, early stopping did NOT fire (patience 15);
    best val loss SI 0.33683 (epoch 27), FM 0.39064 (epoch 28); model.pth = best epoch.
    Trained on `data/lorenz96/data.npz` traj 0–3800 / val 3800–4000.
  - Old results → `$PE/results/_stale/lorenz96_from_scisi-lorenz96_worktree/`:
    gate (convergence_4e6: 10 seq × 8 reps, **PASSED**; calibration N=1e5 32 seq; cross_proposal),
    prior_quality.json (40-ep: SI 0.0680, FM 0.0545, floor 0.0442), prior_rollouts/.
    PF/dynamics/pf_gate code unchanged between b8f2049 and e06ab51 → gate still valid.
- PF references (linear): regenerated here 2026-11-09 (see Finished). Tarballs not used.

## Running
- `rb_final_linear` (relaunched 2026-11-11 12:48, 8fd0964): same chain `final_linear_then_tune_arctan.sh` (SEEDS=0). Skips finished cells; reruns M=100 (17) + M=250 (18), then `tune arctan`. Est. ~4.5 h + 1.5 h. Log `_runs/final_linear_seed0_20261111_1248*.log`.
  **Seed 0 only** (user 2026-11-10). REDUCED vs brief.

- `rb_tune_arctan_cpu` (2026-11-11 12:53, 30d1b39): `CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu THREADS=6 ./run_lorenz96.sh tune arctan` - arctan tuning on CPU (user: only for picking hyperparameters). Est. 15-25 h. Holds `results/lorenz96/arctan/logs/.tune.lock`, so the chain's later GPU `tune arctan` refuses (exit 3) while it runs - expected, the chain then ends non-zero after the linear final.
  NOTE: CPU job shares cores with the GPU linear final -> its per-step timings may be slightly inflated; state it.

- Exp 1 M sweep (user 2026-11-11: KL / W2 vs M figures), CPU, OMP 4 threads/lane: `rb_ms_a` = `exp1_msweep.sh 250`, `rb_ms_b` = `exp1_msweep.sh 10 25 100`. paper system only, K=20, N=1e5, seeds 0-4; base + surge + fc (S = M, shared + Jac-free) per M. M=50 already exists. Est. 4-6 h. `rb_ms_fig` waits for both and runs `make_analytical_multistep_msweep_figures.py` (59c18b0) -> figures/analytical_multistep/msweep_{all_variants,fc_vs_nofc}.*, aggregated/msweep.csv. jacobian_damping analytical = 1.0 at every M (M=10 default 1.0, same).

## 2026-11-16 DelftBlue results import + branch merge
- Merged origin/ICLR-2027-review (29 commits) into review/rebuttal-runs (dc94f89, clean) and fast-forward pushed it to ICLR-2027-review (eefefbd..dc94f89); includes 868156f.
- `rb_snellius_copy` (13:01): `copy_snellius_results.sh` pulls snellius:/scratch-shared/nmucke1/results/ (~155 GB, 15,425 files in MANIFEST.md5) to /export/scratch2/ntm/delftblue_results/results/, then `md5sum -c MANIFEST.md5`. ~46 MB/s -> ~1 h + md5. Resumable (rerun script).
- DONE 13:25: 15,452 files / 155 GB copied; md5 vs MANIFEST.md5 all OK except top-level .DS_Store (manifest stale; our copy == Snellius). Exp 1 added under analytical_multistep/, _figures/analytical_multistep/, _generated/tab_analytical_multistep*.tex (901 files, MANIFEST_analytical_multistep.md5, INDEX.md section; old INDEX in _overwritten/).
- Not done (offered): link it as $PE/results (move local results -> results_local_squamish/), add local-only analytical_multistep/ and _runs/ without overwriting.

## Lorenz 96 HANDED OVER to DelftBlue (2026-11-11 ~14:00, user decision after GPU faults)
- Branch `review/lorenz96-delftblue` pushed (45dada4) with `$PE/LORENZ96_DELFTBLUE_HANDOVER.md`; review/rebuttal-runs fast-forwarded to it (not pushed).
- Inputs tarball `/export/scratch1/ntm/postdoc/lorenz96_delftblue_inputs.tar.gz` (1.1 GB, sha256 0419453...ea96 in `.sha256`): data, checkpoints, linear+arctan references (+pfterms), gates, linear prior_quality + tuning, `linear/_local_rtx3090_seed0/` (52 GPU-run metric files, cross-check only).
- Local arctan CPU tuning stopped; partials -> `results/lorenz96/arctan/_stale_cpu_tuning_20261111/`. No Lorenz 96 jobs run here any more.

## Incident 2 (2026-11-11 13:07-13:53): GPU faults again
- Linear final M=100: surge_unguided_si, surge_unguided_dm (`unspecified launch failure`), flowdas (`illegal memory access`) failed; surge_flowdas hung from 13:17 (log frozen, 100% CPU, GPU 0%). Crash sites differ (async error surfacing) -> GPU/driver, not code. CUDA re-initialises per process this time.
- Chain `rb_final_linear` STOPPED 13:53. Done on GPU since relaunch: M=100 prior_enkf_fm, si/fm/dm_sde, *_fc, surge_si_sde, surge_dm_sde (10/18 M=100 files incl. prior_enkf). Remaining: M=100 surge_unguided_si/dm, flowdas, surge_flowdas, sda, surge_sda, dflow, fig; all 18 at M=250.
- No NVRM/Xid in kernel log (journal may not capture). Temp 63C, 129 W at check. CPU load ~12.6 (arctan CPU tune + Exp1 sweep) during the faults; yesterday's fault had no CPU jobs alongside.
- WAITING FOR USER: how to proceed with GPU work.

## Incident 2026-11-10/11 (GPU fault)
- 18:11 `prior_enkf_fm` M=100: `CUDA error: unspecified launch failure`. Afterwards every process got `CUDA unknown error` and `device: auto` fell back to CPU (samplers AND scorer) until the machine was rebooted 2026-11-11 12:41 (chain killed mid surge_si_sde M=250). No NVRM/Xid lines in the previous boot's kernel journal.
- CPU-fallback outputs (16 cells at M=100, 8 at M=250; metrics/per_step/per_sequence/ensembles) moved to `results/lorenz96/linear/_cpu_fallback_after_gpu_fault_20261110_1811/`. Kept (GPU): M=50, M=25, fixed cells, prior_enkf M=100 (43 files).
- Guard 8fd0964: run_lorenz96.sh exports SCISI_REQUIRE_CUDA=1; driver exits if CUDA unusable. GPU health check after reboot OK (8192^2 matmul stress 10 s).

## Finished
- 2026-11-10 7a30aa9: PF-only KL/energy terms cached per reference (`<ref>.pfterms.pt`, checksum-guarded); scores identical; 12.6 s -> 0.3 s per sequence. M=250 timing (`_runs/time_final_M250_*.log`): linear generative ~15.5 h run at 3 seeds (D-Flow 3.8 h, SURGE/SDA 1-2 h each).
- 2026-11-10 13:02-14:24 **linear tuning DONE** (seed 0, 4 val seq, 20 steps). Applied 14:47 -> 52930eb (+ test 87f66e9). Picks: enkf 1.0/loc4, enkf_large 1.0/loc8, samplers Jacobian-free (damping irrelevant there), *_fc gain 0.75 online, prior_enkf* 1.0/loc4, flowdas 1.0, fig (1,1), dflow 0.02/1e-3, sda 1e-4, surge_si 1.0, surge_dm 0.25, unguided si 0.5 / dm 0.25, surge_sda 1e-5/0.5, surge_flowdas 0.004/0.5.
  SURGE(FlowDAS) stage-2 grid widened x{0.25,1,4} -> x{0.004..4} (narrow grid all collapsed, ESS 1/64; old rows in linear/_stale_tuning_narrow_surge_flowdas_grid/). Guidance >= 0.064 collapses.
  SURGE at E=64, d=16 is under-dispersed (spread 0.7-1.6 vs EnKF 2.6); guided worse than unguided (1 val seq: surge_si min ESS 0.03-0.3, 6-33 resamples/step; unguided 0.4-0.5, 1-4). Report as a negative result.
- 2026-11-10 aa4f462: L96 scoring on GPU (torch sliced W2 / 1-NN KL / energy, float64; match library/scipy to ~2e-15 rel, energy 2e-6 since old cdist used matmul path); tuning horizon 20 steps. EnKF tuning row 9 min -> ~37 s. L96 tests 230 passed. Cause: library sliced W2 np.quantile partition 2.7 s/step.
- 2026-11-10 10:06 rb_exp1_surge DONE (exit 0): SURGE around SI-SDE now KL ~1-2e-4 at all k (paper), unguided 6e-5..3e-4. Gate before it: root 359+2 skip, $PE 433, identity IDENTICAL.
- 2026-11-10 09:19 **fixer merged** (e88c057 ← 095b2a0, 36cfdaf, c4e9dbd): one lineup source, one NS builder, `run_lorenz96.sh {tune|apply|final} {linear|arctan}`; kl_knn on unique PF rows, only core scores count as diverged; opt-in `si_denoiser="follmer"` ON for surge_si_sde/surge_unguided_si/surge_flowdas + multistep SURGE-around-Ours SI rows (OFF: NS, analytical single-step, multistep SURGE(FlowDAS) baseline); SDA one sequence per batch; base-aware stage-2 skip. Agent: every-cell snapshot IDENTICAL after refactor; after fixes only sda/surge_si_sde/surge_unguided_si/surge_flowdas ensembles changed; linear identity IDENTICAL; $PE 433 passed, root 359+2 skipped.
- 2026-11-10 d8b53c5: multistep main-text figure `figures/analytical_multistep/analytical_multistep_main.*` + legend spacing fix.
- 2026-11-09 21:54 → 2026-11-10 (exit 0) **arctan gate PASSED** (10 seq × 8 reps, N=4e6; ESS abs min ≥ 39501, mean-err q95 ≤ 0.056, log-var q95 ≤ 0.053, log-Z sd ≤ 0.26, rate −0.35…−0.50), calibration (correct z² 1.026±0.021; wrong-operator-linear control z² 1.58), cross-proposal; **arctan references** validation 8/8, test 24/24. Table: `_runs/gate_summary.md` (linear + arctan).
- 2026-11-09 18:44–21:28 **Exp 1 done** (exit 0): 80/80 cells OK, aggregate/figures/diagnose. Code drifted during the run (bb58153 → 2d406b7; analytical code untouched, library PF/EnKF changes linear-bitwise). Matches background: plain shared cycle KL 0.068–0.073; +FC SI 5.2e-4 / DM 8.3e-4 / FM ~1.5e-3; EnKF 5.5e-5; SI works from S=1, DM needs S≥10; Jac-free+FC worse (0.076→0.17); SURGE(SDA) K=1 3.0e-4.
  TODO: rerun the 10 `{paper,partial}_surge` cells with the SURGE SI-denoiser fix (move old files aside to `results/analytical_multistep/_stale_surge_velocity_denoiser/`), then re-aggregate/figures/diagnose.
- 2026-11-09 21:00 post-merge checks on 2d406b7: root 349 passed/2 skipped; $PE 51 failed (tests build paths without `__M<M>`; kl_knn NaN on duplicate PF draws) → fixer; linear identity IDENTICAL.
- 2026-11-09 reviews: correctness (kl_knn NaN for arctan → tuning dead; SURGE SI denoiser library bug (b treated as v) → early-ESS collapse is an artefact; stage-2 stale reuse; SDA batch divergence; pf_subsample seed lacks split) and complexity (dedupe NS builders, one lineup source, dispatch table, merged run script, `apply` step).
- 2026-11-09 21:24 KL feasibility (4 val seq, 3 seeds): KL(PF‖ens) floor 1.95±0.02, EnKF 4.73±0.10, oracle 8.18±0.09, pf_matched 9.01±0.33 → usable (gaps ≫ seed std); energy distance orders the same. kl_knn = KL(PF‖ens).
- 2026-11-09 ~20:30 **code merged** on review/rebuttal-runs: arctan 2b423f9, baselines 87b3ea5, SMC 9685021, pipeline 27f7bed → merges d9d315b/3082111/b5e8f0f/a3216c0, integration 2d406b7 (unguided controls = surge_unguided_si/_dm; `cell --num-steps`; `__M<M>` file names; kl_knn = KL(PF‖ens) 1-NN + energy_distance).
  Each agent reported its own tests passing and IDENTICAL linear snapshot. Timings (1 val seq, GPU, M=50, untuned): fig 0.15 s/step, flowdas 0.31, sda 0.33, dflow 4.3 (NFE 1212); surge_* 0.25–2.9 s/step. SURGE on SI prior: ESS → 1/64 at first pseudo-step of every window (reviewer asked to check).
  Arctan: c = 0.2508373 (1/σ_clim, σ_clim=3.986648); PF proposal `linearised` (exact IS); IS oracle P=4096, median ESS 1637; est. arctan refs+gate ≈ 11.5 CPU-h (~2 h wall with 8 workers).
- 2026-11-09 ~20:30 linear results moved: `results/lorenz96/{reference,prior_quality.json}` → `results/lorenz96/linear/`; gate JSONs copied to `linear/gate/`.
- 2026-11-09 17:37 root tests: 337 passed, 2 skipped (MPS-only test + "fixed Q is not localised"). Mac: 338/1; diff = MPS skip. OK.
- 2026-11-09 17:37 `$PE` tests: 278 passed (matches Mac), 8.5 min. Includes analytical multistep tests on the on-disk June analytical_utils.
- 2026-11-09 18:07–18:31 **linear PF references** (N=4e6, 2 reps): validation 8/8, test 24/24 files, ~410–450 s each, 14 workers. In `results/lorenz96/reference/` (to move under `lorenz96/linear/` once operator paths land).
- 2026-11-09 bb58153: analytical KL/W estimators moved into `cases/analytical/_metrics.py` (archive no longer needed; 96 analytical tests pass with archive hidden).
- 2026-11-09 88d328e: L96 M=50 fixed (guard in cell/tune, `--allow-other-m`), prior_enkf 25→50, `observation_model()` helper. Bitwise-identical ensembles vs bb58153 (`identity/before_bb58153.pt`, 11 cells, CPU).
- 2026-11-09 17:48 L96 prior-quality (GPU, 38 s): SI 0.0680, FM 0.0545, floor 0.0442, var SI 0.979 FM 0.969. Identical to the earlier run on this machine → checkpoints load, data check passes. NOTE: SI worse than 8-epoch Mac model (0.0573); FM ~same (0.0524).

## Queued
- After chain: review `results/lorenz96/arctan/tuning/selected_stage{1,2}.md` -> `apply arctan` -> commit -> `SEEDS=0 final arctan`. Then assess seed-0 results with user (more seeds?).
- After linear tuning: review `results/lorenz96/linear/tuning/selected_stage{1,2}.md` → `./run_lorenz96.sh apply linear` → commit config → time cells at M=250 → send compute estimate to user → `./run_lorenz96.sh final linear` → `tune arctan` → apply → `final arctan`.
- Then: REBUTTAL_RESULTS.md, LaTeX tables, results tarball.

## Decisions waiting for user
- (none open)

## Decisions made
- 2026-11-09 user: SURGE SI denoiser → opt-in library fix (default off; NS unchanged); ON for L96 SURGE-SI cells and the multistep SURGE-around-Ours rows; rerun Exp 1's 10 `surge` cells (old ones move aside). NS SURGE(FlowDAS) not rerun: state the bug.
- 2026-11-09 user: L96 final grids at M ∈ {25, 50, 100, 250} for every generative cell, both sweeps, seeds 0–2 (convergence plots). Tuning stays at M=50; knobs reused at all M. M-independent cells (EnKF, PFs, oracle) once.
  To implement at merge: `cell --num-steps M` (allowed set), M in file names (`lorenz96__<cell>__M<M>__traj<s>__.csv`), guard allows only {25,50,100,250}. Pipeline agent informed.
- 2026-11-09 user: KL vs PF must be nonparametric (not Gaussian). Plan: kNN (Perez-Cruz) KL + energy distance as fallback; feasibility study `_runs/kl_feasibility.py` (tmux rb_klstudy).
- 2026-11-09 user: regenerate references here; initial ensembles must not use the true state (verified: all cells and PF draw from N(prior_mean, 1), truth only scored); analytical_utils → moved into case (done); continue with 40-epoch models.
- 2026-11-09 user: L96 full lineup (baselines + SURGE+Ours) and second sweep with arctan obs (c = 1/σ_clim); M = 50 fixed for every generative method (not tuned).
- Gate (linear): keep the existing 10 seq × 8 rep N=4e6 PASS (in `_stale/.../gate`), PF code unchanged since.
- Reuse moved 40-epoch checkpoints and data (config, data and load checks all passed).
