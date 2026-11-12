# Lorenz 96 rebuttal runs: handover to DelftBlue

Written 2026-11-11 on the workstation that ran the work so far. Its RTX 3090 started
throwing CUDA faults mid-run (`unspecified launch failure`, `illegal memory access`, a
hung kernel), so every remaining Lorenz 96 run moves to DelftBlue. This note is for the
agent that runs them there. `PE` below means
`papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments`
(relative to the repo root; mind the spaces when quoting it).

## Ground rules (from the user; keep them)

- The paper is under double-blind review. Never edit the manuscript. Web searches must
  be generic: never search for the paper's title, authors or repository.
- Ask the user before any change to the scientific setup, before pushing, and before
  anything that takes more than about a day of compute.
- No choice may come from the test split. Tuning uses the validation split only.
- Report every number from finished runs, state reduced settings, report negative
  results plainly. Never delete results: move them aside (`_stale_*`).
- Navier-Stokes and uDALES are out of scope: do not run them or change their code.
- Do not touch branch `ICLR-2027-submission` or tag `ICLR-2027-submission-snapshot`.
  A PR, if the user asks for one, targets `ICLR-2027-review`.
- Stop and ask if a test fails, if CUDA results look inconsistent, or if a result
  contradicts what this note says to expect.

## Code

Branch `review/lorenz96-delftblue` (this branch). Environment: Python 3.13, torch 2.9.1
(+CUDA 12.8 wheels were used on the workstation), `uv sync` from the repo root, then

    source .venv/bin/activate
    export PYTHONPATH=$PWD/src MPLBACKEND=Agg

Tests: `cd "$PE" && python -m pytest -q tests -k lorenz96` (230+ tests, ~7 min) and the
root suite `python -m pytest -q tests` (~2 min). On the workstation the full suites last
passed at e88c057 (root 359 + 2 skipped, $PE 433); the Lorenz 96 tests passed again after
the later Lorenz 96 changes (GPU scoring, tuning horizon, PF-term cache, CUDA guard, lock).

## Data (not in git: comes as a tarball the user transfers)

`lorenz96_delftblue_inputs.tar.gz`, unpacked FROM THE REPO ROOT, gives:

- `data/lorenz96/data.npz` (training data; 245 MB)
- `checkpoints/lorenz96/{si,fm}/` (40-epoch SI and FM priors; best epoch saved)
- `$PE/results/lorenz96/linear/`: `reference/` (PF ground truth, N = 4e6, validation 8
  files + test 24 files = 2 replicates x sequences, with `.pfterms.pt` score caches),
  `gate/` (PF convergence gate, PASSED), `prior_quality.json`, `tuning/` (finished linear
  tuning, already applied to the config)
- `$PE/results/lorenz96/arctan/`: `reference/` (same layout), `gate/` (PASSED)
- `$PE/results/lorenz96/linear/_local_rtx3090_seed0/`: the linear final cells the
  workstation finished on its GPU before it failed (metrics / per_step / per_sequence of
  M = 25, 50, the M-independent cells and 10 cells at M = 100, seed 0). For a cross-check
  only; see "Consistency" below. They are NOT in the live result folders, so nothing is
  skipped because of them.

The `.pfterms.pt` files cache the PF-only parts of the KL / energy scores; they carry a
checksum of the reference draws and are rebuilt automatically if they do not match.

## What is decided and done

- Setup: L96 d = 16, F = 8, observation interval 0.2, model noise 1 (one stage), every
  4th site observed, R = 4, 50 assimilation steps; 12 test / 4 validation sequences;
  ensemble E = 64; initial ensembles from N(prior mean, I), never from the true state.
  Two observation sweeps: `linear` (H = every 4th site) and `arctan`
  (h(x) = arctan(c x)/c on the same sites, c = 1/sigma_clim = 0.2508373).
- Generative cells run at M in {25, 50, 100, 250} SDE/ODE steps (convergence plots);
  hyperparameters are tuned at M = 50 and reused at every M.
- Tuning rule: validation split, seed 0, the first 20 of 50 steps
  (`evaluation.tuning_steps`), lowest sliced W2; diverged rows ineligible. Stage 1
  (filters, samplers, baselines) -> selection -> stage 2 (correction and SURGE cells,
  given their parent's stage-1 picks) -> selection -> STOP for review -> `apply`.
- LINEAR tuning is done and applied (`configs/case/lorenz96.yaml`, block `methods:`;
  commit 52930eb). ARCTAN tuning is NOT done (block `methods_arctan:` still holds
  untuned defaults).
- Seeds: the user asked for seed 0 only for now; they will decide on seeds 1-2 after
  seeing the results.
- Scoring runs on the GPU (`SCISI_SCORING_DEVICE` overrides; CUDA when available).
  `run_lorenz96.sh` sets `SCISI_REQUIRE_CUDA=1`: a cell refuses to start if CUDA is not
  usable, instead of silently falling back to the CPU (that happened on the workstation).

## Known results to expect (linear tuning, validation, 20 steps)

Sliced W2 of the picks: EnKF 1.21, EnKF E=2000 0.96, prior_enkf_fm 1.18, SI/FM/DM-SDE
(Jacobian-free) ~1.7, with forecast correction ~1.25-1.29, FlowDAS/FIG ~1.64, D-Flow
1.80, SDA 2.54. Every SURGE cell is worse (2.7-3.6) and under-dispersed (SMC weight
degeneracy with 64 particles in d = 16; guided worse than unguided); SURGE(FlowDAS)
collapses for guidance >= 0.064. These are genuine negative results, not bugs.

## What to run

1. Unpack the tarball, `uv sync`, run both test suites.
2. Linear final grid (seed 0): every generative cell at M in {25, 50, 100, 250} and the
   M-independent cells once, on the 12 test sequences.
3. Arctan tuning: `./run_lorenz96.sh tune arctan` (one GPU, sequential, ~1.5 h). Then
   STOP: send the user `results/lorenz96/arctan/tuning/selected_stage{1,2}.md` with a short
   reading (anything diverged / collapsed / at a grid edge). After the user agrees:
   `./run_lorenz96.sh apply arctan`, commit the config.
4. Arctan final grid (seed 0), as in step 2 with `--obs arctan`.
5. Per sweep, once every cell is done: `SEEDS=0 ./run_lorenz96.sh final <obs>` once more
   on one GPU. It skips every finished cell, adds the `pf_reference` row and runs
   `aggregate_lorenz96.py`, `make_lorenz96_figures.py`, `make_lorenz96_tables.py`.

Cells (`python lorenz96_lineup.py cells generative|fixed`):

- generative, at each M: prior_enkf prior_enkf_fm si_sde fm_ode dm_sde si_sde_fc
  fm_ode_fc dm_sde_fc surge_si_sde surge_dm_sde surge_unguided_si surge_unguided_dm
  flowdas surge_flowdas sda surge_sda dflow fig (18 x 4 = 72 jobs per seed)
- fixed (no M): pf_subsample enkf enkf_large pf_matched oracle oracle_fc (6 per seed)

One job = one cell, one M, one seed (all 12 sequences):

    cd "$PE"
    SCISI_REQUIRE_CUDA=1 python cases/lorenz96/driver.py --obs linear cell \
        --method <cell> --seed 0 [--num-steps <M>] --threads 4

Outputs go to `results/lorenz96/<obs>/{metrics,per_step,per_sequence,ensembles}/` under
`lorenz96__<cell>[__M<M>]__traj<seed>__.*`; a job writes only its own files, so array
jobs can run concurrently. A job whose metrics file exists is done (`run_lorenz96.sh`
uses the same rule to skip).

## Resources: the jobs are small

The individual Lorenz 96 jobs need little memory and time, so they can run on the
smaller GPUs; there is no need for A100s.

- GPU memory: 0.4-0.6 GB observed per job. CPU RAM: a few GB (4 GB is ample; the
  references are ~19 MB per sequence).
- Wall time per job on the RTX 3090 (12 test sequences, including scoring and a
  one-sequence timing rerun), at M = 250: dflow ~45 min; surge_sda, surge_flowdas
  ~25 min; sda ~18 min; surge_si/dm_sde, surge_unguided_* ~10-12 min; flowdas ~4 min;
  everything else < 5 min. Roughly proportional to M, so M = 25 is ~10x faster. The
  fixed cells take minutes (arctan `oracle` uses 4096 importance proposals per member and
  is the slowest of them). A 2 h limit covers every job; 1 h covers all but dflow at
  M = 250.
- Suggested SLURM request per job: 1 GPU (any type), 4 CPUs, 8 GB RAM, 2 h.
- Arctan tuning (step 3) is one sequential job of ~1.5 h on one GPU (~130 rows).

## Consistency

- Run all cells of a sweep on ONE GPU type and record it (`nvidia-smi --query-gpu=name`
  in each job log). A different device gives a different realisation of the stochastic
  samplers; do not mix devices within a sweep without telling the user.
- The `seconds` / per-step cost columns come from a one-sequence rerun inside each job.
  They are only comparable across rows if jobs do not share a GPU; one GPU per job.
- Cross-check: once the DelftBlue linear M = 50 cells exist, compare their
  `mean_error_vs_pf`, `sliced_w2` and `kl_knn` with
  `results/lorenz96/linear/_local_rtx3090_seed0/metrics/`. Expect agreement to a few
  per cent (different device, different random realisation), not bitwise. A large gap
  would mean the workstation GPU corrupted results before it failed visibly: report it.

## Deliverables back to the user

- A results tarball: `$PE/results/lorenz96/{linear,arctan}/{aggregated,metrics,per_step,
  per_sequence,tuning,logs}`, `$PE/figures/lorenz96/`, `$PE/generated/tab_lorenz96*.tex`
  (leave out `ensembles/` and `reference/` unless asked).
- A short status: what ran, on which GPU type, failures, anything surprising.
- Commit only code and docs (results are gitignored). Do not push without asking.

## Where the workstation stopped

The rest of the rebuttal work (Experiment 1, the linear-Gaussian multistep runs and
their M sweep, and `REBUTTAL_RESULTS.md`) stays on the workstation. Its run log is
`$PE/results/_runs/STATUS.md` (not in git).
