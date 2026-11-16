# Lorenz 96 on DelftBlue: handover (2026-11-12)

For the agent taking over the Lorenz 96 rebuttal work on DelftBlue. It follows
`LORENZ96_DELFTBLUE_HANDOVER.md` (the workstation -> DelftBlue note): read that first for
the setup, the ground rules and the tuning protocol. This note says what changed since,
what has run, what is still running, where every file is, and how to fold the Lorenz 96
results into the shared `paper_experiments` results tree. `PE` = `papers/A Unified
Framework for Bayesian Data Assimilation with Generative Models and Observation
Interpolants/code/paper_experiments` (mind the spaces).

## Ground rules (unchanged, from the user)

- Double-blind review: never search for the paper's title, authors or repository.
- The manuscript may only be edited where the user explicitly allows it (see
  "Manuscript" below). Ask before any change to the scientific setup, before pushing,
  and before anything over about a day of compute.
- No choice from the test split. Never delete results: move them aside (`_stale_*`).
- Navier-Stokes / uDALES are not this work; do not touch branch `ICLR-2027-submission`
  or tag `ICLR-2027-submission-snapshot`. Commit only code and docs.
- Report numbers as they are, negative results included. Stop and ask on failures or
  anything inconsistent.

## Where everything is

| What | Path |
|---|---|
| Code (git worktree, branch `review/lorenz96-delftblue`) | `/home/ntmucke/ssi-lorenz96` |
| Main checkout (Navier-Stokes work, uncommitted NS edits: leave alone) | `/home/ntmucke/scientific-stochastic-interpolants` |
| Lorenz 96 scratch root | `/scratch/ntmucke/lorenz96_delftblue/` |
| Results (`results/lorenz96/{linear,arctan}/...`) | `/scratch/ntmucke/lorenz96_delftblue/results/lorenz96` (symlinked to `$PE/results/lorenz96` in the worktree) |
| Figures | `/scratch/ntmucke/lorenz96_delftblue/figures/lorenz96/{linear,arctan,appendix}` (symlinked to `$PE/figures/lorenz96`) |
| LaTeX tables | `$PE/generated/tab_lorenz96_*.tex` in the worktree (gitignored, not on scratch) |
| Data, checkpoints | `/scratch/ntmucke/lorenz96_delftblue/{data,checkpoints}/lorenz96` (symlinked to `data/lorenz96`, `checkpoints/lorenz96`) |
| Venv | `/scratch/ntmucke/lorenz96_delftblue/.venv` (symlinked `.venv`) |
| SLURM logs | `/scratch/ntmucke/lorenz96_delftblue/slurm/logs/` |
| Running status log (chronological, every job id) | `/scratch/ntmucke/lorenz96_delftblue/STATUS.md` |
| SLURM scripts (copies committed in `$PE/slurm_lorenz96/`; the live ones are in `/scratch/ntmucke/lorenz96_delftblue/slurm/`) | see "Scripts" |
| Manuscript appendix branch | worktree `/home/ntmucke/iclr_manuscript_l96`, branch `lorenz96-appendix-results` (not pushed) |
| Inputs tarball (sha256 041945356f...ea96, verified) | `/scratch/ntmucke/lorenz96_delftblue_inputs.tar.gz` |

## Environment

Own venv, not the shared `.venv` of the main checkout (that one has torch 2.15+cu132,
which does not run on V100, plus jax packages that `uv sync` would remove). Built with
`uv venv --python 3.13` + `uv pip install torch==2.9.1 torchvision==0.24.1 --index-url
https://download.pytorch.org/whl/cu126` + `uv pip install -e . "pytest>=8.3.5,<8.4.0"`
(torch 2.9.1 = the workstation's version; cu126 runs on V100 and A100). Keep
`UV_CACHE_DIR` on node-local disk (`/tmp/...`): uv on BeeGFS fails with EBUSY.

    cd /home/ntmucke/ssi-lorenz96 && source .venv/bin/activate
    export PYTHONPATH=$PWD/src MPLBACKEND=Agg
    export PATH=/scratch/ntmucke/texlive/2026/bin/x86_64-linux:$PATH   # pdflatex, latexmk

Tests last run: root suite 359 passed + 2 skipped (45dada4, MIG); `$PE -k lorenz96` 231
passed + 2 failures that were fixed in a25fc18 (both re-run green); the tests added since
(shared mode, online Q) pass.

## SLURM on DelftBlue (what matters here)

- `gpu-a100-small` = A100 MIG 1g.10gb slices: QoS 1 running job per user, <= 2 CPUs per
  task, cpu=4 / gpu=2 / 32 GB per user, 4 h max. Request 4 CPUs as `-n 2 -c 2`. The user
  also runs other small jobs there, which block ours (`QOSMaxJobsPerUserLimit`).
- `gpu-v100` = Tesla V100S, 2 days max, shared with the Navier-Stokes arrays; account
  `research-ceg-gse` allows 64 running jobs per user. No fractional V100s (GresTypes=gpu
  only); to fill a V100, pack several runs into one job (`tune_packed.sbatch`).
- `--mem-per-cpu` is mandatory.

## Decisions taken (all by the user)

1. Seed 0 only (seeds 1-2 to be decided after seeing results).
2. Devices: the linear sweep runs on A100 MIG 1g.10gb, the arctan sweep (tuning and grid)
   on Tesla V100S. Every job logs `nvidia-smi`. Do not mix devices within a sweep.
3. Arctan tuning picks accepted as selected (commit d062a7f).
4. Shared-Jacobian likelihood (`inflated_shared`) added for our three samplers, with and
   without the forecast correction, both sweeps, tuned by the same protocol, final runs
   at all four M (user approved the compute).
5. **The corrected trained cells use the online Q only** (`correction.q_source=online`):
   no trained-prior method is told the true model-noise covariance, to show the methods
   work with minimal tuning. Q itself was never tuned; only online vs exact was a
   choice, and exact is now excluded (commit 86b604e). The oracle + correction keeps the
   exact Q (it is the reference that knows the model).
6. The appendix tables and figures of every run may go into the manuscript appendix
   (done on a branch, not pushed). No other manuscript edits.

## Commits on this branch (since 45dada4, the handover commit)

| Commit | What |
|---|---|
| d062a7f | `apply arctan`: tuned picks into `methods_arctan:` of `configs/case/lorenz96.yaml` |
| a25fc18 | two tests pinned the untuned arctan config values; made them value-independent |
| 1fc45f3 | `run_generative`: `inflated_shared` runs one sequence per call (it shares Sigma_s over the batch, so batching sequences would mix them); own seed and fresh posterior/correction per sequence; every other mode unchanged. Test added |
| fa6cfcc | `make_lorenz96_appendix.py`: appendix tables (M=50 per sweep; vs M per metric; mean +- std over the 12 test sequences, best bold, second underlined, pending rows) and figures (vs M, vs step at M=50; ours and baselines apart) |
| 86b604e | selection rule: corrected trained cells only eligible with online Q; arctan `si_sde_fc` re-picked (online, gain 0.5) |
| 0a15966 | test of that rule |
| (this PR) | this note and `$PE/slurm_lorenz96/` (the SLURM scripts and job lists) |

Shared-Jacobian runs are tagged cells, not new cell names: `driver.py cell --method
si_sde --tag shared --set likelihood_mode="inflated_shared" --set jacobian_damping=0.9
...` writes `lorenz96__si_sde-shared__M50__traj0__.*` with variant `shared`
(`forecast_correction_shared` for `*_fc`). The library's `jacobian_refresh_every` is not
exposed by the driver, so the Lorenz 96 shared runs refresh the shared Jacobian at every
pseudo-step (the manuscript says "every tenth" for the field cases; flag it to the user).

## What has run (all seed 0, 12 test sequences, E=64)

**Linear sweep, A100 MIG 1g.10gb.** All 78 cells (18 generative x M in {25,50,100,250}
+ 6 M-independent) done, 0 failures, 0 diverged; then `SEEDS=0 ./run_lorenz96.sh final
linear` (job 1300170): `pf_reference`, `aggregated/`, `figures/lorenz96/linear/`,
`generated/tab_lorenz96_linear{,_vs_M}.tex`. Cross-check of M=50 against the
workstation's `_local_rtx3090_seed0` (24 cells): max |diff| 3.8% (SURGE cells 1.5-3.8%,
everything else <= 2%), written to `linear/aggregated/crosscheck_delftblue_vs_rtx3090_M50.md`.

**Arctan tuning, Tesla V100S** (job 1293783, 48 min, 6 runs packed per GPU, then the
canonical `./run_lorenz96.sh tune arctan` pass): `arctan/tuning/selected_stage{1,2}.md`.

**Arctan sweep, Tesla V100S.** 74 of 78 cells done (0 failures). `si_sde_fc` at all four M
had used the exact Q: moved to `arctan/_stale_exactQ/` (metrics, per_step,
per_sequence, ensembles) and is being rerun with online Q (see below). Diverged: `fig` at
M=100 (0.3% of members) and M=250 (40%, sliced W2 20.2): a real instability of FIG with
arctan guidance at large M (tuned at M=50, reused), reported as a negative result.

**Shared-Jacobian tuning, MIG** (job 1303718, 1.5 h): `tuning/*__shared-s{1,2}.csv`,
picks in `tuning/selected_shared_stage{1,2}.{md,json}` per sweep (selector:
`slurm_lorenz96/shared_select.py`, same rule as `select_lorenz96_tuning.py`, online Q
only). Stage 1: damping 0.9 everywhere (grid edge). Stage 2 (online only): linear si 0.75,
fm 1.0, dm 1.0; arctan si 1.0, fm 0.75, dm 0.75. The earlier exact-Q selection is kept as
`_stale_exactQ_selected_shared_stage2.*`.

Headline numbers and reading: `results/lorenz96/appendix_tables.md` (all runs, both
sweeps). In short: trained prior + EnKF is best at E=64 in both sweeps; the correction
cuts our sliced W2 by ~20-25% and gives the best KL; every SURGE variant is worse than
its parent; our samplers and the EnKF rows barely move with M; FIG (arctan) and D-Flow
degrade with M. On validation the shared mode is worse than Jacobian-free without the
correction and on par or better with it.

## What is running / queued (check with `squeue -u ntmucke | grep l96`)

| Job | Partition | What | Then |
|---|---|---|---|
| 1305012 (tasks 6,24,42,60) | gpu-v100 | arctan `si_sde_fc` at M=50,25,100,250 with online Q (`cell.sbatch`, `joblist_arctan.txt`) | releases 1302733 |
| 1302733 | gpu-v100 | `SEEDS=0 ./run_lorenz96.sh final arctan` (`final.sbatch`): skips finished cells, runs arctan `pf_reference` (+ `prior-quality` if missing), aggregates, figures, tables. `Dependency=afterok:1305012` | check its log and `arctan/aggregated/` |
| 1304946 (24 tasks) | gpu-a100-small | linear shared cells: 6 cells x 4 M (`shared_cell.sbatch`, `joblist_shared_linear.txt`). Runs one task at a time (MIG QoS); ~10-15 h | - |
| 1304947 (24 tasks) | gpu-v100 | arctan shared cells, same | - |

A task skips itself when its metrics file exists and fails if it is not written, so
re-submitting an array is safe. A shared task reads the picks from
`tuning/selected_shared_stage{1,2}.json` when it starts.

## What is left

1. When 1305012 and 1302733 finish: check `arctan/logs/final_*.log` and the `.exit` file,
   and that `arctan/aggregated/` and `figures/lorenz96/arctan/` exist.
2. When the shared arrays finish: check 24+24 metrics files `*-shared__M*`, no failures,
   one GPU type per sweep in the logs; then re-run the aggregation per sweep so the
   tagged cells reach `aggregated/all.csv` (`SEEDS=0 ./run_lorenz96.sh final <obs>` on
   the sweep's device; it skips all cells) and `python make_lorenz96_appendix.py`
   (fills the pending rows).
3. Results tarball for the user: `results/lorenz96/{linear,arctan}/{aggregated,metrics,
   per_step,per_sequence,tuning,logs}`, `results/lorenz96/appendix_tables.md`,
   `figures/lorenz96/`, `generated/tab_lorenz96*.tex` (no `ensembles/`, `reference/`).
4. Manuscript: copy the refreshed appendix tables/figures onto branch
   `lorenz96-appendix-results` (worktree `/home/ntmucke/iclr_manuscript_l96`, commit
   a761bf6 adds subsection "All runs" at the end of
   `sections_review/appendix_lorenz96_case.tex`, the two `tab_lorenz96_appendix_*.tex`
   with `\revB` captions, figures in `figures_review/lorenz96/appendix/`). Its current
   arctan `si_sde_fc` numbers are the stale exact-Q ones. Ask the user before updating;
   do not push it without asking. Sentences in the manuscript that are out of date (the
   user knows; do not edit unasked): "three seeds", "SURGE/FlowDAS/SDA/D-Flow/FIG not yet
   in the Lorenz 96 driver", the CPU/RTX 3090 device sentence, "shared Jacobian updated
   every tenth pseudo-step".
5. Seeds 1-2: the user decides after seeing the results.

## Merging into one `paper_experiments` results tree

The other cases' results live in
`/scratch/ntmucke/scientific-stochastic-interpolants/paper_experiments/results/<case>/`
(layout: `results/README.md` there). The Lorenz 96 tree is self-contained and uses the
same convention (`results/lorenz96/<obs>/{metrics,per_step,per_sequence,ensembles,
aggregated,tuning,logs,reference,gate}`), so merging is a copy, once no job writes to it:

    rsync -a /scratch/ntmucke/lorenz96_delftblue/results/lorenz96/ <PE root>/results/lorenz96/
    rsync -a /scratch/ntmucke/lorenz96_delftblue/figures/lorenz96/ <PE root>/figures/lorenz96/
    cp /home/ntmucke/ssi-lorenz96/$PE/generated/tab_lorenz96*.tex <PE root>/generated/

No `--delete`, keep the source until the copy is checked (file counts, a few checksums).
`reference/` (PF ground truth, ~1 GB with the `.pfterms.pt` caches) is needed to score
new runs; `_local_rtx3090_seed0/` and `_stale_exactQ/` are provenance (keep, aside). The
code merges through this PR (into `ICLR-2027-review`); the Lorenz 96 scripts read
`results/` relative to `$PE`, so after the copy they work from the merged tree too
(`checkpoints/lorenz96`, `data/lorenz96` must also be reachable from the repo root).

## Scripts (`$PE/slurm_lorenz96/`; the paths inside point at the worktree and scratch)

| Script | Use |
|---|---|
| `cell.sbatch` + `joblist_{linear,arctan}.txt` | one cell per array task: `OBS=<obs> SEED=0 JOBLIST=... sbatch -p <part> [-n 2 -c 2 on MIG] --array=<lines> cell.sbatch` |
| `final.sbatch` | `OBS=<obs> sbatch -p <part> ... final.sbatch`: `SEEDS=0 ./run_lorenz96.sh final <obs>` |
| `tune_packed.sbatch` + `tune_lane.sh` | the arctan tuning, 6 runs per V100, then the canonical tune pass |
| `shared_tune.sbatch` + `shared_tune_lane.sh` + `shared_select.py` | shared-Jacobian tuning, linear and arctan lanes on two MIG slices |
| `shared_cell.sbatch` + `joblist_shared_{linear,arctan}.txt` | shared final cells (`--tag shared`, picks read from the JSON) |
| `tests.sbatch`, `tests_pe.sbatch` | test suites on MIG (note: their exit code does not reflect pytest; read the log) |
| `$PE/make_lorenz96_appendix.py` | appendix tables + figures from the per-cell files |
