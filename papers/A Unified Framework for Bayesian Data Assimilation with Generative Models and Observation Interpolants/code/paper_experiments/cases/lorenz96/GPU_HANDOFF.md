# Lorenz 96 case: handoff to a GPU machine

Written 2026-11-07 on the Mac where the work started. It assumes no access to that
session or that machine's gitignored files. Everything below that is a number comes
from a run that actually finished, unless it is marked as an estimate.

## 1. Status in brief

- Done: solver, EnKF, particle filter (PF), four network variants, training stack,
  evaluation driver, aggregation and figures, tests, the PF convergence gate.
- Partly done: PF references (validation complete, test incomplete), tuning (partial).
- **Not done: the final comparison.** `run_lorenz96_grid.sh` has never run, and there
  are no held-out test results for any method.
- **Early warning:** on validation, every sampler setting finished so far is behind
  EnKF against the PF, and the exact `inflated` likelihood is worse than the
  Jacobian-free one. See section 8 and open question 1.
- **Nothing has been tested on CUDA or on Linux.**
- **Forecast-residual correction (2026-11-09, validation only, section 13):** the
  correction improves every trained sampler (SI-SDE 0.657 -> 0.434 mean error vs PF
  at its tuned gain) but **does not beat the simple baseline** "unguided trained prior
  + localised EnKF" (0.385 SI prior, 0.381-0.392 FM prior), which matches the
  true-model EnKF (0.382).

## 2. Goal

Add a Lorenz 96 data assimilation case for the ICLR 2027 paper under review. Compare
stochastic interpolant (SI) and flow matching (FM) priors, using the paper's
observation-interpolant posterior sampling, against EnKF and a many-sample PF that
serves as ground truth. All methods get the same held-out observation sequences and
the same initial ensemble distribution.

Report per step against the PF: error in posterior mean and marginal variance, and
sliced W2. Report against the true state: RMSE, spread, CRPS. Give mean +/- std over
seeds, wall-clock cost, and an EnKF with ensemble size matched to the generative
methods.

- Branch: `review/lorenz96`, based on commit `6cfa977` ("Reorganize paper experiments
  and fix relocated paths"), the head of `ICLR-2027-review` when this work began.
- Pushed to `origin/review/lorenz96`. Open any PR against `ICLR-2027-review`.
- The manuscript is a separate repository and is not in this clone:
  `https://github.com/nmucke/iclr_da_interpolant_manuscript.git`. Clone it into
  `papers/A Unified Framework .../manuscript/` (gitignored) to read it. Read only.

## 3. Setup from a fresh clone

Paths contain spaces, so set this once and quote it:

```bash
git clone https://github.com/nmucke/scientific-stochastic-interpolants.git
cd scientific-stochastic-interpolants
git switch review/lorenz96
export REPO="$PWD"
export PE="$REPO/papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments"
```

Environment:

- Python must be 3.13 (`requires-python = ">=3.13, <3.14"`). Packaging is `uv`
  (`uv_build`), with workspace members under `external_libs/`, which are tracked.
- **There is no lockfile in the clone.** `.gitignore` has `*.lock`, so `uv.lock` is
  not committed. `uv sync` resolves fresh, and versions may differ from the Mac,
  which ran Python 3.13.9, torch 2.9.0, numpy 2.2.6 and scipy 1.16.3.
- **Version drift:** on 2026-11-07 a clean `uv lock` resolved all 247 packages, but picked
  torch 2.15.0 and triton 3.9.0, while all testing here used torch 2.9.0. Torch's CUDA
  dependencies in that lock: cuda-bindings, cuda-toolkit, nvidia-cudnn-cu13, nvidia-cusparselt-cu13, nvidia-nccl-cu13, nvidia-nvshmem-cu13. Confirm the GPU driver supports that CUDA version. If
  tests or training misbehave, try torch 2.9.0 before suspecting the code. Record the
  installed versions with any results.
- `pyproject.toml` leaves `torch` unpinned and sets no torch index. It has
  `[tool.uv.pip] find-links` for cu126, which as far as I know applies to `uv pip`
  and not `uv sync`; I have not verified that. I expect the default Linux x86_64
  wheel to be a CUDA build, but this was never tried. Check it before anything else:

```bash
uv sync
source .venv/bin/activate
export PYTHONPATH="$REPO/src"
python -c "import torch, scisi; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), scisi.__file__)"
```

If CUDA is false, install a matching CUDA torch wheel into the venv. `uv sync` also
pulls heavy dependencies this case does not use directly (`ray`, `pdetransformer`,
`aurora_lib`, `ffmpeg`); whether they all resolve on Linux is unknown.

Gitignored, so **absent from the clone**:

| Path | Needed? | How to get it |
| --- | --- | --- |
| `data/lorenz96/data.npz` | yes | regenerate, 31 s (step 2) |
| `checkpoints/lorenz96/{si,fm}/` | yes | retrain (step 4); or copy, 2.5 MB each |
| `$PE/results/` (gate, references, tuning) | yes, as evidence | regenerate, or copy (187 MB when written) |
| `archive/` | no | only named in two docstrings; no import found |
| `uv.lock` | no | see above |

`external_libs/` is tracked. `src/scisi/data/datasets.py` imports a CFD library, but
this case does not go through that dataset. `driver.py` does import
`cases/navier_stokes/_ns_pipeline.py` lazily for `attach_nfe_counter`, so that module's
imports must work when generative cells run. The trainer imports `trackio` at module
level, so it must be installed, but `train.py` uses a local file tracker and nothing
goes online.

`common/paths.py`: `REPO_ROOT` is found by walking up to a directory with both
`pyproject.toml` and `src/scisi`. `RESULTS_ROOT` is `$PE/results`. So data and
checkpoints resolve under the repository root, and results under `$PE`, wherever you
launch from. `train.py` changes directory to `REPO_ROOT` because the trainer writes to
a relative `checkpoints/`. Confirm `scisi.__file__` points into the clone: a stale
install elsewhere would silently run other code.

## 4. Device handling

**Nothing has run on CUDA.** Training and sampling ran on MPS. All filters ran on CPU.

`auto` means CUDA, then MPS, then CPU (`resolve_device` in `_l96_system.py`, added at
the freeze; a short `train.py` run on the Mac resolved it to `mps`, and the CUDA branch
has never executed).

| Code | How the device is chosen | For CUDA |
| --- | --- | --- |
| `train.py` | `trainer.device` in `config/lorenz96_*.yaml`, set to `auto`. `scisi.utils.device_utils.set_device` is not used | nothing should be needed; override with `trainer.device=cuda` |
| `driver.py` samplers, `prior-quality` | `evaluation.device` in `configs/case/lorenz96.yaml`, set to `auto` | nothing should be needed |
| `driver.py` EnKF, oracle, PF, references | CPU, not configurable | see below |
| `pf_gate.py` | CPU, not configurable. `--workers` processes, one thread each | see below |

The PF and EnKF are **CPU-only**, and that is not a small fix, so I left it. They use
a CPU `torch.Generator`; `randn`, `rand` and `randperm` calls in
`src/scisi/particle_filter/bootstrap.py` and `enkf.py` pass no device, and
`_randn_like` in `src/scisi/dynamics/lorenz96.py` passes a device with that CPU
generator, which fails on CUDA tensors. `run_pf` and `_reference_task` also build the
particles on CPU. A GPU port needs a device-side generator, `device=` on each draw, and
a fresh check against `tests/test_filters.py`; it also changes the random streams.

So a GPU alone speeds up training and sampling, not the PF. The PF benefits from CPU
cores instead: raise `--workers`. It is memory bound, gains nothing from torch threads,
and each worker needs about 1 GB at N=4e6.

Other notes: `trainer.mixed_precision_warmup` is 0, so no `GradScaler`. Sampler seeds
use `torch.manual_seed`, so sampler output will not match the Mac bit for bit.
Keep `batch_size = S*E` in `assimilate`: the sampler passes unsliced observation rows
to each batch.

## 5. Problem definition and settings

`configs/case/lorenz96.yaml` is the source of truth, and `Problem` defaults must match
it (a test checks this).

| Setting | Value | Why |
| --- | --- | --- |
| `dim_state` | 16 | largest d that passed; see section 6 |
| `forcing` | 8 | standard chaotic regime |
| `dt` | 0.01 | RK4 step |
| `interval` | 0.2 | 0.4 collapsed ESS more often at the same d |
| `noise_std` | 1.0 | deterministic L96 has a degenerate transition density; noise 2 at interval 0.1 gave a posterior wider than climatology |
| `noise_stages` | 1 | kernel is exactly N(Phi(x), noise_std^2 I), so the optimal proposal and the one-step oracle are exact. 2 stages gave much worse ESS (comparison not noise-matched) |
| `obs_stride` | 4 | every fourth site, 4 observations |
| `obs_variance` | 4.0 | R=1 failed the gate; R=4 passed |
| `prior_std` | 1.0 | around an attractor state |
| `num_steps` | 50 | assimilation steps per sequence |

Layout: states are `[B, 1, 1, d]`, history `[B, 1, 1, d, 1]`. `observations[j]` observes
`truth[j+1]`. Models work in standardised units (mean 2.2297, std 3.9866, from the
training split), and `R` is divided by `std^2`.

PF: fully adapted auxiliary PF using the optimal proposal: weights
N(y; Hz, qHH^T + rI), resampling before the kick, conditional kick by Matheron's rule.
Reference N = 4e6.

Evaluation: 12 test sequences, 4 validation sequences, E = 64, large EnKF E = 2000,
seeds 0, 1, 2. Sequences are fixed across seeds, so the std over seeds is sampler noise
only; `aggregated/sequence_spread.csv` holds the spread over sequences.

Limits of this setting, which should be stated in any write-up:

- With `noise_stages=1` the one-step Gaussian closure is exact given a perfect prior,
  so this case does not stress the closure.
- Skewness and excess kurtosis of the posterior are modest: 0.31 and 0.49 at N=1e5.
  These are upper bounds, since PF sampling inflates them and samples share ancestors.
- Observations are weak. Posterior std is 2.86 against a climatological 3.99.
- EnKF and PF use the true model; the generative methods use learned priors.
- The system is chaotic: common-noise leading Lyapunov estimate about 1.84 per unit time.
- It is non-Gaussian enough for EnKF to differ from the PF: at N=1e5 the EnKF gap is
  0.39 posterior std against a PF replicate gap of 0.15, EnKF variance ratio 1.22,
  RMSE 3.07 (EnKF, E=2000) against 2.84 (PF). This is `gate/screen_final.csv`, 6
  sequences, first 10 steps skipped. Gaps there include the reference's own MC error.

## 6. The PF gate

`pf_gate.py` has `PASS_CRITERIA`, fixed in advance and never changed afterwards:

| Criterion | Threshold |
| --- | --- |
| `mean_error_q95` | <= 0.10 |
| `log_var_error_q95` | <= 0.10 |
| `log_evidence_sd` | <= 0.50 |
| `ess_abs_min` | >= 500 |
| convergence rate of mean error in N | between -0.70 and -0.30 |

Errors are for a single run, per entry, in local posterior-std units (log units for
variance), over all steps, against the mean of the other replicates at the top rung.
I chose them by judgement, before the runs: a reference error of a tenth of a posterior
std is small against the differences being measured, and the rate must look like
Monte Carlo (-0.5). They are not derived from theory. Do not loosen them; if a setting
fails, change the setting or raise N.

Replicate agreement cannot catch a filter that is consistently wrong, hence `calibrate`
(truth against posterior, with wrong-model controls) and `cross-proposal`.

Evidence, in order:

| System (interval 0.2, sigma 1, 1 stage) | Run | Result |
| --- | --- | --- |
| d=40, 32 | early screens at N=1e5 | collapsed or marginal |
| d=24, 20, interval 0.4, R=1 | N=1e6, T=30, 1 seq | all failed |
| d=24, stride 2 and 4, R=1 | N=1e6, T=30, 1 seq | failed |
| d=20, stride 2, R=1 | N=1e6, T=30, 1 seq | failed `ess_abs_min` |
| d=20, stride 4, R=1 | N=1e6, T=30, 1 seq | **passed, then failed below** |
| d=20, stride 4, R=1 | N=1e6, T=50, 5 seq, 8 reps | **all 5 failed**: `ess_abs_min` 4-64, rate about -0.3 |
| d=12-20, R=1,2,4 | N=1e5, T=50, 5 seq, 6 reps | only d=16 stride 4 R=4 projected to pass |
| **d=16, stride 4, R=4** | N=1e6, T=50, 5 seq, 8 reps | 2 pass; 3 fail only `log_evidence_sd` (0.547, 0.727, 0.554) |
| **d=16, stride 4, R=4** | **N=4e6, T=50, 5 seq, 6 reps** | **all 5 pass** |

Lesson: a single-sequence pass means little. Failures come from rare steps where ESS
collapses, so always use several sequences at the full length.

Final rung (`gate/convergence_4e6.json`, ladder 1e5 and 4e6):

| Seq | ESS frac mean | ESS abs min | mean q95 | log-var q95 | corr q95 | log-Z sd | rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1427 | 24486 | 0.031 | 0.035 | 0.019 | 0.175 | -0.425 |
| 1 | 0.1238 | 3021 | 0.047 | 0.044 | 0.026 | 0.306 | -0.406 |
| 2 | 0.1234 | 17324 | 0.035 | 0.036 | 0.020 | 0.138 | -0.444 |
| 3 | 0.1277 | 2282 | 0.055 | 0.056 | 0.031 | 0.176 | -0.362 |
| 4 | 0.1338 | 54689 | 0.030 | 0.027 | 0.017 | 0.106 | -0.422 |

Worst single entry: mean error 0.397 posterior std (sequence 3), so the q95 hides
rare larger errors.

Calibration (N=1e5, 32 sequences, T=50; z^2 should be 1; max deviation of a rank decile
from 0.1):

| Filter | z^2 all | z^2 observed | z^2 unobserved | rank dev |
| --- | --- | --- | --- | --- |
| correct | 0.998 +/- 0.023 | 1.005 +/- 0.020 | 0.996 +/- 0.028 | 0.003 |
| obs variance x2 | 0.818 +/- 0.015 | 0.688 | 0.862 | 0.021 |
| model noise x0.5 | 2.388 +/- 0.120 | 1.732 | 2.607 | 0.100 |
| forcing 6 | 1.346 +/- 0.032 | 1.193 | 1.397 | 0.070 |

Cross-proposal, bootstrap against optimal, d=8, stride 2, R=4, T=20, N=1e6, 8 reps, 2
sequences (a system where bootstrap also converges): RMS t of means 1.15 and 0.96, of
variances 1.30 and 1.20, |t| of log-evidence 1.30 and 0.03. No disagreement.

Caveats on the gate:

1. The N=4e6 rung used 6 replicates, not the default 8, to fit local compute.
2. The rate at that rung is a two-point fit (1e5, 4e6). The 3-point ladder to 1e6 gave
   -0.34 to -0.42.
3. **Mild selection effect:** this system was chosen after the N=1e5 probe on gate
   sequences 0-4, and the final gate reused the same indices. Run `--sequences 10` so
   that sequences 5-9 are fresh.
4. Calibration ran at N=1e5, not at N=4e6.
5. Sliced W2 between runs is recorded in the JSON and has no pass criterion.

Held-out references, checked on their own (N=4e6, 2 replicates each):

| Sequence | ESS frac mean | ESS abs min | log-Z, 2 replicates | q95 mean difference |
| --- | --- | --- | --- | --- |
| validation 0 | 0.1144 | 30154 | -550.81 / -550.75 | 0.051 |
| validation 1 | 0.1455 | 44956 | -525.71 / -525.73 | 0.031 |
| validation 2 | 0.1830 | 8214 | -510.98 / -510.84 | 0.032 |
| validation 3 | 0.1376 | 72936 | -528.96 / -528.79 | 0.027 |
| test 0 | 0.1334 | 49424 | -536.57 / -536.69 | 0.045 |
| test 1 | 0.1414 | 17031 | -539.06 / -538.53 | 0.042 |
| test 2 | 0.1475 | 9200 | -536.50 / -536.94 | 0.050 |

Test sequences 1 and 2 have log-Z differences of 0.53 and 0.44 between replicates,
which is near what a 0.5 limit on the sd allows. Six of the 12 test sequences had both
replicates on disk at the freeze; only these three were checked, and the rest of the
job was left running.

## 7. Pipeline from nothing

Set up as in section 3 first. `RAN` = done here at these settings. `RAN (tiny)` = only
through a reduced test. `NEVER` = has not run.

**Step 0. Tests. RAN.** A few minutes on an idle machine.

```bash
cd "$REPO" && python -m pytest tests/test_lorenz96_solver.py tests/test_filters.py tests/test_periodic_1d.py tests/test_metrics.py tests/test_lorenz96_data.py -q
cd "$PE" && MPLBACKEND=Agg python -m pytest tests/test_lorenz96_case.py tests/test_lorenz96_gate.py tests/test_paths.py -q
```

**Step 1. Gate. RAN on CPU.** About 25 min for the first command block and about
50 min for the 4e6 rung, with 6 workers on 10 cores, machine awake (estimates from
per-run times). Outputs land in `results/lorenz96/gate/`, a few KB each. A failed gate
means stop before training.

```bash
cd "$PE"; export OMP_NUM_THREADS=1
python cases/lorenz96/pf_gate.py converge --ladder 1e5,4e6 --replicates 8 --sequences 10 --workers 8 --threads 1 --out results/lorenz96/gate/convergence_4e6.json
python cases/lorenz96/pf_gate.py calibrate --n 100000 --sequences 32 --workers 8 --threads 1 --out results/lorenz96/gate/calibration.json
python cases/lorenz96/pf_gate.py cross-proposal --problem '{"dim_state":8,"obs_stride":2,"num_steps":20}' --n 1000000 --replicates 8 --sequences 2 --workers 8 --threads 1 --out results/lorenz96/gate/cross_proposal.json
```

The command above is what I recommend (8 replicates, 10 sequences). What ran here used
`--replicates 6 --sequences 5 --workers 6` for the 4e6 rung. `results/` must exist; the
scripts create what they need below it. Expect `GATE PASSED` in stdout.

**Step 2. Data. RAN.** 31 s, one thread.

```bash
python cases/lorenz96/simulate_data.py --threads 4
```

Writes `$REPO/data/lorenz96/data.npz`, 256 MB: 4000 trajectories x 1000 steps x 16, seed
label `lorenz96-training-data`. Trajectories 0-3799 train (about 3.8M pairs), 3800-3999
validate. Mean 2.2297, std 3.9866.

**Step 3 (optional). Architecture check. RAN, one short run each.** See below.

**Step 4. Train SI and FM. RAN on MPS, 8 epochs.**

```bash
python cases/lorenz96/train.py --config lorenz96_stochastic_interpolant --name si
python cases/lorenz96/train.py --config lorenz96_flow_matching --name fm
```

Trailing `key=value` arguments override the config. Note for a GPU: **8 epochs was a cut
for local compute, from a planned 40.** Recommended on a GPU:

```bash
python cases/lorenz96/train.py --config lorenz96_stochastic_interpolant --name si -- trainer.num_epochs=40 trainer.early_stopping.patience=15
python cases/lorenz96/train.py --config lorenz96_flow_matching --name fm -- trainer.num_epochs=40 trainer.early_stopping.patience=15
```

The scheduler follows `num_epochs` by interpolation. With 8 epochs, `patience: 8` meant
early stopping could not fire. The best validation checkpoint is what `model.pth`
holds. Each epoch is 3708 steps of batch 1024. MPS alone ran about 13k samples/s.
Larger batches were not faster on MPS (4096: 10.8k samples/s), and CUDA is unknown;
benchmark before changing `batch_size`, and rescale `lr` if you do. Val loss is
dominated by irreducible noise and hardly moves, so judge models by step 5, not by
loss. Outputs: `checkpoints/lorenz96/<name>/{model.pth,config.yaml,metrics.jsonl}`.
`elapsed` in `metrics.jsonl` from the Mac run is meaningless (see section 8).

Architecture: `PeriodicResNet1d` (C=96, 4 blocks, kernel 7, 623k parameters), chosen
because it had the lowest validation loss in a one-epoch check on 400k pairs (SI
objective, MPS) and is exactly shift-equivariant under every lattice shift:

| Network | Params | Val loss | Time | Note |
| --- | --- | --- | --- | --- |
| `PeriodicResNet1d` | 623k | 0.348 | 73 s | used |
| `PeriodicUNet1d` | 873k | 0.353 | 87 s | equivariant only under shifts of 4 |
| `PeriodicTransformer1d` | 496k | 0.356 | 103 s | |
| `MLP1d` | 592k | 0.491 | 17 s | not equivariant |

The first three are within what a single short run can separate, so this is weak
evidence. After this comparison, `PeriodicUNet1d`, `PeriodicTransformer1d`, `MLP1d`
(and their helpers) were removed from `src/scisi/architectures/periodic_1d.py`, so
step 3 cannot be rerun from this tree; they are in the git history at commit caa08af.
`PeriodicResNet1d` was kept unchanged, and the trained checkpoints load strictly and
give bitwise identical outputs after the removal. The ResNet's receptive field covers up to 31 sites, enough for d=16 and any d
up to 31. `CircularConv1d` is a matmul over unfold because `nn.Conv1d` was about 100x
slower on the Mac's CPU build without oneDNN; that may not apply on CUDA.

**Step 5. Prior quality. RAN.** About 3 min on MPS.

```bash
python cases/lorenz96/driver.py prior-quality
```

**Step 6. Validation references. RAN.** 8 runs of 320-543 s, 6 at once.

```bash
python cases/lorenz96/driver.py reference --split validation --workers 8
```

Each file is about 13 MB in `results/lorenz96/reference/validation/`. Existing files are
skipped, and a reference made for a different problem raises an error when loaded.

**Step 7. Tuning on validation only. PARTIAL here.**

```bash
python cases/lorenz96/driver.py tune --method enkf --grid '{"inflation":[1.0,1.02,1.05,1.1,1.2],"localisation":[null,4,8]}' --threads 2
python cases/lorenz96/driver.py tune --method enkf_large --grid '{"inflation":[1.0,1.02,1.05],"localisation":[null,4,8]}' --threads 2
for m in si_sde fm_ode dm_sde; do
  python cases/lorenz96/driver.py tune --method $m --grid '{"likelihood_mode":["dps_jacobian_free","inflated"],"num_steps":[25,50]}' --threads 2
done
```

Selection rule, fixed before results: **lowest validation `sliced_w2`**. Then write the
values into `methods:` in `configs/case/lorenz96.yaml`, which currently holds
**defaults, not tuned values**, and is labelled so. A GPU makes it reasonable to extend
`num_steps` to 100 and 200 and to try `jacobian_damping` in 0.9, 0.95 (see open
question 1). Results append to `results/lorenz96/tuning/<method>__<keys>.csv`, so
delete a stale file before a rerun. Each row records the base `methods:` settings
(`base`, JSON) and `diverged_steps`, the (sequence, step) entries with a non-finite
score (in practice: fewer than two members left). A row with `diverged_steps > 0` has infinite
accuracy scores and cannot be selected. Appending to a file with other columns is
refused; the partial CSVs from the Mac predate these two columns, so move them aside
before tuning again. `enkf_large` was never tuned here.

**Step 8. Final grid. NEVER at full size; every cell RAN (tiny) in tests.** Do not
start until tuning is written into the config.

```bash
cd "$PE" && PY=python SEEDS="0 1 2" ./run_lorenz96_grid.sh
```

That script itself has never run at full size (a dry run with a stub interpreter
checked the skip and rerun logic). It does test references (24 runs of about 11 min, 6
at once: about 45 min on the Mac), `prior-quality`, 14 cells x 3 seeds (the 8 plain
cells first, then the 6 correction cells of section 13), the `pf_reference` cell once, then aggregation and figures. Cells run one at a time by
default (`JOBS=1`), because concurrent cells inflate each other's `seconds`. A cell
whose `results/lorenz96/metrics/` file exists is skipped, so after a failure just
rerun the same command; `FORCE=1` reruns every cell. The script stops after the cell
loop if any cell failed, before `pf_reference`, aggregation and figures. Cost per generative cell at
batch 768 on MPS: `inflated` M=25 about 14.5 s per assimilation step (12 min per cell,
24 min at M=50), `dps_jacobian_free` M=25 about 1 s (under a minute). Each cell also
reruns one sequence for like-for-like timing, about 1/12 more. Measure again on the GPU.
For honest `seconds` values keep `JOBS=1`; if you raise it, say so.

Outputs under `results/lorenz96/`: `metrics/`, `per_step/`, `per_sequence/`,
`ensembles/` (about 4 MB per cell). Estimated total under 1 GB.

**Step 9. Aggregation and figures. RAN (tiny)**, by a test with real cell output and a
NaN sequence. Included in the grid script.

```bash
python aggregate_lorenz96.py
python make_lorenz96_figures.py
```

Writes `results/lorenz96/aggregated/{all,per_step,sequence_spread}.csv` and
`figures/lorenz96/`. The `run.py` entry (`python run.py case=lorenz96 method=si_sde
scenario=lorenz96`) composes and its runner contract is tested, but it never ran a cell.

What a GPU changes: training and samplers get faster, so run 40 epochs and larger M,
more seeds, or more test sequences. PF cost does not change unless the PF is ported
(section 4). Raising N is only useful with more CPU cores or that port. A larger d is a
new project stage, needing new data, new training and a new gate: d=20 stride 4 R=4 at
N=1e5 had `ess_abs_min` 6-19 and mean q95 up to 0.41, which suggests needing at least
1e7 to 1e8 particles (extrapolation). Keep d=16 first.

## 8. Results so far

**None of this is a final result.** There are no test-split scores.

Training, 8 epochs on MPS:

| Epoch | SI train / val | FM train / val |
| --- | --- | --- |
| 0 | 0.3524 / 0.3469 | 0.4048 / 0.4035 |
| 3 | 0.3443 / 0.3428 | 0.3963 / 0.3983 |
| 6 | 0.3419 / 0.3426 | 0.3946 / 0.3945 |
| 7 | 0.3415 / 0.3399 | 0.3942 / 0.3936 |

**The Mac slept for most of training.** The lid was closed on battery from 20:43 on
Nov 6 and the machine ran only in short maintenance wakes, so 8 epochs took about 23
hours. This does not affect the models: the processes were paused, not restarted, no
step was lost, and the schedule counts epochs, not time. It does make `elapsed` in
`metrics.jsonl` and every wall-clock figure from that period meaningless, including
the N=4e6 gate times. Gate results are unaffected, as they do not depend on time.
The per-run costs quoted here come from periods when the machine was awake.

Prior quality against the true kernel (64 validation states, 512 draws, 100 sampler
steps, `results/lorenz96/prior_quality.json`, true variance 1):

| Prior | Mean error | Sampling floor | Variance |
| --- | --- | --- | --- |
| SI | 0.0573 | 0.0442 | 0.980 |
| FM | 0.0524 | 0.0442 | 0.966 |

So both priors learned the transition well, and the network is not the obvious
limitation at one step.

Partial validation tuning (4 sequences, seed 0, E=64, `results/lorenz96/tuning/`). The
final selection is **not** made, and these are from reduced settings:

| Method | Setting | mean err vs PF | var err vs PF | sliced W2 | RMSE | spread | diverged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EnKF | infl 1.0, no loc | 0.448 | 0.384 | 1.458 | 3.112 | 2.721 | 0 |
| EnKF | infl 1.0, loc 4 | 0.382 | 0.384 | 1.286 | 3.038 | 2.811 | 0 |
| EnKF | infl 1.0, loc 8 | 0.399 | 0.377 | 1.338 | 3.028 | 2.716 | 0 |
| EnKF | infl 1.02, loc 4 | 0.388 | 0.421 | 1.312 | 3.000 | 2.985 | 0 |
| SI-SDE | Jac-free, M=25 | 0.657 | 0.442 | 1.809 | 3.309 | 2.836 | 0 |
| SI-SDE | Jac-free, M=50 | 0.656 | 0.457 | 1.812 | 3.322 | 2.956 | 0 |
| SI-SDE | inflated, M=25 | 0.962 | 0.785 | 2.358 | 3.588 | 3.465 | 0.0027 |
| SI-SDE | inflated, M=50 | 0.966 | 0.841 | 2.417 | 3.597 | 3.623 | 0.004 |
| FM-ODE | Jac-free, M=25 | 0.628 | 0.476 | 1.786 | 3.296 | 2.999 | 0 |
| FM-ODE | Jac-free, M=50 | 0.627 | 0.476 | 1.788 | 3.290 | 3.028 | 0 |
| FM-ODE | inflated, M=25 | 0.962 | 0.832 | 2.385 | 3.576 | 3.560 | 0.0002 |
| DM-SDE | Jac-free, M=25 | 0.680 | 0.499 | 1.853 | 3.317 | 3.075 | 0 |
| DM-SDE | Jac-free, M=50 | 0.677 | 0.515 | 1.860 | 3.333 | 3.121 | 0 |
| DM-SDE | inflated, M=25 | 0.976 | 0.887 | 2.456 | 3.611 | 3.707 | 0.0014 |

These are 8-epoch models at `jacobian_damping` 1.0. E=64 has a floor: a perfect sampler
would not score zero, and `pf_subsample` measures that floor but has not run. The FM-ODE
and DM-SDE `inflated` M=50 rows and the rest of the EnKF grid (8 of 15 rows done) were
still running at the freeze; the CSVs hold any later rows if they are copied.

Stale or reduced numbers:

- The d=16 N=1e6 gate and `screen_d16*.csv` are superseded. Early d=16 screens predate
  a seed fix and report mean-of-minima for `ess_min`.
- Nothing above is stale from review fixes: data, models and references were not
  affected by them (section 9), and references store the problem and refuse a mismatch.
- A retrained model invalidates the tuning table. References do not depend on models.

## 9. Review status

Three independent review passes ran. **Review 3 did not finish** its test runs or its
end-to-end run; I ran those tests myself afterwards. **No re-review of the fixes was
done.** Review 1 predates `integrate_state_major`, process-parallel replicates and the
final system. Review 2 finished without its planned cycling comparison against the PF.

Review 1: solver, filters, gate evidence.

| Finding | Status |
| --- | --- |
| Gate had no external correctness check (doubled R scored better) | Fixed: `calibrate` with negative controls, `cross-proposal`; both run |
| Ratios, pooling and skip hid errors | Fixed: per-entry single-run quantiles, all steps, coded criteria |
| Oracle inner SIR degenerate at stages > 1 | Moot at `noise_stages=1`, which uses the exact path; documented, SIR test added |
| Untested paths, one tautological test | Fixed |
| Seed reuse between prior and filter | Fixed: separate streams |
| Docstring overclaimed kernel non-Gaussianity | Fixed |
| `ess_min` was a mean of minima | Fixed in `rung_stats` |
| Skew and kurtosis inflated | Stated here as upper bounds |
| Gap comparability | Stated in section 5 |
| Lucky test tolerance | Fixed: t-statistics |
| Taper not positive semi-definite | Fixed: chordal distance |
| Comment and terminology errors | Fixed |
| Asks: more sequences, ESS on test sequences, sample and covariance convergence | Done: 5 sequences, section 6, `correlation_error` and between-run W2 |

Review 2: generative models, sampling, leakage. Sampler wiring, standardisation,
per-member observations and split separation were verified with exact analytic drifts.

| Finding | Status |
| --- | --- |
| M1: `oracle`, `pf_matched`, `pf_subsample` crash | Fixed, test covers all cells |
| M2: divergence loses data, hides failures, differs by method | Fixed: members outside finite or \|x\| < 100 dropped per step, `diverged_fraction` metric for every method, finite means. DM-SDE asserts on a NaN drift, which counts the whole cell as diverged (conservative; library not changed) |
| M3: NFE undercounted | Fixed: counts forward calls with `attach_nfe_counter`, emits the NFE row |
| `inflated_shared` mixes sequences | Fixed: `assimilate` refuses it when batching sequences |
| Seconds not like for like | Fixed: each cell times one sequence at its own E. Reference seconds still come from parallel workers |
| `evaluation.seeds` unread; missing grid script and README | Fixed |
| Tuning not done | Open: see step 7 |
| `tune` CSV header misalignment | Fixed: file keyed by grid names |
| Discretisation under-dispersion at M=50 | Open: M is in the tuning grid; extend it on the GPU |
| Finite-ensemble floors only at E=64; PF W2 non-zero | Accepted: limit of the metric; state it when reporting `enkf_large` W2 |
| Seeds measure sampler noise only | Fixed: `sequence_spread.csv` |
| `_NAMES` has no PF entry; `enkf_large` unreachable from `run.py` | Fixed for PF; `enkf_large` is driver-only by design |
| Data and config match unchecked | Fixed: `Standardisation.from_data` compares them |
| Unsliced observations in `base_posterior.py` | Rejected as a change: existing library; guarded by `batch_size = S*E`, commented and tested |
| Step 1 may be out of distribution | Open, not measured |
| EnKF and PF use the true model | Stated in section 5 |

Review 2's agent also exceeded its limits: several PF checks ran 8-49 minutes during the
gate. Nothing was written to the repository.

Review 3: quality and repository fit. Existing cases: tracked changes are additive, and
nothing iterates the enums (by reading). The `make_tables --demo` before-and-after
comparison was **not** run.

| Finding | Status |
| --- | --- |
| 1: three cells crash | Fixed (same as M1) |
| 2: `configs/method/{enkf,particle_filter}.yaml` do not fit the new classes | Rejected: pre-existing, nothing instantiates `method.filter`, and editing them risks existing cases. The driver builds the filters. Still a trap for anyone instantiating them |
| 3: `run.py ... method=particle_filter` KeyError | Fixed: maps to `pf_matched`, others raise a clear error |
| 4: Hydra entry ignores its config | Fixed: runner passes `cfg.case`, uses `evaluation.seeds`, rejects another scenario |
| 5: NaN crashes aggregation | Fixed and tested |
| 6: config claimed the gate passed before it had | Fixed: comment reworded; the gate has since passed |
| 7: NFE definition | Fixed (same as M3) |
| 8: grid script missing | Fixed: script added, never run |
| 9: test gaps | Fixed: every cell, `assimilate` layout, runner, aggregation and figures. `build_posterior` is covered through the cells |
| Dead `CellOutput.nfe` | Fixed: now used |
| `tune --method` too broad | Fixed |
| Stale references reused silently | Fixed: stored problem compared on load. Particle count is not compared |
| Use `RESULTS_ROOT` | Fixed. `data/lorenz96/data.npz` remains in `driver.py`, `simulate_data.py` and two configs |
| Duplicated mean logic | Fixed |
| `pf_reference` seed-independent | Fixed: grid runs it once |
| `load_curves` duplicates the shared loader; private `_save_fig` | Partly: missing file handled. Kept local because the shared loader filters by M and series the case does not have |
| `KEY_METRICS[:4]` magic slice | Fixed: removed |
| `Standardisation` duplicates `load_preprocesser` | Rejected: different tensor layouts; small |
| Oracle repeats the kick update; class name; three result types | Rejected for now: no behaviour impact, and the filters are verified as they stand |
| Schema docstrings | Fixed |
| Lint | Fixed: unused import, mutable default. Rejected: `zip(strict=)` (repo does not use it), B023 closures (called within their own iteration), one unused loop variable |
| Configs hard-coded device; patience equals epochs | Device fixed (`auto`); patience handled by the step 4 overrides |
| Gate evidence only in ignored `results/` | Addressed: the numbers are in this note |

Review findings did not make any data, model or reference stale. Tuning ran after the
fixes.

Review 4 (code review of caa08af, fixed on `review/lorenz96`): tuning could select a
setting that diverged early (now `diverged_steps`, infinite scores); EnKF settings keys
unchecked; NFE from a partial count; an over-broad `except AssertionError`; tuning and
screen CSVs mixing runs (header check, `base` column); `reference_replicates` not a
real option (must be 2); grid script ran 3 cells at once and could not resume; gate
statistics untested. `driver._reference_task` now calls `pf_gate.run_pf`, and the PF
refuses non-finite log weights. PF, reference, EnKF, oracle and checkpoint outputs
were checked bitwise identical before and after, so the gate, references, models and
data stay valid. The partial tuning CSVs need moving aside (step 7).

## 10. Rules

- No tuning on the test split: `driver.py tune` reads validation only. Do not choose
  anything, including M or the likelihood mode, from test numbers.
- Report negative results honestly. If SI or FM trails EnKF, say so.
- Numbers only from runs that finished. Say which settings were reduced.
- Do not edit the manuscript. Do not touch `ICLR-2027-submission` or the tag
  `ICLR-2027-submission-snapshot`.
- Review work targets `ICLR-2027-review`, not `dev` and not `main`.
- Do not use online tracking; `train.py` writes `metrics.jsonl` locally.
- Additions must not change behaviour or numbers of existing cases.
- Do not loosen `PASS_CRITERIA`. If the problem block changes, data, training, gate and
  references must all be redone.

## 11. Open questions and known problems

1. **Why is `inflated` worse than Jacobian-free?** This is the main scientific question.
   One hypothesis, not tested: the paper's Algorithm 1 draws from p(x^n | y^n, x^{n-1})
   per member without the weights p(y^n | x^{n-1}), so even perfect networks target
   something other than the filtering posterior; the manuscript's conclusion admits this
   ("one-step autoregressive assimilation neither reweights past states"). A more exact
   one-step posterior would then approach that biased target. The evidence is suggestive
   only: on 6 gate sequences the exact `OneStepPosteriorOracle` had gap 0.79, variance
   ratio 1.58 and RMSE 3.67, close to `inflated` (0.96, RMSE 3.59), but those are
   different sequences, different N and M=25. Jacobian-free may look better by offsetting
   errors. To settle it: run the `oracle` cell on validation, and compare. The other
   possibility is a real problem with trained, non-symmetric Jacobians; the exact-drift
   check used a symmetric one. `jacobian_damping` is untuned.
   Related evidence, from the separate branch `review/multistep-linear-gaussian`
   (linear-Gaussian, exact Kalman reference, not yet pushed when this was written):
   there the per-member update provably misses the filtering posterior. In the scalar
   case its steady variance is 2/3 against Kalman's 0.618, more sampling does not close
   the gap, and resampling by the predictive likelihood does. So the hypothesis above
   is established for the linear case but still untested here. Applying SURGE to the
   SDE samplers was under test on that branch as a possible fix; check with the user
   before adding it to this case.
2. The oracle row must be in the final table. It separates the unweighted-scheme bias
   from network error, and the paper's claims need to be read against it.
3. `inflated` diverges slightly at damping 1.0 (fraction 0.0002-0.0027). Damping may
   help.
4. EnKF is ahead in the partial numbers, and the setting is only mildly non-Gaussian, so
   the case may show EnKF winning. That is a legitimate result.
5. The gate caveats in section 6: replicate count, fresh sequences, calibration at 1e5.
6. CUDA, Linux and a fresh `uv sync` are all untested.
7. `run_lorenz96_grid.sh` and the `run.py` path have never run at full size.
8. `make_tables --demo` before-and-after comparison was not run, and the case has no
   LaTeX table: `make_tables.py` was not extended.
9. Step 1 may be out of distribution for the priors, since x^0 is an attractor state plus
   N(0,1) noise. Not measured.
10. The architecture comparison is one short run each, and only with the SI objective.

## 12. Tests on the frozen tree

Also repeated from a clean clone of this commit with the same counts (31 for the case
and path tests together). Run on the Mac (Python 3.13, torch 2.9.0, CPU) at the freeze, with PF and tuning jobs
running alongside:

| Command | Outcome |
| --- | --- |
| `tests/test_lorenz96_solver.py`, `test_filters.py`, `test_periodic_1d.py`, `test_metrics.py` (repo root) | 241 passed, 20 xfailed, 3 warnings; 182 s |
| `tests/test_lorenz96_case.py` (in `$PE`, final version) | 23 passed |
| `tests/test_paths.py` (in `$PE`) | passed, in a run of 31 with the earlier case tests |

The 20 xfails were strict and expected: they recorded that the U-Net is equivariant
only under shifts of 4. They went with the U-Net (section 7, step 3).

After review 4, on Linux (Python 3.13, torch 2.9.0, CPU only, one core): the four
root files above 119 passed (60 s), all of `$REPO/tests` 123 passed (adds
`test_lorenz96_data.py`), `$PE` case and path tests 39 passed, and all of `$PE/tests`
except `test_analytical_multistep.py` 54 passed (adds `test_lorenz96_gate.py`).
`test_analytical_multistep.py` fails to collect in this clone because
`analytical_utils` is not on disk; it fails the same way before review 4.

At the freeze, a 1-epoch `train.py` smoke run after the device change finished and
was deleted. No test touches CUDA, and the generative cells in tests use untrained
networks on CPU with the Jacobian-free likelihood.

After merging the forecast-residual correction (section 13) onto review 4, on the
Mac (CPU, `OMP_NUM_THREADS=2`): the five root files of step 0 123 passed (22 s);
`$PE` `test_lorenz96_case.py` 57 passed (review 4's 31, with `test_every_cell...` now
over 15 cells, plus the correction tests), and step 0's `$PE` command 80 passed (25 s).

## 13. Forecast-residual correction (added 2026-11-09)

Background and derivation: `FORECAST_CORRECTION.md` (sections 2, 4, 5). The library
code is `src/scisi/posterior_models/forecast_correction.py`; this case only wires it in.

### New cells

| Cell | What | Tidy label (method, variant) |
| --- | --- | --- |
| `si_sde_fc`, `fm_ode_fc`, `dm_sde_fc` | the plain sampler (its settings **and** its seed stream) + variant A correction | `Ours (...)`, `forecast_correction` |
| `prior_enkf` / `prior_enkf_fm` | unguided SI / FM prior forecast + the `enkf` cell's update (`EnsembleKalmanFilter.analyse`) | `EnKF`, `trained_prior_si` / `trained_prior_fm` |
| `oracle_fc` | exact one-step kernel per member (as `oracle`) + correction with true-model forecasts and exact K_q | `One-step oracle`, `forecast_correction` |

Every cell now also writes two per-sequence scalars: `innovation_lag1` (pooled lag-1
autocorrelation of normalised innovations) and `innovation_msq`. Both use the
**true-model** forecast of the analysis ensemble, mean_i Phi(x_i) with variance
H Cov(Phi) H' + q + r, so they are the same measure for every method
(`driver.innovation_scores`). Zero lag-1 means white innovations. The E=64
PF-subsample floor is -0.05. Lag-1 is not mean-removed (a constant bias counts as
correlation, as in the library's diagnostics) and pairs only adjacent finite steps.
Both statistics are exact for `noise_stages: 1` only and are NaN (with a logged
warning) otherwise.
Plain cells skipped by a resumed grid keep their review-4 files without these keys, and
`sequence_spread.csv` keeps only metrics every seed has: re-run those cells (`FORCE=1`)
for their innovation metrics to appear for all seeds.

Configuration lives in `configs/case/lorenz96.yaml` (`methods.<cell>_fc.correction`, keys
documented there): `forecast_num_steps` (null means the sampler's M), `localisation`
(GC radius in sites), `gain_scale`, `q_source` (`online` | `exact`), `update_cap`,
`enabled`; unknown keys are refused. `exact` is the library's `q_source="fixed"` with
`q_variance` = the true transition variance (closed-form K_q); on this case it gives
bitwise the same update as the earlier residual bank `sqrt(q d) e_j` it replaces,
with and without localisation (H is a selection). The forecast is always the
posterior's own unguided SDE/ODE (the old `forecast_mode` is gone). A `_fc` cell inherits its plain cell's sampler block, so tuning
the plain cell also moves the corrected one. Plain cells never read the blocks. Off
by default: only the `_fc` cells use it. CLI overrides use dotted keys:

```bash
python cases/lorenz96/driver.py cell --method si_sde_fc --seed 0 --tag fs5_g0.5 \
    --set correction.forecast_num_steps=5 --set correction.gain_scale=0.5
python cases/lorenz96/driver.py tune --method fm_ode_fc --tag gain \
    --set likelihood_mode=dps_jacobian_free --set num_steps=25 \
    --grid '{"correction.localisation":[4],"correction.gain_scale":[0.5,0.75,1.0]}'
```

`cell --set` **requires `--tag`** (it is refused without one). The tag goes into
every output path (`lorenz96__si_sde_fc-fs5_g0.5__traj0__.csv`, the per-sequence
JSON, `ensembles/si_sde_fc-fs5_g0.5__seed0.pt`) and into the tidy variant
(`forecast_correction_fs5_g0.5`), so a sweep cell never overwrites the default cell
and the aggregator reports it as its own row. A
`correction.enabled=false` run of a correction cell needs a tag (e.g. `off`), also
when set in the YAML, so that row does not read as the corrected cell; a tag giving
the label of an untagged cell (`si_sde --tag forecast_correction`) is refused. The
figures plot the untagged default variants only. Tags take letters, digits, `.`, `=`,
`-` and single `_`.

`tune` accepts the cells that read settings: the samplers, their `_fc` cells,
`prior_enkf*`, `enkf`, `enkf_large` and `oracle_fc`. `oracle`, `pf_matched`,
`pf_subsample` and `pf_reference` take no settings, so `tune` does not offer them
and `cell --set` refuses them. `--innovations` adds lag-1, msq and NFE columns (on by
default for the new cells), `--set` changes the base settings (recorded in the
`base` column), and `--tag` suffixes the CSV name
(`tuning/<cell>__<keys>__<tag>.csv`), keeping parallel jobs and differently
configured runs apart. As for every cell (step 7), appending to a CSV with other
columns is refused, and a setting with `diverged_steps > 0` gets infinite scores.

> **The final grid will NOT reproduce the validation table below.** The gains and
> radii below were chosen with the sampler at `dps_jacobian_free` M=25, but the
> `methods:` block still holds the untuned `inflated` M=50 defaults, and the `_fc`
> cells inherit those. Running `run_lorenz96_grid.sh` as is scores the corrected
> cells at `inflated` M=50 with a gain tuned for another sampler. First write the
> chosen sampler settings into `methods:` (step 1 of "What the GPU run should add"),
> then re-tune the correction.

**Off-identity.** Plain `si_sde`, `fm_ode` and `dm_sde` (Jacobian-free, M=25, 4
validation sequences, real priors on MPS), plus `enkf` and `oracle`, were bitwise
identical to the code before the correction (`ec4f97c`). After the merge onto review
4, every plain cell (`si_sde`, `fm_ode`, `dm_sde`, `enkf`, `enkf_large`, `oracle`,
`pf_*`) gives bitwise the ensembles, per-step and per-sequence values of review 4
(`856074f`) on the tiny case (untrained networks, CPU); the only additions are the
two innovation records. So is each `_fc` cell with `correction.enabled: false`
(tested). The plain grid commands are unchanged (dry run).

The table below was measured before the library rewrite of the correction and the
merge onto review 4, and was not rerun since.

### Validation results (reduced settings)

4 validation sequences, seed 0 only, E=64, the 8-epoch priors, sampler
`dps_jacobian_free` M=25 (the best plain setting in section 8; the `methods:`
block still holds the untuned `inflated` M=50 defaults). Mean and variance error are
in PF posterior-std and log units. Lag-1: the PF subsample gives -0.05.

| Method | mean err | var err | sliced W2 | RMSE | spread | lag-1 | NFE/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PF reference, 64 draws (floor) | 0.127 | 0.190 | 0.596 | 2.773 | 2.665 | -0.05 | - |
| EnKF, true model, infl 1.0, loc 4 | **0.382** | 0.384 | **1.286** | 3.038 | 2.811 | 0.06 | - |
| SI prior + EnKF (`prior_enkf`, M=25, loc 4) | 0.385 | 0.381 | 1.302 | 3.054 | 2.625 | 0.06 | 25 |
| FM prior + EnKF (`prior_enkf_fm`, M=25) | 0.392 | 0.380 | 1.313 | 3.034 | 2.764 | 0.05 | 25 |
| FM prior + EnKF, M=5 | **0.381** | 0.413 | 1.313 | 3.037 | 2.514 | 0.07 | 5 |
| SI prior + EnKF, M=5 | 0.465 | 0.620 | 1.605 | 3.059 | 2.069 | 0.11 | 5 |
| oracle (exact kernel), plain | 0.963 | 0.778 | 2.352 | 3.592 | 3.457 | 0.36 | - |
| oracle_fc, loc 4, gain 1, exact K_q | 0.404 | 0.390 | 1.345 | 3.041 | 2.833 | 0.05 | - |
| SI-SDE plain | 0.657 | 0.442 | 1.809 | 3.309 | 2.836 | 0.22 | 25 |
| SI-SDE + corr, loc 4, gain 1 | 0.537 | 0.483 | 1.579 | 3.163 | 2.620 | -0.08 | 75 |
| SI-SDE + corr, loc 4, gain 0.5 (chosen) | 0.434 | 0.511 | 1.427 | 3.068 | 2.502 | 0.06 | 75 |
| FM-ODE plain | 0.628 | 0.476 | 1.786 | 3.296 | 2.999 | 0.20 | 25 |
| FM-ODE + corr, loc 4, gain 1 | 0.574 | 0.497 | 1.603 | 3.115 | 2.804 | -0.07 | 75 |
| FM-ODE + corr, loc 4, gain 0.5 (chosen) | 0.437 | 0.550 | 1.418 | 3.082 | 2.613 | 0.06 | 75 |
| FM-ODE + corr, loc 4, gain 1, exact K_q | 0.421 | 0.542 | 1.376 | 3.040 | 2.620 | 0.05 | 50 |
| DM-SDE plain | 0.680 | 0.499 | 1.853 | 3.317 | 3.075 | 0.24 | 25 |
| DM-SDE + corr, loc 4, gain 1 | 0.510 | 0.455 | 1.492 | 3.058 | 2.808 | -0.05 | 75 |

Tuning grids: EnKF inflation {1.0, 1.05} x radius {2, 4, 8}. Correction radius {2, 4, 8}
at gain 1, then gain {0.25 (SI only), 0.5, 0.75} at radius 4. Oracle radius {none, 2,
4, 8}. Selection by lowest sliced W2 (section 7):
- radius 4 for every cell (8 is equal for SI and the oracle);
- gain 0.5 for SI and FM. For SI, 0.25 was worse (0.513), so 0.5 is interior. For FM,
  0.5 is the grid edge.

DM-SDE was not gain-tuned and inherits the FM value. Gain 0.75 gives lag-1 ~0 (SI
0.008, FM -0.012) but a slightly higher mean error (0.443, 0.464). Gain 1 over-corrects
(lag-1 -0.07 to -0.08, below the PF floor). Exact K_q helps FM (0.574 -> 0.421 at gain
1) and SI less (0.537 -> 0.506), so part of the gap comes from the online Q-hat or the
sampler's realised gain differing from K_q (FORECAST_CORRECTION.md 4.4).

Forecast-steps sweep (radius 4, **gain 1**, online Q-hat; the sweep ran before the gain
was chosen, so it is not at the chosen gain), mean error / sliced W2 / lag-1 / NFE:

| `forecast_num_steps` | SI-SDE + corr | FM-ODE + corr |
| --- | --- | --- |
| 1 | 0.688 / 1.901 / 0.23 / 27 | 0.747 / 1.916 / -0.13 / 27 |
| 2 | 0.428 / 1.389 / 0.04 / 29 | 0.733 / 1.877 / -0.12 / 29 |
| 5 | 0.524 / 1.554 / -0.05 / 35 | 0.653 / 1.746 / -0.09 / 35 |
| 10 | 0.521 / 1.544 / -0.06 / 45 | 0.608 / 1.654 / -0.09 / 45 |
| 25 (= M) | 0.537 / 1.579 / -0.08 / 75 | 0.574 / 1.603 / -0.07 / 75 |

FM-ODE gets steadily worse with fewer steps. Few-step Euler ODE draws are
under-dispersed and biased, which is consistent with the note. The SI value at 2 steps
(0.428) is an outlier with one seed. A plausible reading, not tested: the 2-step SDE
forecasts are over-dispersed (the note measured 1.1 Q), which acts like a lower
effective gain, near the gain-0.5 optimum. Do not choose 2 steps from this alone.

Cost: NFE per assimilation step = M + 2 x `forecast_num_steps` with the online
Q-hat, or M + `forecast_num_steps` with `q_source: exact`. `prior_enkf` costs M (5
suffices for FM). Wall-clock per cell (4 sequences batched, MPS, up to 4 jobs at once
on a loaded machine, so only a rough guide): plain 30-34 s, corrected at 25 forecast
steps 55-69 s, `prior_enkf` 10-14 s at M=25 and 2-5 s at M=5. Scoring (sliced W2
against 4096 PF draws) took about 4 min per run here and dominates validation time.

### Reading

- **The correction is not an advantage here.** The best corrected sampler (0.43) still
  trails `prior_enkf` (0.38-0.39) on mean error, sliced W2 and RMSE, at 3x the NFE, and
  its variance error is worse (0.51-0.55 vs 0.38). That is the outcome
  FORECAST_CORRECTION.md section 5 counts against the method. It matches the note's
  expectation: this case has an exactly Gaussian kernel (`noise_stages=1`), so the
  sampler's full-rank, non-Gaussian handling of Q buys nothing.
- The dropped weight dominates the plain gap: the exact-kernel oracle goes from 0.963
  to 0.404. Network error is the rest: trained+corrected 0.42-0.44 vs oracle_fc 0.40,
  and `prior_enkf` 0.385 vs EnKF 0.382.
- Corrected ensembles are under-dispersed (spread 2.50-2.62 vs PF 2.67 and EnKF 2.81).
- The scratch-study figure of 0.41 for trained SI at gain 1 was not reproduced (0.537
  here). The scratch scripts are not in the repository, so the difference is unexplained.

### What the GPU run should add

1. Write the chosen sampler settings into `methods:` (the corrected cells inherit
   them), then tune on validation again with 40-epoch priors: gain {0.25, 0.5, 0.75,
   1.0} for all three samplers, `q_source` {online, exact}, and the
   `forecast_num_steps` sweep **at the chosen gain**, 3 seeds.
2. Tune `prior_enkf` (inflation {1.0, 1.02, 1.05}, radius {4, 8}, M {5, 10, 25}). It
   is the bar the correction must clear.
3. Final grid: `SEEDS="0 1 2" ./run_lorenz96_grid.sh` now includes the six new
   cells. Report `prior_enkf` next to the corrected cells in any table.
4. If the paper keeps the correction, it needs a case where it beats `prior_enkf`:
   a non-Gaussian kernel (`noise_stages > 1`, which needs a new gate, and new
   innovation statistics: the current ones are NaN there) or E small for d.
   State plainly if none is found.

## 14. Arctan observations (added 2026-11-09)

A second problem on the same system: h(x) = arctan(c L x) / c on the same sites
(L = every 4th site), in PHYSICAL units, same R = 4, 50 steps, same truths, same
observation-noise draws e (`make_sequence` draws everything before h is applied:
y_arctan = h(truth) + e), same seeds, E and initial ensembles. Selected with
`--obs arctan` (before the subcommand) on `driver.py` and `pf_gate.py`, with
`driver.case_config("arctan")`, or `scenario=lorenz96_arctan` in `run.py`.

- **Strength.** c = 1 / sigma_clim = 0.2508373. sigma_clim from the training split
  of `data.npz` (3.8M states): per site 3.9838-3.9900, pooled sqrt(mean_i var_i) =
  3.986648. Not swept. The arctan slope is 0.76 at the attractor mean (2.23) and
  0.14 at x = 10, so observations of large |x| carry little.
- **Config.** `observations: {linear, arctan}`, `methods_arctan:` (a verbatim, NOT
  TUNED copy of `methods:`; copy new cells' lines across) and
  `oracle_arctan.proposals`. Linear leaves `Problem.to_dict()` exactly as before, so
  data and linear references keep loading. `Standardisation.from_data` compares the
  data with the problem's linear version (dynamics only). An arctan reference stores
  `observation` in its problem dict; the two operators' references refuse each other.
- **Paths.** Every output moved to `results/lorenz96/<obs>/` (linear too):
  `reference/`, `gate/`, `tuning/`, `metrics/`, `per_step/`, `per_sequence/`,
  `ensembles/`, `prior_quality.json`. **Existing linear results under
  `results/lorenz96/{reference,gate,tuning,...}` must be moved into
  `results/lorenz96/linear/`.** Tidy scenario `lorenz96_arctan`
  (`Scenario.LORENZ96_ARCTAN`, listed in `NONLINEAR_STUDY_SCENARIOS`, so `SCENARIOS`
  is unchanged). `aggregate_lorenz96.py` and `make_lorenz96_figures.py` take
  `--obs`; `run_lorenz96_grid.sh` takes `OBS=arctan`.
- **Samplers.** `observation_model` gives `StandardisedArctanOperator`, a
  `PointwiseObservationOperator` whose g folds in the de-standardisation,
  g(w) = arctan(c (std w + mean)) / c: h acts on standardised states and returns
  physical observations; `to_model` is the identity and the variance is R = 4.
  Overriding only g / g' keeps `forward/jvp/vjp/gram/jacobian/linearize` and the
  direct g / dg / inner calls of `inflated_shared` consistent; `inner` (the linear
  selection) places observations for localisation.
- **h, not L, per method.** `InterpolantGaussianLikelihood` with a nonlinear
  operator takes `_score_nonlinear`: residual beta (y - h(xhat1)) and J_h at xhat1
  (`dps_jacobian_free`: `gram`; `inflated`: `jvp`/`vjp`; `inflated_shared`: per-member
  pointwise route via g, g', inner). Checked numerically: the samplers give the same
  output with the analytic operator and with a forward-only h whose derivatives come
  from autograd, and differ from the linear run. Correction (`*_fc`): takes the
  likelihood's operator (EnKF-style gains with h(f_i), online K_q); `oracle_fc`
  passes the physical arctan operator and its `q_source: exact` linearises h at each
  group's forecast mean (library). EnKF, `enkf_large`, `prior_enkf*`: stochastic EnKF
  with predicted observations g(H x_i), gain C_xy (C_yy + R)^-1, localisation
  C_xy * (rho H'), C_yy * (H rho H'); equal to the linear update for g = identity,
  same random draws.
- **PF and references.** The optimal proposal needs linear-Gaussian observations.
  The arctan reference uses `linearised`: per particle the optimal proposal of h
  linearised at z (the forecast before the kick) is an importance proposal, with the
  exact weight N(y; h(x), R) N(x; z, qI) / q(x | z, y), resampling after the kick.
  Exact for any h (SIR with a valid proposal, same target); for g = identity the
  weight is the optimal proposal's N(y; Hz, (q+r)I). The weights have finite
  variance when the proposal variance P > q/2 on each observed site, i.e.
  r > q g'^2, which holds here (P/q >= 0.8). `pf_matched` uses the same filter.
- **Oracle.** No closed form: `ImportanceOneStepOracle` draws, per member,
  P = `oracle_arctan.proposals` = 4096 candidates from N(Phi(x_i), q I), weights
  N(y; h(x), R), picks one. Per-member ESS is recorded (`oracle_ess_mean`,
  `oracle_ess_min`, per step and per sequence). Measured on 4 validation sequences,
  E = 64, seed 0: ESS median 1637 (0.40 P), mean 1750, 1% quantile 92, minimum 8.8,
  1.2% of member-steps below 100 (P = 1024: median 410, 8% below 100).
- **Innovations.** y - E[h(x^f)] over the true-model forecast mixture, divided by
  sqrt(Var[h(x^f)] + r), per site. With one noise stage the kick is independent per
  site and g is elementwise, so the per-member moments are 1D Gaussian integrals,
  computed by 40-node Gauss-Hermite (checked against Monte Carlo); the variance is
  var_i(m_i) + mean_i v_i + r, the linear formula for g = identity. These are the
  exact predictive moments, so for an exact filter the normalised innovations are
  still uncorrelated (martingale differences) with unit mean square. What changes:
  they are not Gaussian, so the PF-subsample lag-1 floor must be re-measured for
  arctan, not taken from the linear problem.
- **Gate.** Same commands, thresholds and procedure with `--obs arctan`; outputs in
  `results/lorenz96/arctan/gate/` (a bare `--out` name goes there). `converge` runs
  the `linearised` filter. `cross-proposal` compares `linearised` with `bootstrap`
  (the report's `*_optimal` keys hold the linearised numbers; `proposals` names
  them). `calibrate` adds the negative control `wrong_obs_operator_linear` (a filter
  assuming h = L x). `screen` is linear only. Only tiny smoke runs were done; **the
  arctan gate has not run.**

Single-sequence probe (validation 0, N = 1e5, one thread), ESS fraction mean / abs
min: linear optimal 0.131 / 11; arctan linearised 0.209 / 2061; arctan bootstrap
0.175 / 1262. At N = 1e6 the arctan abs min was 14212. Arctan is easier for the PF
than the linear problem, mainly because its observations are weaker. Cross-proposal
system (d = 8, stride 2, T = 20, N = 1e6): ESS fraction 0.34 linearised, 0.28
bootstrap.

Cost estimate (this 16-core machine, one thread per run, measured while other jobs
ran; cost is linear in N: 8.2 s at 1e5, 86 s at 1e6, so about 345 s per 50-step run
at 4e6): references, 16 sequences x 2 replicates = 32 runs, about 3.1 CPU-hours;
`converge --ladder 1e5,4e6 --replicates 8 --sequences 10`, 80 runs at each rung,
about 7.9 CPU-hours; `calibrate` (1e5, 32 sequences, 5 filters) about 0.4 CPU-hours;
`cross-proposal` (1e6, 2 sequences, 8 replicates, 2 proposals, about 24 s each)
about 0.2 CPU-hours. Total about 11.5 CPU-hours; with 8 workers about 1.5-2 h wall
clock if memory bandwidth does not limit (about 1 GB per worker at 4e6).

```bash
python cases/lorenz96/pf_gate.py --obs arctan converge --ladder 1e5,4e6 --replicates 8 --sequences 10 --workers 8 --threads 1 --out convergence_4e6.json
python cases/lorenz96/pf_gate.py --obs arctan calibrate --n 100000 --sequences 32 --workers 8 --threads 1
python cases/lorenz96/pf_gate.py --obs arctan cross-proposal --problem '{"dim_state":8,"obs_stride":2,"num_steps":20}' --n 1000000 --replicates 8 --sequences 2 --workers 8 --threads 1
python cases/lorenz96/driver.py --obs arctan reference --split validation --workers 8
python cases/lorenz96/driver.py --obs arctan reference --split test --workers 8
```
