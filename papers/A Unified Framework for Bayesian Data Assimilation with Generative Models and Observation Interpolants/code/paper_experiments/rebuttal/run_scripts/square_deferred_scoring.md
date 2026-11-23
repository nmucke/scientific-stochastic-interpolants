# L96 `--obs square`: run cells before the PF reference exists (2026-11-18)

`driver.py cell` has two mutually exclusive flags (uncommitted, cases/lorenz96/driver.py):

- `--defer-scoring`: run the method and save, without reading the reference,
  - `results/lorenz96/<obs>/ensembles/<cell>[-<tag>][__M<M>]__seed<s>.pt`: float32 `[S=12, T=50, E, d=16]`, the same file a normal run writes
  - `...__seed<s>.meta.pt`: what scoring needs besides the ensembles: settings, split, E, M, NFE, VJP, SMC/oracle-ESS records, the per-step seconds from the usual 1-sequence timing rerun, and the innovation statistics.
  It writes no metrics, per_step or per_sequence files and prints `scoring deferred (cell --rescore)`.
- `--rescore`: no run. It loads the ensembles and meta file, refuses them if their settings or split differ from the current config + `--set`/`--num-steps`, then scores them against reference replicate 0 and writes `metrics/`, `per_step/` and `per_sequence/` exactly as a normal `cell` run does (same file names and records; `seconds` comes from the deferred run).
  Scores come from the float32 file, so they can differ from inline scoring in the last digits (tested: rel < 1e-4).
- A normal run (neither flag) also writes the `.meta.pt`. If the reference turns out to be missing, the run fails after the ensembles + meta are saved, so `--rescore` can recover it.

Truth and observations are rebuilt from the unchanged seeding (`make_sequence(problem, derive_seed("lorenz96-evaluation","sequence","test",i))`), so rescoring is consistent.

**Pass the same `--obs`, `--method`, `--seed`, `--num-steps`, `--tag` and `--set` to `--rescore` as to the deferred run.** They determine the file names and the settings check.

## Commands (from "$PE", after `source "$PE/results/_runs/env.sh"`)

```bash
# EnKF (CPU, done 2026-11-18 17:50, 18 s; log results/_runs/sq_enkf_20261118_174948.log)
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 24-31 \
  $PY cases/lorenz96/driver.py --obs square cell --method enkf --seed 0 --threads 8 --defer-scoring
# SI-SDE (GPU), e.g. M=50
$PY cases/lorenz96/driver.py --obs square cell --method si_sde --seed 0 --num-steps 50 --defer-scoring

# once results/lorenz96/square/reference/test/sequence{0..11}_replicate{0,1}.pt exist:
$PY cases/lorenz96/driver.py --obs square cell --method enkf   --seed 0 --rescore
$PY cases/lorenz96/driver.py --obs square cell --method si_sde --seed 0 --num-steps 50 --rescore
```

Rescoring is cheap and runs on the scoring device (CUDA if visible, else CPU).
Test: `tests/test_lorenz96_case.py::test_deferred_scoring_then_rescore_matches_inline_scoring` (enkf, si_sde).
