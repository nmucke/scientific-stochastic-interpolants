# slurm_bench: smoke tests, benchmarks and checks (NS rebuttal, DelftBlue)

Scripts only; their outputs stayed on DelftBlue in
`/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/` (copy in
`paper_experiments/results/_smoke_and_bench/slurm_bench/`). Paths inside are DelftBlue paths,
kept as a record of how the checks ran. See `../REBUTTAL_EXPERIMENTS.md` §5.

- `equiv/`: old-vs-new code equivalence on NS (`equiv.sh` original, `equiv2.sh` with SDA moved
  to V100; `compare.py` reports per-step and state differences, old1-vs-old2 as noise floor).
- `surge_ours/smoke.sh`: SURGE on our SI/DM samplers vs plain, 2 assimilation steps.
- `fc/bench.sh`: forecast-correction cost and peak memory.
- `mig_pair.sh`: two commands on two MIG 1g.10gb slices in one job.
- `gpucheck.sbatch`, `jaxcheck.*`, `bench.sbatch`, `cell.sh`, `v100/check.sh`,
  `hook/sitecustomize.py`: GPU / JAX / torch environment checks.
