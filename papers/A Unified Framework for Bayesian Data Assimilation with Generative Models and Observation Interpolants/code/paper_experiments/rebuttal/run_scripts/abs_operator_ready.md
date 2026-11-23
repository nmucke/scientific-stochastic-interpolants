# L96 abs operator (h(x) = |L x|) ready for method cells — 2026-11-18

`--obs abs` works end to end for method cells (uncommitted, branch review/rebuttal-runs).
Same system / sites / truths / noise draws as linear/arctan/square; h = |x| (exact, not
smoothed) on every 4th site, physical units, no strength, obs noise N(0, 4 I).
Sign-blind -> bimodal observed-site posteriors (modes near +-y).

Wiring (mirrors square): `Observation("abs")`, `abs_nonlinearity` (|w|, sign w),
`StandardisedAbsOperator` (g = |std w + mean|, g' = std sign(std w + mean)),
config `observations.abs`, `oracle_abs`, `evaluation_abs: {test_sequences: 3}`,
`methods_abs:` (verbatim copy of `methods_arctan:`, not tuned), scenario `lorenz96_abs`.
Results under `results/lorenz96/abs/`.

Smoke-tested (CPU, 1 test sequence, E=64, `--defer-scoring`): EnKF, SI-SDE (shared) and
SI-SDE (shared) + corr. (M=10 via --allow-other-m) run and write finite ensembles
[S, 50, 64, 16]; members land on both signs.

## Cell commands (from paper_experiments, after `source results/_runs/env.sh`)

CPU: prefix `CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu`.

```bash
# EnKF E=64
python cases/lorenz96/driver.py --obs abs cell --method enkf --seed 0 --defer-scoring
# SI-SDE (shared), M in {25, 50, 100, 250}
python cases/lorenz96/driver.py --obs abs cell --method si_sde --seed 0 --num-steps 100 \
    --tag shared --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 --defer-scoring
# SI-SDE (shared) + corr.
python cases/lorenz96/driver.py --obs abs cell --method si_sde_fc --seed 0 --num-steps 100 \
    --tag shared --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 \
    --set correction.gain_scale=1.0 --set correction.q_source=online --defer-scoring
# separate tree / first N sequences / other E:
python cases/lorenz96/quick_cell.py --root results/_runs/<name> --sequences 1 --ensemble-size 64 -- \
    --obs abs cell --method si_sde --seed 0 --num-steps 100 --tag shared \
    --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 --defer-scoring
# rescore later: the SAME command with --rescore instead of --defer-scoring
# (needs results/lorenz96/abs/reference/test/sequence{0,1,2}_replicate{0,1}.pt)
```

PF reference: exact optimal proposal `abs_optimal` (library); launching in tmux
`rb_abs_ref` (`driver.py --obs abs reference --split test --workers 6`, N=4e6, 3 seqs x 2
replicates, cores 20-31). Files appear in `results/lorenz96/abs/reference/test/`.
