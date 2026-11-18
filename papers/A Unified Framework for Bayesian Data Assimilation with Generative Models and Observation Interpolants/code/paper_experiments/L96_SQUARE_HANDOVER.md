# Lorenz 96 with h(x) = x² — handover (2026-11-18)

New Lorenz 96 experiment: the same system as the linear/arctan cases (16 sites, every 4th
observed, obs noise N(0, 4 I), 50 assimilation steps), but observed through **h(x) = (L x)²**
elementwise (`--obs square`, no strength). The sign is unobserved, so observed-site
posteriors are bimodal (modes near ±√y). **Purpose: test whether our samplers represent the
bimodal posterior.**

Lineup: PF reference, EnKF (E=64), Ours SI-SDE (shared), Ours SI-SDE (shared) + corr.

## Code (branch `review/rebuttal-runs`)

- `src/scisi/particle_filter/bootstrap.py`: proposal `square_mixture` (exact importance
  sampling; per observed site a defensive mixture of the transition kick (weight 0.1) and
  Laplace components at **both** local modes of the one-site posterior, modes from a cubic).
  New reference fields `final_samples` (1e5 equal-weight draws at the last step) and
  `positive_fraction` [T, d] (P(x>0) from all N particles). Linear/arctan outputs unchanged.
  Plain bootstrap collapses here (ESS abs min 1 at N=1e5 and 1e6), so it is not used.
- `cases/lorenz96/_l96_system.py`: `Observation("square")`; per-operator evaluation overrides
  `evaluation_<obs>` (merged into `evaluation`).
- `cases/lorenz96/_l96_generative.py`: `StandardisedSquareOperator` (g = (std w + mean)²,
  g' = 2 std (std w + mean)) for the samplers' linearised guidance.
- `cases/lorenz96/driver.py`: `cell --defer-scoring` (run + save ensembles and
  `<cell>__seed<s>.meta.pt`, no reference needed) and `cell --rescore` (score saved
  ensembles later; same arguments as the deferred run). `oracle_<obs>` config lookup.
- `cases/lorenz96/pf_gate.py`: square uses `square_mixture`; `converge` reports mode-mass error.
- `cases/lorenz96/quick_cell.py` (new): runs driver.py with `--root DIR` (outputs under
  DIR/lorenz96/<obs>/), `--sequences N` (first N test sequences) and `--ensemble-size E`.
- `configs/case/lorenz96.yaml`: `observations.square`, `oracle_square`,
  `evaluation_square: {test_sequences: 3}` (**currently 3**, set to 12 for the full test
  split), `methods_square:` (verbatim copy of `methods_arctan:`, not tuned on square).
- `make_l96_square_figures.py` (new): metrics table + final-step marginal figure
  (PF vs EnKF vs SI-SDE shared vs + corr.), sign-mass diagnostics.
- `lorenz96_lineup.py`, `results_schema.py`, `run_lorenz96.sh`, `make_lorenz96_tables.py`,
  `configs/scenario/lorenz96_square.yaml`, tests: `square` added. `cases/lorenz96/GPU_HANDOFF.md` §16.
- Tests: L96 suite (`tests/test_lorenz96_*.py`) and library `tests/test_filters*.py` pass
  (incl. new square-mixture-vs-grid-filter and deferred-scoring tests).

## Settings used

All from the arctan picks (`methods_square:` / arctan `selected_shared_stage{1,2}.json`):

| cell | args |
|---|---|
| EnKF E=64 | `--method enkf --seed 0` (inflation 1.0, localisation 4) |
| SI-SDE (shared) | `--method si_sde --seed 0 --num-steps M --tag shared --set likelihood_mode=inflated_shared --set jacobian_damping=0.9` |
| SI-SDE (shared) + corr. | as above with `--method si_sde_fc` and `--set correction.gain_scale=1.0 --set correction.q_source=online` |

Output names: `ensembles/enkf__seed0.pt`, `ensembles/si_sde-shared__M<M>__seed0.pt`,
`ensembles/si_sde_fc-shared__M<M>__seed0.pt` (+ `.meta.pt`), shape [S, 50, E, 16].
**The ensemble size is not in the file name**: run other E into another `--root`.

## Commands (from paper_experiments, after `source results/_runs/env.sh` or the cluster env)

```bash
# PF reference (CPU; one task = one sequence x replicate, ~16 min at N=4e6 on one core)
python cases/lorenz96/driver.py --obs square reference --split test --workers 10

# EnKF / SI-SDE on the main tree, scoring deferred (GPU: export SCISI_REQUIRE_CUDA=1)
python cases/lorenz96/driver.py --obs square cell --method si_sde --seed 0 --num-steps 100 \
    --tag shared --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 --defer-scoring
python cases/lorenz96/driver.py --obs square cell --method si_sde_fc --seed 0 --num-steps 100 \
    --tag shared --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 \
    --set correction.gain_scale=1.0 --set correction.q_source=online --defer-scoring

# a separate tree, first sequence only, E=256 (link the reference in for --rescore)
mkdir -p results/_runs/myrun/lorenz96/square
ln -s "$PWD/results/lorenz96/square/reference" results/_runs/myrun/lorenz96/square/reference
python cases/lorenz96/quick_cell.py --root results/_runs/myrun --sequences 1 --ensemble-size 256 -- \
    --obs square cell --method si_sde --seed 0 --num-steps 100 --tag shared \
    --set likelihood_mode=inflated_shared --set jacobian_damping=0.9 --defer-scoring

# rescore: the SAME command with --rescore instead of --defer-scoring
# figures + table:
python make_l96_square_figures.py --obs square --root <root>/lorenz96 --M 100 \
    --num-sequences 1 --sequences 0 --rows 1 --out <dir>
```

Each cell also reruns one sequence for timing at the end (it roughly doubles a 1-sequence run).
SI-SDE `inflated_shared` runs the sequences one after another.

## PF reference status

`results/lorenz96/square/reference/test/sequence{0..4}_replicate{0,1}.pt` exist (N=4e6,
915-958 s per task on one core). Not in git (`results/` is ignored): copy it to the cluster
(about 200 MB) if you want to rescore there, e.g.
`rsync -a squamish:<PE>/results/lorenz96/square/reference/ <cluster PE>/results/lorenz96/square/reference/`.

**The reference does not pass the gate.** Replicate 0 vs 1: ESS fine (abs min >= 1837), but
final-step mode masses P(x>0) differ by up to 0.31 (seq 2; seq 0: 0.03), mean error
0.25-0.64 posterior sd (gate 0.10), log-Z differs by up to 4.9. Likely path degeneracy (a
thinned sign branch is not re-seeded). Accepted for now (user, 2026-11-18); options: larger N
(1.6e7+), or MCMC moves after resampling. Show replicate 1 vs 0 (the `pf_reference` row) as
the reference floor.

## Runs on squamish (RTX 3090 machine), 2026-11-18

GPU: repeated `CUDA error: unspecified launch failure` (also with `CUDA_LAUNCH_BLOCKING=1`,
at an ordinary `layer_norm` in the prior network, `periodic_1d.py:78`), so everything ran on
the CPU (`CUDA_VISIBLE_DEVICES= SCISI_REQUIRE_CUDA=0 SCISI_SCORING_DEVICE=cpu`).

| run | root | status |
|---|---|---|
| EnKF E=64, seq 0 | `results/_runs/sq1_quick` | done, scored |
| SI-SDE shared / + corr., M=50, seq 0 | `results/_runs/sq1_quick` | done, scored |
| SI-SDE shared / + corr., M=100 and M=250, seq 0, E=64 | `results/_runs/sq1_quick` | running (tmux `rb_sq1_si_sde{,_fc}_M{100,250}`) |
| EnKF E=256, seq 0 | `results/_runs/sq1_quick_E256` | done |
| SI-SDE shared / + corr., M=100, E=256, seq 0 | `results/_runs/sq1_quick_E256` | running (tmux `rb_sqE256_si{,_fc}`) |
| EnKF E=64, seqs 0-11 / 0-4 | `results/lorenz96/square/_12seq_enkf`, `_5seq_enkf` | kept, unused |

Figures/tables: `/export/scratch2/ntm/delftblue_results/results_tables/figures/l96_square/quick1*/`.

First result (seq 0, M=50, E=64, final step): the PF marginal at observed site x0 is
bimodal (P(x>0)=0.48). SI-SDE (shared) puts members in both modes (0.70) and has the
smallest mode-mass error at bimodal sites (0.12 vs EnKF 0.25, + corr. 0.23); the EnKF is
unimodal with a long tail (0.86) but has the best time-averaged sliced W2 / kNN-KL / energy
distance (2.00 / 9.54 / 3.48 vs shared 2.67 / 11.0 / 5.30, + corr. 2.27 / 9.85 / 4.18).
One sequence only.
