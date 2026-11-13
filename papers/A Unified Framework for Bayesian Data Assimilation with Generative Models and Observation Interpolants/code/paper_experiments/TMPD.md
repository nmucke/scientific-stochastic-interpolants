# TMPD baseline

TMPD is from Boys et al., *Tweedie Moment Projected Diffusions for Inverse
Problems* (arXiv 2310.06721; reference code github.com/bb515/tmpdjax). It was
added because the reviewers said our method overlaps with it.

## Where it lives

- `src/scisi/likelihood_models/tmpd.py`: `TMPDLikelihood`.
  - The Tweedie mean is x̂₁ = (x_τ + σ_τ² s)/β_τ, and the covariance is C = (σ_τ²/β_τ) ∇x̂₁.
  - C is held fixed; only the mean is differentiated (reverse-mode VJPs through the network).
  - `covariance`:
    - `full`: d_y VJPs per step, plus a dense solve.
    - `diagonal`: d_y VJPs per step.
    - `row_sum`: 2 VJPs per step. This is the paper's DTMPD, used in all of its image experiments.
  - Linear observation operators only.
- `src/scisi/posterior_models/tmpd_posterior.py`: `TMPDPosterior`, the guided reverse SDE.
  - The guidance weight is the score-substitution weight a_τ + ½g², i.e. s is replaced by s + f^y. On TMPD's VP SDE this equals their g².
  - `guidance_weight="g_squared"` (SDA's weight) is for ablation only. On our √(αβ) schedule it is biased: a Gaussian test gives covariance error ≈ 0.9 vs ≤ 0.012 for the correct weight.
- `configs/method/tmpd.yaml`, `Method.TMPD`, and the new `Metric.VJP` (backward passes, which NFE does not count).
- Tests: `tests/test_tmpd.py` and `tests/test_tmpd_wiring.py` in this directory.

## Wired cases

TMPD runs on the FM prior read as a diffusion model. It uses the same √(αβ) diffusion
schedule and full first step as Ours (DM-SDE), so that head-to-head is fair.

- **Analytical** (single-step and multistep). Opt-in:
  `run.py case=analytical seeds=[0,1,2,3,4] num_steps=$M '+analytical_methods=["TMPD"]' results_file=...`
  For multistep, use `case=analytical_multistep`.
- **Lorenz 96, linear sweep only.** The `tmpd` cell is a `baselines` row of
  `lorenz96_lineup.ROWS` with `sweeps=("linear",)`. It runs through
  `_l96_baselines.run_tmpd` (posterior: `_l96_generative.build_tmpd_posterior`), and its
  settings are `methods.tmpd` in `configs/case/lorenz96.yaml`. It has no
  `methods_arctan` entry.
  - One cell at one M, for M in `lorenz96_lineup.M_VALUES` = 25, 50, 100, 250:
    `python cases/lorenz96/driver.py cell --method tmpd --seed S --num-steps M`.
    The default `--obs linear` applies.
  - Every cell of the sweep: `SEEDS="0" ./run_lorenz96.sh final linear` runs `tmpd`
    at every M. It skips cells that are already done.
  - A non-default covariance needs a tag, for example
    `... cell --method tmpd --seed 0 --num-steps 50 --tag rowsum --set covariance=row_sum`.
  - Not tuned: there is no `GRIDS` row in `run_lorenz96.sh`, and `num_steps` and
    `diffusion_scale` are fixed as for `dm_sde`.
  - In the arctan sweep the cell is not listed (`lorenz96_lineup.py cells generative
    arctan`), and `driver.py --obs arctan cell --method tmpd` is refused.
  - It writes a `vjp` metric next to `nfe`.

On linear-Gaussian problems TMPD and our method are both exact, so the analytical rows
should agree by construction. Only learned priors (Lorenz 96, NS, urban) separate them.

## Navier–Stokes and urban (uDALES): wired, opt-in

- `build_posterior` → `_build_tmpd` (`cases/navier_stokes/_ns_pipeline.py`), reused by urban:
  the FM model, `endpoint_vanishing_diffusion` (√(αβ), `tmpd.yaml` `diffusion_term.multiplier`
  = 1.0 as `dm_sde.yaml`), the τ=0 node unguided (velocity kept), as Ours (DM-SDE).
  Covariance `likelihood_model.covariance_ns_urban: row_sum`; `covariance: full` stays the
  analytical default. Nonlinear scenarios raise `NotImplementedError` (n/a rows).
- `Method.TMPD` is in both drivers' `METHOD_CONFIG_NAME`/`WIRED_METHODS`, not in
  `NS_METHODS`/`URBAN_METHODS`: run with `+ns_methods=["TMPD"]` / `+urban_methods=["TMPD"]`.
- `run_assimilation` reports `vjp_per_step` (likelihoods with `vjp_count` only) and the
  drivers write a `vjp` row; other methods' rows are unchanged.
- E=64 in one batch does not fit a 9.5 GiB MIG slice (the backward through the UNet):
  use `+posterior_batch_size` (NS 16, urban 8; new opt-in knob on NS). Members are
  independent in TMPD, so chunking changes nothing but the noise draw order.
- Launchers: `slurm_tmpd/{ns_wrong_ic,ns_trueic,udales}/submit.sh` (`DRY_RUN=1`),
  smoke `slurm_tmpd/smoke.sh`; tests `tests/test_tmpd_ns_urban.py`.

MIG smoke (2026-11-13, 1g.10gb, E=64, M=100 NS / 50 urban): NS 32² finite (RMSE 0.85 over 2
steps vs 0.55 Ours DM-SDE jacfree), 276 s/step, 2.9 GiB; NS sparse 5% NaN in the first step
(the row-sum diagonal is negative for up to ~10% of sensors at τ<0.35 with the
non-symmetric learned Jacobian, guidance spikes to ~10³); urban 1.5625% finite but each
update increases the error (RMSE 0.70 → 3.2 over 2 updates vs 0.62 Ours), 147 s/step,
8.0 GiB at batch 8 (~6 h per 150-step cell: gpu-v100, not MIG).

## Smoke results (Mac, small settings; not paper numbers)

- **Analytical** (M = 50, 2 seeds): KL 0.0022, sliced W2 0.020. Ours (DM-SDE, shared) is
  0.0023 and SDA 0.17.
- **Lorenz 96** (2 validation sequences, E = 32, M = 50): all metrics finite.
  - `full`: RMSE 3.70, 33 s.
  - `row_sum`: RMSE 3.73, 15 s.
