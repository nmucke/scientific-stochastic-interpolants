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

## Not wired: Navier–Stokes and urban (uDALES)

The library classes need no change; only the drivers do.

1. Add a `"TMPD"` branch to `build_posterior` in `cases/navier_stokes/_ns_pipeline.py`
   (about l.876, next to SDA). Urban reuses this dispatch through `_urban_pipeline.py`.
2. Add `Method.TMPD: "tmpd"` to `METHOD_CONFIG_NAME` in `cases/navier_stokes/driver.py`
   (about l.128) and `cases/urban/driver.py` (about l.147). Then add it to the method
   lineups and grid scripts, and emit the `vjp` metric alongside NFE.

Decisions to make when wiring:

- **Covariance: use `row_sum`.** `full` and `diagonal` need one backward pass per
  observation, about 800–1000 per step on NS and more on urban, which is infeasible. Full
  also stores a d_x × d_y matrix per member. Row-sum is what the TMPD paper itself used for
  high-dimensional data.
- **Diffusion schedule.** The NS SDA branch uses `diffusion_term=None`, i.e. the diffusion
  model's default g = α. The analytical case, Lorenz 96 and Ours (DM-SDE) use the
  endpoint-vanishing √(αβ). For a fair comparison against Ours (DM-SDE), build TMPD as on
  Lorenz 96: FM model plus `endpoint_vanishing_diffusion`.
- **Nonlinear scenarios** (`sparse_5_arctan`, `sparse_5_speed`, `sparse_1p5_arctan_vel`):
  TMPD raises `NotImplementedError`, because nonlinear h is a stated limitation of the
  paper. Report these rows as n/a. Adding an EKF-style linearisation would no longer be
  faithful TMPD.
- **Stability risk.** At NS scale our method needs Jacobian damping (λ = 0.9–0.95),
  because the learned UNet Jacobian is non-symmetric and λ = 1 diverges. Faithful TMPD has
  no such setting and may be unstable there. On short Lorenz 96 smoke runs about 0.2% of
  members already diverged with `full`/`row_sum`, as with Ours (DM-SDE, inflated). If TMPD
  diverges on NS, report that; do not add a damping knob to the baseline.

## Smoke results (Mac, small settings; not paper numbers)

- **Analytical** (M = 50, 2 seeds): KL 0.0022, sliced W2 0.020. Ours (DM-SDE, shared) is
  0.0023 and SDA 0.17.
- **Lorenz 96** (2 validation sequences, E = 32, M = 50): all metrics finite.
  - `full`: RMSE 3.70, 33 s.
  - `row_sum`: RMSE 3.73, 15 s.
