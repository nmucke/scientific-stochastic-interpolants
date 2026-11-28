# Stochastic Navier–Stokes experiment scripts

Run everything from the repository root with `.venv/bin/python`. Default output
paths still point to `research/ito_maps/results/...`, where the new runs will
write. The old results are in `research/ito_maps/archive/results/`. Defaults
marked **old** name the initial-stage runs: checkpoint names such as
`ito_rollout_seed0`/`flow_rollout_seed0`, and fixed protocols such as test
trajectories 180–199 or E=16. The checkpoints still exist under
`checkpoints/stochastic_navier_stokes/`. Update those defaults before the new
experiments.

## Training launchers
- `train_psd_pilots.sh`: PSD stage for the Itô map and flow map (seed 0), via `scisi.bin.main_map_model_train` → `checkpoints/stochastic_navier_stokes/{ito,flow}_psd_seed0`. **old** hard-coded budgets and checkpoint dirs.
- `train_rollout_stage.sh`: teacher-rollout fine-tuning starting from `*_psd_seed0`, then a validation `compare.py` run → `*_rollout_seed0`, `results/validation_rollout`. **old** names and budgets.
- `train_replications.sh [K] [seeds...]`: full two-stage training for extra seeds, with an optional K-refinement through `refine_path_features.py`. **old** checkpoint naming.
- `refine_path_features.py`: refines Brownian path features (K → finer K) without changing the map's function; `--checkpoint-dir` → `--output-dir` (both required).
- `archive_training.py`: writes lightweight provenance and lineage hashes for checkpoint dirs → `results/training/`. Used by `verify_confirmation.py` and `tests/test_archive_training.py`.

## Forecast evaluation (one-step / short rollout)
- `compare.py`: evaluates the same held-out cases for solver and map checkpoints (`--checkpoints` names under `checkpoints/stochastic_navier_stokes`, `--steps`, `--origins`, `--horizon`) → `<output>/<model>/evaluation.json`. **old** default checkpoints `stochastic_interpolant`/`flow_matching`, data config from the `stochastic_interpolant` checkpoint, splits val=160, test=180.
- `persistence_baseline.py`: CPU persistence baseline on the same cases → `--output`. **old** default config is the `stochastic_interpolant` checkpoint, test offset 180.
- `report_results.py`: figures and tables from `evaluation.json` files (`--inputs`, `--output`). Library module for the scripts below.
- `paired_comparison.py`: paired trajectory-bootstrap differences between chosen model/budget pairs (`--output` required).
- `training_seed_report.py`: separates training-seed variation from test-trajectory uncertainty, using checkpoint metadata. Covered by `tests/test_map_report.py`.
- `final_report.py`: combined one-step and ten-lead test report from completed evaluations (`--one-step`, `--rollout`, optional seeds and persistence, `--output`). **old** frozen protocol defaults: test 180–199, E=16, origins 5/45/85, rollout from origin 45, predeclared pairs.
- `plot_forecasts.py`: qualitative plots for a predefined held-out case (`--checkpoints` paths, `--steps`, `--output`). **old** fixed case: trajectory 180, origin 45, E=16, seed 20261007.

## Long-horizon autonomous NS rollout
- `audit_ns_time_reference.py`: recovers dataset timestamps and the held-out enstrophy climatology → `results/long_rollout/data_reference_audit.json` and `results/long_horizon/reference_enstrophy.json`. Run this first; the rollout and the reports read its output.
- `long_horizon_ns.py`: resumable autonomous rollout with streamed diagnostics and sparse raw states (`--checkpoint-dir`, `--output`, `--steps`, `--horizon`, `--ensemble`, `--time-audit`, `--resume`). Covered by `tests/test_long_horizon_ns.py`.
- `run_long_horizon.sh`: runs four models through `long_horizon_ns.py` → `results/long_horizon/<name>`, then `report_long_horizon.py`. **old** checkpoints `flow_rollout_seed0`/`ito_rollout_seed0`/`stochastic_interpolant`/`flow_matching`, budgets 1/8/64/64, horizon 4096, E=8.
- `report_long_horizon.py`: boundedness and skill report for finished rollouts (`--inputs`, `--reference`, `--output`).
- `spatial_enstrophy_spectra.py`: Parseval-normalized spatial enstrophy spectra from saved raw states (`--run-dirs`, `--leads`, `--output`). Default trajectory 180.
- `audit_long_horizon_units.py`: independent CPU re-check of finished rollouts (hashes, units, restart consistency) (`--run-dirs`, `--output`).
- `finalize_long_horizon.py`: CPU finalization; runs the audit, spectra and report in order over `results/long_horizon`. **old**: hard-codes the four model names, budgets 1/8/64/64, trajectory 180, origin 5, E=8, horizon 4096. Covered by `tests/test_finalize_long_horizon.py`.
- `simulate_assumed_ns_reference.py`: CPU PDE continuation using viscosity, drag and noise that you supply; these are not verified truth (`--output` required). Covered by `tests/test_assumed_ns_reference.py`.
- `benchmark_assumed_ns_reference.py`: CPU timing of that PDE solver (jax_cfd) → `results/long_horizon/cpu_reference_benchmark.json`.

## Path-compression and analytic diagnostics
- `ns_path_compression.py`: estimates the endpoint variance a frozen teacher leaves unexplained, given K Brownian features → `results/ns_path_compression`. **old** default teacher `stochastic_interpolant`, validation trajectory 160.
- `brownian_compression.py`: analytic linear-SDE Brownian-feature compression → `experiments/results/brownian_compression` (the old outputs are archived).
- `gaussian_score_calibration.py`: analytic Gaussian proper-score vs. calibration illustration → `experiments/results/gaussian_score_calibration` (the old outputs are archived).

## Old-protocol audit, kept because tests import it
- `verify_confirmation.py`: audits the old frozen three-seed confirmation against `results/frozen_protocol.json` (now in `archive/results/`). It applies only to that protocol and would need a new protocol file to be useful. Kept only for `tests/test_verify_confirmation.py`.

Archived to `research/ito_maps/archive/experiments/`: `run_confirmation.sh`, the exact frozen confirmation run, and `rebuild_confirmation_report.sh`, which rebuilt the old `final_report`.

## WP1 step 1: coloured driver and matched schedule (handover §4 WP1)
- `estimate_noise_spectrum.py`: writes the `FourierDiagonalCovariance` spectrum file C_k = κ Σ̄_k / q(1) for a dataset config, from `increment` (y − c, upper bound), `residuals` of a `models=conditional_mean` regressor on validation (preferred), `anomalies` of an ensemble file, or a precomputed `power` array (Phase 0 D1). Floor, mode smoothing, `--zero-mean` for the WP2 invariant. Covered by `tests/test_estimate_noise_spectrum.py`.
- `wp1_euler_exact_noise.py`: Euler drift + exact integrated noise of an SI checkpoint at N = 1, 2, 4, 8 against its own N = 128 on one shared path per validation case: spread ratio, fair CRPS, same-path RMSE, shell spectra. Covered by `tests/test_wp1_euler_exact_noise.py`.
- `ns_path_compression.py` now draws the teacher's own driver (covariance and variance clock), so it runs unchanged on coloured processes.
- `wp1_step1_report.py` (+ `slurm_wp1/report.sbatch`, CPU): WP1 step-1 report from finished outputs only: path-compression omitted fraction per process (SE across histories, paired vs white), Euler/exact-noise spread ratio, fair CRPS, spread/skill and anomaly band ratios with a trajectory bootstrap (paired across models), residual-spectrum shell profile vs another resolution, training curves. Results for 64×64 seed 0 in `research/ito_maps/results/wp1_stochastic_navier_stokes_64/`. Covered by `tests/test_wp1_step1_report.py`.
- `slurm_wp1/`: `train_si.sbatch` (VARIANT = white_t2 | coloured_t | coloured_matched, SPECTRUM), `eval_si.sbatch` (path compression K = 1, 4 and the Euler diagnostic), `estimate_spectrum.sbatch`, `train_regressor.sbatch`, and `smoke.sbatch` (all of them at tiny budgets on a MIG slice). Config-driven: `CONFIG` (default `stochastic_navier_stokes_64`, the 64×64 development dataset) selects `--config-name`, checkpoints go to `checkpoints/$CONFIG/<run>` and spectra/hydra/trackio/eval outputs to `/scratch/.../results_itomap/wp1/$CONFIG/`; `CONFIG=stochastic_navier_stokes` gives the 128×128 setup (its older outputs sit directly under `wp1/`). Run names: `wp1_si_<variant>_e<EPOCHS>_seed<SEED>`, `wp1_conditional_mean_seed<SEED>`. On 64×64, from the worktree root: `sbatch slurm_wp1/train_regressor.sbatch`, then `MODE=residuals REGRESSOR=checkpoints/stochastic_navier_stokes_64/wp1_conditional_mean_seed0 sbatch --dependency=afterok:<reg> slurm_wp1/estimate_spectrum.sbatch`, then `VARIANT=coloured_matched SPECTRUM=.../wp1/stochastic_navier_stokes_64/spectra/residuals_kappa1.npz sbatch --dependency=afterok:<spec> slurm_wp1/train_si.sbatch` (same for `coloured_t`; `white_t2` needs no spectrum), then `RUN=<run> sbatch --dependency=afterok:<si> slurm_wp1/eval_si.sbatch`. Submit from the worktree root.
