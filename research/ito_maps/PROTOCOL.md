# Experimental protocol: few-step Itô-map improvements on stochastic Navier–Stokes

Written 2026-11-23, before the first training run of the improvement work
(`notes/ideas/HANDOVER.md`, §6). Any later change is recorded at the bottom
with its date and reason; results produced before a change keep the old
protocol label.

## Data and splits

- Archive: `data/stochastic_navier_stokes/data.npz`, array `state`,
  200 trajectories × 100 snapshots × 128², float32.
- Whole trajectories: train 0–159, validation 160–179, test 180–199.
- History length 5. Dataset windows as in `StochasticNavierStokesDataset`
  (count `T − L − 1`, kept for comparability with all models).
- Normalisation: mean 0, std 3.09969 (from the config; the train-split std
  is 3.09824, a 0.05% difference that is left as is). Metrics are reported on
  denormalised fields; spectra and budgets in float64.

## Selection rule (fixed in advance)

- Everything (hyperparameters, checkpoint, K, κ, refresh time, …) is chosen
  on **validation**.
- Primary criterion: lowest validation fair CRPS of the one-step forecast at
  1 map step. Tie-break (differences inside the paired bootstrap 95% interval):
  RMS spread–skill ratio closer to 1, then extreme-rank fraction closer to
  2/(E+1).
- Checkpoint: the final checkpoint of the training budget (no early-stopping
  selection), so that one checkpoint serves all step budgets.
- The test split is opened once, at the end, with frozen configs and hashes.
  Anything informed by test is labelled exploratory.

## Evaluation cases

- Origins 5, 25, 45, 65, 85 per trajectory (as in
  `config/stochastic_navier_stokes.yaml`), all 20 validation trajectories.
- Ensemble size E = 16 for scores and calibration; E = 32 for spectra and
  per-wavenumber spread ratios.
- Step budgets: maps 1, 2, 4, 8 (one checkpoint for all budgets). SI solver
  16, 64, 128 steps, Euler–Maruyama and Euler drift with exact integrated
  noise.
- Rollouts: leads 1–10 from every origin with ≤ 85 (truth exists up to lead
  95 from origin 5); long runs of about 500 leads from origin 5 on a fixed
  subset of validation trajectories (160, 165, 170, 175), E = 8. Beyond the
  last lead with truth, only stationary statistics and the budget residual.
- Same-path comparisons share one Brownian path object explicitly.

## Metrics

- Fair CRPS and energy score (unbiased pair terms, not clamped).
- RMS spread with the √((E+1)/E) correction, RMS spread–skill ratio, rank
  histograms, extreme-rank fraction.
- Shell spectra of members (not of the ensemble mean), per-wavenumber spread
  ratio; conventions of `notes/key_findings.md` §4.
- Rollouts: the above per lead; pooled enstrophy, spectrum and lag-1
  autocorrelation for long runs; budget residual (WP7) per member and lead.
- Network calls and synchronised wall-clock latency at equal E and precision.

## Uncertainty

- Average origins within a trajectory, then bootstrap whole trajectories
  (2,000 resamples, one shared index matrix for all models and leads).
- Differences are paired on shared resamples.
- Headline comparisons use three training seeds (0, 1, 2); seed variation is
  reported separately from trajectory variation.

## Training

- Full precision for map losses; dropout 0 for map training (backbone and
  attention).
- Objectives are compared at equal updates; wall-clock is reported. Any
  simulation-based refinement counts towards cost.
- When the noise process changes (WP1), the SI baseline is retrained on the
  same process.

## Baselines (rerun under this protocol)

- Multi-step SI solver at 16, 64, 128 steps, with Euler–Maruyama and with exact
  integrated noise.
- Gaussian-base flow map (PSD).
- Current Itô map: white driver, β = t², PSD, local increments, K = 4,
  `gamma_multiplier` 1.0.

The original training-grid map checkpoints are not available on DelftBlue.
The baseline maps are retrained from the committed grid configs
(`results/training_grid/hydra/ns_grid_{ito_psd_local,flow_psd}`) with one
change: dropout 0 (the grid runs used the backbone's 0.1 for PSD). They are
re-creations, not the original runs.

## Changes

(none yet)
