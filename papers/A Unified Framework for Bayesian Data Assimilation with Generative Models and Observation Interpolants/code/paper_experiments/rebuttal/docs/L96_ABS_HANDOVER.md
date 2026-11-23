# Lorenz 96 with h(x) = |x|: results handover (2026-11-18, updated 2026-11-19)

This note covers the new Lorenz 96 experiment with absolute-value observations. None of it is in the manuscript yet.

## Setup

- **System:** the same Lorenz 96 system as the linear and arctan cases:
  - 16 sites, with every 4th site observed;
  - observation noise N(0, 4 I);
  - 50 assimilation steps.
- **Observation operator:** h(x) = |x|, elementwise and exact (not smoothed).
- **Why this case:** the sign of an observed site is not observed, so its posterior can be bimodal. The test is whether the samplers place mass in both modes, with the right weights.
- **Reference:** a particle filter (PF) with the exact optimal proposal (`abs_optimal`).
  - For one site, the posterior is a mixture of two truncated Gaussians, one on each half-line.
  - Both signs are drawn with their exact masses, and the filter is fully adapted.
  - The two PF replicates agree to about 0.01 in P(x>0).
  - The "PF reference" row in the tables below is replicate 1 scored against replicate 0. It shows the sampling noise of the reference itself.
- **Methods:** settings are taken unchanged from the arctan picks; nothing was tuned on |x|. Each sampler is run in four variants:
  - Jac-free;
  - Jac-free + corr.;
  - shared (`likelihood_mode=inflated_shared`, `jacobian_damping=0.9`);
  - shared + corr. (`q_source=online`; correction gain 1.0 for SI, 0.75 for DM).
  - The baseline is an EnKF with inflation 1.0 and localisation radius 4.
- **Metrics:** sliced W2, kNN-KL and energy distance to the PF samples, averaged over the 50 steps (lower is better). "% vs EnKF" is (EnKF − method) / EnKF; positive means better than the EnKF.
- **Scope:** section 0 covers all methods on 5 test trajectories and is the main result. Sections 1–3 are earlier results on 1 or 3 trajectories.
- **Hardware:** everything ran on the CPU of squamish, because the RTX 3090 gives CUDA launch failures.

## 0. Main result: all methods, 5 test trajectories (0–4), E = 64 (added 2026-11-19)

This sweep supersedes the 1- and 3-trajectory numbers in sections 1–3, which are kept as earlier, preliminary results.

- **Lineup:** every Lorenz 96 method from the manuscript (Table `tab:ms_l96`) except TMPD, which the manuscript runs only for linear observations.
  - Generative cells run at M = 25, 50, 100, 250.
  - All settings are the arctan picks, unchanged; nothing was tuned on |x|.
- **PF reference:** extended to sequences 3–4, with the same `abs_optimal` proposal, 2 replicates and N = 4e6.
- **Runs:** 93 cells, all on CPU, with no failures (2026-11-18 22:31 to 2026-11-19 08:54).
- **Metrics:** time-averaged over the 50 steps, then mean ± sd over the 5 sequences. The % is the improvement over the EnKF (E=64); positive means better.

### Summary

- **shared + corr. is the only variant that beats the EnKF on all three metrics, for every sampler and every M.**
  - Sliced W2 improves by +3 to +6%, kNN-KL by +8 to +16%, and energy distance by +3 to +12%.
  - FM-ODE shared + corr. is the best: about +5% / +13% / +11%.
- **Jac-free + corr. clearly beats the EnKF only on kNN-KL** (+10 to +23%). It is level with the EnKF on sliced W2 and mixed on energy distance (−9 to +6%).
- **SI prior + EnKF and FM prior + EnKF are level with the EnKF** (within about ±10%).
- **Our variants without the correction, and FIG, FlowDAS and SDA, are clearly worse than the EnKF:** sliced W2 is 22–55% worse and energy distance 67–186% worse.
- **Two groups look suspicious and should be checked before they are used:**
  - **Every SURGE variant is extremely poor.** This covers ours, FlowDAS + SURGE and SDA + SURGE (sliced W2 113–242% worse than the EnKF). The reweighting may not handle the sign-blind |x| likelihood, or there may be a bug; this has not been checked.
  - **D-Flow SGLD gets worse as M increases** (sliced W2 is 46% worse than the EnKF at M=25 and 252% worse at M=250). The SGLD step settings carried over from arctan probably do not suit |x|.
- **The gains are smaller than on trajectory 0 alone.** There, shared + corr. reached up to +22%. The spread over sequences is about ±0.05–0.13 in sliced W2, so the sliced W2 gains are within one sd; the kNN-KL and energy-distance gains are clearer.
- **M has almost no effect on our methods.**
- **No generative method reaches the EnKF with E=2000** (+18% / +34% / +12% over the EnKF with E=64).

### Full table

| Method | Variant | M | n | Sliced W2 | kNN-KL | Energy dist. |
|---|---|---|---|---|---|---|
| Ours SI-SDE | shared + corr. | 25 | 5 | 1.25 ± 0.13 (+3%) | 3.46 ± 0.25 (+16%) | 0.952 ± 0.26 (+3%) |
| Ours SI-SDE | shared + corr. | 50 | 5 | 1.23 ± 0.13 (+5%) | 3.57 ± 0.3 (+13%) | 0.919 ± 0.29 (+6%) |
| Ours SI-SDE | shared + corr. | 100 | 5 | 1.24 ± 0.055 (+4%) | 3.73 ± 0.19 (+10%) | 0.891 ± 0.12 (+9%) |
| Ours SI-SDE | shared + corr. | 250 | 5 | 1.25 ± 0.076 (+3%) | 3.8 ± 0.23 (+8%) | 0.95 ± 0.18 (+3%) |
| Ours SI-SDE | shared | 25 | 5 | 1.71 ± 0.081 (-32%) | 4.59 ± 0.18 (-11%) | 2.07 ± 0.24 (-111%) |
| Ours SI-SDE | shared | 50 | 5 | 1.71 ± 0.067 (-32%) | 4.77 ± 0.14 (-15%) | 2.03 ± 0.21 (-107%) |
| Ours SI-SDE | shared | 100 | 5 | 1.72 ± 0.064 (-33%) | 4.9 ± 0.14 (-19%) | 2.06 ± 0.17 (-109%) |
| Ours SI-SDE | shared | 250 | 5 | 1.77 ± 0.074 (-37%) | 5.04 ± 0.19 (-22%) | 2.19 ± 0.21 (-123%) |
| Ours SI-SDE | Jac-free + corr. | 25 | 5 | 1.3 ± 0.078 (-0%) | 3.19 ± 0.19 (+23%) | 1.01 ± 0.19 (-3%) |
| Ours SI-SDE | Jac-free + corr. | 50 | 5 | 1.31 ± 0.043 (-1%) | 3.39 ± 0.16 (+18%) | 1.05 ± 0.072 (-7%) |
| Ours SI-SDE | Jac-free + corr. | 100 | 5 | 1.29 ± 0.075 (+0%) | 3.54 ± 0.24 (+14%) | 1.01 ± 0.2 (-3%) |
| Ours SI-SDE | Jac-free + corr. | 250 | 5 | 1.29 ± 0.032 (+0%) | 3.53 ± 0.16 (+14%) | 0.992 ± 0.088 (-1%) |
| Ours SI-SDE | Jac-free | 25 | 5 | 1.61 ± 0.015 (-25%) | 3.91 ± 0.11 (+5%) | 1.77 ± 0.08 (-80%) |
| Ours SI-SDE | Jac-free | 50 | 5 | 1.58 ± 0.055 (-22%) | 3.94 ± 0.17 (+5%) | 1.67 ± 0.15 (-70%) |
| Ours SI-SDE | Jac-free | 100 | 5 | 1.59 ± 0.022 (-23%) | 4.04 ± 0.16 (+2%) | 1.66 ± 0.081 (-69%) |
| Ours SI-SDE | Jac-free | 250 | 5 | 1.58 ± 0.065 (-22%) | 4.22 ± 0.22 (-2%) | 1.64 ± 0.19 (-67%) |
| Ours DM-SDE | shared + corr. | 25 | 5 | 1.24 ± 0.078 (+4%) | 3.67 ± 0.19 (+11%) | 0.927 ± 0.17 (+6%) |
| Ours DM-SDE | shared + corr. | 50 | 5 | 1.24 ± 0.099 (+4%) | 3.74 ± 0.22 (+9%) | 0.912 ± 0.18 (+7%) |
| Ours DM-SDE | shared + corr. | 100 | 5 | 1.25 ± 0.075 (+4%) | 3.69 ± 0.11 (+11%) | 0.928 ± 0.19 (+6%) |
| Ours DM-SDE | shared + corr. | 250 | 5 | 1.24 ± 0.088 (+4%) | 3.77 ± 0.26 (+9%) | 0.909 ± 0.18 (+8%) |
| Ours DM-SDE | shared | 25 | 5 | 1.68 ± 0.038 (-30%) | 4.79 ± 0.18 (-16%) | 1.95 ± 0.11 (-99%) |
| Ours DM-SDE | shared | 50 | 5 | 1.71 ± 0.067 (-32%) | 4.85 ± 0.23 (-17%) | 2.01 ± 0.18 (-104%) |
| Ours DM-SDE | shared | 100 | 5 | 1.73 ± 0.055 (-33%) | 4.92 ± 0.18 (-19%) | 2.06 ± 0.16 (-110%) |
| Ours DM-SDE | shared | 250 | 5 | 1.72 ± 0.074 (-33%) | 4.85 ± 0.22 (-18%) | 2.04 ± 0.2 (-107%) |
| Ours DM-SDE | Jac-free + corr. | 25 | 5 | 1.26 ± 0.06 (+3%) | 3.6 ± 0.18 (+13%) | 0.93 ± 0.12 (+5%) |
| Ours DM-SDE | Jac-free + corr. | 50 | 5 | 1.26 ± 0.088 (+2%) | 3.72 ± 0.19 (+10%) | 0.928 ± 0.19 (+6%) |
| Ours DM-SDE | Jac-free + corr. | 100 | 5 | 1.26 ± 0.083 (+2%) | 3.68 ± 0.14 (+11%) | 0.942 ± 0.15 (+4%) |
| Ours DM-SDE | Jac-free + corr. | 250 | 5 | 1.27 ± 0.052 (+2%) | 3.71 ± 0.11 (+10%) | 0.951 ± 0.13 (+3%) |
| Ours DM-SDE | Jac-free | 25 | 5 | 1.64 ± 0.035 (-27%) | 4.33 ± 0.11 (-5%) | 1.78 ± 0.081 (-81%) |
| Ours DM-SDE | Jac-free | 50 | 5 | 1.63 ± 0.063 (-26%) | 4.27 ± 0.19 (-4%) | 1.77 ± 0.14 (-80%) |
| Ours DM-SDE | Jac-free | 100 | 5 | 1.63 ± 0.051 (-26%) | 4.32 ± 0.14 (-5%) | 1.78 ± 0.17 (-81%) |
| Ours DM-SDE | Jac-free | 250 | 5 | 1.61 ± 0.029 (-25%) | 4.29 ± 0.1 (-4%) | 1.75 ± 0.06 (-78%) |
| Ours FM-ODE | shared + corr. | 25 | 5 | 1.22 ± 0.062 (+5%) | 3.52 ± 0.17 (+15%) | 0.875 ± 0.11 (+11%) |
| Ours FM-ODE | shared + corr. | 50 | 5 | 1.21 ± 0.06 (+6%) | 3.59 ± 0.14 (+13%) | 0.863 ± 0.14 (+12%) |
| Ours FM-ODE | shared + corr. | 100 | 5 | 1.23 ± 0.074 (+5%) | 3.67 ± 0.21 (+11%) | 0.877 ± 0.17 (+11%) |
| Ours FM-ODE | shared + corr. | 250 | 5 | 1.23 ± 0.05 (+5%) | 3.67 ± 0.18 (+11%) | 0.885 ± 0.1 (+10%) |
| Ours FM-ODE | shared | 25 | 5 | 1.66 ± 0.05 (-28%) | 4.63 ± 0.12 (-12%) | 1.91 ± 0.14 (-94%) |
| Ours FM-ODE | shared | 50 | 5 | 1.68 ± 0.035 (-30%) | 4.7 ± 0.15 (-14%) | 1.94 ± 0.11 (-97%) |
| Ours FM-ODE | shared | 100 | 5 | 1.68 ± 0.044 (-30%) | 4.71 ± 0.094 (-14%) | 1.93 ± 0.13 (-97%) |
| Ours FM-ODE | shared | 250 | 5 | 1.68 ± 0.024 (-30%) | 4.7 ± 0.088 (-14%) | 1.92 ± 0.068 (-96%) |
| Ours FM-ODE | Jac-free + corr. | 25 | 5 | 1.32 ± 0.048 (-2%) | 3.49 ± 0.13 (+16%) | 1.06 ± 0.12 (-8%) |
| Ours FM-ODE | Jac-free + corr. | 50 | 5 | 1.33 ± 0.064 (-3%) | 3.57 ± 0.16 (+14%) | 1.07 ± 0.13 (-9%) |
| Ours FM-ODE | Jac-free + corr. | 100 | 5 | 1.31 ± 0.048 (-1%) | 3.57 ± 0.13 (+14%) | 1.01 ± 0.096 (-3%) |
| Ours FM-ODE | Jac-free + corr. | 250 | 5 | 1.29 ± 0.049 (+0%) | 3.57 ± 0.1 (+13%) | 0.983 ± 0.089 (-0%) |
| Ours FM-ODE | Jac-free | 25 | 5 | 1.62 ± 0.042 (-25%) | 4.08 ± 0.13 (+1%) | 1.72 ± 0.12 (-75%) |
| Ours FM-ODE | Jac-free | 50 | 5 | 1.64 ± 0.036 (-27%) | 4.21 ± 0.14 (-2%) | 1.78 ± 0.11 (-81%) |
| Ours FM-ODE | Jac-free | 100 | 5 | 1.59 ± 0.067 (-23%) | 4.17 ± 0.18 (-1%) | 1.64 ± 0.16 (-67%) |
| Ours FM-ODE | Jac-free | 250 | 5 | 1.61 ± 0.046 (-24%) | 4.22 ± 0.16 (-2%) | 1.7 ± 0.13 (-73%) |
| Ours SI-SDE + SURGE |  | 25 | 5 | 4.42 ± 0.09 (-242%) | 11.1 ± 0.25 (-170%) | 16.6 ± 0.57 (-1587%) |
| Ours SI-SDE + SURGE |  | 50 | 5 | 4.31 ± 0.1 (-233%) | 10.9 ± 0.46 (-164%) | 15.8 ± 0.79 (-1511%) |
| Ours SI-SDE + SURGE |  | 100 | 5 | 4.16 ± 0.12 (-222%) | 10.5 ± 0.54 (-154%) | 14.6 ± 0.95 (-1382%) |
| Ours SI-SDE + SURGE |  | 250 | 5 | 4.31 ± 0.27 (-233%) | 11 ± 0.76 (-167%) | 15.7 ± 1.7 (-1496%) |
| Ours DM-SDE + SURGE |  | 25 | 5 | 4.24 ± 0.18 (-228%) | 10.6 ± 0.63 (-157%) | 15.2 ± 1.1 (-1449%) |
| Ours DM-SDE + SURGE |  | 50 | 5 | 4.22 ± 0.29 (-226%) | 10.5 ± 1.1 (-153%) | 15 ± 2 (-1430%) |
| Ours DM-SDE + SURGE |  | 100 | 5 | 4.37 ± 0.25 (-238%) | 10.9 ± 0.99 (-165%) | 16.1 ± 1.7 (-1541%) |
| Ours DM-SDE + SURGE |  | 250 | 5 | 4.29 ± 0.2 (-232%) | 10.8 ± 0.56 (-161%) | 15.7 ± 1.3 (-1496%) |
| FlowDAS |  | 25 | 5 | 1.64 ± 0.085 (-27%) | 3.83 ± 0.23 (+7%) | 1.87 ± 0.25 (-90%) |
| FlowDAS |  | 50 | 5 | 1.67 ± 0.092 (-29%) | 4.08 ± 0.21 (+1%) | 1.96 ± 0.27 (-99%) |
| FlowDAS |  | 100 | 5 | 1.65 ± 0.059 (-28%) | 4.24 ± 0.28 (-3%) | 1.92 ± 0.22 (-95%) |
| FlowDAS |  | 250 | 5 | 1.67 ± 0.07 (-29%) | 4.37 ± 0.38 (-6%) | 1.94 ± 0.24 (-98%) |
| FlowDAS + SURGE |  | 25 | 5 | 2.76 ± 0.13 (-113%) | 6.69 ± 0.27 (-62%) | 6.31 ± 0.63 (-542%) |
| FlowDAS + SURGE |  | 50 | 5 | 2.92 ± 0.18 (-126%) | 7.36 ± 0.45 (-78%) | 7.26 ± 0.77 (-639%) |
| FlowDAS + SURGE |  | 100 | 5 | 3.06 ± 0.086 (-137%) | 7.53 ± 0.51 (-82%) | 7.73 ± 0.48 (-687%) |
| FlowDAS + SURGE |  | 250 | 5 | 3.58 ± 0.21 (-177%) | 8.98 ± 0.47 (-117%) | 10.8 ± 1.2 (-1002%) |
| SDA |  | 25 | 5 | 1.95 ± 0.028 (-51%) | 5.87 ± 0.2 (-42%) | 2.68 ± 0.13 (-173%) |
| SDA |  | 50 | 5 | 2 ± 0.068 (-54%) | 6.1 ± 0.22 (-48%) | 2.8 ± 0.21 (-185%) |
| SDA |  | 100 | 5 | 2 ± 0.061 (-55%) | 6.14 ± 0.19 (-49%) | 2.81 ± 0.2 (-186%) |
| SDA |  | 250 | 5 | 1.99 ± 0.05 (-54%) | 6.09 ± 0.18 (-48%) | 2.78 ± 0.15 (-182%) |
| SDA + SURGE |  | 25 | 5 | 3.09 ± 0.19 (-139%) | 7.69 ± 0.46 (-86%) | 8.02 ± 1.1 (-716%) |
| SDA + SURGE |  | 50 | 5 | 3.06 ± 0.17 (-137%) | 7.64 ± 0.63 (-85%) | 7.9 ± 0.86 (-704%) |
| SDA + SURGE |  | 100 | 5 | 3.03 ± 0.26 (-135%) | 7.39 ± 0.8 (-79%) | 7.61 ± 1.4 (-674%) |
| SDA + SURGE |  | 250 | 5 | 3.11 ± 0.26 (-140%) | 8.05 ± 0.79 (-95%) | 8.05 ± 1.3 (-719%) |
| D-Flow SGLD |  | 25 | 5 | 1.89 ± 0.058 (-46%) | 5.52 ± 0.12 (-34%) | 2.25 ± 0.13 (-129%) |
| D-Flow SGLD |  | 50 | 5 | 2.1 ± 0.05 (-62%) | 6.14 ± 0.3 (-49%) | 2.48 ± 0.11 (-152%) |
| D-Flow SGLD |  | 100 | 5 | 2.88 ± 0.071 (-122%) | 8.07 ± 0.27 (-96%) | 3.65 ± 0.19 (-271%) |
| D-Flow SGLD |  | 250 | 5 | 4.55 ± 0.083 (-252%) | 10.7 ± 0.35 (-158%) | 6.44 ± 0.26 (-556%) |
| Guided FM (FIG) |  | 25 | 5 | 1.66 ± 0.068 (-28%) | 3.79 ± 0.15 (+8%) | 1.76 ± 0.2 (-79%) |
| Guided FM (FIG) |  | 50 | 5 | 1.72 ± 0.052 (-33%) | 4.02 ± 0.16 (+3%) | 1.85 ± 0.15 (-88%) |
| Guided FM (FIG) |  | 100 | 5 | 1.74 ± 0.055 (-35%) | 4.12 ± 0.11 (+0%) | 1.9 ± 0.14 (-93%) |
| Guided FM (FIG) |  | 250 | 5 | 1.76 ± 0.065 (-36%) | 4.15 ± 0.088 (-1%) | 1.93 ± 0.15 (-96%) |
| SI prior + EnKF |  | 25 | 5 | 1.31 ± 0.11 (-1%) | 3.75 ± 0.27 (+9%) | 1.06 ± 0.21 (-8%) |
| SI prior + EnKF |  | 50 | 5 | 1.28 ± 0.074 (+1%) | 3.87 ± 0.19 (+6%) | 1 ± 0.14 (-2%) |
| SI prior + EnKF |  | 100 | 5 | 1.29 ± 0.061 (+0%) | 4.01 ± 0.23 (+3%) | 1.02 ± 0.15 (-3%) |
| SI prior + EnKF |  | 250 | 5 | 1.33 ± 0.1 (-3%) | 4.14 ± 0.3 (-0%) | 1.1 ± 0.2 (-12%) |
| FM prior + EnKF |  | 25 | 5 | 1.26 ± 0.056 (+3%) | 3.96 ± 0.16 (+4%) | 0.937 ± 0.087 (+5%) |
| FM prior + EnKF |  | 50 | 5 | 1.3 ± 0.071 (-0%) | 4.1 ± 0.2 (+1%) | 1.03 ± 0.15 (-5%) |
| FM prior + EnKF |  | 100 | 5 | 1.32 ± 0.065 (-2%) | 4.09 ± 0.19 (+1%) | 1.07 ± 0.17 (-9%) |
| FM prior + EnKF |  | 250 | 5 | 1.3 ± 0.036 (-0%) | 4.12 ± 0.17 (+0%) | 1.01 ± 0.076 (-3%) |
| EnKF (E=64) |  | – | 5 | 1.29 ± 0.032 (+0%) | 4.13 ± 0.19 (+0%) | 0.982 ± 0.082 (+0%) |
| EnKF (E=2000) |  | – | 5 | 1.06 ± 0.046 (+18%) | 2.74 ± 0.17 (+34%) | 0.862 ± 0.083 (+12%) |
| PF (E=64) |  | – | 5 | 2.25 ± 0.19 (-74%) | 5.51 ± 0.42 (-33%) | 4.15 ± 0.86 (-323%) |
| Sampling floor (PF, 64 draws) |  | – | 5 | 0.706 ± 0.011 (+45%) | 2.28 ± 0.047 (+45%) | 0.0158 ± 0.0082 (+98%) |
| PF reference (2nd replicate) |  | – | 5 | 0.135 ± 0.0093 (+90%) | 0.0442 ± 0.018 (+99%) | 0.00197 ± 0.0031 (+100%) |

**Files.** Run root: `$PE/results/_runs/abs5/` (`$PE` is defined under "Files and code"). The table is in `abs5/queue/summary.{md,csv}`, regenerated by `abs5/queue/summary.py`. The queue scripts are `abs5/queue/{jobs.txt,lane.sh,run_all.sh}`, with per-cell logs in `abs5/queue/logs/`. The ensembles are in `abs5/lorenz96/abs/ensembles/`, with shape [5, 50, 64, 16]. The config change is `evaluation_abs: {test_sequences: 5}`.

No figures have been made for the 5-trajectory runs yet.

## 1. Main sweep: trajectory 0, E = 64, all samplers × variants × M

EnKF (E=64), time-averaged: sliced W2 **1.26**, kNN-KL **3.95**, energy distance **0.927**.

Each cell gives the % improvement over the EnKF for sliced W2 / kNN-KL / energy distance.

| Sampler | Variant | M=25 | M=50 | M=100 | M=250 |
|---|---|---|---|---|---|
| SI-SDE | shared + corr. | +7 / +18 / +14 | +8 / +14 / +22 | +5 / +10 / +6 | +8 / +9 / +16 |
| SI-SDE | Jac-free + corr. | +2 / +22 / −1 | −1 / +15 / −8 | +4 / +18 / +7 | −1 / +14 / −8 |
| SI-SDE | shared | −30 / −12 / −103 | −32 / −16 / −113 | −31 / −19 / −105 | −30 / −19 / −98 |
| SI-SDE | Jac-free | −28 / +2 / −92 | −24 / 0 / −81 | −24 / −3 / −78 | −31 / −7 / −100 |
| DM-SDE | shared + corr. | +5 / +12 / +8 | +5 / +10 / +11 | +3 / +9 / +5 | +6 / +8 / +13 |
| DM-SDE | Jac-free + corr. | −1 / +8 / −11 | 0 / +11 / −2 | 0 / +11 / −4 | +4 / +11 / +9 |
| DM-SDE | shared | −29 / −14 / −97 | −27 / −18 / −88 | −35 / −22 / −120 | −30 / −18 / −97 |
| DM-SDE | Jac-free | −23 / −4 / −69 | −26 / −7 / −79 | −29 / −9 / −91 | −27 / −8 / −83 |
| FM-ODE | shared + corr. | +9 / +12 / +20 | +9 / +12 / +20 | +6 / +8 / +14 | +1 / +4 / −2 |
| FM-ODE | Jac-free + corr. | +4 / +16 / +7 | −2 / +11 / −13 | −1 / +8 / −11 | −2 / +10 / −7 |
| FM-ODE | shared | −31 / −16 / −103 | −33 / −15 / −105 | −34 / −23 / −116 | −35 / −21 / −117 |
| FM-ODE | Jac-free | −28 / 0 / −86 | −27 / −2 / −81 | −29 / −6 / −86 | −28 / −7 / −82 |

The absolute values are in `summary.md` and `summary.csv`. Their location is listed under "Files and code".

What the table shows:

- **shared + corr. beats the EnKF on all three metrics for every sampler and every M**, with one exception: FM-ODE at M=250 is 2% worse on energy distance.
  - The largest gains are SI-SDE at M=50 (energy distance +22%) and FM-ODE at M=25–50 (+9% sliced W2, +20% energy distance).
- **Jac-free + corr. beats the EnKF on kNN-KL in every cell** (+8 to +22%). On sliced W2 and energy distance it is roughly level with the EnKF.
- **The variants without the correction are clearly worse than the EnKF:** sliced W2 is 23–35% worse and energy distance 70–120% worse. The correction step is what makes the samplers competitive on |x|.
- **Increasing M gives no consistent gain.** M=25–50 is as good as M=250 or better, which suggests the remaining error comes from the likelihood approximation, not from discretising the sampler.
- **The three samplers behave similarly.** With shared + corr., SI-SDE and FM-ODE are slightly ahead of DM-SDE.

### Final step and sign mass, SI-SDE, trajectory 0, E = 64

- **Mode-mass error:** |ΔP(x>0)| is the gap between the method's and the PF's P(x>0), averaged over the 4 observed sites.
- **Bimodal sites:** at step 50, three of the four observed sites have PF mass of at least 0.2 on each sign.
- **Missed modes:** no method missed a mode, meaning no method put fewer than 5% of its members on one side of 0 at a bimodal site.

| M | Method | sliced W2 (step 50) | kNN-KL (step 50) | energy distance (step 50) | mode-mass error, time-avg. | mode-mass error, bimodal sites (step 50) |
|---|---|---|---|---|---|---|
| – | PF reference | 0.126 | −0.056 | ≈0 | 0.007 | 0.011 |
| – | EnKF | 0.915 | 3.60 | 0.371 | 0.096 | 0.047 |
| 25 | shared + corr. | 1.02 | 2.23 | 0.374 | 0.096 | 0.107 |
| 50 | shared + corr. | 0.836 | 2.67 | 0.211 | 0.095 | 0.101 |
| 100 | shared + corr. | 0.888 | 2.44 | 0.292 | 0.112 | 0.067 |
| 250 | shared + corr. | 0.778 | 2.47 | 0.124 | 0.101 | 0.107 |
| 100 | shared | 1.10 | 3.68 | 0.744 | 0.127 | 0.057 |
| 100 | Jac-free | 1.06 | 3.41 | 0.641 | 0.134 | 0.067 |

At the last step, shared + corr. is better than the EnKF on kNN-KL at every M, and on energy distance at M = 50–250. Over time, its mode-mass error is the same as the EnKF's (about 0.10). At the bimodal sites at step 50, the EnKF has the smaller mode-mass error; with one trajectory, these per-site numbers are noisy.

## 2. Larger ensemble: trajectory 0, E = 256, M = 100 (SI-SDE)

All values are time-averaged. The % compares against the EnKF with E = 256.

| Method | sliced W2 | kNN-KL | energy distance |
|---|---|---|---|
| EnKF (E=256) | 1.08 | 3.32 | 0.863 |
| SI-SDE shared + corr. | 1.03 (+5%) | 2.99 (+10%) | 0.839 (+3%) |
| SI-SDE Jac-free | 1.49 (−38%) | 3.23 (+3%) | 1.73 (−100%) |
| SI-SDE shared | 1.55 (−44%) | 3.89 (−17%) | 1.90 (−120%) |

The ranking is the same as with E = 64.

## 3. Consistency check: 3 trajectories (0–2), E = 64, M = 100 (SI-SDE)

Values are time-averaged, mean ± standard deviation over the 3 trajectories. The % compares against the EnKF.

| Method | sliced W2 | kNN-KL | energy distance |
|---|---|---|---|
| PF reference (step 50) | 0.131 | 0.067 | ≈0 |
| EnKF | 1.29 ± 0.03 | 4.12 ± 0.24 | 0.999 ± 0.097 |
| SI-SDE shared + corr. | 1.26 ± 0.04 (+2%) | 3.74 ± 0.19 (+9%) | 0.945 ± 0.12 (+5%) |
| SI-SDE Jac-free | 1.58 ± 0.03 (−22%) | 4.00 ± 0.13 (+3%) | 1.64 ± 0.11 (−64%) |
| SI-SDE shared | 1.70 ± 0.07 (−32%) | 4.85 ± 0.17 (−18%) | 2.02 ± 0.19 (−102%) |

At step 50, 8 of the 12 observed sites (4 per trajectory) are bimodal in the PF. No method missed a mode on any of them. The ranking matches trajectory 0.

## Caveats and what is not done

- **Few trajectories.** Section 0 uses 5 of the 12 test sequences; sections 1–3 use 1 or 3. Sliced W2 differences of a few percent are within one sd.
- **No tuning on |x|.** All settings are the arctan picks.
- **Figures cover SI-SDE only, and only the 1- and 3-trajectory runs.** The final-step marginal and metrics-vs-step figures exist for SI-SDE. DM-SDE and FM-ODE have metrics, but no figures yet.
- **Legend label.** The marginal figure still labels the observation lines "±y/c"; for |x| this should read "±y".
- **Code not committed.** The |x| code is still uncommitted on the code branch `review/rebuttal-runs`. This covers:
  - the `abs_optimal` proposal;
  - `StandardisedAbsOperator`;
  - the config;
  - the tests;
  - the figure scripts.

## Files and code

The paper-experiments directory is `PE = papers/A Unified Framework for Bayesian Data Assimilation with Generative Models and Observation Interpolants/code/paper_experiments` on squamish, in the code repo. The results are not in git.

- **PF reference:** `$PE/results/lorenz96/abs/reference/test/sequence{0,1,2}_replicate{0,1}.pt`
- **Main sweep (E=64):**
  - run root: `$PE/results/_runs/abs1_quick/`;
  - table: `abs1_quick/overnight/summary.{md,csv}`;
  - queue scripts: `overnight/{jobs.txt,lane.sh,run_all.sh,summary.py}`.
- **E=256:** `$PE/results/_runs/abs1_quick_E256/`
- **3 trajectories:** `$PE/results/lorenz96/abs/`
- **Figures** (final-step marginals for 3 observed + 3 unobserved sites with the PF replicate, metrics vs step, metric tables): `$PE/results/figures/l96_abs/{quick1_M25,quick1_M50,quick1_M100,quick1_M250,quick1_E256_M100,seq3_M100}/`
- **Commands:** see `$PE/L96_SQUARE_HANDOVER.md`, section "Sibling experiment: h(x) = |x|". Use the `--obs square` commands with `--obs abs`.
