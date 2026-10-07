# Mathematical implementation audit before final evaluation

Date: 2026-10-07. Scope: current `BaseMapModel`, `FollmerItoMapModel`, `FlowMapModel`, `MapTrainer`, Brownian path implementation, and manuscript methodology. No model changes made in this audit.

## Findings

No substantive mathematical mismatch was found in the core conditional-map construction for the unmasked stochastic Navier–Stokes experiments.

- **Itô diagonal:** `training_pair` reuses the original stochastic interpolation and its SDE drift target, including gamma-prime times sqrt(t) noise. With a pretrained teacher, the actual diagonal target is the teacher's frozen drift on these interpolant samples. This is correctly distinguished from raw-data regression in the manuscript.
- **Drift versus probability-flow velocity:** the stochastic diagonal is not the deterministic probability-flow velocity, and no extra derivative of sqrt(t) is inserted. The Gaussian-base image-generation target from the source Itô-map paper is not transplanted into this model.
- **Composition:** the first EMA map generates the intermediate state; the second uses this generated state and the same history. Both use one shared weighted-Brownian path. Average drifts are weighted by their interval lengths, and the complete EMA target is detached. EMA parameters are updated after student optimization; they remain frozen during target evaluation.
- **Path representation:** raw local increments sum to the explicit noise residual. The Gaussian bridge cache samples exact finite-query path laws; only the finite path conditioning of the drift integral is compressed. Child features may resolve details absent from the parent features, so finite-feature PSD generally has nonzero residual and no automatic exact semigroup guarantee. The manuscript states this limitation.
- **Deterministic baseline:** `FlowMapModel.training_pair` uses independent Gaussian noise as its source. At forecast time, the inherited `gaussian_base=True` causes a fresh Gaussian field to be drawn when no explicit base is supplied. History provides conditioning, not the stochastic source. Thus deterministic maps still generate ensembles, and no point-mass-to-nondegenerate deterministic transport is claimed.
- **Teacher rollout:** training integrates the frozen teacher drift by Euler steps and adds exact weighted-noise increments for the stochastic case. Subtracting the total noise from endpoint displacement gives its finite-step average-drift target. This is simulation-based and is labeled accordingly. The drift teacher, teacher grid, and numerical noise integration determine the distilled reference.
- **Sampling and validation:** validation uses the full *range* of interval lengths, not exclusively the full [0,1] endpoint. It uses the same endpoint probability as training but disables the interval curriculum and loss ramp. Fixed random draws and teacher trajectories support reproducible checkpoint selection. Validation loss is not a true-distribution score.
- **Föllmer terminology:** the implemented gamma(t)=a(1−t) diffusion follows the repository's default class, not the distinct KL-optimal corrected Föllmer diffusion. The manuscript explicitly distinguishes these processes. A trained map cannot change diffusion after training without learning the corresponding path-dependent drift transformation.

## Exact interval distribution to record

At curriculum maximum L:

1. Draw length ℓ uniformly on [0.01L,L].
2. Draw start s uniformly on [0,1−ℓ], then t=s+ℓ.
3. With configured endpoint probability replace (s,t) by (0,L).
4. Draw split fraction f uniformly on [0.25,0.75] and set u=s+f(t−s).

The curriculum grows L from 0.125 to 1 over 500 updates in the initial pilot. Stage two has L=1 and endpoint probability0.5. This is not uniform sampling of an ordered time triangle. Off-diagonal intervals have a positive minimum length; zero-length diagonal behavior is supervised separately. Population uniqueness statements requiring all intervals do not transfer automatically to this finite training distribution.

## Interpretation limits retained

Architecture, feature compression, numerical teacher error, diagonal approximation, optimization, and physical forecast error remain distinct. PSD-to-rollout improvement does not isolate the objective from extra compute. The conditional NS path experiment establishes a one-history finite-teacher information limit, not a dataset-wide guarantee. Additional map steps need not monotonically improve quality; the validation results already show nonmonotonic cases.

The main claims should be based on frozen methodology, all held-out trajectories, measured timing including Brownian construction, and multiple student seeds. Bootstrap uncertainty over test trajectories does not replace uncertainty over independently trained models. No further implementation changes are needed solely for these interpretive issues.
