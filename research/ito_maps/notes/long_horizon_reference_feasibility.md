# Long-horizon truth and physical-reference feasibility

Follow-up audit, 2026-10-07. This extends the earlier `navier_stokes_provenance.md` audit; it does not modify existing data or trained models.

## Recovered timestamp and resolution provenance

The companion `data/stochastic_navier_stokes/data_file.pt` contains a two-element list: float32 vorticity `[200,100,256,256]` and a float32 vector of 100 stored times. A complete comparison verifies that **every value** of `data.npz['state']` equals `full_states[:,:,::2,::2]`, with maximum absolute difference zero. The model archive is spatial subsampling, not average pooling. The full-resolution archive and model archive are SHA-256 hashed in `results/long_rollout/data_reference_audit.json`.

Stored times run from 49.9899978638 to 99.4800949097. Their mean spacing is 0.4998999701605903; float32 adjacent differences range from 0.4998931885 to 0.4999084473. The sensible nominal spacing is **0.4999 stored time units**. This is compatible with 4,999 solver substeps of size 1e-4, including the current generation script's floating-point floor-division behavior, but does not identify the actual generating revision, viscosity, drag, noise amplitude, or absolute time-origin convention. The newly recovered timestamps justify an approximate secondary time axis; they do not resolve all physical provenance.

With history length 5, the initial state is snapshot index 4. The archive supports exactly **95 subsequent verified truth fields**, ending at snapshot index 99. The separate `external_libs/jax_cfd_lib/trajectory.npz` contains only 100 fields at 256² and supplies neither a longer matched continuation nor the missing parameter manifest.

For plots, retain forecast lead as the primary axis. A secondary axis can show nominal elapsed stored time, `lead × 0.4999`; mark it as extrapolated past lead 95. At 1,024 and 4,096 leads this is approximately 511.8976 and 2,047.5904 time units. Forecast error, skill, spread–skill, and truth-based calibration stop at lead 95. Ensemble spread, generated enstrophy, and numerical diagnostics can continue beyond it.

## Descriptive held-out reference distribution

`results/long_horizon/reference_enstrophy.json` pools all 20 held-out trajectories × 100 stored snapshots. Enstrophy is `0.5 * mean_grid(omega**2)` in raw vorticity units. Its pooled mean is 4.6653943657, median 4.5642520786, and 5%/95% quantiles 3.1516720961 / 6.5265774403. The artifact also contains the 20×100 values and timestamps.

This is a descriptive historical range, **not** future truth, a predictive interval, a conditional reference ensemble, or proof that the data are stationary. The held-out trajectories share the dataset's generation procedure and are not independent samples at every snapshot. Do not turn the pooled 2,000 values into an independence-based confidence interval.

## Current-simulator CPU feasibility

`benchmark_assumed_ns_reference.py` benchmarked the current repository Euler–Maruyama solver on CPU only, four-core affinity, initialized from held-out trajectory 180 at snapshot 4. Parameters were explicitly set to viscosity 1e-3, drag 0.1, forcing amplitude 1, timestep 1e-4. These are **assumed diagnostic parameters**, not recovered data-generation parameters. Each timing trial used 256 substeps. All tested fields remained finite.

| Grid | Median projected seconds per 4,999-substep interval | One path, 1,024 leads | One path, 4,096 leads |
|---|---:|---:|---:|
| 128² | 1.13 | 19 minutes | 1.29 hours |
| 256² | 6.28 | 1.79 hours | 7.15 hours |

These are linear runtime projections from short CPU benchmarks, not completed long simulations or established long-term stability. Eight 256² members would require roughly 14–57 serial CPU-hours for 1,024–4,096 leads; vectorization/hardware may change costs. A 128² PDE run changes the numerical dynamics and cannot replace the archived 256² generator merely because it is faster.

Root decision: use the verified 95-lead comparison followed by a 4,096-lead model stress study. **No long assumed-parameter PDE continuation was launched.** A reproducible helper is available for future, separately labelled diagnostics once the user supplies or independently verifies the missing physical parameters.

## Future diagnostic helper and RNG handling

`simulate_assumed_ns_reference.py` requires explicit viscosity, drag, and forcing amplitude. It labels all output `is_verified_dataset_continuation: false`, forces the CPU backend, starts from an archived full-resolution state, keeps raw vorticity, and stores every-lead diagnostics plus sparse 128² output snapshots. Existing data cannot be overwritten. It does not claim original-forcing replay or a verified forecast target.

The stochastic PDE is Markov in the complete vorticity state. Independent future forcing is generated with a separate key for each member and physical lead, using nested `fold_in` operations; the existing solver splits each interval into its substeps. The number of requested future leads does not change earlier forcing draws. Initial conditions are shared across reference members; future RNG streams differ. For paired discretization/convergence studies, this helper's independent seed reuse across different substep counts is insufficient: one must explicitly aggregate the same underlying Wiener increments onto both meshes.

Reproduction:

```bash
.venv/bin/python research/ito_maps/experiments/audit_ns_time_reference.py
.venv/bin/python research/ito_maps/experiments/benchmark_assumed_ns_reference.py
# Tiny diagnostic execution check only, not a matched physical reference:
.venv/bin/python research/ito_maps/experiments/simulate_assumed_ns_reference.py \
  --output /tmp/ns_assumed_diagnostic_check --viscosity .001 --drag .1 \
  --forcing-scale 1 --inner-steps 2 --horizon 2 --members 2 --grid 128 --save-every 1
```

## Independent production-unit audit

`audit_long_horizon_units.py` independently recomputed every saved raw FlowMap snapshot available in the committed 2,064-lead production snapshot. It uses NumPy and the frozen affine scale, rather than calling the runner's metric functions. All raw-file hashes verified. Mean/member enstrophy, enstrophy quantiles, enstrophy of the mean field, unbiased RMS spread, paired-truth RMSE, and finite-ensemble-corrected spread–skill agreed to at most 1.8e-15. The independent identity `mean_member_enstrophy = enstrophy_of_mean + 0.5*(E-1)/E * RMS_spread**2` also passed. Truth index alignment is `origin + lead - 1`; truth-dependent fields are null after lead 95. The production timestamp-audit hash remains unchanged. Evidence is stored in `results/long_horizon/independent_flow_unit_audit.json`.

The tiny assumed-parameter PDE helper check also verified bitwise-identical common prefixes for requested horizons 2 and 3, identical initial conditions across members, distinct future forcing draws, and finite outputs. These are implementation checks using only two numerical substeps per lead, not long-horizon physics validation; see `assumed_reference_helper_validation.json`.

### Completed-map audit

A separate final audit, `results/long_horizon/independent_completed_map_audit.json`, covers both completed 4,096-lead maps: 8,192 scalar rows and all 700 saved raw files. Independent physical-metric differences are at most 1.8e-15. Data/model/config/time-audit hashes, input histories and truth ordering, journals, cumulative timing/NFE, and the final plus lead-1,024 restart states are consistent. Restart-history endpoints exactly equal the corresponding saved normalized states. CPU Torch/NumPy/Python RNG payloads restore into local generators, and CUDA RNG payload structures are valid; no GPU RNG replay or model inference was performed. The earlier 2,064-lead FlowMap audit remains unchanged as a historical artifact.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  research/ito_maps/experiments/audit_long_horizon_units.py \
  --run-dirs research/ito_maps/results/long_horizon/flow_map \
             research/ito_maps/results/long_horizon/follmer_ito_map \
  --output /tmp/ns_completed_map_audit.json
```

The final audit accepts multiple completed, finite NS runs and requires a new output filename, protecting prior audit evidence. It independently rescores saved raw fields and checks scalar identities at the remaining leads; it cannot reconstruct unsaved full fields from scalar summaries.
