# Navier–Stokes data provenance audit

Checked 2026-10-07. The supplied data must not be silently assigned every parameter appearing in either the original paper or the repository's current simulator.

## Verified local evidence

- `data/stochastic_navier_stokes/data.npz`: a ZIP containing only `state.npy`, no parameter/time metadata. Current compressed archive size is 1,208,781,160 bytes. Experiment inspection establishes 200 trajectories, 100 snapshots, 128×128 scalar vorticity fields.
- `config/stochastic_navier_stokes.yaml`: train trajectory indices `[0,160)`, validation `[160,180)`, test `[180,200)`; history length 5; base/target mean 0, standard deviation 3.09969; quadratic Brownian SI with gamma multiplier 1. These are configuration choices, not physical data-generation metadata.
- `external_libs/jax_cfd_lib/src/jax_cfd_lib/bin/simulate_stochastic_ns_main.py`: currently uses viscosity 1e-3, drag 0.1, periodic domain `[0,2π]^2`, **256×256** grid, high-frequency time step 1e-4, saved spacing 0.5, final time 50, five trajectories, initial max velocity 7, and seed 0. Saves only the first trajectory as `trajectory.npz`, key `trajectory`. This does **not** match the supplied archive shape, key, or filename, so it is not a sufficient generation manifest for the existing training data.
- `external_libs/jax_cfd_lib/src/jax_cfd_lib/navier_stokes_forward_model.py`: stochastic forcing has eight independent real Gaussian coefficients multiplying sine/cosine modes `(6,0)`, `(7,0)`, `(5,5)`, `(8,8)`; forcing scale default 1.0. The stochastic branch of `set_up_forward_model` chooses `forward_euler_maruyama`, despite the calling script's stale comment mentioning Crank–Nicolson RK4. The implementation has viscosity and drag in its explicit drift. Dealiasing applies a spectral brick-wall filter.

## Manuscript decision

State the verified array shape, split, normalization, and forecast interval in **stored snapshot units**. Describe the periodic stochastic-vorticity setting, but label current simulator parameters as corroborating context whose connection to this archive is unverified. Do not claim saved spacing 0.5, viscosity 1e-3, spinup duration, forcing strength, or an exact physical-solver comparison for the supplied archive until its generation manifest is recovered.

Potential recovery routes: original creation script/git revision, dataset download source, stored notebooks/logs, or author-provided provenance. No user interruption is necessary to run the authorized paired-data comparisons; missing generation provenance limits the final paper's reproducibility statement.
