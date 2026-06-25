# Case 3 — Urban airflow CFD over a building array (uDALES)

Applied, multi-variable realism (spec Section 6): incompressible airflow + heat
transport over an array of buildings (bluff bodies). Coupled **velocity** (`u, v`)
and **temperature** (`T`) fields, with boundary layers, wakes, and recirculation
behind obstacles — anisotropic, spatially heterogeneous statistics.

## Data (author-provided — GAP Section 6)

The uDALES runs are **author-provided** as `.nc` + `mask.npz`; no in-repo CFD
generator. Set `data_path` / `mask_path` in `configs/case/urban.yaml`. The scope
here is: loader, **solid-cell masking** (applied consistently in the model, the
obs operator, and every metric — solid cells excluded from RMSE/CRPS/KL),
channel-count fix, and metrics. Domain/grid, BCs, `Re`/`Pr`, stochasticity
source, normalisation, and per-variable noise `sigma` are **TODO** (specify with
user). Same generative setup as NS but multi-channel.

## Observation scenarios

Super-res `32^2->128^2` and sparse `5%`; sparse sensors at **physically
plausible** locations (street level + façades). Observe velocity and temperature.

## Deliverables

- **`tab:urban_accuracy`** — velocity RMSE, temperature RMSE, KL-at-points ×
  {`32^2->128^2`, `5%`}. → `generated/tab_urban_accuracy.tex`.
- **`tab:urban_calibration_cost`** — CRPS, `|1-spread/skill|`, NFE, s/step.
  → `generated/tab_urban_calibration_cost.tex`.
- **`fig:urban_fields`** — geometry + truth/prior/posterior for velocity and
  temperature; mark building footprints and sensor locations. *(Figure: TODO.)*

Pay attention to reconstruction in **unobserved wake regions** — report KL at
unobserved points there. Same method list as NS.

## Tidy rows this case emits

`case=urban`; `scenario ∈ {32^2->128^2, sparse 5%}`; `metric ∈ {rmse_velocity,
rmse_temperature, kl_points, crps, spread_skill}` plus `nfe`/`seconds`.

## TODO seams

- `driver.py::UrbanRunner.evaluate` — **GAP E2** (author data + solid-cell
  masking + channel-count fix), **E1** (obs operators), **E4** (multi-channel
  unified sampler), plus the shared new metrics (E7/E9/E10/E11).
