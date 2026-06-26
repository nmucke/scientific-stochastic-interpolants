# GPU-Machine Handoff — Paper-Sync Experiments

**Audience:** a fresh agent (or engineer) on the GPU machine that has the trained model weights.
**Branch:** `sync-with-paper`. **Goal:** run the full-scale Navier–Stokes (and later urban) experiments
that cannot run on the laptop (no GPU, no `model.pth` weights there).

Read this top-to-bottom before running anything. The companion documents are:
- `paper_new/GAP_ANALYSIS.md` — the full paper↔code gap analysis and the 6-phase plan (the "why").
- `paper_new/EXPERIMENTS_IMPLEMENTATION_SPEC.md` — the engineer-facing experiment spec (cases, metrics, scenarios).
- `paper_experiments/README.md` — how the experiment harness is organized.

---

## 1. What this branch did (state at handoff)

A three-layer rebuild aligned the implementation with the manuscript in `paper_new/`:

**Layer 1 — affine-path foundation (`src/scisi/models/`)** — DONE, reviewed, math verified to ~1e-13.
General-in-the-schedules helpers on the interpolation classes: `velocity_score_coeff` (a_τ),
`score_from_velocity`/`velocity_from_score`, and `FlowMatchingModel.score` (Eq. fm_score). Everything
is general in α,β,γ (the SI model uses quadratic-β, per author decision — see §5).

**Layer 2 — unified posterior sampler (`src/scisi/posterior_models/`, `likelihood_models/`)** — DONE, reviewed, fixed.
The paper's unified drift `b_post = b_prior + w_τ·G_τ·S̄` with `w_τ = a_τ + ½g²`, instantiated as the
three samplers SI-SDE / FM-SDE / FM-ODE. The likelihood (`InterpolantGaussianLikelihood`) has THREE
config-selectable modes (see §4). Validated against the closed-form analytical Gaussian posterior:
the `inflated` mode recovers it (mean error ~0.003–0.01, decreasing with ensemble size).

**Layer 3 — experiments (`paper_experiments/`)** — analytical DONE with real numbers; NS wired + smoke-passed.
New library capabilities live in `src/scisi/` (metrics, super-res operator); experiment *definitions*
live in `paper_experiments/`.

### What is already a real, final result (ran on the laptop, no training needed)
**Case 1 (analytical linear–Gaussian)** — `tab:analytical_results` is filled with real numbers:

| Method | KL to exact | Sliced-W₂ |
|---|---|---|
| Ours (SI-SDE / FM-SDE / FM-ODE) | 0.001 / 0.002 / 0.001 | 0.017 / 0.024 / 0.020 |
| FlowDAS | 0.299 | 0.352 |
| Guided FM / Guided diffusion | 0.104 / 0.174 | 0.185 / 0.194 |
| EnKF / Particle filter | 0.001 / 0.003 | 0.021 / 0.028 |

Figures: `paper_experiments/figures/results/analytical/` (the 7 panels of `fig:analytical_panels` + a
dimensionality-convergence panel). The dimensionality study confirms `inflated` converges to exact while
the DPS modes plateau — the key finding (see §5).

### What is wired but needs THIS machine (real weights + GPU)
**Case 2 (Navier–Stokes)** — full pipeline wired (`paper_experiments/cases/navier_stokes/`): loads a
trained prior, builds the per-scenario observation operator, runs autoregressive assimilation with the
three samplers + FlowDAS, computes the full metric set (ensemble-mean RMSE, log-KE-spectrum RMSE, CRPS,
spread–skill, KL-at-points, NFE, wall-clock), emits tidy results, fills `tab:ns_accuracy` /
`tab:ns_calibration_cost` / `tab:ablation`, and renders `fig:ns_trajectories` / `fig:ns_diagnostics`.
It passed a smoke run with RANDOM weights (proves wiring). **It needs real `model.pth` weights and a GPU.**

---

## 2. THE most important things to do here (in order)

1. **Point the config at the real weights** (§3.1). Both priors are already trained on the GPU box:
   `checkpoints/stochastic_navier_stokes/stochastic_interpolant_small/` (SI) and
   `checkpoints/stochastic_navier_stokes/flow_matching/` (FM). The config already names these; just set
   `require_weights: true` so a missing `model.pth` hard-fails instead of falling back to random weights.
2. **Run the full-scale NS headline + ablation** (§3 commands / `run_gpu_ns.sh`), then regenerate the tables (§3.4).
3. **Decide the canonical likelihood mode** from the NS results (§5) — this is a deferred author decision.
4. **Implement the remaining baselines** (§6.2) and the **urban case** when its data arrives (§6.3).

---

## 3. How to run the full-scale Navier–Stokes experiments

> **The one-command path:** `bash paper_experiments/run_gpu_ns.sh` runs the sanity gate → headline grid →
> ablation → table regeneration. Edit the variables at the top of that script (device, checkpoint run names,
> E/M/seeds/scenarios) first. The manual steps below are what the script does, for reference.

### 3.0 Environment
- Python env: the repo's `.venv` / `uv` (see `pyproject.toml`). Confirm `torch` sees CUDA:
  `python -c "import torch; print(torch.cuda.is_available())"`.
- Data: `data/stochastic_navier_stokes/data.npz` (key `state`, shape ~`(200,100,128,128)`). ~1.1 GB.
- Weights: place trained checkpoints under `checkpoints/stochastic_navier_stokes/<run_name>/model.pth`
  (the dir already holds the matching `config.yaml`s).

### 3.1 Point at real weights
Both priors are already trained on the GPU box, at
`checkpoints/stochastic_navier_stokes/stochastic_interpolant_small/model.pth` (SI) and
`checkpoints/stochastic_navier_stokes/flow_matching/model.pth` (FM). The config already names these:
```yaml
checkpoints:
  si_run: stochastic_interpolant_small   # trained SI prior (SI-SDE / FlowDAS)
  fm_run: flow_matching            # trained FM prior (FM-SDE / FM-ODE)
require_weights: true              # hard-fail if a configured dir / model.pth is missing
reference_ensemble_size: 1024
device: cuda
```
The loader builds the FM model from the `flow_matching` checkpoint's OWN `config.yaml` and loads its
`model.pth` (no SI-reuse fallback needed). Set `require_weights: true` on the GPU box so a missing
`model.pth` is a hard error rather than a silent random-weights run. Confirm the SI and FM checkpoints
share the data/architecture/schedules used for a fair comparison (spec §5).

### 3.2 Headline run (Hydra multirun over the 3 samplers × scenarios; seeds loop internally)
```bash
python paper_experiments/run.py --multirun case=navier_stokes \
  method=si_sde,fm_sde,fm_ode scenario=superres_32,sparse_5 \
  ensemble_size=64 num_steps=50 likelihood_mode=inflated \
  case.require_weights=true case.device=cuda
# appendix columns: scenario=superres_16,sparse_1p5
```

### 3.3 Ablation run (fills tab:ablation; correction axis = likelihood_mode + g_τ/M/E sweeps)
```bash
python paper_experiments/run.py case=navier_stokes ablation=true \
  ablation_smoke=false ablation_scenario="sparse 5%" \
  ensemble_size=64 num_steps=50 likelihood_mode=inflated \
  case.require_weights=true case.device=cuda
```
The correction-axis maps to the modes: No-correction (G=I) → `inflated`; Full gain → `dps_full`;
Jacobian-free → `dps_jacobian_free`. The g_τ sweep includes `g=0` (== FM-ODE).

### 3.4 Regenerate the LaTeX tables (from the UNION of all case CSVs)
```bash
# combine per-case tidy CSVs (keeps the already-real analytical rows) and emit every tab_*.tex
head -1 paper_experiments/results/analytical_results.csv > paper_experiments/results/all_results.csv
tail -q -n +2 paper_experiments/results/*_results.csv >> paper_experiments/results/all_results.csv
python paper_experiments/make_tables.py --results paper_experiments/results/all_results.csv
```
Snippets land in `paper_experiments/generated/tab_*.tex` for the paper to `\input`.

### 3.5 Sanity gate before the full grid
Run ONE cheap cell first (e.g. SI-SDE, sparse 1.5625% so N_y=256, seed 0) and confirm the log prints
"Loaded trained weights from ..." (NOT the random-weights warning) and RMSE is at a sane physical scale,
before launching the full grid. `run_gpu_ns.sh` does this automatically.

---

## 4. The three likelihood modes (config: `likelihood_mode`)

`InterpolantGaussianLikelihood` (in `src/scisi/likelihood_models/gaussian_likelihood.py`):

- **`inflated`** — ΠGDM-style: `Σ̄ = β²R + H Σ_s Hᵀ` (full Σ_s), front factor `Σ_s/σ_τ²`, **no gain (G=I)**.
  **Exact for the Gaussian case** (analytical gate ~0.005). **Provisional default / headline rows.**
  Cost: at full scale it does ~N_y JVPs per pseudo-time step (the dominant cost — see §7).
- **`dps_full`** — the paper's multiplicative-gain DPS surrogate: same `S̄`, then `× G_τ = I + (1/β²)Σ_s HᵀR⁻¹H`.
  An ablation variant; biased (~0.3 in the analytical case, M-independent).
- **`dps_jacobian_free`** — Corollary cheap_drift: isotropic `Σ_s = ρI`, cheap, inaccurate at moderate τ.

Deprecated keys still map for back-compat (`gain: full→dps_full`, `gain: jacobian_free→dps_jacobian_free`,
`correct_likelihood_score: true→inflated`), but new configs should use `likelihood_mode`.

---

## 5. Open scientific decision (deferred to the author, informed by NS results)

A review finding (verified against the closed-form posterior): **only the `inflated` (ΠGDM) covariance
recovers the exact analytical posterior; the paper's multiplicative-gain Theorem (`S_τ = G_τ S̄`) yields the
less-accurate DPS surrogate.** The manuscript currently presents the DPS surrogate as the method and defers
the inflated covariance to "future work," but `results.tex` claim (i) ("reproduces the exact posterior")
only holds for `inflated`. **No paper text has been changed** (author's instruction). After the NS runs,
the author will decide which mode is the canonical/default and whether to update the methodology/results
prose. The code supports all three modes so this can be decided empirically.

Also: the SI model uses **quadratic-β** (`β=t²`), per author decision — all schedule-derived formulas are
general in α,β,γ, so this is consistent; just don't assume rectified-flow reductions for SI.

---

## 6. Remaining work (prioritized)

### 6.1 FM prior for NS — already trained (no action needed beyond pointing at it)
Both priors exist on the GPU box: SI at `checkpoints/stochastic_navier_stokes/stochastic_interpolant_small/`
and FM at `checkpoints/stochastic_navier_stokes/flow_matching/`. The config (§3.1) names them and the
loader builds the FM model from the FM checkpoint's own `config.yaml`. **Verify once before the full run:**
the FM checkpoint instantiates a `FlowMatchingModel` and its `model.pth` `load_state_dict` succeeds without
key/shape errors, and that SI and FM share the data/architecture/schedules (spec §5) for a fair comparison.
(If you ever need to retrain, the pipeline is `config/flow_matching_stochastic_navier_stokes*.yaml` +
`src/scisi/bin/main_train.py`.)

### 6.2 Baselines (deferred "until after Phase 1" — now is the time)
Wired today: FlowDAS (real), our 3 samplers. **Missing** (each lists as a row in the NS/urban tables):
- **Guided diffusion / DPS** — `paper_experiments/configs/method/guided_diffusion.yaml` exists but
  `DPSGaussianLikelihood` is NOT implemented (TODO marker in the config).
- **EnKF** — exists OFF-pipeline in `external_libs/jax_cfd_lib`; wrap it as a benchmark method sharing the
  same truth+obs+mask.
- **Bootstrap particle filter** — `src/scisi/particle_filter/` is an empty stub; implement (the analytical
  case has a minimal PF in `paper_experiments/cases/analytical/samplers.py` to reference).
- **SDA** and **ensemble score filter** — entirely absent (second wave).
Classical filters (EnKF/PF) need the true NS solver for propagation (note this as their advantage over the
generative methods, per spec §1.2).

### 6.3 Urban case (Case 3) — pending author-provided data
`paper_experiments/cases/urban/` is a stub. The author will provide uDALES `.nc` runs + `mask.npz`. When
they arrive: place under `data/udales/`, set `data_size` to the real channel count (velocity + temperature),
ensure solid-cell masking is applied in the obs operator AND all metrics, and reuse the NS driver pattern.

### 6.4 After NS results
Decide the canonical likelihood mode (§5); then (author) update the manuscript prose accordingly; fill the
NS/urban tables and figures into `paper_new/sections/results.tex` (replace the `\figbox{}` placeholders with
`\includegraphics`, `\input` the generated `tab_*.tex`).

---

## 7. Performance caveat for the `inflated` full-scale run

The `inflated` mode forms `H Σ_s Hᵀ` via ~N_y Jacobian-vector products per pseudo-time step. Per trajectory
the UNet-eval count ≈ `N_y × M × (physical_steps − L)`; for super_res 32→128 that's `1024 × 50 × 20 ≈ 1e6`
per seed per method — large. Budget GPU-hours accordingly; for appendix scenarios consider smaller M or
batching the JVP columns. The `dps_jacobian_free` mode is far cheaper (no JVPs) if a fast approximate run is
needed first. (The NS-hardening commit cached the device-resident observation matrix so `seconds/step` is
representative — confirm no per-call CPU→GPU copies remain.)

---

## 8. Reproducibility checklist (spec §9)
- Fixed seed list; tables report mean ± std over seeds (`aggregate_over_seeds`).
- Identical truth + observation sequences + sensor masks across all methods per scenario (seeded;
  masks independent of method/seed).
- SI and FM priors must share architecture/data/schedules; log the training configs.
- Generative baselines reuse the shared prior; classical baselines use the true solver (note it).
- NFE and wall-clock logged at matched ensemble size; exclude solid cells from all urban metrics.
- KL-at-points uses a large-E reference ensemble (Case 2/3), the analytic posterior (Case 1).
