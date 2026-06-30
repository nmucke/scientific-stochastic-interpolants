# GPU-Machine Handoff — Paper-Sync Experiments

> **2026-06-29 evening — a run is LIVE.** The NS step-count benchmark
> (`run_ns_stepbench.sh`, 640 runs, M∈{50,100,250,500}) is running detached on the
> GPU; see `RUN_STATUS.md` (top) and `NEXT_SESSION_PROMPT.md` (START HERE block)
> for progress, the ~120–150 GPU-h ETA, the FlowDAS-regression flag, and the
> tomorrow checklist. Don't start the D-Flow/SURGE sweeps or headline runs until
> the step-bench frees the GPU (or run them after it).



**Audience:** a fresh agent (or engineer) on the GPU machine that has the trained model weights.
**Branch:** `sync-with-paper`. **Goal:** run the full-scale Navier–Stokes (and later urban) experiments
that cannot run on the laptop (no GPU, no `model.pth` weights there).

> ## Session update — 2026-06-29
>
> **This document is largely SUPERSEDED.** Read `PROJECT_HANDOFF.md` and
> `paper_experiments/RUN_STATUS.md` first; they are the current source of truth.
> The corrections that matter here:
> - **Method lineup is final** (9 generative + classical): Ours SI-SDE / FM-ODE /
>   FM-SDE ("FM-SDE (DM)"); FlowDAS; Guided FM (FIG); Guided FM (OT-ODE); D-Flow
>   SGLD; SDA; SURGE. Classical (NS only): EnKF (E=1000 non-localized = ground-truth
>   posterior / KL reference; E=64 localized = baseline), LETKF, particle filter,
>   ensemble score filter. The legacy "Guided FM" (one-step DPS-on-flow) and "Guided
>   diffusion" (DPS) are **DROPPED**.
> - **The multiplicative gain `G_τ` was dropped** — there are no longer three "modes"
>   to choose between for accuracy; the relevant axis is the covariance
>   (`inflated` / `inflated_shared` vs isotropic Jacobian-free). `dps_full`/`G_τ` are
>   kept off-by-default in code only.
> - **The FM prior is a dedicated trained checkpoint** (`flow_matching` /
>   `flow_matching_big`), NOT the SI drift reused. The SDA/SURGE **diffusion prior is
>   built from the FM model** (`diffusion_from_fm: true`), since the trained diffusion
>   checkpoint is weak.
> - **All baselines are implemented** (FlowDAS, FIG, OT-ODE, D-Flow SGLD, SDA, SURGE +
>   classical filters). The "missing baselines" list in §6.2 is stale.
> - **Analytical case is DONE** with the full faithful lineup; three baseline bugs
>   (FlowDAS, D-Flow SGLD, SDA) were found and fixed (details in `RUN_STATUS.md`).
> - **NS + urban headline runs are PENDING the GPU** — manuscript NS/urban tables are
>   `\tbd`; old NS baseline numbers are stale and must be re-run with the fixed
>   methods. Urban is generative-only, sparse-only, no KL/energy.
> - The live paper is `manuscript/` (compiles clean), not `paper_new/`.

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
**Case 1 (analytical linear–Gaussian)** — `tab:analytical_results` is filled with real numbers for the
full faithful lineup. KL to exact (mean over 5 seeds): SI-SDE 0.0009, FM-SDE 0.0016, FM-ODE 0.0011;
FlowDAS 0.080; Guided FM (OT-ODE) 0.0021; D-Flow SGLD 0.079; SDA 0.019; SURGE 0.0021; EnKF 0.0012; PF
0.0030; Guided FM (FIG) collapsed (structurally degenerate). Headline message: with faithful
implementations essentially every baseline recovers the linear-Gaussian posterior (an exactness check,
not strawmen) — methods separate on the nonlinear fluid cases.

> **Note:** the OLD numbers that used to be here (FlowDAS 0.299, Guided FM 0.104, Guided diffusion 0.174)
> are obsolete — they predate the baseline bug fixes and the dropped DPS baselines.

Figures: `manuscript/figures/analytical/` (analytical_case.pdf, analytical_kl_vs_steps.pdf,
analytical_covariance_ablation.pdf).

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

## 4. The likelihood modes (config: `likelihood_mode`)

> **2026-06-29:** the multiplicative-gain mode `dps_full` was DROPPED from the paper
> (kept off-by-default in code). The meaningful axis is the covariance: per-member
> `inflated`, shared `inflated_shared`, or isotropic `dps_jacobian_free`. See
> `PROJECT_HANDOFF.md`.

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

## 5. The gain decision — RESOLVED (2026-06-29): the multiplicative gain is dropped

The multiplicative-gain Theorem (`S_τ = G_τ S̄`, mode `dps_full`) did NOT improve accuracy; accuracy
comes from **inflating the covariance** (`inflated` / `inflated_shared`), not the gain. The gain is
therefore dropped from the paper (the methodology drift is now `b_prior + w_τ S̄`) and kept only as an
off-by-default code option. The ablation is a **covariance** comparison, not a gain comparison. This
section is retained for history; the decision is closed.

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

### 6.2 Baselines — ALL IMPLEMENTED (2026-06-29; this §6.2 list is stale)
All baselines are wired: FlowDAS, Guided FM (FIG), Guided FM (OT-ODE), D-Flow SGLD
(`DFlowPosterior`), SDA, SURGE (`SurgePosterior`); and the true-solver classical
filters EnKF, LETKF, particle filter, ensemble score filter (`enkf_baseline.py`).
Classical filters need the true NS solver for propagation (their advantage over the
generative methods, per spec §1.2) — so they run on NS only, not urban.
**What remains is RUNNING them at headline scale on the GPU** and locking the D-Flow
SGLD / SURGE NS hyperparameters, not implementing them.

### 6.3 Urban case (Case 3) — IMPLEMENTED; headline runs pending the GPU (2026-06-29)
Data + models arrived and the case is wired (`cases/urban/{driver.py,_urban_pipeline.py}`,
`configs/case/urban.yaml`). Data is **4-channel `(u, v, w, thl)`** (NOT 3). Generative-only,
**sparse 5% and sparse 1.5625% only (no super-res)**, no KL, no energy/enstrophy — per-variable RMSE +
split CRPS + spread-skill against the ground-truth state. Solid-cell masking is applied in the obs
operator and all metrics. `urban.yaml`: si_run `stochastic_interpolant_big_gamma1`, fm_run
`flow_matching_big`, `diffusion_from_fm: true`, test sims 170–178. **The headline runs are not done.**

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
