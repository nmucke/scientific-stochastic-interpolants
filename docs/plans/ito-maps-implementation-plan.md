# Plan: Itô Maps for scisi

Implementation plan for the features described in [Itô Maps for Any-Step SDEs (arXiv:2606.11156)](https://arxiv.org/abs/2606.11156): a new Itô map model module, a trainer refactor into a base trainer with subclasses, from-scratch and distillation training, and flow-matching support.

**PR ordering:** this plan is **PR 1**. The deterministic-models plan (`docs/plans/deterministic_models_and_trainer_refactor.md`) is **PR 2** and builds on the trainer refactor that lands here. The deterministic-to-Itô-map fine-tuning plan (`docs/plans/deterministic_to_ito_map_finetuning.md`) is **PR 3** and builds on both — it imposes two small design requirements on this PR, marked "PR 3 note" below. The trainer split (Phase 1 below) is shared infrastructure and is implemented exactly once, in this PR, using the file layout agreed with the deterministic plan (back-compat `Trainer` alias in `trainer.py`, no mass config edits). PR 2 adds only `DeterministicTrainer` and the `deterministic_models/` submodule on top.

## What the paper adds, in this repo's terms

An Itô map learns a **two-time stochastic flow map**

```
X̂ₛ,ₜ(x, W) = x + (t−s)·Ĝₛ,ₜ(x, W) + (Mₜ − Mₛ)
```

where `Ĝₛ,ₜ` is a learned average drift conditioned on (a compressed encoding of) the Brownian path `W`, and `Mₜ = ∫₀ᵗ σᵤ dWᵤ` is the martingale part (computable in closed form from the path, not learned). Once trained, it jumps from any time `s` to any time `t` in **one network evaluation**, while still sampling from the correct endpoint law — i.e., one-step (or few-step) generation of the SDE that the current `FollmerStochasticInterpolant` integrates with ~100 Euler–Maruyama steps. With `σ ≡ 0` it degenerates to a deterministic flow map, which is exactly the one-step version of flow matching — that is how FM support falls out naturally.

Training has two loss terms:

- **Diagonal loss** `ℒ_SI`: at `s = t`, `Ĝₜ,ₜ` has a closed-form regression target — the SDE drift of the interpolant process (drift + `(σₜ²/2)·score`), which this repo can already compute from its interpolants via `forward_diff` and the score identities in `AffineGaussianPathMixin`. In the paper's canonical setting (linear interpolant, Gaussian base, `σₜ = √(2(1−t))`) this reduces to `Gₜ,ₜ = E[X₁ − 2X₀ | Iₜ]` with per-sample target `X₁ − 2X₀`.
- **Self-distillation loss** (`ℒ_LSD` Lagrangian, or `ℒ_PSD` progressive): enforces two-time consistency, either via a time-derivative match (`Ĝₛ,ₜ + (t−s)∂ₜĜₛ,ₜ` vs. the diagonal drift at `t`, using `torch.func.jvp`) or via semigroup composition `Fₛ,ₜ = Fᵤ,ₜ ∘ Fₛ,ᵤ`.

**Distillation mode** replaces the diagonal target with a frozen teacher's drift: for a velocity teacher (flow matching), `Gₜ,ₜ(x) = (1+t)·v(x,t) − x` in the paper's setting; in general it is `b + (σ²/2)s` with the score `s` recovered from `v` via `score_from_velocity`, which the repo already implements. For a Föllmer drift teacher the conversion uses the existing `_drift_with_prior_score` machinery.

Two structural facts about the repo make this cheap to integrate: (1) all model-specific math already lives in `model.forward()` returning `(pred, target)`, so the trainer is nearly generic; (2) the two things Itô maps need that the current trainer cannot express — two-time sampling, Brownian simulation, and multi-evaluation losses with `jvp` — are exactly why the trainer split is needed.

---

## Phase 1 — Trainer refactor (no behavior change)

Shared infrastructure for this plan **and** the deterministic-models plan (PR 2). It lands here, once, with the file layout below; PR 2 does not re-touch these files.

Split `src/scisi/training/trainer.py` into:

### New: `src/scisi/training/base_trainer.py` — `BaseTrainer`

Owns everything generic (moved from the current `Trainer`):

- `SCHEDULERS_THAT_REQUIRE_LOSS` and the `EarlyStopping` dataclass move here.
- epoch loop, `train()`, validation loop, trackio logging, checkpointing + config dump, early stopping, scheduler handling, mixed-precision warmup + `GradScaler`, `EmaGradientClipper` hookup, device movement.
- Two overridable hooks, which are the *only* subclass responsibilities:
  - `_prepare_batch(batch) -> batch` — base implementation: device transfer only. Subclasses inject sampled quantities.
  - `_compute_loss(batch) -> Tensor` — default: `pred, target = self.model(**batch); return self.loss_fn(pred, target)`.
- Two additions worth making while we are here (PR 2 inherits both; they are off-by-default / behavior-preserving):
  - make checkpoint saving work when `tracker is None` (currently `_save_checkpoint` silently no-ops without a tracker);
  - an **optional EMA of model weights** (off by default) — self-distillation losses are much more stable with an EMA/stop-grad target.
- **PR 3 note:** include a no-op `_on_epoch_start(epoch)` hook called at the top of the epoch loop — one behavior-preserving line; PR 3's staged unfreezing needs it.

### Rewrite: `src/scisi/training/trainer.py` — `StochasticInterpolantTrainer` + back-compat alias

Module path kept so the 25+ existing configs **and checkpointed `config.yaml`s** referencing `scisi.training.trainer.Trainer` / `...trainer.EarlyStopping` keep working (checkpointed configs cannot be edited, so an alias is mandatory, not optional):

```python
from scisi.training.base_trainer import BaseTrainer, EarlyStopping  # re-export

class StochasticInterpolantTrainer(BaseTrainer):
    def _prepare_batch(self, batch):
        batch = super()._prepare_batch(batch)
        batch["t"] = torch.rand(batch["base"].shape[0], 1, device=self.device)
        batch["noise"] = torch.randn(batch["base"].shape, device=self.device)
        return batch

Trainer = StochasticInterpolantTrainer  # existing configs/checkpoints keep working
```

Serves Föllmer SI, flow matching, and diffusion models unchanged. **No mass config updates** — existing configs keep `scisi.training.trainer.Trainer`; only new configs use explicit subclass targets.

### Explicitly deferred to PR 2

`DeterministicTrainer` and the `deterministic_models/` submodule are **not** part of this PR — they are specified in `docs/plans/deterministic_models_and_trainer_refactor.md` and slot in as one new file each on top of this refactor.

### Refactor guard

A golden-run test — fixed seed, 2 epochs on a tiny synthetic dataset, assert the loss trajectory is bit-identical between old `Trainer` and new `StochasticInterpolantTrainer` before deleting the old code. Also assert the back-compat imports: `scisi.training.trainer.Trainer is StochasticInterpolantTrainer` and `EarlyStopping` importable from `scisi.training.trainer` (PR 2's `tests/test_trainers.py` re-checks these; keeping them here means PR 2 can't regress PR 1).

## Phase 2 — Itô map module: `src/scisi/models/ito_maps/`

### `brownian.py` — path simulation and encoding (the genuinely new numerics)

- `BrownianPathSampler`: simulates increments of `W` on a uniform grid over `[0,1]` and accumulates `Mₜ = ∫σᵤ dWᵤ`. For PDE fields `W` lives per-pixel in `[B, C, H, W]`, so a 200-point path is ~200× state memory — provide a configurable grid (paper uses 200–256; start at 64 for 64×64 fields) and a **closed-form KL mode** that never materializes the path: sample K i.i.d. Gaussian coefficient fields `ξₙ` and evaluate `Mₜ − Mₛ = Σₙ ξₙ ∫ₛᵗ σᵤ φₙ′(u) du` with precomputed scalar integrals.
- Two `BrownianEncoder`s per the paper:
  - `KLEncoder` — K≈5 sine-basis Karhunen–Loève coefficients per dimension → output `[B, K·C, H, W]`;
  - `DyadicEncoder` — Haar/dyadic martingale increments, depth 4–5.
  Both produce channel-stacked tensors so they ride the **existing `field_cond` input pathway** — no new network plumbing required for the baseline UNet.
- `sigma_schedule` abstraction with the paper's default `σₜ = √(2(1−t))` and a `gamma`-matched option so it composes with the repo's stochastic interpolants; `σ ≡ 0` for the flow-matching / deterministic case.

### `ito_map_model.py` — `ItoMapModel(BaseModel)`

- Wraps a standard `drift_model` network (same contract as today) parametrizing `Ĝₛ,ₜ(x, ...)`.
- **Two-time conditioning:** pass `cond = [s, t]` as `[B, 2]` — the FiLM time embedding in `src/scisi/architectures/u_net.py` needs a small, backward-compatible change to accept 2-dim `cond` (embed each and concatenate). Brownian features enter via the `field_cond` channel-concat path (or a dedicated `brownian_features` kwarg defaulting to `None`, so existing nets are untouched).
- `map(x, s, t, w_feats, martingale_increment, ...)` returning `x + (t−s)Ĝ + (Mₜ−Mₛ)`.
- `sample(...)`: one-step endpoint prediction `X̂ₛ,₁`. `sample_trajectory(...)`: reuses `BaseModel._sample_trajectory`'s autoregressive rollout with a configurable number of inner steps (`num_steps=1` is the headline any-step case; N-step partitions reuse the same Brownian path, matching the paper's Algorithm 2).
- Diagonal drift helper `G_diag_target(base, target, noise, t)` computing the closed-form regression target: interpolant `forward_diff` plus the `(σ²/2)·score` correction, reusing `AffineGaussianPathMixin`. Keeping it general (rather than hardcoding `X₁ − 2X₀`) lets it work with the repo's Föllmer / point-mass-base interpolants.
- **Distillation constructors**, mirroring the existing `DenoiseDiffusionModel.from_flow_matching` pattern:
  - `ItoMapModel.from_flow_matching(fm_model, sigma_schedule)` — teacher drift `G = (1+t)v − x` (general form via `score_from_velocity`).
  - `ItoMapModel.from_stochastic_interpolant(si_model, sigma_schedule)` — teacher drift from the Föllmer `b_theta` with a prior-score correction to the chosen `σₜ`.
  - Both do **weight surgery** to warm-start the student net from the teacher net: copy all matching parameters; zero-init the new input channels (Brownian features) and the extra time-embedding rows, so at init the student ignores the new inputs and exactly reproduces the teacher's diagonal behavior. They also stash the frozen teacher for the trainer's distillation target.

## Phase 3 — `src/scisi/training/ito_map_trainer.py` — `ItoMapTrainer(BaseTrainer)`

Inherits every base feature (logging, checkpointing, early stopping, scheduler, MP warmup, grad clipping, optional EMA) for free. Adds:

- `_prepare_batch`: diagonal `t ~ U(0,1)`; off-diagonal `(s, t)` from a configurable distribution (uniform-reordered, or the paper's logit-normal `t = sigmoid(0.6 + Zₜ)`, `s = t·sigmoid(Zₛ)`); interpolant noise; Brownian path simulation + feature extraction + martingale increments (all delegated to `brownian.py`).
- `_compute_loss` = `ℒ_SI + λ·ℒ_consistency`, with `consistency_mode: lsd | psd`:
  - **LSD**: `∂ₜĜₛ,ₜ` via `torch.func.jvp` over the `t` conditioning input (with a finite-difference fallback flag, since `jvp` through some wrapped architectures — PDETransformer, Aurora — may not be traceable); stop-grad (or EMA-model) diagonal target at `Ĝₜ,ₜ(Xₜ)`.
  - **PSD**: two composed student evaluations vs. one direct, `s < u < t`, stop-grad on the composition branch.
- **Teacher mode**: optional frozen `teacher` (built via the Phase 2 classmethods). When present, the diagonal and consistency targets use the converted teacher drift instead of the closed-form interpolant target (paper's Algorithm 4, teacher-guided Lagrangian). `teacher: null` → from-scratch training. Same trainer class handles both. **PR 3 note:** the teacher interface must be duck-typed — any object exposing `drift(x, t, field_history, field_cond, pars_cond)` — not `isinstance`-gated on the SI/FM model classes, so PR 3 can plug in a non-neural analytic teacher (`GaussianShellTeacher`).
- `_compute_val_loss` inherits from base and calls the same `_compute_loss` — no special casing needed.

## Phase 4 — Configs, entry points, flow matching

- `bin/main_train.py`: one addition — if the config has a `pre_trained_model` block (same shape as the posterior configs), instantiate + load the teacher and build the Itô map via the appropriate `from_*` classmethod before constructing the trainer.
- New configs:
  - `config/ito_map_stochastic_navier_stokes.yaml` — from scratch, stochastic (`σₜ = √(2(1−t))` or gamma-matched).
  - `config/ito_map_distill_navier_stokes.yaml` — `pre_trained_model` pointing at a trained Föllmer SI checkpoint.
  - `config/ito_map_flow_matching_navier_stokes.yaml` — `sigma_schedule: zero`, deterministic flow map; trainable both from scratch and by distilling a `FlowMatchingModel`. With `σ ≡ 0`, Brownian features and martingale increments are identically zero, LSD reduces to deterministic flow-map matching, and one-step sampling is consistency-style FM — no special-case code, just config.
- Optional follow-up (not blocking): a `posterior_models/ito_map_posterior.py`, since cheap differentiable endpoint samples are the paper's headline use for steering / data assimilation — flag as future work in the PR.

## Phase 5 — Tests and validation

1. **Trainer refactor golden test** (Phase 1, described above).
2. **Brownian unit tests**: increment variance / KL coefficient statistics; `Mₜ` variance matches `∫σ²`; KL closed-form vs. simulated-path agreement; encoder output shapes.
3. **Diagonal-target test**: on a 1D/2D Gaussian toy with linear interpolant, verify `G_diag_target` equals the analytic `E[X₁ − 2X₀ | Iₜ]` posterior mean.
4. **Teacher-conversion test**: analytic Gaussian case where the FM velocity is known in closed form — check `(1+t)v − x` equals the analytic SDE drift.
5. **End-to-end smoke tests**: 2-epoch from-scratch and distillation runs on a tiny synthetic dataset (both `lsd` and `psd`, both stochastic and `σ = 0`); one-step sample shape + finiteness; autoregressive rollout via `sample_trajectory`.
6. **Scientific validation** (uses existing infra): train on the analytical case, compare one-step Itô map samples against 100-step Euler–Maruyama Föllmer samples using the existing `metrics/` (CRPS, spread-skill, sliced-W2, spectra) — this is the result that actually tells you the feature works.

## Build order and risk notes

Phases land as separate PRs in order 1 → 2 → 3 → 4 → 5 (tests written alongside each phase). Main risks, ranked:

- **Memory of Brownian paths on fields** — mitigated by the closed-form KL mode; decide grid resolution empirically on Navier-Stokes first.
- **`jvp` through wrapped third-party nets** (PDETransformer, Aurora) — finite-difference fallback covers it; the UNet path should trace fine.
- **Base distribution mismatch with the paper**: the paper's theory is stated for `X₀ ~ N(0,I)`; the repo's Föllmer setup uses a point-mass base (previous PDE state) with `γ(t)` noise. The general diagonal target (`forward_diff + (σ²/2)·score`) handles this, but the `(1+t)v − x` teacher shortcut only holds in the Gaussian-base case — the code should use the general mixin-based conversion and treat the shortcut as a special case.

## Explicit follow-up after PR 1 lands

**Scientific validation (Phase 5.6) is deferred out of PR 1** — the unit and smoke
tests show the method trains finitely, not that it works. Owner: **nmucke**.
Task: train an Itô map on the analytical / stochastic Navier-Stokes case and
compare one-step Itô map samples against 100-step Euler–Maruyama Föllmer samples
using the existing `metrics/` (CRPS, spread-skill, sliced-W2, spectra), including
a 1-epoch GPU bf16 mixed-precision run before any long training (the bf16 LSD
path only has CPU-autocast and skip-if-no-CUDA test coverage).

## Alignment with the follow-up plans (PR 2, PR 3)

`docs/plans/deterministic_models_and_trainer_refactor.md` (PR 2) and `docs/plans/deterministic_to_ito_map_finetuning.md` (PR 3) are implemented after this plan, in that order, as separate PRs. PR 3 additionally extends two PR 1 files additively (`ito_map_model.py` gains `from_deterministic`; `ito_map_trainer.py` gains freeze/unfreeze/teacher-warmup options) — the two "PR 3 note" design requirements above (duck-typed teacher, `_on_epoch_start` hook) are what make those extensions retrofit-free. Division of labor for PR 1/PR 2:

| Piece | Owned by |
|---|---|
| `training/base_trainer.py` (`BaseTrainer`, `EarlyStopping`, `SCHEDULERS_THAT_REQUIRE_LOSS`) | PR 1 (this plan, Phase 1) |
| `training/trainer.py` rewrite (`StochasticInterpolantTrainer`, `Trainer` alias, re-exports) | PR 1 |
| Optional weight-EMA + tracker-less checkpointing in `BaseTrainer` | PR 1 |
| `training/ito_map_trainer.py`, `models/ito_maps/`, Itô configs | PR 1 |
| `training/deterministic_trainer.py` | PR 2 |
| `deterministic_models/` submodule, deterministic config, Aurora guard in `main_train.py` | PR 2 |

Known shared touchpoints (sequential PRs, trivial merges):

- **`bin/main_train.py`**: PR 1 adds the `pre_trained_model` teacher block; PR 2 adds the `"drift_model" in cfg.model` guard on the Aurora special-case. Independent edits to nearby code — PR 2 rebases on PR 1.
- **`architectures/u_net.py` cond embedding**: PR 1 extends the FiLM time embedding to accept `[B, 2]` cond **backward-compatibly** — `[B, 1]` behavior is unchanged, so PR 2's `DeterministicModel._step` (which passes `cond = zeros(B, 1)`) is unaffected.
- **Trainer back-compat tests**: PR 1's golden test already asserts the `Trainer` alias and `EarlyStopping` re-export; PR 2's `tests/test_trainers.py` re-asserts them alongside its `DeterministicTrainer` tests — deliberate overlap, not duplication to remove.
