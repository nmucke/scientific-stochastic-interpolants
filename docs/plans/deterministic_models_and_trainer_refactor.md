# Deterministic time-stepping models + trainer refactor

> **PR ordering / prerequisite:** this plan is **PR 2**, implemented after the Itô maps plan (`docs/plans/ito-maps-implementation-plan.md`, **PR 1**). PR 1's Phase 1 lands the `BaseTrainer` / `StochasticInterpolantTrainer` split with exactly the file layout specified in Part 2 below — so by the time this plan is implemented, Part 2 reduces to adding `deterministic_trainer.py` only. Part 2 keeps the full split spec as the reference contract (and as the fallback if this plan is ever implemented standalone).
>
> **Downstream consumer (PR 3):** `docs/plans/deterministic_to_ito_map_finetuning.md` fine-tunes a trained `DeterministicModel` into an Itô map. It consumes this module read-only, but treats two things as load-bearing contracts: the `_step(x, field_history, field_cond, pars_cond)` signature (with `cond = zeros(B,1)` internally) and the `residual` flag semantics (`x + out` vs `out`). Don't change these without updating PR 3.

## Context

The repo trains stochastic interpolant (SI) generative models for time-stepping (`x_n → x_{n+1}`) in scientific applications. We want a simpler deterministic baseline: `x_{n+1} = network(x_n, context, params)` trained with plain MSE. Requirements:

- New `src/scisi/deterministic_models/` submodule; models reuse the **same architectures** (`scisi.architectures.u_net.UNet`, `pde_transformer.PDETransformerWrapper`) and the **same batch dict structure** (`base`, `target`, `field_history`, `field_cond`, `pars_cond`) as the stochastic models.
- Single-step prediction **and** autoregressive trajectory rollout, compatible with `main_test.py`'s `model.sample_trajectory(...)` call.
- Split `src/scisi/training/trainer.py` into a `BaseTrainer` (mixed precision + warmup, EMA grad clipping, early stopping, checkpointing, trackio logging, scheduler handling) with `StochasticInterpolantTrainer` and `DeterministicTrainer` subclasses.

Key facts driving the design:
- The trainer's SI-specific code is confined to two methods: `_prepare_batch` (injects `batch["t"]`, `batch["noise"]`) and `_compute_loss` (`pred, true = self.model(**batch); loss_fn(pred, true)`). The tuple-forward contract is already MSE-generic.
- `BaseModel._sample_trajectory` (`src/scisi/models/base_model.py:174-221`) contains exactly the autoregressive rollout we need (history rolling, per-step `field_cond`/`pars_cond` indexing, trajectory stacking) and only calls `self.sample(...)` per physical step. `BaseModel` is `nn.Module` (not ABC), so its `@abstractmethod` decorators are inert — we can subclass without implementing stepper machinery.
- Everything is wired via Hydra `_target_` strings; no registry. 25 existing configs + checkpointed `config.yaml`s reference `scisi.training.trainer.Trainer` and `scisi.training.trainer.EarlyStopping` → keep those import paths working.
- Datasets already yield `base` = x_n, `target` = x_{n+1}, `field_history` (+ optional `field_cond`/`pars_cond`). No dataset changes needed.

## Part 1 — `src/scisi/deterministic_models/`

**New: `src/scisi/deterministic_models/__init__.py`** — docstring + `from scisi.deterministic_models.deterministic_model import DeterministicModel`.

**New: `src/scisi/deterministic_models/deterministic_model.py`** — `class DeterministicModel(BaseModel)`:

```python
def __init__(self, network: nn.Module, residual: bool = False, mask_path: Optional[str] = None)
```
Subclassing `BaseModel` gives us mask handling, `device` property, `_prepare_batch` batch repetition, and the `_sample_trajectory` rollout for free. Any deterministic model = this class + an architecture from config (same pattern as `drift_model` for SI models).

Methods:
- `_step(x, field_history=None, field_cond=None, pars_cond=None) -> Tensor` — the one-step map. `cond = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)` as pseudo-time placeholder (architectures require a `cond` arg; keeps arch configs with `cond_dim: 1` unchanged). `out = self.network(x, cond, field_history, field_cond, pars_cond)`; return `x + out` if `self.residual` else `out`. (`residual=False` default = exactly the requested formulation; flag is opt-in.)
- `forward(base, target, field_history=None, field_cond=None, pars_cond=None, **kwargs) -> (pred, target)` — returns `(self._step(base, ...), target)`; `**kwargs` swallows `t`/`noise` defensively. Tuple return keeps the generic trainer `_compute_loss` working.
- `drift(...)` — `raise NotImplementedError` (no drift for a direct next-step predictor).
- `sample(field_history, base=None, batch_size=1, field_cond=None, pars_cond=None, return_field_history=False, **kwargs) -> Tensor` — single step. `**kwargs` absorbs and ignores `num_steps`, `stepper`, `diffusion_term`, `gaussian_base`. If `base is None`, use `field_history[..., -1]` (main_test.py passes `base=None` for non-Follmer models). Reuse inherited `self._prepare_batch` when `batch_size > 1 and field_history.shape[0] == 1`. Move inputs to `self.device`, run `_step` under `torch.no_grad()`, return result `.cpu()`. When `return_field_history=True`, roll history like `BaseModel._sample` does (`cat([field_history[..., 1:].cpu(), pred.unsqueeze(-1)], -1)`) and return `(pred, field_history)` — the contract `_sample_trajectory` relies on.
- `sample_trajectory(field_history, base=None, batch_size=1, num_physical_steps=10, field_cond=None, pars_cond=None, **kwargs) -> Tensor` — delegates to inherited `self._sample_trajectory(..., num_steps=1, gaussian_base=False)`. Output `[B, C, H, W, num_physical_steps]`, same as stochastic models. Docstring note: with `batch_size > 1` all ensemble members are identical (deterministic).

## Part 2 — Trainer split (lands in PR 1 — the Itô maps plan)

**Already done when this PR starts** (PR 1, Phase 1 of the Itô maps plan). The contract this PR relies on:

**`src/scisi/training/base_trainer.py`** — moved from trainer.py:
- `SCHEDULERS_THAT_REQUIRE_LOSS`, `EarlyStopping` dataclass.
- `class BaseTrainer` = current `Trainer` minus the t/noise injection: same `__init__` signature, `_train_step_mixed_precision`, `_train_step_full_precision`, `train`, `_compute_val_loss`, `_log_with_tracker`, `_check_early_stopping`, `_update_scheduler`, `_save_checkpoint`, `_print_info`, and the generic `_compute_loss`.
- `BaseTrainer._prepare_batch(batch)` = device move only.
- Note: PR 1 also adds two behavior-preserving extras to `BaseTrainer` — checkpoint saving with `tracker=None`, and an **optional** weight-EMA (off by default). `DeterministicTrainer` inherits both; neither changes anything in this plan (the "move verbatim" framing predates PR 1).

**`src/scisi/training/trainer.py`** (module path kept for backwards compat):
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

**New (this PR): `src/scisi/training/deterministic_trainer.py`** — `class DeterministicTrainer(BaseTrainer)`; body essentially empty (inherited `_prepare_batch` already does the right thing). Exists for config readability and future divergence (e.g. rollout-based validation). This is the only trainer file this PR adds.

## Part 3 — Config + entry points

**New: `config/deterministic_navier_stokes.yaml`** — copy of `config/stochastic_navier_stokes.yaml` with:
```yaml
model:
  _target_: scisi.deterministic_models.deterministic_model.DeterministicModel
  residual: false
  network:            # identical UNet block as in stochastic_navier_stokes.yaml
    _target_: scisi.architectures.u_net.UNet
    ...
trainer:
  _target_: scisi.training.deterministic_trainer.DeterministicTrainer
  # early_stopping / gradient_clipper sub-configs unchanged
```
No `interpolation` block; `loss_fn` stays `torch.nn.MSELoss`.

**Edit: `src/scisi/bin/main_train.py`**:
1. Guard the Aurora special-case (line ~117): `if "drift_model" in cfg.model and "AuroraWrapper" in cfg.model.drift_model._target_:` — the deterministic config has no `drift_model` key.
2. No config-switch code change needed: run with `--config-name deterministic_navier_stokes` (add a commented `config_name` line matching existing convention).

**`src/scisi/bin/main_test.py`** — no changes: `num_steps`/`stepper` are absorbed by `sample_trajectory`'s `**kwargs`, and `base=None` (set for non-Follmer models) falls back to `field_history[..., -1]`.

## Part 4 — Tests

**New: `tests/test_deterministic_model.py`** (CPU, tiny: B=2, C=1, H=W=16, L=3; UNet `hidden_channels=[4, 8]`):
- `forward` returns `(pred, target)`, pred shape `[B,1,16,16]`; extra `t`/`noise` kwargs ignored.
- `residual=True` output differs from `residual=False` and equals `base + network_out`.
- `sample(field_history, base=None)` → `[B,1,16,16]`; `return_field_history=True` → rolled history with prediction as last slice.
- `sample_trajectory(field_history, base=None, num_physical_steps=6, num_steps=50, stepper=object())` → `[B,1,16,16,6]`, first L frames equal seeded history (proves unused kwargs are ignored).

**New: `tests/test_trainers.py`**:
- Synthetic dataset yielding `{"base","target","field_history"}`; `DataLoader(batch_size=2)`.
- `DeterministicTrainer` + `DeterministicModel` + `AdamW` + `StepLR` (required — `BaseTrainer.__init__` calls `scheduler.get_last_lr()`) + `EmaGradientClipper` + `EarlyStopping(patience=2)`, `tracker=None`, `device="cpu"`, `num_epochs=2` → `train()` runs; loss on a fixed batch decreases.
- `StochasticInterpolantTrainer._prepare_batch` adds `"t"` `[B,1]` and `"noise"` matching `base.shape`.
- Backwards-compat guard: `scisi.training.trainer.Trainer is StochasticInterpolantTrainer`, `EarlyStopping` importable from `scisi.training.trainer`.

## Implementation order

1. Rebase on the Itô maps PR; verify `training/base_trainer.py` + the rewritten `training/trainer.py` (subclass + aliases) landed as specified in Part 2. (Only if implementing this plan standalone: create them per Part 2 first.)
2. `training/deterministic_trainer.py`.
3. `deterministic_models/` submodule.
4. `config/deterministic_navier_stokes.yaml` + `main_train.py` guard. (PR 1 also edits `main_train.py` — the `pre_trained_model` teacher block; the Aurora guard here is an independent edit nearby, apply on top.)
5. Tests. (The backwards-compat assertions in `tests/test_trainers.py` overlap with PR 1's golden test on purpose — keep both.)

## Verification

1. `pytest tests/` — new tests + existing `test_metrics.py` pass.
2. Import smoke: `python -c "from scisi.training.trainer import Trainer, EarlyStopping; from scisi.training.deterministic_trainer import DeterministicTrainer; from scisi.deterministic_models import DeterministicModel"`.
3. Hydra composition smoke: compose `deterministic_navier_stokes.yaml` and `hydra.utils.instantiate(cfg.model)` on CPU; call `forward` and `sample_trajectory` on random tensors.
4. If NS data is available locally: 1-epoch `main_train.py --config-name deterministic_navier_stokes trainer.num_epochs=1 trainer.device=cpu`, then `main_test.py` against the checkpoint to confirm the rollout end-to-end.
5. Regression: existing SI config still trains (1 epoch) through the refactored `Trainer` alias.
