# Map API compatibility and supported settings

Audit date: 2026-10-07, after freezing `results/frozen_protocol.json`. The audit led to one explicitly authorized safety guard: MapTrainer now rejects nontrivial masks before initialization. The frozen unmasked NS numerical behavior, objectives and evaluation remain unchanged.

## Supported and used in the frozen experiments

The map classes support normalized dense 2D state tensors `[B,C,H,W]` and histories `[B,C,H,W,L]`, with fixed history length matching the backbone. The Navier–Stokes configuration uses one vorticity channel, five history fields, no static or parameter conditioning, and the scalar mask1.

`sample(field_history=..., base=..., batch_size=..., num_steps=..., return_field_history=...)` returns CPU samples `[B,C,H,W]`, optionally accompanied by an updated CPU history. `batch_size` is the total requested ensemble batch, not an extra ensemble dimension. A singleton history/base/condition broadcasts to that batch; an existing batch must match it. The model casts inputs to its parameter device/dtype. It does not implicitly call `eval()`: callers should load/evaluate the model in evaluation mode, as the provided entry points do.

`FollmerItoMapModel` defaults to the latest observed field as its base. A fresh weighted-Brownian path supplies stochasticity for each forecast and ensemble member. `FlowMapModel` defaults to an independent Gaussian field as its base; history remains conditional information. An explicit base overrides the default for that forecast.

`sample_trajectory` returns `[B,C,H,W,T]` with the supplied history included. Its `num_physical_steps=T` is total output length, not the count of newly generated forecasts. It shifts history after each new field. A supplied explicit base is used only at the first forecast; subsequent Itô forecasts start at the latest generated field and subsequent flow forecasts draw fresh Gaussian bases. The generator advances across forecasts rather than resetting. The provided test evaluator performs equivalent repeated `sample` calls and uses its own `forecast_horizon` to count new lead steps.

The existing dataset/window preprocessing is reused. Inputs and targets must already be normalized consistently; models do not own a preprocessor and do not automatically invert normalization. Returned samples and trajectory histories remain in model coordinates. The frozen NS configuration uses identical base/target scale3.09969, so copying generated targets into history is consistent.

## Shared concepts that are intentionally different APIs

- Map `forward(x,s,t,...)` evaluates a finite-time map. Solver model `forward(base,target,t,noise,...)` returns a training prediction/target pair. Use `MapTrainer`, not ordinary `Trainer`, to train maps.
- `model` exposes the trainable network in both families, but the map network consumes two-time conditioning and optional path channels. Teacher backbone parameters are converted through `MapConditionedNetwork.load_from_teacher`; solver and map state dictionaries are not interchangeable.
- `num_steps` counts neural map calls in map sampling and numerical integration steps in the solver. Equal counts do not guarantee equal runtime. Map default is one step, versus the solver default100.
- Existing FSI methods place `base` before `field_history` positionally, whereas `BaseMapModel.sample` places history first. Keyword calls are the compatible usage; existing positional calls may need adjustment.
- Maps do not expose solver controls `stepper`, arbitrary `diffusion_term`, `ode_drift`, or solver-specific `.drift`/`.score` interfaces. Existing posterior/guidance wrappers that require those methods cannot consume maps unchanged. Arbitrary post-training diffusion changes are not supported mathematically by the learned finite-time map.
- `BaseMapModel` is a separate `nn.Module` abstraction, not a subclass of `BaseModel`. Code using strict `isinstance(model, BaseModel)` tests needs a common protocol or explicit map handling. The new entry points already distinguish the relevant models.
- Low-level `map/average_velocity` calls expect caller-managed tensor batches/devices and, for nonzero Itô intervals, a coherent `WeightedBrownianPath`. Convenience `sample` constructs and expands these inputs. A manually supplied path must match state shape, device, diffusion amplitude and desired coupling.

## Conditioning that can be used with explicit shapes

Single-step `sample` passes static spatial fields `[B,C_cond,H,W]` and parameter vectors `[B,D]` through the existing backbone interfaces; singleton condition batches broadcast to the state batch. The configured backbone must have matching conditioning dimensions. Ordinary conditioning channels are kept separate from the appended path channels by the adapter.

For `sample_trajectory`, spatial conditioning may be static `[B,C_cond,H,W]` or a sequence `[B,C_cond,H,W,T_future]`; the latter selects index i for the ith new forecast. It must contain every requested future condition. This ordering begins at the first generated forecast, not at the beginning of the supplied history. No validation of sequence length is currently added beyond indexing behavior.

Parameter conditioning during trajectories inherits the legacy expression `pars_cond[:, i:i+1]`. It supports the narrow convention of one scalar parameter per future time, stored as `[B,T_future]`, giving `[B,1]` to a backbone configured for one parameter. It **does not support general static `[B,D]` parameter vectors or `[B,T_future,D]` parameter trajectories correctly**: the former is truncated/changed each step, and the latter retains an extra singleton time dimension. This is a real API limitation shared with the existing solver trajectory helper. Use repeated single-step calls with explicit correctly shaped `pars_cond` until a general time-axis convention is selected and tested. It has no effect on the frozen unconditioned NS experiments.

## Masks require additional methodological work

Map sampling multiplies each learned drift-plus-noise increment by the loaded mask, preserving the incoming value where the mask is zero. The mask is a registered buffer and moves with the module. However, the legacy Euler and Euler–Maruyama steppers currently ignore the mask they receive, and `MapTrainer`'s interpolant/teacher/composition targets are unmasked.

Therefore a nontrivial mask changes map sampling dynamics without a matching training construction, and does not reproduce the legacy teacher's mask behavior. `MapTrainer` now raises a clear `ValueError` for any mask that is not identically one, preventing silent masked-training misuse. Sampling retains its existing mask behavior. Do not claim general masked-domain support from the presence of `mask_path` alone. Establish masked interpolation, source noise and teacher dynamics consistently before using it for a scientific experiment. The frozen NS masks are identically one, so this mismatch is inactive. No numerical behavior has been changed after freeze.

## Other boundaries

`FollmerItoMapModel` currently accepts the repository Wiener-process interpolants with gamma(t)=a(1−t). A different time schedule, state-dependent diffusion, or colored/spatially correlated physical noise is not automatically a supported Brownian driver. The generative Brownian noise is independent per field coordinate; it is distinct from the physical NS forcing.

The convenience sampling and trajectory interfaces are written for 2D fields. Although the Brownian utility supports more general state ranks, the UNet, conditioning adapter, and trainer reshape paths are not advertised as generic arbitrary-dimensional models. Alternate backbones must expose the expected conditional encoder/input block or provide their own compatible map-network implementation.

If base and target normalization differ, or targets represent tendencies rather than full next states, appending the target-space output directly to a base-space history is generally incorrect. A caller must reconstruct the physical next state and transform it into history coordinates between forecasts. This is an existing general pipeline limitation and is inactive for the equal-normalization full-state NS setup.

Epoch-boundary resume restores model/EMA/optimizer/scheduler/clipping/RNG state, but exact replay assumes deterministic data transforms or workers recreated with deterministic seeds. Persistent DataLoader workers' private RNG states are not serialized. These limits are documented in the trainer.
