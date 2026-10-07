# Flow maps and implementation design

Research date: 2026-10-07. Status: research/design only; no implementation changes in this task.

## Sources and distinction between mathematical objects

Primary source: Boffi, Albergo, Vanden-Eijnden, *Flow map matching with stochastic interpolants: A mathematical framework for consistency models*, [arXiv:2406.07507v2](https://arxiv.org/html/2406.07507v2), revised 2025-06-03. Local HTML and extracted text are in `research/ito_maps/sources/2406.07507v2.*`. Use v2 equation numbering.

A deterministic flow map transports an initial random state along an ODE. It remains a probabilistic forecaster when Gaussian initial states are sampled conditional on a fixed physical history. Starting that deterministic map from a fixed historical field would instead collapse the ensemble. This matters because the existing Föllmer model starts from the most recent field, while the existing flow matching model starts from Gaussian noise.

For fixed conditioning history `c`, define `F(s,t,x;c)` and enforce

    F(s,t,x;c) = x + (t-s) v_theta(x,s,t;c).

This supplies the diagonal identity exactly. Sampling uses compositions over a user-selected pseudo-time grid; physical forecasting then shifts the conditioning history and starts a fresh generative process.

## Correct training objectives

The cited paper's Lagrangian distillation loss (Eq. 3.11) is

    E || partial_t F(s,t,I_s;c) - b_teacher(t,F(s,t,I_s;c);c) ||².

Its direct flow map matching objective (Eq. 3.17) uses

    y = F(t,s,I_t;c)
    E || partial_t F(s,t,y;c) - dot I_t ||²
      + E || F(s,t,y;c) - I_t ||².

The partial derivative holds `y` fixed as a time argument; parameter gradients still pass through the inverse evaluation. Direct training needs reverse maps and symmetric time-pair coverage. Progressive map distillation (Eq. 3.23) matches a student jump to compositions of a fixed teacher. Naively substituting noisy interpolant velocities into Eulerian distillation adds a map-dependent conditional variance penalty (Sec. 3.6).

Implementation implications derived for this repository:

- Expose objective names explicitly (`direct`, `lagrangian_distillation`, optionally `trajectory_distillation`) rather than calling all supervised endpoint regression “FMM.”
- For direct FMM with `LinearDeterministicInterpolation`, draw Gaussian `z`, calculate `I_t=(1-t)z+t target` and `dot I_t=target-z`; the physical historical base is conditioning only.
- A PyTorch JVP closure over only the outer target time gives the required partial derivative. Construct `y` outside that closure. Do not detach `y` from parameters for exact Eq. 3.17, and do not differentiate `y` with respect to the JVP time variable.
- The two times should be independent uniform draws for full-domain direct FMM. Restricting to forward pairs while requiring an untrained inverse is not equivalent.
- Freeze teacher parameters for distillation but retain gradients of teacher outputs with respect to mapped state. `torch.no_grad()` around the entire teacher call changes the objective gradient for Lagrangian distillation.
- A numerical teacher endpoint objective can be useful operationally and avoids mixed second derivatives, but is a separate, simulation-based distillation route and must be labeled as such in the manuscript and logs.
- A diagonal regression auxiliary term can warm-start drift behavior. Annealing maximum jump length is a practical option; report it, and make training coverage match sampled jump lengths.
- Never regress a deterministic map from a fixed historical field directly to random physical next states and expect calibrated distributions: the conditional MSE optimum is the mean.

## Existing repository contracts

`src/scisi/models/base_model.py` implements common mask handling, batched conditional replication, pseudo-time integration, CPU outputs, sliding histories, and trajectories. `FollmerStochasticInterpolant` and `FlowMatchingModel` add interpolation losses and choose SDE/ODE steppers. Both expose `.model` as an alias of `.drift_model`; downstream architecture/checkpoint code relies on that convention.

Shape contract:

- field/state/base/target: `[B,C,H,W]`;
- history: `[B,C,H,W,L]`;
- conditioning field: `[B,C_cond,H,W]`;
- scalar parameters: `[B,D]`;
- pseudo-time: `[B,1]`;
- trajectories: `[B,C,H,W,T]`.

`Trainer` owns batch transfer, random pseudo-time/noise, optimizer/scheduler/clipping, epochs, validation, tracking and checkpoints. Its model forward returns `(prediction, target)` for simple drift regression. A map loss has multiple components and often derivatives, so this is the natural narrow extension point.

`StochasticNavierStokesDataset`, loaders and `Preprocesser` already yield exactly the conditioning and target tensors needed. Reuse these rather than introducing a map-specific dataset. Existing split is trajectories 0:160 train, 160:180 validation, 180:200 test. Existing normalization uses std 3.09969. Establish whether this constant was derived on training data; do not silently change normalization across model comparisons.

## Minimum implementation proposal

### BaseMapModel

Implement a sibling base abstraction sharing/inheriting `BaseModel` utility behavior, with an explicit `map(x,s,t,field_history,field_cond,pars_cond,...)` primitive. Avoid invoking numerical ODE/SDE steppers from the map sampling loop. A map jump is one backbone evaluation (unless explicitly documented otherwise).

Reuse `_prepare_batch`, masks, physical history/trajectory convention and CPU output convention. Override pseudo-time sampling/integration and physical trajectory dispatch: the existing `_sample_trajectory` passes `stepper` and `diffusion_term` into `sample`, which has different meaning for a map. A shared implementation should preserve signatures for callers without silently accepting nonsensical solver settings.

Validation needs positive `num_steps`, endpoint-exact grids, correct broadcasting of singleton time tensors, base migration to model device/dtype, and clear shape errors. Preserve masks after every map step. Prefer a registered mask buffer if doing so does not break existing checkpoints.

Provide `.model` alias for the map backbone. Keep trainable map parameters and frozen teacher ownership explicit; checkpointing a teacher registered under the map can otherwise duplicate a large teacher and inflate apparent parameter counts.

### FlowMapModel

Use Gaussian initialization like `FlowMatchingModel`; accept explicit `base` for reproducible sampling and never replace it silently. Define `map=x+(t-s)*map_model(...,cond=cat([s,t],-1),...)`. Reuse `LinearDeterministicInterpolation`. Support at least direct FMM or Lagrangian distillation with accurate names and documentation; direct FMM is the cleanest interpretation of the requested standalone model.

The existing `UNet` already accepts `cond_dim=2` through its general MLP encoder. No backbone change is required for a minimal implementation. This differs from the paper's separate Fourier time embeddings; it is a transparent architecture choice. Any warm start from a single-time UNet requires explicit weight adaptation for the changed encoder, not `strict=False` with unreported failures.

### FollmerItoMapModel

Keep historical-state initialization and the exact interpolation/drift conventions of `FollmerStochasticInterpolant`. Replace differential integration with a learned state-and-path map. The path features and known stochastic increment must belong to the same Brownian realization. Independent noise at each neural evaluation is incompatible with same-path composition.

For time-dependent additive diffusion, the known increment is `integral_s^t gamma(r)dW_r`, not automatically `gamma(s)*(W_t-W_s)`. The latter is a numerical approximation and should not be passed off as exact any-step noise. For the current affine `gamma(r)=gamma_multiplier*(1-r)`, exact weighted Gaussian increments and consistent subdivision can be derived analytically, or piecewise path quadrature can be documented as an approximation.

A map backbone should receive Brownian/path channels separately from state/history; reuse the existing `field_cond` route or a small input adapter. Do not redefine `UNet.in_channels` to `(K+1)C` without inspecting `get_init_conv`: input channels are also multiplied by history length. Combining the Brownian and existing conditioning field channels is possible while preserving the existing UNet.

The exact Itô-map objective and path representation must follow the companion derivation task. Endpoint noise alone generally does not determine nonlinear SDE trajectories; evaluate feature-resolution convergence and separate this approximation from network error.

### MapTrainer

Subclass `Trainer` and reuse epoch lifecycle, optimizer and clipping. Override `_prepare_batch` to sample time pairs plus path information and `_compute_loss` to combine named components. Prefer `model.loss(...) -> scalar or named components` over overloading `(prediction,target)` with unrelated tensors. Keep scalar total loss and component logs.

A robust offline checkpoint path is necessary for experiments. Base Trainer currently assumes a tracker when saving; map experiments need persisted resolved config, best/last model state, optimizer/scheduler state, seed and run metadata without external services. Validation must compute required derivatives; a JVP can work under `no_grad` via forward AD, while reverse-over-reverse JVP cannot be assumed to work. Set the model to `eval` and explicitly select derivative semantics.

Configs should mirror the existing stochastic Navier–Stokes configuration and reuse normalization/splits/U-Net. New train/test entrypoints should use Hydra rather than hard-coded checkpoint globals, and parameterize checkpoint path, seed, step counts, ensemble size, horizon and output directory.

## Compatibility hazards found in the baseline

1. `Trainer.__init__` unconditionally calls `scheduler.get_last_lr`; `_update_scheduler` also assumes a scheduler. `_print_info` unconditionally uses tracker fields despite tracker being optional.
2. `Trainer._compute_val_loss` wraps all evaluation in `no_grad`; derivative map objectives require testing this explicitly.
3. Existing mixed precision clips scaled gradients without calling `scaler.unscale_` first. Map derivative losses should initially use full precision. CPU/device-index strings also differ from AMP device type strings.
4. The existing `EarlyStopping` uses `val_loss < best_loss + delta`, allowing slight degradations to count as improvement. This affects fair checkpoint selection; use a documented stricter criterion for new experiments or fix it consistently.
5. `StochasticNavierStokesDataset._prepare_train_sample` breaks when `preprocesser=None` because `sample` remains a tensor then is indexed by strings. Configuring preprocessing avoids it; a narrow bug fix is reasonable.
6. Its window count is `T-L-1` instead of all `T-L` valid target windows. Preserving this gives baseline comparability; changing it should be shared across comparisons.
7. UNet dropout must be zero during derivative consistency training, or dropout draws must be deliberately shared. Separate stochastic masks corrupt map composition/JVP targets.
8. GPU fused scaled-dot-product attention may not support forward-mode AD or second derivatives. Use math attention backend for derivative training, test a real U-Net gradient step, and use the same attention inference settings for timing.
9. `BaseModel.to` is a device-only override, unlike standard `nn.Module.to`, and mask is not a registered buffer. Reusing it blindly can break dtype conversions.
10. `BaseModel.sample` returns CPU arrays, so synchronize CUDA before measuring compute and separately report transfer overhead. Counting pseudo-time steps alone is not a compute comparison.
11. Existing FM and Föllmer configs use different widths (FM 16/32/64/128 versus Föllmer 8/16/32/64), so their pretrained runtimes are not architecture-matched evidence.
12. Current stochastic interpolation's `forward_diff` is the Itô drift target `alpha' base + beta' target + gamma' sqrt(t) z`, deliberately excluding the derivative of `sqrt(t)`. Do not change it to the ordinary path derivative when extending to Itô maps.

## Validation priorities

- Identity at equal times, shape/batch replication, conditional forwarding, history shifting and distinct stochastic versus Gaussian initialization.
- Analytic constant and affine ODE map/JVP losses; verify parameter gradients through inverse composition.
- Additive SDE with known constant or affine drift: Brownian coupling, increment variances, composition consistency, zero-diffusion reduction and ensemble spread.
- Real Navier–Stokes data loader and one optimizer step through the actual configured U-Net. Check that checkpoint reload preserves fixed-seed outputs.
- Pilot quality/cost curves over 1/2/4/8/16 map steps versus matched Föllmer and FM numerical solvers at larger step counts. Report wall time, NFE, RMSE of ensemble mean, CRPS/energy score, coverage/spread and spectra. Include physical rollout horizons, not only one-step images.

No experiments have been run as part of this research note. Faster sampling at comparable quality is the hypothesis, not an established outcome.

## Agreed practical route: progressive self-distillation

After discussion with lead researcher, the preferred initial training route is PSD for both map classes, avoiding JVP/attention second-derivative support and making fuller use of existing trained drift networks. This is supported by flow-map progressive distillation and the Itô paper's Algorithm 3. For `s<u<t`, write

    r = (u-s)/(t-s)
    g1 = G_teacher(s,u,x,path)
    x_u = x + (u-s)*g1 + M_u-M_s
    g2 = G_teacher(u,t,x_u,path)
    loss_psd = ||G_student(s,t,x,path) - stopgrad(r*g1+(1-r)*g2)||².

Use one shared martingale path in every term. The whole right hand side is a target with stopped gradient, including `x_u`. For deterministic flow maps set all martingale increments and path features to zero. Teacher can initially mean the current network evaluated without gradients; an EMA target is an optional, documented practical variant. Add the diagonal loss against the appropriate interpolant drift target or pretrained drift. PSD alone does not identify the right dynamics.

Mathematical audit: the state-map residual is `(t-s)` times the average-velocity residual. Thus their squared losses differ by `(t-s)^2`. The Itô paper describes them as equivalent; they share exact zeros, but unweighted velocity PSD is a different time weighting. We should report the chosen velocity loss accurately instead of asserting equality of unweighted loss values.

Suggested exact warm start architecture:

- Wrap the original single-time `UNet` in `MapConditionedNetwork`; keep its channels, history handling and Fourier encoder unchanged.
- Replace its time encoder by a `TwoTimeEncoder` containing the original encoder and an additional gap branch. For input `[s,t]`, output `original(s)+(t-s)*gap_mlp(t-s)`, with zero final layer initially. This starts with the teacher's time embedding and preserves the diagonal embedding exactly.
- For stochastic path inputs, wrap its initial block by `original_init(x,history,physical_field_cond)+path_projection(path_features)`, with a zero-initialized 1x1 projection into the first hidden width. The wrapper can route concatenated physical/path channels via the existing `field_cond` argument and separate them internally. On diagonal calls path features are zero.
- Do not simply enable `field_cond_channels` in a previously unconditioned UNet: `InitConvWithFieldCondAndHistory` inserts a further history-processing block and changes the architecture beyond a padded input convolution.
- Explicit teacher transplant remaps time-encoder and initial-block keys into the wrapped modules, checks all copied shapes, and asserts exact diagonal output agreement in evaluation mode. Remaining backbone parameters copy unchanged.
- Disable dropout for the map training configs so conditional maps and teacher targets represent deterministic functions of state, times and path. This is especially important for self-distillation.

These wrappers are implementation choices, not claims that the papers used this precise architecture. They preserve trained physical forecasting capability while introducing only interval and Brownian conditioning.
