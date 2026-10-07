# Resumable long Navier–Stokes forecast runner

`experiments/long_horizon_ns.py` reuses the existing trained models, dataset loader and affine preprocessing. It never retrains, clips, replaces, or reinitializes generated states. Each member feeds its own previous predictions back into its five-frame conditioning window. All four selected seed-zero models start from held-out trajectory180, observed frames0–4, with eight ensemble members and the same sampling seed20261007. This common seed is not a claim of identical Brownian paths across models or solver budgets.

The primary requested horizon is4,096 stored snapshot intervals. The chosen existing checkpoints/budgets are:

| Output alias | Checkpoint under `checkpoints/stochastic_navier_stokes/` | Network evaluations per physical step |
|---|---|---:|
| `flow_map` | `flow_rollout_seed0` | 1 |
| `ito_map` | `ito_rollout_seed0` | 8 |
| `stochastic_interpolant` | `stochastic_interpolant` | 64 |
| `flow_matching` | `flow_matching` | 64 |

A model's output directory contains an immutable `protocol.json`, streamed `diagnostics.jsonl`, a checkpoint summary `diagnostics.json`, latest `restart.pt`, a retained `restart_001024.pt` milestone, and sparse raw normalized float32 states in `raw_states/lead_<N>.npy`. Raw snapshots have shape `[initial_cases,ensemble,channel,height,width]`. Save every first100 frames, every16th later frame, and the invocation's final or failing frame. At4,096 leads this is350 raw snapshots, approximately175MiB per E8 scalar128² model. The complete scalar diagnostic series is retained at every lead; this intentionally does not save every raw field throughout the entire run.

The latest restart atomically stores generated history, Python/NumPy/torch CPU/CUDA random states, committed diagnostics, raw-state hashes, and the protocol hash. Resume verifies all protocol/input/checkpoint provenance and saved raw-state hashes. It can increase the requested horizon while preserving the original protocol. An unfinished invocation can stop at a nominated lead using `--stop-after-lead`; resume starts after its committed lead. Crash recovery rolls the diagnostic journal back only to the last committed checkpoint; any existing uncommitted raw snapshots must agree exactly with deterministic regeneration. A model that has already produced nonfinite output cannot silently resume through that failure.

Finite amplitude or enstrophy excursions do **not** stop sampling. First crossings of10,100 and1,000 times initial mean member enstrophy are recorded as descriptive flags. The only automatic dynamics stop is nonfinite model output, whose raw values and affected member indices are preserved. Neither a finite run nor absence of threshold crossings establishes invariant-measure correctness.

Diagnostics use the original archive's vorticity units after the inverse affine transformation and compute spatial moments in float64. Enstrophy is `0.5 * mean_grid(vorticity**2)` for each member, with its ensemble mean/quantiles and the separately recorded enstrophy of the ensemble-mean field. RMS spread is the square root of spatially averaged unbiased member variance. Where paired truth exists, spread–skill is `sqrt((E+1)/E) * RMS_spread / RMSE`. This finite-ensemble correction is explicit; the raw spread is also retained.

Only95 paired future frames exist from origin5 in the stored100-frame trajectories. RMSE, spread–skill and truth enstrophy become null after lead95; no generated PDE continuation is presented as matching truth. The complete timestamp/downsampling audit in `results/long_rollout/data_reference_audit.json` verifies the archive correspondence and mean stored spacing0.4998999701605903. `--time-audit` records this audit's hash and derives nominal elapsed time as `lead * mean_spacing`, relative to the last observed frame at index4. This is not an absolute timestamp; extrapolation beyond stored timestamps and unverified PDE parameters remain explicitly identified. Preserve the timestamp audit unchanged after production protocols reference it.

Generation runtimes measure the model sampler and synchronization only. These runs may share the RTX3090 with video training, and their timings are diagnostic wall times rather than controlled model-speed comparisons. No CPU-versus-GPU speedup should be inferred from them.

Example production invocation (root schedules GPU jobs):

```bash
.venv/bin/python research/ito_maps/experiments/long_horizon_ns.py \
  --checkpoint-dir checkpoints/stochastic_navier_stokes/flow_rollout_seed0 \
  --output research/ito_maps/results/long_horizon/flow_map \
  --steps 1 --horizon 4096 --ensemble 8 --trajectories 180 --origin 5 \
  --device cuda --num-threads 2 \
  --time-audit research/ito_maps/results/long_rollout/data_reference_audit.json \
  --concurrent-workload 'CLEVRER training on shared RTX3090; diagnostic runtime only'
```

Use the same command plus `--resume` to continue. A first16-lead check can add `--stop-after-lead 16` while keeping the originally planned `--horizon 4096`. The512/1,024/4,096 horizon milestones require no retraining; each interval is one autonomous next-frame forecast.

Five focused CPU tests passed: enstrophy/spread identity, truthful skill cutoff, continued finite excursions and sparse raw storage, bit-exact restarted forecasts across all three RNG streams, horizon extension, and nonfinite failure preservation. Actual E8 checkpoint CPU feasibility checks with two threads measured approximately1.469seconds/lead for FlowMap1 and11.940seconds/lead for ItôMap8. These are short uncontrolled feasibility profiles, not performance benchmarks; records are in `results/long_horizon/cpu_setup_profile.json`. Their roughly1.67-hour and13.6-hour projected4,096-lead costs favor GPU execution. No production GPU jobs were launched by the implementation agent.

## Active queue scheduling (2026-10-07)

Root session32086 owns the production queue. FlowMap4096 completed. The queue shell PID2126968 is temporarily SIGSTOPped while its Itô child PID2130505 continues to4096. After that child finishes, no NS GPU work runs until root sends `kill -CONT 2126968`; this is planned after exclusive-GPU video evaluation. Both4096-step solver baselines remain required. This scheduler pause changes no generated trajectory or RNG state.
