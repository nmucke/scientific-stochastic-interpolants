# Training runtime context

All recorded stage times include the trainer's optimization and validation work and exclude model/data setup before trainer construction. They are operational wall times, not controlled comparisons of training efficiency.

KTH training completed before the long-horizon Navier–Stokes production runs began. CLEVRER's stochastic-interpolant teacher also completed before that overlap. Around14:11 local time (Europe/Amsterdam) on2026-10-07, the root started the additional user-requested Navier–Stokes stability study on the shared RTX3090. The final portion of the CLEVRER flow-matching teacher run and later map stages therefore share GPU resources with stability sampling. The amount of overlap varies across stages. Do not interpret their recorded wall-time differences as intrinsic differences in model training efficiency or compare them as isolated GPU benchmarks.

Independent CPU review of both completed CLEVRER teachers found:

| Teacher | Updates / epochs | Selected validation loss | Recorded stage seconds |
|---|---:|---:|---:|
| Föllmer-named stochastic interpolant | 20,000 / 200 | 0.004465026424441021 | 845.8103458529804 |
| Flow matching | 20,000 / 200 | 0.004911090203677304 | 921.9039655399974 |

These validation losses use different regression targets and are not cross-model forecast-quality scores. Both checkpoints strictly load on CPU, every stored model tensor is finite, and `last.pth` histories match the JSON histories exactly. Both use seed-zero random initialization,2,917,498 parameters, RGB inputs, two observed frames,240,000 training windows,15,000 validation windows, and a final optimizer learning rate2e-5. The official train/validation split and full CLEVRER manifest are preserved. No teacher configuration, checkpoint, or saved metadata was altered during review. Lightweight archives will be created only by the freezer after all six CLEVRER training stages finish.

For the planned video evaluation, the root will finish or pause NS GPU sampling first. Reported inference-speed comparisons must use that GPU-exclusive evaluation phase. Long-horizon NS runtime metadata separately labels its concurrent workload and must not be mixed with the controlled video inference timings.
