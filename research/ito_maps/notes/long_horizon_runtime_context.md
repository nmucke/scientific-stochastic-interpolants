# Long-horizon execution context

The four-model stability queue is separate from the controlled latency experiments. Its protocol strings conservatively mark timing as diagnostic and mention CLEVRER training overlap during the initial rollout work. That common string was supplied to every family by the queue script; it should not be read as a claim that every individual family overlapped training.

Both map trajectories finished before the GPU-exclusive video evaluations concluded. Their diagnostic sampling times may include interference from concurrent video training. The queue shell was paused while the map child finished, preventing either solver reference from starting during video timing.

After both video audits passed, the verified watcher resumed the original NS queue at 2026-10-07 16:48:42 Europe/Amsterdam (UTC+2), as recorded in `../../video_forecasting/results/ns_resume_watch.log`. SI64 started then; FM64 follows serially. No other GPU workload was launched by this team after that resumption. CPU report generation, audits and manuscript checks continued concurrently.

This note clarifies execution history without modifying frozen per-run protocols or their hashes. Keep all stability-run wall times diagnostic. The manuscript's speedup claims use the separate controlled fluid and video latency measurements, not ratios of these long-run times.
