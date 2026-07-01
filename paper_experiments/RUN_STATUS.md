# Reduced paper grid — 2026-07-01 (CURRENT)

The experiment set was **reduced and restructured** into a clean per-case results
tree. Source of truth for the layout, lineup, scenarios, steps, trajectories and
saving policy: **`results/README.md`**. Everything below this section is the older
(headline/multitraj/stepbench) history, kept for provenance.

**Reduced lineup (13 rows/case):** Ours (SI-SDE/DM-SDE/FM-ODE) × {jacfree, shared}
(the two likelihood-covariance modes, tagged by the tidy `variant` column) +
FlowDAS, FlowDAS+SURGE (`SURGE (FlowDAS)`), SDA, SDA+SURGE (`SURGE (SDA)`),
D-Flow SGLD + EnKF, Particle filter (NS & analytical only — urban is
generative-only). Dropped: Guided FM (FIG/OT-ODE), standalone SURGE, LETKF, EnSF.

**Grid:** NS scenarios {superres 16/32, sparse 5%/1.5625%}; urban {sparse 5%/1.5625%};
analytical {joint}. Steps M ∈ {50,100,250,500} (generative). NS/urban: 5 test
trajectories (`test_index=1..5`, one seed each), `num_physical_steps=20` (5 history
+ 15 DA). Analytical: 5 seeds averaged in-run.

**Saving:** raw states for trajectory 1 ONLY (all methods, **both** Ours modes —
`variant` is in the filename); per-step metric curves + timings (seconds/NFE) for
EVERY trajectory (`results/<case>/per_step/`).

**Run it (each is env-overridable — see the script headers):**
```bash
# analytical (CPU, minutes):
bash paper_experiments/run_analytical_grid.sh
# NS KL reference (E=1000 EnKF, GPU) then the NS grid:
setsid nohup bash paper_experiments/run_ns_reference.sh >run_ns_reference.log 2>&1 & disown
setsid nohup bash paper_experiments/run_ns_grid.sh      >run_ns_grid.log      2>&1 & disown
# urban grid (GPU):
setsid nohup bash paper_experiments/run_urban_grid.sh   >run_urban_grid.log   2>&1 & disown
```
**Track / aggregate:**
```bash
.venv/bin/python paper_experiments/status.py                 # -> results/STATUS.md
.venv/bin/python paper_experiments/aggregate_grid.py         # -> results/<case>/aggregated/all.csv
```

**Reduced-grid smoke-test (CPU, no GPU/weights) — proven operational 2026-07-01:**
```bash
# analytical full lineup:
STEPS="50 100" bash paper_experiments/run_analytical_grid.sh
# NS / urban tiny wiring check (random weights):
DEVICE=cpu REQUIRE_W=false E=2 NP=7 STEPS=2 TRAJ=1 SCENARIOS="sparse 5%" \
  bash paper_experiments/run_ns_grid.sh          # generative groups
```
---

_Pre-restructure history (stepbench/headline/multitraj notes, 2026-06-26…30) moved to_ `archive/RUN_STATUS_history.md`_._
