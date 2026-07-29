# Reduced paper grid — 2026-07-01 (CURRENT)

The experiment set was **reduced and restructured** into a clean per-case results
tree. Source of truth for the layout, lineup, scenarios, steps, trajectories and
saving policy: **`results/README.md`**. Everything below this section is the older
(headline/multitraj/stepbench) history, kept for provenance.

**Reduced lineup (13 rows/case):** Ours (SI-SDE/DM-SDE/FM-ODE) × {jacfree, shared}
(the two likelihood-covariance modes, tagged by the tidy `variant` column) +
FlowDAS, FlowDAS+SURGE (`SURGE (FlowDAS)`), SDA, SDA+SURGE (`SURGE (SDA)`),
D-Flow SGLD, Guided FM (FIG) + EnKF, Particle filter (NS & analytical only — urban
is generative-only). Dropped: Guided FM (OT-ODE), standalone SURGE, LETKF, EnSF.

**Urban is SURGE-only (2026-07-25):** where a baseline has a SURGE variant, urban
runs only that variant — so bare `FlowDAS` / `SDA` are off the urban grid and
`SURGE (FlowDAS)` / `SURGE (SDA)` stand in. Both stay wired in the driver so
`run_urban_tuning.sh` can still name them. Urban also drops NS's
`energy_spec_rmse` (KE spectrum is ill-posed with buildings in the domain) on top
of the long-standing `kl_points` omission (no ground-truth posterior).

**Grid:** NS scenarios {superres 16/32, sparse 5%/1.5625%}; urban {sparse 5%/1.5625%};
analytical {joint}. Steps M ∈ {25,50,100,250} for NS/urban (analytical still
{50,100,250,500}). NS/urban: 5 test trajectories (`test_index=1..5`, one seed each),
`num_physical_steps=20` (5 history + 15 DA). Analytical: 5 seeds averaged in-run.

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
**Track / aggregate:** (per-case; run → aggregate → make figures)
```bash
.venv/bin/python paper_experiments/status.py                 # -> results/STATUS.md
.venv/bin/python paper_experiments/aggregate_analytical.py   # -> results/analytical/aggregated/all.csv
.venv/bin/python paper_experiments/aggregate_ns.py           # -> results/navier_stokes/aggregated/{all,per_step}.csv
.venv/bin/python paper_experiments/aggregate_urban.py        # -> results/urban/aggregated/{all,per_step}.csv
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
