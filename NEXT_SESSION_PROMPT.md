# Next-session kickoff prompt

> ## ⏩ START HERE — live run as of 2026-06-29 evening
> **The NS step-count benchmark is RUNNING (detached, survives logout).** Script:
> `paper_experiments/run_ns_stepbench.sh`; live log: `<scratchpad>/run_ns_stepbench.log`.
> - **What it is:** 8 step-based methods × M∈{50,100,250,500} × 4 scenarios × 5 trajectories
>   = 640 runs. Metrics (RMSE/CRPS/spread-skill/KL-vs-EnKF-E1000) for all 640 in
>   `paper_experiments/results/stepbench/csv/` (one CSV per scenario×M×traj).
>   States + per-step metric curves + timing saved for **trajectory 1 only** →
>   128 `.npz` under `paper_experiments/results/stepbench/states/traj1/M*/` (~8.2 GB).
>   D-Flow SGLD is EXCLUDED (optimiser; infeasible at high M). 0 crashes so far.
> - **ETA caveat:** slower than hoped (~26–40 min/cell at M=50; SDA/SURGE ~9 min each;
>   shared GPU). M=500 cells are ~10× slower → total ≈ 120–150 GPU-h. **M=50/100 should
>   be done by morning; M=250 and M=500 will take SEVERAL MORE DAYS.** Don't assume it's finished.
> - **First check tomorrow:**
>   1. `pgrep -f run_ns_stepbench.sh` (alive?), `grep -c CRASHED <log>`, count CSVs (`ls .../stepbench/csv | wc -l`, target 80) and npz (`find .../stepbench/states -name '*.npz' | wc -l`, target 128).
>   2. When enough M-levels are done: `.venv/bin/python paper_experiments/make_ns_stepbench_figure.py`
>      → one 4-panel figure per scenario (RMSE|CRPS|spread-skill|KL vs M) in `manuscript/figures/navier_stokes/`.
>   3. Use these 5-traj-averaged numbers to fill the `\tbd` cells in the NS table (`manuscript/sections/results.tex`, `tab:ns_accuracy`).
> - **⚠ Watch FlowDAS:** first traj-1 super-res cell gave RMSE **0.784** (the fixed,
>   faithful guidance) vs the old buggy table's 0.411 — i.e. the fix may be a *regression*
>   on dense-obs NS. Confirm with the 5-traj average; if real, FlowDAS likely needs a
>   regime-appropriate guidance scale (as OT-ODE did). SDA, by contrast, improved (0.259
>   super-res vs old buggy 0.819 — fix transfers to NS).
> - **Still PENDING (GPU, after step-bench):** D-Flow + SURGE NS hyperparameter sweeps
>   (configs not locked); NS + urban **headline 5-traj runs** at M=50 (fill remaining
>   `\tbd` cells; the old FlowDAS/SDA NS numbers are stale post-fix); urban runs.

Copy the block below into a fresh session to continue the project. It is kept in
sync with `PROJECT_HANDOFF.md` (the authoritative state) — read that file first.

---

You are taking over the **observation-interpolant data assimilation** project
(branch `sync-with-paper`) at `/export/scratch1/ntm/postdoc/scientific-stochastic-interpolants`.
Use `.venv/bin/python`. **Read `PROJECT_HANDOFF.md` end-to-end before doing anything** —
it is the single source of truth (its "Session update — 2026-06-29" section is current).
`paper_experiments/RUN_STATUS.md` is the live run log. The paper lives in `manuscript/`
(compiles clean; `cd manuscript && latexmk -pdf main.tex`).

## Where things stand (2026-06-29)
- **Method lineup is FINAL.** Generative (9): Ours **SI-SDE / FM-ODE / FM-SDE**
  (the FM-SDE is shown as "FM-SDE (DM)"); **FlowDAS**; **Guided FM (FIG)**; **Guided
  FM (OT-ODE)**; **D-Flow SGLD**; **SDA**; **SURGE**. Classical (NS only): **EnKF**
  (E=1000 non-localized = ground-truth posterior / KL reference; E=64 localized =
  baseline), **LETKF**, **particle filter**, **ensemble score filter**. The legacy
  "Guided FM" (one-step DPS-on-flow) and "Guided diffusion" (DPS) are **DROPPED**
  (enum kept for back-compat, removed from the run registries + `make_tables`).
- **The multiplicative gain `G_τ` is dropped** (kept off-by-default in code). The
  ablation axis is the covariance (`inflated` / `inflated_shared` vs Jacobian-free).
- **The SDA/SURGE diffusion prior is built from the FM model** (`diffusion_from_fm:
  true`) because the trained diffusion checkpoint is weak. The FM prior is its own
  trained checkpoint.
- **Analytical case is DONE** — all 11 methods are self-contained closed-form
  samplers (`cases/analytical/samplers.py`); real KL numbers (SI-SDE 0.0009 …
  FlowDAS 0.080, OT-ODE 0.0021, D-Flow 0.079, SDA 0.019, SURGE 0.0021, EnKF 0.0012,
  PF 0.0030, FIG collapsed). Three figures under `manuscript/figures/analytical/`.
  Headline message: with faithful baselines essentially every method recovers the
  linear-Gaussian posterior (exactness check, not strawmen); methods separate on the
  nonlinear fluid cases.
- **Baseline audit (this session):** three real bugs found + fixed and validated on
  the analytical case — FlowDAS (non-faithful surrogate → paper Algorithm-2
  guidance, KL 0.299→0.080), D-Flow SGLD (RMSProp cold-start → Adam bias correction,
  0.624→0.079), SDA (over-normalized/under-weighted → `fm_coeff`, no step-norm,
  0.436→0.019). FIG is faithful but structurally collapses on full noisy obs.
- **Manuscript:** tables list the full lineup with NO per-row `\cite` (cited in prose
  + appendix "Method descriptions", `sections/appendix_methods.tex`). New bib:
  yan_fig_2025, ben-hamu_d-flow_2024, parikh_d-flow_2026, wei_surge_2026. Analytical
  table is filled; NS + urban tables are `\tbd` pending GPU runs.

## Prioritized remaining work (all need the GPU; details in `PROJECT_HANDOFF.md` Part B)
1. **Run the NS headline 5-trajectory grid** with the FIXED methods (E=64, M=50;
   E=1000 only for the ground-truth EnKF). This fills every `\tbd` in `tab:ns_accuracy`
   / the NS calibration table. The OLD NS baseline numbers are stale — re-run.
2. **Lock the D-Flow SGLD + SURGE NS hyperparameters** (their sweeps never completed;
   D-Flow pSGLD is ~17 min/config from gradient-checkpointing recompute), then include
   them in the grid.
3. **Run the urban headline grid** — generative-only, **sparse 5% + sparse 1.5625%
   only (no super-res)**, per-variable RMSE + split CRPS + spread-skill (no KL, no
   energy/enstrophy). `urban.yaml` is configured (big matched pair, `diffusion_from_fm`).
4. **Step-count benchmark** (M=100 / 500) — still deferred.
5. After numbers land: rebuild `all_results.csv`, regenerate tables, sync the
   manuscript `\tbd` cells, recompile.

## Golden rules
- `.venv/bin/python` (bare python has no torch). torch 2.9.1+cu128.
- Background jobs are killed at session boundaries — use `setsid nohup … & disown` +
  a logfile, and salvage per-cell `[NS] … {metrics}` lines from logs if a run dies
  before writing its CSV.
- The NS driver writes its CSV only at the END of the grid; `num_physical_steps >
  len_field_history` (use ≥ 8). Ablation flag is `+ablation=true`.
- Point each run at its OWN `results_file=` (concurrent writers corrupt one file).
- Commit/push only when the author asks.
- Keep `PROJECT_HANDOFF.md`, `paper_experiments/RUN_STATUS.md`, and this prompt current
  as work lands.
