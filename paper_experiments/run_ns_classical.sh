#!/bin/bash
# =============================================================================
# Navier--Stokes CLASSICAL group -- EnKF + Particle filter  (2026-07-31)
#
# WHY THIS EXISTS. `Particle filter` has ZERO rows anywhere in
# results/navier_stokes/ -- the classical group has never been run. The EnKF rows
# that DO exist (180 of them) come from `ref_*__traj<N>.csv`, i.e. the E=1000
# reference runs of run_ns_reference.sh, NOT from the grid; they carry no Particle
# filter and are not the E=64 grid cells the paper table needs. status.py therefore
# reports `MISSING Particle filter` for all four scenarios, and the classical row of
# the NS lineup is empty.
#
# WHAT IT RUNS. Nothing new -- it is a thin, documented wrapper around
# `run_ns_grid.sh` with `GRPS=classical`. All the logic (localization, KL reference
# wiring, states policy, skip-if-exists) stays in the grid script, so this cannot
# drift from how the rest of the grid is produced. Specifically the grid's classical
# block (run_ns_grid.sh:203-211) runs, per (traj, scenario):
#
#     +ns_methods=["EnKF","Particle filter"]  num_steps=50   (M is INERT here:
#         classical filters propagate the ensemble with the TRUE solver, so there is
#         no sampler-step axis -- one cell per (traj, scenario), no M sweep)
#     +enkf_localization_radius=20   for the SPARSE scenarios only
#         (super-res stays non-localized -- see the grid script)
#     +kl_reference_states=results/navier_stokes/reference/traj<N>/gt
#         verified present: 4 files for each of traj 10..14
#
# GRID: 4 scenarios x 5 trajectories (10..14) = 20 cells ->
#     results/navier_stokes/metrics/<scen>__traj<N>__classical.csv
# None of these exist yet, so all 20 run. The script is skip-if-exists, so an
# interrupted run resumes by relaunching this command verbatim.
#
# COST: not measured. The E=1000 EnKF reference cells took ~26 min each; at E=64
# this should be far cheaper, but the Particle filter has never been timed here at
# all. Expect a few minutes per cell plus ~3 min startup/JIT -- order 2-3 h for all
# 20 -- and treat that as a guess until the first cell lands.
#
# Launch detached:
#   setsid nohup bash paper_experiments/run_ns_classical.sh \
#     >run_ns_classical.log 2>&1 & disown
# Track:  .venv/bin/python paper_experiments/status.py --case navier_stokes
#
# Env overrides pass straight through to run_ns_grid.sh: TRAJ, SCENARIOS(|-sep),
# E, NP, DEVICE, REQUIRE_W, SAVE_TRAJ, DIV_GUARD, ROOT.
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants

# The ONLY thing this wrapper decides. Everything else is the grid's own defaults
# (TRAJ=10..14, E=64, NP=20, DEVICE=cuda, SAVE_TRAJ=11, DIV_GUARD=10.0).
export GRPS=classical

echo "[ns-classical] $(date '+%F %T') launching run_ns_grid.sh with GRPS=$GRPS"
echo "[ns-classical] expecting 20 cells: 4 scenarios x traj ${TRAJ:-10 11 12 13 14}"
exec bash paper_experiments/run_ns_grid.sh
