# Case 4 — Stochastic Lorenz 96

Learned SI and FM priors with observation-interpolant posterior sampling, against
EnKF and a converged particle filter that serves as the reference posterior.

## System

Lorenz 96 on a ring of 16 sites, F = 8, RK4 with dt = 0.01. Between observations
(0.2 time units) the state moves by the deterministic flow plus additive
N(0, I) noise, so the transition kernel is exactly N(Phi(x), I). Every fourth site is
observed with variance 4. All settings live in `configs/case/lorenz96.yaml`.

## Protocol

`pf_gate.py` decides whether the particle filter can serve as ground truth;
`simulate_data.py` and `train.py` build the priors; `driver.py` computes references,
tunes on validation sequences and scores every method on the same held-out sequences;
`aggregate_lorenz96.py` and `make_lorenz96_figures.py` reduce and plot.
`run_lorenz96_grid.sh` runs the final comparison.

The current status, the evidence and the exact commands are in `GPU_HANDOFF.md`.
