# Stage 1 — Agent D (Experiments & reproducibility) raw report

Status: received, NOT yet verified by orchestrator (Stage 3 pending).

---

**Location**: `manuscript/sections/appendix_ns_case.tex` §H.3, Table 4 (`tab:ablation`), "Run in progress; cells (--) are placeholders"
**Severity**: blocking
**Problem**: Appendix H.3 promises five ablation axes (covariance jacfree/shared incl. refresh cadence k and damping λ; g_τ sweep; M sweep; E sweep) and every data cell is `--`. What the results tree already contains: (a) the **M sweep** (M∈{25,50,100,250}, jacfree + shared_jac10, all 4 scenarios, 5 trajectories) is fully present in `paper_experiments/results/navier_stokes/aggregated/all.csv` and already plotted in Figs. 11–13; (b) the **covariance axis** is Table 1 itself; (c) a **partial λ sweep** exists (`results/navier_stokes/tuning_lambda_check/`, λ∈{0.9,0.95,1.0}, k=10, M=25, scenarios 16² and sparse 5% only); (d) the **k sweep is not run** — `paper_experiments/run_ns_ablation.sh` is written (k∈{1,5}, M=100, 5 traj) but `results/navier_stokes/ablation/` does not exist; (e) **no g_τ sweep** and **no E sweep** data anywhere.
**Confidence**: high
**Fix**: Fill the M and covariance rows from existing aggregates today (zero compute). For the rest, run `run_ns_ablation.sh` restricted to one scenario (~15–50 GPU-h) plus small one-scenario g_τ and E sweeps (~20–60 GPU-h), or delete the axes not run from the ablation paragraph and table before submission — an empty table with six promised axes is worse than a smaller filled one.

---

**Location**: `results.tex` §5 baselines, "both SDA and FlowDAS perform better when combined with SURGE, so we report only the SURGE-enhanced versions"
**Severity**: major
**Problem**: The repo's own NS grid contradicts the citation-based justification. Bare SDA (same grid, same tuned γ_sda, M=100, 5 trajectories; rows in `results/navier_stokes/aggregated/all.csv`) beats SDA+SURGE in **every** scenario — e.g. sparse 5%: RMSE 0.105±0.009 vs 0.162±0.016, CRPS 0.053 vs 0.113, spread–skill 0.12 vs 0.94 — and would beat every row in Table 1 in the sparse columns, including bolded SDA+SURGE 1.616 and Ours SI-SDE (shared) 1.731. Bare FlowDAS also beats FlowDAS+SURGE in both sparse scenarios (0.694 vs 0.863; 0.871 vs 1.190). The citation's claim holds in the analytical case but reverses at field scale; Table 3 additionally shows FlowDAS+SURGE diverging at M=500. (The 2026-07-22 SURGE-SDA score-bug fix and rerun affected only the SURGE wrapper, not bare SDA, so the bare rows are current.)
**Confidence**: high for the data contradiction (verified in the aggregate); medium that the authors intend these bare rows as final (same frozen grid).
**Fix**: Add the bare FlowDAS and SDA rows to Table 1 (data exists, zero compute) and change the justification sentence — SURGE helps in the exact-posterior regime but hurts at field scale (consistent with the paper's own ensemble-collapse observation). Reporting only the weaker variants of two baselines is the finding a reviewer is most likely to consider adversarial if discovered from released code.

---

**Location**: `results.tex` "details are given in Appendix F" + `statements.tex` "Appendix F specifies each baseline and its tuning protocol" vs `appendix_methods.tex`
**Severity**: major
**Problem**: Appendix F contains **no tuning protocol, budgets, search ranges, or chosen values** — only qualitative remarks. The configs contain per-(case, scenario, M) tuned tables that never appear in the paper: FlowDAS ζ (NS sparse 5%: 0.002 at M=100 but 1.0 at M=500; urban 5e-4), SDA γ_sda (1e-4–1e-2 NS; 5e-4 urban), FIG (k=3, c=10 NS sparse; c=160 urban — a 16× change), D-Flow (η=0.005, noise 1e-3, λ_reg 1e-4 NS / 1e-3 urban), ours λ (0.9 superres, 0.95 sparse NS and urban, 1.0 analytical). Equal-tuning-budget fairness cannot be assessed from the paper.
**Confidence**: high
**Fix**: Add one hyperparameter table to Appendix F transcribed from `paper_experiments/configs/method/*.yaml` (writing only), plus 2–3 sentences on the protocol.

---

**Location**: Table 1 and Table 2; e.g. bold "1.616" vs "1.731" in the 5% RMSE column
**Severity**: major
**Problem**: No error bars anywhere. The aggregate CSV carries cross-trajectory std (n=5): SI-SDE (shared) 1.731±0.198 vs SDA+SURGE 1.616±0.161 (×10⁻¹) — the bolded winner is well inside one std of the runner-up; several other bold/non-bold distinctions are similarly within noise. Urban is a single trajectory (std=0 in `results/urban/aggregated/all.csv`), yet conclusions (DM-SDE best velocity, SI-SDE best temperature, jacfree "close to shared") rest on differences of 0.01–0.05 with n=1. Analytical Table 3 reports mean over 5 seeds without σ.
**Confidence**: high
**Fix**: Cheapest: print ± std (or paired per-trajectory win counts — per-trajectory rows exist in `results/navier_stokes/metrics/`) in Tables 1 and 3, and soften "best"-claims within spread. For urban, `run_urban_grid.sh` already defaults to TRAJ="1 2 3 4 5"; completing trajectories 2–5 is ~200–250 GPU-h. If not run, extend the one-trajectory caveat to the conclusion sentence citing urban rankings.

---

**Location**: Abstract sentence 2 / C2 vs `appendix_methods.tex` last paragraph and `configs/case/*.yaml` `diffusion_from_fm: true`
**Severity**: minor
**Problem**: No natively-trained diffusion prior is ever evaluated: the DM-SDE/SDA/SURGE "diffusion prior" is always constructed from the FM velocity (the dedicated NS diffusion checkpoint exists but is unused; config comment: "poorly trained"). Disclosed only in the final paragraph of Appendix F.
**Confidence**: high
**Fix**: One sentence in §5's baseline paragraph, or soften "any" in the abstract.

---

**Location**: Abstract sentence 5 / C7 "essential for sparse observations" vs Table 2
**Severity**: minor
**Problem**: "Essential" supported by NS sparse (2–4×) but urban — also sparse — shows 3–5% gains at 3–4× cost; conclusion itself concedes "modest".
**Confidence**: high
**Fix**: Qualify to "can be decisive for sparse observations" or "essential when prior correlations dominate sparse reconstruction (Navier–Stokes)".

---

**Location**: `appendix_methods.tex` EnKF paragraph vs `run_ns_grid.sh` (`+enkf_localization_radius=20` on sparse scenarios)
**Severity**: minor
**Problem**: The E=64 EnKF rows in Table 1 used covariance localization (radius 20) in both sparse scenarios; the E=1000 reference was non-localized. The paper describes a plain stochastic EnKF with no mention of localization (config still carries "TODO: report localisation radius"). Irreproducible as described; localized/non-localized asymmetry undisclosed.
**Confidence**: high
**Fix**: One sentence in Appendix F stating the localization settings.

---

**Location**: whole paper — unstated quantities needed to reproduce (checked against `configs/` and run scripts)
**Severity**: major (as a set)
**Problem**: Used by the reported runs but appear nowhere in the manuscript: (1) damping λ per case/scenario (0.9 superres, 0.95 sparse NS + all urban, 1.0 analytical); (2) FlowDAS ζ values and J=25 MC samples; (3) SDA γ_sda values; (4) FIG (k, c) values; (5) D-Flow η=0.005, Langevin noise 1e-3, λ_reg, chain length/burn-in, ensemble batch 16; (6) urban observation-noise level — σ=0.05 in per-channel *normalized* units (never stated, nor that it lives in normalized space); (7) DM-SDE diffusion proportionality constant (multiplier 1.0; analytical g_0=1.0) unstated; (8) divergence guard (abort + NaN-pad at ensemble RMSE > 10) armed in every NS/urban run; (9) SURGE particle count (=E); (10) Table 1's five held-out trajectories and step-averaging stated only in Appendix H, not the caption.
**Confidence**: high (all traced to configs/scripts; manuscript grepped for each)
**Fix**: The experimental-setup table the authors already planned (TODO block at `results.tex` lines 29–32) — one appendix table per case. Writing only.

---

**Location**: `results.tex` §5.1, "diverge as $M$ grows at fixed per-step hyperparameters"
**Severity**: nit
**Problem**: For FlowDAS(+SURGE) the analytical driver resolves ζ per M from the config table (0.004 at M=50 → 1.0 at M=500), so hyperparameters were retuned per M; divergence at M=500 occurred despite retuning. Accurate for D-Flow (η fixed).
**Confidence**: medium
**Fix**: "…diverge as $M$ grows even with per-$M$ retuned step sizes (FlowDAS+SURGE) or at a fixed step size (D-Flow)".

---

## Checked, no finding
- Code vs §4: `gaussian_likelihood.py` implements Eqs. 13–15 exactly; Eq. 16 jac-free mode matches; shared Jacobian at ensemble mean with refresh cadence k and correct invalidation; damping multiplies only `c_jac`, so λ=0 retains the analytical −γ⁴τ²A_τβ̇I term for SI exactly as Eq. 17 claims.
- Algorithm 1 τ=0 skip and w_τ=κ_τ+½g_τ² implemented verbatim in both posterior models; FM score-from-velocity matches Appendix A.1.
- Extra code knobs (`drift_time_shift`, `sigma_bar_eig_floor`, `isotropic_front_factor`, `residual_step_cap`, SMC `resample`) all disabled/0/null in configs used for reported runs; divergence guard produced zero hits in current NS and urban grid logs — no reported cell NaN-padded.
- Table 1 numbers spot-checked against `aggregated/all.csv`: exact matches; "2–4×", "one-tenth the cost", "3–4× lower cost" arithmetic verified.
- Urban tuning used test-split trajectory 6, disjoint from evaluated trajectories — consistent with "tuned on separate short trajectories"; NS γ_τ multiplier 1.0, stated schedule accurate.
- Main-text urban protocol (145 steps, every-3rd, 49 updates, foreign IC, M=50, E=64) matches active `run_urban_grid.sh` defaults exactly.
