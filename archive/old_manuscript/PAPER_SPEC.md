# Manuscript rewrite — shared spec (the "bible") for all writing agents

**Goal.** Rewrite the paper *"A General Observation-Interpolant Method for Data
Assimilation with Flow-Based Generative Models"* into this `manuscript/` folder with the
structure below. The central message: **SI, flow matching, and SDE-flow-matching
(= denoising diffusion) are one family of posterior samplers**; the paper makes explicit
**which efficiency/accuracy trade-offs to make and when they matter vs. when methods coincide.**

**Source material.** The previous version's sections are already in `sections/*.tex` (you are
rewriting them in place). They are high quality — **reuse equations, theorems, lemmas, and
notation verbatim where possible**; this is largely a reorganization + new emphasis + new
figures + filled-in results, NOT a from-scratch rewrite. Do not invent new notation.

**Hard constraints.**
- **Main body ≤ 10 pages** (intro → conclusion, before `\appendix`). Move extended
  derivations, extra results, and long comparisons to the appendix.
- Use ONLY the existing macros in `style/macros.tex` (e.g. `\bx,\state,\obs,\obsmatrix,\obscov,
  \fmvel,\drift,\genscore,\gdiff,\gweight,\vscoef,\sigpath,\interpolantobs,\srcmean,\srccov,
  \postdrift,\bXstar,\conddist,\dist`). Do not redefine them.
- Keep every `\label{}` that other sections cross-reference (see "Stable labels" below).
- Compile target: `pdflatex`+`bibtex` via `latexmk` (local shims for gensymb/animate/listingsutf8
  are present). Citations are in `library_NTM.bib` — reuse existing `\cite` keys; do not invent keys.

---

## Target structure (Sections, in order)

1. **Introduction** — keep close to current `introduction.tex` (the "unified view" + contributions
   are good). Add a forward-reference to **Figure 1 (visual abstract)** — the figure itself is
   built separately (TikZ, `figures/visual_abstract.tex`); just `\input` it and reference it.
2. **Related work** — keep current content but **merge "Guidance in deterministic flow ODEs" INTO
   "Inverse problems with generative models"**, renaming the merged subsection
   **"Posterior sampling with flow-based generative models"**. Keep the other subsections.
3. **Preliminaries** — three subsections, mostly reuse current `preliminaries.tex`:
   (a) Bayesian inverse problems / DA (current §"Bayesian data assimilation");
   (b) generative modelling with **stochastic interpolants**;
   (c) **flow matching as a special case of SI without the Wiener process in training** — and
   crucially, **first present the FM ODE, then show how the equal-marginal SDE family lifts it to a
   diffusion-model SDE sampler** (Eq. `general_sde_family`). Make the "FM-SDE = denoising diffusion"
   connection explicit.
4. **Methodology** — subsections:
   - **Posterior sampling (SDE and ODE).** Condition the affine path on the observation; show the
     conditioned path is again affine Gaussian; the conditional drift = prior drift + `w_τ·G_τ·S̄`.
     Write the **posterior SDE for SI and FM identically**, differing only in the weight/coefficient,
     and the ODE as the `g_τ=0` member — so similarities/differences are visually obvious. Include
     the **theorem that the marginal-preserving family yields exact posterior samples in the ideal
     setting** (current `appendix_proof_posterior` / methodology Eq. `unified_posterior_drift`).
   - **Approximating the likelihood score.** Briefly state the problem (exact tilt `Φ_τ^obs`
     intractable — needs trajectory backprop). Introduce the **observation interpolant**
     `ȳ_τ = α_τ H a₀ + β_τ y`. Include the **two lemmas**: interpolated-observation likelihood
     (`lem:interpolated_likelihood`) and source conditional moments (`lem:source_moments`).
   - **Multiplicative correction.** The `S_τ = G_τ S̄` theorem (`thm:multiplicative_correction`) +
     Corollary (`cor:cheap_drift`).
   - **Comparison of posterior samplers.** Very explicitly write out the **three samplers**
     (SI-SDE, FM-SDE, FM-ODE) — same observation-interpolant likelihood + same multiplicative
     correction; differ only in `g_τ` (native γ_τ / lifted / 0), the weight
     `w_τ=a_τ+½g_τ²`, and the source/anchor. Keep Tables `tab:samplers` and `tab:si_vs_fm`.
   - **Approximations and simplifications.** A short **subsubsection per computational bottleneck**,
     each explicitly stating *what true quantity is approximated, how, and the assumption*: (i)
     Gaussian likelihood surrogate (DPS) + uninflated vs inflated (ΠGDM) covariance; (ii) Jacobian-free
     isotropic gain (Corollary); (iii) **ensemble-shared source covariance** (evaluate the
     source-covariance Jacobian once at the ensemble mean → single N_y×N_y solve, O(E·N_y)→O(N_y),
     exact as the ensemble tightens); (iv) fixed/detached covariances; (v) pseudo-time clamping.
     (Most of this exists in current methodology `subsec:approximations` — keep + sharpen.)
   - **Summary of choices.** Keep the summary prose + add **Figure 2 (decision guide)**: a flow/
     decision diagram — *if high-dimensional state → …; if high-dimensional (dense) observations → …;
     if sparse observations → inflated covariance matters; if cheap inference needed → Jacobian-free;
     if deterministic sampler → ODE; …*. Figure built separately (TikZ, `figures/decision_guide.tex`).
5. **Comparison to alternative methods** (`\section`, short). How we differ from FlowDAS, FIG/OT-ODE,
   DPS/ΠGDM, SDA, EnSF, EnKF/PF; the trade-offs. Move long derivations/extended comparison to appendix.
6. **Implementation.** Keep the 4 algorithms (`alg:unified`, `alg:si_sde`, `alg:fm_sde`, `alg:fm_ode`)
   + implementation considerations (pseudo-time clamping, ensemble draws, solver choice, cost).
7. **Results.** Rewrite to report the real findings (below). Use `\input{../generated/tab_*.tex}` is
   NOT available across folders — instead reproduce the tables and fill with the numbers in
   "Real results" below. Keep figure placeholders (`\figbox`) for NS/urban field figures (not yet
   generated) but USE REAL NUMBERS in the tables that have them.
8. **Conclusion.** Leave essentially empty (a one-line placeholder: "\TODO Conclusion.").

(The numbering 1–8 above maps to the user's 1,2,3,4,7,8,9,10; there is no separate 5/6.)

---

## The unification narrative (must come through everywhere)
- One affine Gaussian path `x_τ = α_τ a₀ + σ_τ ε + β_τ x₁`. Conditioning on `y` tilts the target to
  the posterior but keeps the path affine Gaussian; conditional score = prior score + `S̄`;
  conditional velocity = prior velocity + `a_τ S̄` (velocity–score duality).
- The equal-marginal SDE family (free `g_τ≥0`) gives, from ONE guidance term, a whole family of
  posterior samplers. Three members: **SI-SDE** (`g_τ=γ_τ`, native), **FM-SDE** (`g_τ>0`, lifted —
  this is exactly a **denoising-diffusion / score-SDE sampler** built on the FM prior), **FM-ODE**
  (`g_τ=0`, deterministic guided probability-flow ODE).
- They share ONE observation-interpolant likelihood and ONE multiplicative correction; they differ
  only in `g_τ`, the weight `w_τ=a_τ+½g_τ²`, and the source/anchor.

## When the choices matter (the practical thesis — back with results)
- **Dense / informative observations (super-resolution):** the likelihood dominates; the cheap
  Jacobian-free isotropic gain (`dps_jacobian_free`) is already excellent and all samplers nearly
  coincide → use the cheapest.
- **Sparse observations:** the prior covariance structure matters; the **inflated covariance** is
  markedly more accurate, but exact inflated is O(E·N_y) network-Jacobians → intractable at field
  scale → use the **ensemble-shared** approximation.
- **High-dimensional state:** form `Σ̄` once (shared Jacobian), solve the N_y×N_y system; never the
  N_u×N_u state covariance.
- **High-dimensional / dense observations (large N_y):** prefer the Jacobian-free gain (precompute
  `HᵀR⁻¹H`); the inflated N_y×N_y solve grows with N_y.
- **Deterministic vs stochastic:** ODE (g=0) is cheapest per step and admits high-order solvers;
  SDE samplers give better-calibrated spread.

---

## Real results (USE THESE NUMBERS)

**Case 1 — analytical linear–Gaussian** (closed form; complete, real): KL to exact / Sliced-W₂:
- Ours SI-SDE 0.001 / 0.017 · FM-SDE 0.002 / 0.024 · FM-ODE 0.001 / 0.020
- FlowDAS 0.299 / 0.352 · Guided FM 0.104 / 0.185 · Guided diffusion 0.174 / 0.194
- EnKF 0.001 / 0.021 · Particle filter 0.003 / 0.028
- **Finding:** the inflated covariance recovers the exact posterior (KL≈1e-3, → exact as E grows);
  the DPS-surrogate baselines plateau (KL≈0.1–0.3, step-independent bias). EnKF/PF are near-exact
  here because the system is linear–Gaussian (their ideal regime).

**Case 2 — stochastic Navier–Stokes** (learned prior, 128², `dps_jacobian_free`, E=64, M=50, 3 seeds;
the 3 "Ours" rows are REAL, mean over seeds, tight std):

| method | 32²→128² rmse | 16²→128² rmse | sparse 5% rmse | sparse 1.5625% rmse |
|---|---|---|---|---|
| Ours SI-SDE | 0.066 | 0.349 | 0.744 | 0.865 |
| Ours FM-SDE | 0.064 | 0.338 | 0.736 | 0.874 |
| Ours FM-ODE | 0.065 | — | 0.759 | — |

CRPS (SI-SDE): 0.035 / 0.197 / 0.432 / 0.497. Cost: SI-SDE NFE=50 ~5 s/step; FM-SDE/ODE NFE=100
~16 s/step (FM needs velocity+score per step). KL-at-points ~0.02–0.03 (super-res), larger for FM.
- **Headline finding:** `dps_jacobian_free` is excellent when observations are dense
  (rmse 0.066 super-res) and degrades as they thin (rmse 0.74–0.87 sparse). The three samplers are
  consistent with each other.
- **Inflated-covariance finding (key):** at sparse 1.5625%, the **ensemble-shared inflated**
  covariance gives rmse **0.137** vs **0.865** for the Jacobian-free gain (E=8, M=10 comparison) —
  the inflated covariance is dramatically more accurate exactly when observations are sparse. Exact
  inflated is intractable at scale (O(E·N_y) UNet-Jacobians, ~days/cell); the ensemble-shared
  approximation makes it feasible (~hours).
- Baseline + ablation NS numbers are being generated by a running job; the Results agent should
  leave clearly-marked slots (or `--`) for: FlowDAS, Guided FM, Guided diffusion, SDA, EnSF, EnKF,
  PF, and the ablation (gain axis: full `dps_full` vs Jacobian-free vs inflated-shared; g/M/E sweeps).

**Case 3 — urban airflow:** not run (data pending); keep the subsection brief with placeholders.

---

## Two figures (built by the figure owner, do not write them in section files)
- **Figure 1 — visual abstract** (`figures/visual_abstract.tex`, TikZ, `\input` near the top of
  Intro, full-width `figure*`). Show the posterior-sampling process: a source distribution on the
  left → the affine interpolation path over pseudo-time τ → the posterior sample on the right; the
  observation `y` entering through the observation-interpolant likelihood + multiplicative
  correction `w_τ G_τ S̄`; the three samplers as a small branch (SI-SDE native noise / FM-SDE lifted
  noise / FM-ODE deterministic). Clean, schematic, self-contained vector graphics. Label
  `\label{fig:visual_abstract}`.
- **Figure 2 — decision guide** (`figures/decision_guide.tex`, TikZ flow/decision diagram). The
  efficiency/accuracy choices: state dim, observation density (dense→cheap gain; sparse→inflated /
  ensemble-shared), N_y size (→Jacobian-free), deterministic vs calibrated (ODE vs SDE), steps M.
  Label `\label{fig:decision_guide}`.

## Stable labels (keep; cross-referenced across sections)
`sec:introduction, sec:related_work, sec:preliminaries, sec:methodology, sec:implementation,
sec:results, sec:conclusion, subsec:flow_models, subsec:obs_interpolation,
sec:multiplicative_correction, subsec:posterior_dynamics, subsec:supplying_diffusion,
subsec:summary_table, subsec:approximations, lem:interpolated_likelihood, lem:source_moments,
thm:multiplicative_correction, cor:cheap_drift, eq:unified_posterior_drift, eq:general_sde_family,
eq:general_velocity_of_score, eq:SI_score, eq:fm_score, eq:interpolated_observation,
tab:samplers, tab:si_vs_fm, tab:approximations, fig:visual_abstract, fig:decision_guide,
appendix:score_velocity, appendix:proof_posterior, appendix:proof_multiplicative,
appendix:simple_test_case`. Appendices keep their current content.
</content>
