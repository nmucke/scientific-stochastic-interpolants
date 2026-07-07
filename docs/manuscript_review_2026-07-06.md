# Manuscript review — 2026-07-06

Full-pass review of `manuscript/` (main text, all active appendices, macros, TikZ
figures, build log, `PAPER_SPEC.md`). Results are pending, so table content was not
reviewed. **All checked mathematics is sound**: Tweedie identities (Lemma A.1),
velocity–score duality incl. the rectified-flow special case, the Fokker–Planck
marginal-preservation argument, the conditioning proposition (the ε ⊥ y | x₁
argument), the SI score/weight algebra (w_τ = τγ(β̇γ/β − γ̇) checks out), the
interpolated-observation likelihood moments, and the analytical test case
(drift, K = ½I). Notation is consistent between main text and appendices.
Main body currently ends on page 9.

## Errors (must fix)

1. **Two undefined references** (confirmed in `main.log`). `main.tex:97` comments
   out `appendix_proof_posterior`, but it is still referenced:
   - `sections/appendix_score_velocity_duality.tex:105` → `\ref{theorem:posterior_sde}`
   - `sections/appendix_proof_conditioning.tex:63` → `\ref{appendix:proof_posterior}`

2. **Orphaned contribution claim.** Intro bullet 2 (`introduction.tex:14`) promises
   the Doob h-transform guidance-weight identification, but the Doob remark
   (`methodology.tex:41–43`) and its appendix are commented out. The SDA method
   description (`appendix_methods.tex:195`) also leans on the "Doob weight g²"
   contrast with no derivation left in the paper.
   **Recommended fix for 1+2: re-include `appendix_proof_posterior`** (already
   written); otherwise scrub all three references and the contribution clause.

3. **Citation formatting.** Plain `\cite` is used everywhere and renders textually
   ("… interpolant path Albergo et al. (2023); …"). Nearly all citations are
   parenthetical and need `\citep` (ICLR requires `\citep`/`\citet`). Global pass.

4. **Duplicate FIG bib entry with wrong authors.** `library_NTM.bib` has
   `yan_fig_2024` (correct: Yan, Yici; Zhang, Yichi; Meng, Xiangming; Zhao, Zhizhen)
   and `yan_fig_2025` (authors "Yan, Hanyu; Zhang, Hao; Meng, Tao; Zhao, Yan" —
   look fabricated). `appendix_methods.tex:105` cites the bad one → same paper
   appears twice in the references with contradictory metadata. Cite
   `yan_fig_2024` everywhere; delete `yan_fig_2025`.

5. **Cost-unit error in abstract and intro.** Abstract (`main.tex:58`) and intro
   bullet 4 say the ensemble-shared Jacobian reduces cost "from O(E N_y) network
   **Jacobians** per step to a single one". Methodology (`methodology.tex:123`)
   correctly counts O(E·N_y) Jacobian–**vector products** (= E Jacobians).
   Abstract/intro overstate by a factor N_y.

6. **Baseline lineup mismatch.** Appendix F fully describes FIG, OT-ODE, LETKF,
   and standalone SURGE, but none appear in the Baselines paragraph
   (`results.tex:20`) or any table. PAPER_SPEC also lists "Guided FM / Guided
   diffusion" rows. Decide the final lineup and make results §, tables, and
   appendix agree.

7. **Algorithm 1 vs. pseudo-time clamping.** `implementation.tex` says guidance is
   applied from τ = Δτ (w_τ singular at τ=0 since β₀=0); `tab:approximations` says
   "[Δτ, 1−Δτ]"; Algorithm 1 (`appendix_extra.tex:11`) starts the guidance at τ=0.
   Add the clamp to the pseudo-code and unify the interval.

8. **β_τ = τ² schedule stated nowhere except a table row.** The
   "Quadratic-β SI schedule" row of `tab:approximations` points to Section 4,
   which never mentions it; the experimental schedules (α, β, γ, σ per model
   class) are given nowhere. Add to Implementation or an experimental-setup
   appendix.

## Inconsistencies / minor

- "ODE is cheapest per step" (`methodology.tex:128`, decision guide) contradicts
  Implementation's "each step costs one model evaluation … identically for all
  three samplers" (and spec timings show FM ≈ 2× SI NFE). Restate as "cheapest
  overall via high-order solvers".
- NS equation symbol clashes (`results.tex:52`): bold v = `\fmvel`; α (friction)
  vs. schedule α_τ; ε vs. latent noise; ξ (forcing) vs. model noise of Eq. (1).
- OT-ODE "γ = 4" (`appendix_methods.tex:143`) undefined and collides with SI γ_τ.
- Metrics text promises NFE (`results.tex:23`); tables only carry s/step.
- Bib hygiene: empty publisher (`evensen_data_2022`), empty booktitle
  (`rozet_score-based_2023`, `yan_fig_2024`).
- `fig:analytical_panels` still uses `\figbox` placeholders although matching
  figures (`an_prior.pdf`, `an_like.pdf`, `an_true.pdf`, `an_sampled.pdf`,
  `an_kl_diff.pdf`, `an_kl_steps.pdf`, `an_slices.pdf`) already exist in
  `figures/analytical/` — can be dropped in now.
- Template leftovers: NeurIPS "Hippocampus / Cranberry-Lemon" author block in
  `main.tex` (ignored while anonymous, clean before camera-ready);
  `neurips_2025.sty`; hyperref loaded twice (packages.tex + main.tex); obsolete
  `fixltx2e`; unused packages (`animate`, `todonotes`, `lineno`, `moreverb`,
  `listings`).
- Score arguments drift between s(x) and s(x, x₀) across equations.
- 14 overfull hboxes, all ≤ 6 pt.

## Suggested improvements

1. Theorem 1 states no regularity hypotheses (the FP-uniqueness caveat lives only
   in the proof of Prop. A.3). Add a one-clause hypothesis to preempt reviewers.
2. Page budget: main body ends exactly on p. 9 with an empty results skeleton;
   the pending findings paragraphs + conclusion will spill. Likely lever:
   compress Sections 2–3.
3. Conclusion: add one line of future work on nonlinear observation operators
   (the linear-H assumption is the most binding limitation).
4. `sections/appendix_correction_factor_new.tex` is dead (never `\input`) — move
   to `sections_old/`.
5. SURGE "approximation-free" claim is exact only up to time-discretization of
   the Girsanov weights; hedge one word.

**Highest-leverage fix:** re-include `appendix_proof_posterior` (resolves errors
1–2 and grounds the SDA/SURGE discussions), then the global `\cite` → `\citep`
pass.
