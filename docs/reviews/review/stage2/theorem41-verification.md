# Stage 2 — Independent verification of Theorem 4.1 proof chain (SI + g≡0 degeneracy)

Adjudication of Agent B's flagged gap by an independent Stage-2 agent. Orchestrator summary: **finding CONFIRMED as under-guarded (major, not blocking); all other proof steps PASS.**

## (a) Mathematical claim — CONFIRMED
- Target marginals p_τ^y (Prop B.1) are absolutely continuous for τ∈(0,1) — never a point mass.
- A well-posed deterministic flow (g=0) pushes δ_{x_0} to δ_{Φ_τ(x_0)}, so Law(X*_τ) = p_τ^y is impossible for that member.
- Break point localised: Prop A.3's FP algebra survives; the final uniqueness invocation fails — with initial datum δ_{x_0} the continuity equation has at least two distributional solutions (p_τ^y and the push-forward δ-curve); near τ=0 the velocity behaves like (x−α_τx_0)/(2τ), a classic non-uniqueness point (solutions α_τx_0 + c√τ for every c). The well-posedness hypothesis "cannot hold" for SI+g=0, so the proof silently excludes it.
- The commented-out remark at `appendix_proof_conditioning.tex:66–68` states exactly this; a complementary FM-side remark sits uncompiled at `appendix_score_velocity_duality.tex:110–112`.

## (b) Wrong vs under-guarded — UNDER-GUARDED (major, not blocking)
- Theorem asserts nothing false; the member is excluded by the stated well-posedness hypothesis (vacuously true there).
- Aggravating: post-theorem sentence "every admissible diffusion schedule gives exact posterior samples" reads as "any measurable g≥0 in L²"; intro claim C19 inherits the ambiguity.
- Mitigating: the three displayed members never instantiate the bad combination.

## (c) Fix — CONFIRMED sound; single-sentence option preferred
Caveat: the commented remark's clause "the theorem requires g_τ>0 near τ=0" misdescribes the live theorem (exclusion happens via well-posedness); reword if reinstated. Exact sentence to add after methodology.tex line 27:

> Admissibility is not vacuous: for the point-mass SI source $p_0=\delta_{\bx_0}$ the deterministic choice $\gdiff_\tau\equiv0$ violates the well-posedness hypothesis — a well-posed deterministic flow transports $\delta_{\bx_0}$ to a point mass and cannot realise the non-degenerate marginals $p_\tau^{\obs}$ — so the guided ODE member requires a Gaussian source, and SI priors are used only with $\gdiff_\tau=\gamma_\tau>0$.

## (d) Remaining proof steps — ALL PASS
- B.1 ε⊥y|x_1 argument: PASS.
- B.1 Bayes step (y⊥x_τ|x_1): PASS (τ∈(0,1)).
- B.1 velocity shift by κ_τ: PASS (affine part cancels; needs only preamble conditions).
- Thm 4.1 via A.3 on conditioned path: PASS with the (a) caveat. Unflagged nit: A.3 preamble assumes prior E‖x_1‖²<∞; posterior inherits it (bounded Gaussian likelihood) — no real gap.
- Initial-law argument Φ_0^y constant: PASS.
- Substitution giving w_τ=κ_τ+½g_τ²: PASS.

## Recommendation
Severity: **major**. Minimal edit: the single guard sentence above; optionally reinstate the appendix remark with its misdescribing clause reworded. No change needed to theorem statement, displayed members, or proofs.
