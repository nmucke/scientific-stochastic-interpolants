# Stage 1 — Agent B (Mathematical correctness) raw report

Status: received, NOT yet verified by orchestrator (Stage 3 pending).

Scope: every stated result and proof in `preliminaries.tex`, `methodology.tex`, `appendix_score_velocity_duality.tex`, `appendix_proof_conditioning.tex`, `appendix_proof_interpolant_likelihood_covariance.tex`, `appendix_analytical_case.tex`.

## Findings

**Location:** Theorem 4.1, `methodology.tex` lines 16–27 + "With an exact prior and likelihood tilt, every admissible diffusion schedule gives exact posterior samples"; proof in `appendix_proof_conditioning.tex` lines 55–64; commented-out remark lines 66–68.
**Severity:** major
**Problem:** The live theorem statement silently includes the degenerate combination SI point-mass source (p_0 = δ_{x_0}) with g_τ ≡ 0. For that member the conclusion is impossible: a deterministic flow started from a point mass has marginals δ_{X(τ)}, which cannot equal the positive C^{1,2} densities p_τ^y the theorem asserts. The theorem is *technically* protected only because its well-posedness/uniqueness hypothesis fails for this member — the SI probability-flow drift contains (β̇/β)x = (2/τ)x, non-Lipschitz at τ→0, and both the flow-pushforward δ-curve and p_τ^y solve the same continuity equation from δ_{x_0} (non-uniqueness). Nothing in the live text tells the reader; "every admissible diffusion schedule gives exact posterior samples" invites reading SI + g=0 as admissible. The authors clearly know: the commented-out remark states exactly this degeneracy but is not compiled.
Key steps: (i) for SI, p_0^y = p_0 = δ_{x_0}; (ii) with g=0, Eq. (9)'s drift equals the posterior velocity v^y, so the sampler is the posterior probability-flow ODE; (iii) a well-posed ODE maps δ_{x_0} to δ_{X(τ)} ≠ p_τ^y; hence for SI either well-posedness fails or the conclusion fails, and the hypotheses quietly select the former.
**Confidence:** high (all three displayed members below the theorem avoid the combination, so no downstream result is wrong; the issue is that the statement's guard is invisible).
**Fix:** Reinstate the commented remark as a compiled Remark, or add after the three displayed members: "For a point-mass source (SI), the deterministic member g_τ=0 violates the uniqueness hypothesis — no well-posed deterministic flow can split δ_{x_0} into the non-degenerate marginals p_τ^y — so the guided ODE is available only for Gaussian-source (FM) priors; SI is instantiated with its native g_τ=γ_τ>0."

---

**Location:** Algorithm 1, `methodology.tex` lines 170–178: "Set the likelihood correction c_τ = 0" vs. the update line using (κ_τ + ½g_τ²) ŝ_τ.
**Severity:** minor
**Problem:** The τ=0 branch sets c_τ = 0, but c_τ is never used: the update on line 178 references ŝ_τ and κ_τ directly. At τ=0, ŝ is never computed and κ_0 is undefined for FM (κ_τ = (1−τ)/τ), so as written the first iteration evaluates undefined quantities. The intended (commented-out) line 176 sets c_τ = (κ+½g²)ŝ.
**Confidence:** high.
**Fix:** Uncomment line 176 and change the update to X* ← X* + (d_τ^g + c_τ)Δτ + g_τ√Δτ z.

---

**Location:** Proposition B.1, `appendix_proof_conditioning.tex` lines 9–25.
**Severity:** minor
**Problem:** The statement carries no τ-domain or regularity assumptions of its own. Eqs. (34)–(35) need: τ ∈ (0,1) with the standing Appendix-A assumptions (β_τ>0, σ_τ>0, C¹ schedules); Φ_τ^y positive and differentiable in x; and E[‖x_1‖² | y, x_0] < ∞ so the posterior velocity exists (Appendix A only assumes the prior second moment).
**Confidence:** high that assumptions are needed; high that they hold in all cases the paper uses.
**Fix:** Add: "Assume the standing conditions of Appendix A hold for τ∈(0,1), that p(y|·) is bounded, positive, and such that Φ_τ^y is differentiable in x, and that E[‖x_1‖²|y,x_0]<∞; the identities hold on (0,1)."

---

**Location:** Lemma 4.3, `methodology.tex` lines 87–94.
**Severity:** minor
**Problem:** The identities involve s_τ and ∇s_τ, which require σ_τ>0; for SI σ_0=0 and the score does not exist at τ=0. Lemma stated without the τ∈(0,1) qualifier that Appendix A carries.
**Confidence:** high.
**Fix:** Insert "for τ∈(0,1) (where σ_τ>0)" into the lemma statement.

---

**Location:** Sampler list after Theorem 4.1, "DM-SDE (g_τ>0)" (line 34); implemented g_τ = g_0√(σ_τβ_τ).
**Severity:** nit
**Problem:** The implemented endpoint-vanishing diffusion has g_0 = g_1 = 0, contradicting the label g_τ>0. (Theorem needs only g≥0; drift verified finite: ½g²s = ½g_0²τ(τv−x) — the 1/(1−τ) in Eq. (31) cancels exactly.)
**Confidence:** high (symbolically verified).
**Fix:** Change to "g_τ>0 on (0,1)".

---

**Location:** Eq. (15), Σ̄^{-1}; Lemma 4.2 with R "positive (semi)definite".
**Severity:** nit
**Problem:** Eq. (15) requires Σ̄ ≻ 0. For τ→0⁺ on SI, Σ̄ ≈ γ²τHH^⊤ + β²R with β²=τ⁴; if R only PSD and H rank-deficient this can be singular. Experiments use R = rI ≻ 0, so nothing fails in practice, but the invertibility assumption is unstated.
**Confidence:** high.
**Fix:** Add "R ≻ 0" to Lemma 4.2's hypotheses (or "assume Σ̄_τ invertible" after Eq. (15)).

---

**Location:** Eq. (5)/(24), `preliminaries.tex` line 50 "at pseudo-times where β̇σ − σ̇β ≠ 0".
**Severity:** nit
**Problem:** The inversion also divides by β_τ (and σ_τ); the stated condition alone does not exclude β_τ=0 (FM at τ=0). Appendix A states β_τ>0 on (0,1] but the main text does not.
**Confidence:** high.
**Fix:** "…where β_τσ_τ(β̇_τσ_τ − σ̇_τβ_τ) ≠ 0".

## Results verified with no findings

- Lemma A.1 (Tweedie identities incl. covariance identity): re-derived; correct.
- Prop A.2 (velocity–score identity), Eqs. (5)/(23)–(25): re-derived; consistent.
- Prop A.3 (Fokker–Planck computation): cancellation exact; conclusion correctly conditional on FP uniqueness.
- Prop B.1 proof (ε ⊥ y | x_1 argument): sound; Bayes step uses y ⊥ x_τ | x_1, which holds.
- Theorem 4.1 proof body and initial-law argument: correct given B.1 + A.3; Φ_0^y constant in x, p_0^y=p_0 — apart from the g=0/SI issue above.
- Lemma 4.2 cross-term subtlety: checked carefully — η independent of the PAIR (χ_τ, x_τ) since x_τ contains no η, which implies Cov(η, χ | x_τ, x_0) = 0. Lemma and proof correct.
- Lemma 4.3 algebra: exact via Lemma A.1.
- SI specialisations (methodology lines 97–113): re-derived; both match, using ∇_x c = β̇I. λ=0 claim correct: γ²τ − γ⁴τ²A_τβ̇ = τ²(1−τ)²/(2−τ) > 0 on (0,1) — SI at λ=0 retains a positive isotropic term distinct from Jacobian-free γ²τI. (∇μ̄)^⊤ = Σ_χH^⊤/σ² (Algorithm 1 line 175) verified.
- Eq. (15) stopped-covariance caveat correctly disclosed; "exact in linear–Gaussian regime" right.
- Endpoint singularities for experimental schedules verified symbolically: SI κ_τ=(1−τ)(3−τ)/2 finite; Wronskian τ^{3/2}(3−τ)/2 ≠ 0 on (0,1]; A_τ diverges at endpoints but only appears in bounded products; Σ̄→R at τ=1. FM: κ=(1−τ)/τ diverges at τ=0 but residual ȳ−μ̄ = O(τ) cancels; Algorithm skip of τ=0 plus cancellation covers first step; last step regular (DM drift finite — Eq. 31 blow-up cancelled by g²∝σβ, as claimed in App. A.1).
- Analytical case (App. G): SI drift, FM velocity, posterior K=½I, and α+β=1 remark all re-derived and correct.
