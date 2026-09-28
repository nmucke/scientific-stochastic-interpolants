# Stage 2 — SDA paper read (arXiv:2306.10574) vs manuscript characterisation

Orchestrator summary: **manuscript's App F SDA "key limitation" claim INACCURATE; "all-at-once score" clause PARTIALLY inaccurate; mechanics of the likelihood description accurate but "DPS-style" label imprecise.** Agent E's finding CONFIRMED (and previously re-derived by orchestrator).

## What SDA actually does (quotes verified against the paper)
- VP diffusion, cosine schedule; samples the **native reverse SDE** (Eq. 4) with exponential-integrator predictor (Eq. 16) + LMC corrector (Eq. 17, τ=0.25–0.5).
- Likelihood approximation (Eq. 15): `p(y|x(t)) ≈ N(y | A(x̂(x(t))), Σ_y + (σ(t)²/μ(t)²) A Γ A^T)`, in practice Γ = 10⁻² I (constant, tunable) — i.e. an **inflated** ΠGDM-style covariance, not raw DPS ("γ_sda" is the manuscript's own name for Γ).
- Guidance enters by adding ∇log p(y|x(t)) to the prior score inside the native reverse SDE — the h-transform/Doob form; no separate weight.
- Trajectory handling: learns **local segment scores** (Eq. 14, pseudo-Markov blanket) composed into an approximate joint trajectory score (Algorithm 2); generation is joint/non-autoregressive.

## Adjudication
1. "DPS-style guidance … isotropic denoiser variance scaled by tunable γ_sda" — ACCURATE in mechanics; "DPS-style" mildly off-key (SDA explicitly distances itself from DPS's uninflated/rescaled variant; it uses the inflated form).
2. "learns an all-at-once score over the whole trajectory and conditions jointly" — PARTIALLY ACCURATE: conditions jointly, yes; but the score is learned **locally over segments** and composed — that is SDA's headline contribution.
3. "guidance enters with the Doob weight g² … so the intermediate marginals are not p_τ^y" — **INACCURATE**: SDA uses the native diffusion, for which κ_τ = ½g² and hence w_τ = g² — Doob weight and marginal-path weight coincide; with an exact tilt the h-transform marginals ARE p_τ^y. SDA's real inexactness is the Gaussian–Tweedie surrogate with isotropic Γ (that final clause of the manuscript survives). The trajectory-vs-marginal distinction is real only for non-native g_τ (including the ODE member) — where the manuscript's point genuinely holds.

## Minimal edits (replacement text)
Replace in the SDA paragraph:
- "Its original form learns an all-at-once score over the whole trajectory and conditions jointly;" →
  "Its original form composes locally learned segment scores into a joint trajectory score and conditions all states at once;"
- The *Key limitation* sentence →
  "\emph{Key limitation:} SDA is tied to the native reverse diffusion, for which the Doob weight coincides with the marginal-path weight ($\vscoef_\tau=\tfrac12\gdiff_\tau^2$, so $\gweight_\tau=\gdiff_\tau^2$); its bias therefore stems entirely from the Gaussian--Tweedie likelihood surrogate, whose isotropic covariance carries no model source information and requires tuning $\gamma_{\mathrm{sda}}$ per scenario. Because the guidance weight is inherited from the fixed diffusion rather than derived from a probability path, it does not extend to samplers with modified or vanishing diffusion ($\gweight_\tau\neq\gdiff_\tau^2$), unlike Theorem~\ref{theorem:unified_posterior}."

**VERDICT: manuscript claim 3 INACCURATE.**
