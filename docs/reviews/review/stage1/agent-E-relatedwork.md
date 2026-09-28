# Stage 1 — Agent E (Related work & positioning) raw report

Status: received; SDA finding independently re-derived by orchestrator (confirms); FIG finding based on abstract+repo (PDF bot-blocked); Stage-2 SDA-paper read dispatched.

---

**Location**: Appendix F, SDA paragraph — "conditions trajectories rather than marginals … so the intermediate marginals are not $p_\tau^{\obs}$"
**Severity**: major
**Problem**: The critique is mathematically off target for SDA as published. For a diffusion prior sampled with its *native* reverse SDE (SDA's setting, α_τ=0), the paper's own coefficient gives κ_τ = ½g²_native (from κ_τ = σ²β̇/β − σσ̇ and the forward-SDE identity), so the Doob weight g² *equals* the marginal-path weight w_τ = κ_τ + ½g². With the exact tilt, the h-transform of the reverse diffusion has marginals exactly p_τ^y — conditioning a process on an event yields both conditioned trajectories *and* conditioned marginals. SDA's inexactness comes from the Gaussian/Tweedie tilt approximation (which the paper shares), not from the weight. The claim only bites for *non-native* diffusion coefficients (e.g., the authors' reimplementation on the FM-derived prior, or the ODE limit). The cited support is loose: Särkkä & Solin §7.5 gives the h-transform drift form, and Wu et al. criticize the likelihood approximation, not the g² weight.
**Confidence**: high on the math (own derivation; ALSO independently re-derived by orchestrator: matching a_s=β_{1−s}, v_s=σ²_{1−s} to dv/ds=2fv+g² gives g²=2κ_τ exactly); SDA's likelihood form verified from sources quoting the paper, not the full PDF (Stage-2 read dispatched).
**Fix**: Replace the clause with: "the guidance weight g_τ² is tied to the prior diffusion's native coefficient (for which g_τ² = w_τ); it does not transfer to other equal-marginal samplers of the same path, and the isotropic Tweedie covariance carries no model source information." Delete "so the intermediate marginals are not p_τ^y" (or restrict explicitly to non-native g_τ).

---

**Location**: Related work — "Closest to our work is Yan et al., who interpolate the observations within a flow-matching formulation"
**Severity**: major
**Problem**: FIG is not FM-only. The FIG abstract states it uses "a pre-trained diffusion or flow-matching model as a prior," and the official repo (riccizz/FIG) ships FIG-flow (RectifiedFlow) *and* FIG-diffusion (DAPS/DDNM checkpoints). The delta "we generalize to all three generative models" vs "FIG = FM only" is overstated on the FM/DM axis; true remaining deltas vs FIG: derived closed-form moments (bias + source-inflated covariance vs heuristic corrector), the SI/Föllmer member, the SDE members with the analytic weight, and autoregressive DA.
**Confidence**: high that FIG covers diffusion priors (abstract + official code); full paper not read (OpenReview PDF bot-blocked).
**Fix**: "…who interpolate the observations to guide deterministic sampling of flow-matching (and flow-adapted diffusion) priors in one-shot inverse problems, with a heuristic corrector rather than a derived likelihood covariance."

---

**Location**: Related work / §4.2 — observation-interpolant novelty framing
**Severity**: major
**Problem**: Missing closely-related work: Dou & Song, "Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective," ICLR 2024 (spotlight). Constructs a *noised measurement sequence* y_t paired with the diffusion states and treats posterior sampling as filtering with SMC correction — the diffusion-side precedent for evolving the observation along the generative path.
**Confidence**: high (located, ICLR 2024 proceedings; relevance from abstract).
**Fix**: Add after the FIG sentence: "Relatedly, \citet{dou_fps_2024} pair the state with a noised measurement path and correct the resulting guidance with sequential Monte Carlo." + bib entry.

---

**Location**: Related work — "Most use a Tweedie-type estimate … \citep{song_score-based_2021,rozet,qu,pokle,martin_pnp-flow_2025}"
**Severity**: minor
**Problem**: Song et al. (2021) does not use Tweedie-differentiated likelihood guidance; PnP-Flow explicitly *avoids* backpropagating through the denoiser/ODE. The archetype, DPS (chung_diffusion_2023), is absent from the list.
**Confidence**: high.
**Fix**: `\citep{chung_diffusion_2023,rozet_score-based_2023,qu_deep_2024,pokle_training-free_2024}`; optionally "while plug-and-play variants avoid network backpropagation entirely~\citep{martin_pnp-flow_2025}".

---

**Location**: Related work — ensemble score filter coverage; `si_latent-ensf_2024` only in commented-out text
**Severity**: minor
**Problem**: Compiled bibliography contains no Latent-EnSF (dropped with the old paragraph). bao_ensemble_2024 listed under "diffusion-model posterior sampling" is a slight misfit (EnSF is training-free).
**Confidence**: high (bbl checked).
**Fix**: "…and training-free ensemble score filters~\citep{bao_ensemble_2024, si_latent-ensf_2024}" as its own clause.

---

**Location**: Related work, generative-DA sentence
**Severity**: minor
**Problem**: Missing the two weather-scale generative-DA works reviewers cite by default: Huang et al. DiffDA (ICML 2024, arXiv:2401.05932) and Manshausen et al. sparse-station generative DA (arXiv:2406.16947; JAMES 2025) — the latter is the same sparse-sensor regime.
**Confidence**: high (both located with venues).
**Fix**: "Diffusion-based DA has also been demonstrated at weather scale~\citep{huang_diffda_2024, manshausen_generative_2025}." + two bib entries.

---

**Location**: Appendix F SURGE paragraph / SMC positioning
**Severity**: minor
**Problem**: The SMC-corrected-guidance line is cited only via wu_practical_2023. Cardoso et al. MCGDiff (ICLR 2024, arXiv:2308.07983) gives asymptotically exact SMC posterior sampling for exactly the linear-Gaussian setting of the analytical case.
**Confidence**: high.
**Fix**: "in the line of SMC-corrected diffusion guidance~\citep{cardoso_mcgdiff_2024, wu_practical_2023}".

---

**Location**: Appendix F "FlowDAS…, the closest prior method" vs Related work "Closest to our work is FIG"
**Severity**: nit
**Problem**: Two different "closest" works named in two places.
**Fix**: Appendix F: "the closest prior method *among the DA baselines*".

---

**Location**: Appendix F SDA paragraph h-transform citations
**Severity**: nit
**Problem**: If h-transform discussion survives, DEFT (Denker et al., arXiv:2406.01781) and Didi et al. (arXiv:2312.09236) are natural cites.
**Confidence**: med.
**Fix**: Optionally add.

---

**Location**: Related work ¶1 — "flow matching~\citep{fotiadis_stochastic_2024}"
**Severity**: nit
**Problem**: Cited work is conditional downscaling/super-resolution, not next-state forecasting, while the sentence is about next-state sampling.
**Confidence**: med.
**Fix**: Drop or widen the clause.

---

**Location**: Introduction ¶2 — "Denoising diffusion~\citep{ho, song} corresponds to SDE sampling of an FM-type model."
**Severity**: nit
**Problem**: ho/song support "denoising diffusion", not the correspondence claim; equivalence is albergo/ma_sit territory.
**Confidence**: high.
**Fix**: Move citations before "Denoising diffusion" or append `\citep{ma_sit_2024}` after "corresponds to".

---

## No-finding areas (each checked)
- FlowDAS characterisation (App. F): verified against arXiv:2501.16642 full HTML — MC marginalization, softmax weights, constant step size ζ. Accurate. (high, full text)
- wei_surge_2026 Results claim: verified against arXiv:2605.18745 full HTML — SDA+SURGE and FlowDAS+SURGE both beat plain counterparts in the cited paper's Lorenz/NS/weather tables; mechanics as described. Accurate. (high, full text) [Orchestrator note: the claim is accurate TO THE CITATION; the repo's own NS aggregates contradict it at field scale — see agent D finding.]
- chen_probabilistic_2024: "Föllmer formulation" accurate; Proposition B.9 confirmed in arXiv:2403.13724v2. (high)
- parikh_d-flow_2026: exists (arXiv:2602.21469); App. F description matches. (high, full text)
- ΠGDM "inflates" / DPS "uninflated": accurate. (high)
- Albergo et al. credit: correctly scoped. (high)
- FIG guidance formula "c(1−τ)/τ, normalised residual": could not verify against paper text (bot-blocked) — rests on authors' own reading/implementation.
- Contribution 1 delta (general-g conditioned-interpolant weight): no prior statement located.
- Contribution 3 (ensemble-shared Jacobian): no prior work located; ΠGDM kinship acknowledged.
