# Stage 1 — Agent A (Claims vs. evidence) raw report

Status: received, NOT yet verified by orchestrator (Stage 3 pending).

## Findings

**Location** — Appendix H.3, Table 4 (`tab:ablation`), caption: "Run in progress; cells (--) are placeholders."
**Severity** — blocking.
**Problem** — Section H.3 asserts "we isolate the effect of the components introduced in Section 4 … Table 4 collects the results," but every data cell is `--`. A submitted PDF would contain an explicitly unfinished experiment, and the covariance/cadence/damping/diffusion/M/E ablation claims are entirely unsupported.
**Confidence** — high (verified in `sections/appendix_ns_case.tex` lines 196–218).
**Fix** — Either fill the table from the planned NS ablation run before the deadline, or delete Table 4 and rewrite H.3 to point at the ablation evidence that does exist (Table 1 shared-vs-Jac-free columns, Appendix G covariance paragraph, Figs. 11–13 for M). Deleting is a 10-line edit; running is the standard NS grid on one scenario.

**Location** — C24; Conclusion sentence 3, "our methods achieve the best accuracy under dense observations"; Results 5.2, "our samplers achieve the best RMSE and CRPS" and "the $E=1000$ EnKF reference attains the lowest sparse-scenario errors."
**Severity** — major.
**Problem** — Contradicted by Table 1 itself: at $16^2$ the EnKF reference has RMSE 1.442 and CRPS 0.779 versus the best "Ours" 2.024/1.062, so the reference is lowest in three of four scenarios ($16^2$, 5%, 1/64), not only the sparse ones. The bolding also silently excludes the reference row (2.024 is bolded as "best value in each column" while 1.442 sits below it), which the caption's rule does not license.
**Confidence** — high (numbers read directly from `results.tex` Table 1).
**Fix** — Three small edits: (i) conclusion → "our methods achieve the best accuracy under dense observations among methods that do not use the true solver"; (ii) 5.2 → "…attains the lowest errors in every scenario except $32^2$, but requires the true solver…"; (iii) caption → add "the $E=1000$ EnKF reference row is excluded from bolding."

**Location** — C2; Abstract sentence 2, "converts any pretrained flow-based generative model—stochastic interpolant, flow matching, or diffusion—into a posterior sampler."
**Severity** — major.
**Problem** — Scope creep on two axes. (i) No natively trained diffusion prior is ever used: Appendix F states "SDA, SURGE, and DM-SDE require a diffusion prior; we construct it from the trained FM velocity," so the "diffusion" leg of "any pretrained" is demonstrated only via an FM network. (ii) "Any" quietly requires the affine Gaussian path of Eq. (3) conditioned on $\bx_0$; the practical method further requires linear Gaussian observations (Lemma 4.2, admitted in the limitations).
**Confidence** — high for (i) (Appendix F, `appendix_methods.tex` line 27); high for (ii).
**Fix** — Abstract: "converts any generative model built on the affine interpolant path—stochastic interpolant, flow matching, or diffusion—into a posterior sampler"; and add one sentence in Section 5 or Appendix D: "the diffusion sampler is instantiated from the FM prior via the velocity–score identity; a natively trained score network is interchangeable." No new experiment needed, though a small natively-trained-DM run on NS would fully close (i).

**Location** — C7; Abstract sentence 5, "covariance inflated by the model's source covariance, which is essential for sparse observations."
**Severity** — major.
**Problem** — Contradicted within the paper: the conclusion says prior correlations "provide only modest gains in the urban case" (Table 2: SI-SDE T-RMSE 0.38/0.43 → 0.36/0.41), and even in sparse NS the "essential" effect is specific to SI-SDE (DM-SDE shared, 3.424, still loses to SDA+SURGE, 1.616). Also, the ablated quantity is the Jacobian (prior-correlation) part of the covariance — the Jacobian-free variant Eq. (16) is *also* source-inflated — so the abstract attributes the effect to the wrong component.
**Confidence** — high (Tables 1–2 and Eq. `eq:cov_cheap`).
**Fix** — Abstract: "…source-inflated covariance whose Jacobian term carries prior correlations, which is decisive under sparse observations in our Navier–Stokes experiments."

**Location** — C26; Conclusion sentence 4 and Results 5.3 / Table 2 (`tab:urban_accuracy_M50`), "SI-SDE best limits error growth in temperature inferred from velocity alone."
**Severity** — major.
**Problem** — The headline urban superlatives rest on **one** held-out trajectory (caption: "One held-out trajectory") with margins of 0.01–0.02 over D-Flow SGLD (T-RMSE 0.36/0.41 vs 0.37/0.43) and no variability estimate; "best" at that margin and n=1 is not evidenced. The NS case, by contrast, uses five trajectories.
**Confidence** — high that n=1 (caption + aggregate CSV per paper map); medium that the ranking would survive more trajectories.
**Fix** — Cheapest: hedge — conclusion → "on the trajectory tested, SI-SDE gave the lowest temperature error"; 5.3 → add "on this single held-out rollout." Better: run the remaining held-out trajectories through the existing `run_urban_grid.sh` pipeline (cost: a few GPU-days) and report means.

**Location** — Results 5.2, "Hyperparameters are tuned on separate short trajectories; details are given in Appendix~F"; Section 4.3, damping "$\lambda\in[0,1]$."
**Severity** — major.
**Problem** — The promised details do not exist: Appendix F names tunable knobs (FlowDAS $\zeta$, D-Flow $\lambda$, SDA $\gamma_{\mathrm{sda}}$) but gives no tuned values, grids, or budget; the damping $\lambda$ used for the headline "shared" rows and the DM-SDE diffusion amplitude $\gdiff_0$ are stated nowhere in the paper. The results are therefore not reproducible from the text, despite the Reproducibility statement's claims. (A TODO comment in `results.tex` lines 29–32 acknowledges exactly this gap.)
**Confidence** — high (grepped all active appendices; no numerical $\lambda$, $\zeta$, $\gdiff_0$ values).
**Fix** — Add a half-page per-case hyperparameter table (values of $\lambda$, $k$, $\gdiff_0$, baseline knobs, tuning grid) to Appendix D or F; the values exist in `paper_experiments/configs/`.

**Location** — Results 5.2, "SI-SDE (shared) also achieves the best CRPS in three of four scenarios."
**Severity** — minor.
**Problem** — Strictly, SI-SDE shared has the best CRPS in two scenarios ($16^2$: 1.062; 5%: 0.872). The third ($32^2$: 0.347 vs FM-ODE 0.327, a 6% gap) counts only under the caption's tie rule applied to *unscaled* values (0.03 = 0.03), while the displayed $\times10^{-1}$ numbers differ at 2 decimals — the same ambiguity makes bolds 1.128 ($16^2$ CRPS), 0.347 ($32^2$ CRPS), and 0.638 ($32^2$ RMSE) look wrong to a reader applying the rule to the printed digits.
**Confidence** — high on the numbers; medium on which tie convention was intended.
**Fix** — Caption: "…every value within 0.01 of it *in unscaled units*"; text: "the best or near-best CRPS in three of four scenarios."

**Location** — C17; Introduction "unified view," "a deterministic guided ODE, including FIG and OT-ODE."
**Severity** — minor.
**Problem** — Appendix F contradicts membership: FIG's corrector "is not derived from a probability path, so the guidance magnitude $c(1-\tau)/\tau$ is not data-adaptive" — i.e., FIG uses a heuristic weight, not the family's $\gweight_\tau=\vscoef_\tau$, so it is a precedent for, not an instance of, the $\gdiff_\tau=0$ member.
**Confidence** — high.
**Fix** — "…a deterministic guided ODE, with FIG and OT-ODE as closely related deterministic precedents."

**Location** — Results 5.2, "Classical filters with $E=64$ are not competitive."
**Severity** — minor.
**Problem** — Overbroad: at 5% sensors the $E=64$ EnKF (RMSE 2.768, CRPS 1.384) beats DM-SDE shared (3.424/1.823), FM-ODE shared (3.828/2.051), every Jac-free variant (~6.8–7.0), and FlowDAS+SURGE. It loses only to SI-SDE shared, SDA+SURGE, and D-Flow there.
**Confidence** — high.
**Fix** — "The classical filters at $E=64$ never attain the best accuracy in any scenario" or "are not competitive with the best methods."

**Location** — Results 5.2 preamble, "both SDA and FlowDAS perform better when combined with SURGE, so we report only the SURGE-enhanced versions."
**Severity** — minor.
**Problem** — The paper's own Table 3 contradicts both halves: it *does* report plain SDA and FlowDAS, and at $M=500$ FlowDAS+SURGE (KL $5\times10^8$) is catastrophically worse than plain FlowDAS (3.85).
**Confidence** — high.
**Fix** — "…so we report only the SURGE-enhanced versions in the field cases (both variants appear in Table 3; SURGE's advantage degrades at large $M$, Appendix G)."

**Location** — C8/C21; Abstract sentence 6, "make the method tractable at high dimension"; contribution 3, "making the sparse-observation regime feasible at field scale."
**Severity** — minor.
**Problem** — Evidence is runtime columns at $128^2$ states only; no scaling study, and the paper's own limitation (C32) notes shared-covariance cost grows with $N_y$. The contribution bullet also lumps the Jacobian-free covariance into sparse feasibility, though Table 1 shows it fails precisely there (6.97 vs 1.73).
**Confidence** — high on what the evidence covers.
**Fix** — Abstract: "tractable at the field scales we consider"; bullet: "a Jacobian-free covariance for dense observations and an ensemble-shared Jacobian that makes the sparse regime feasible at field scale."

**Location** — C19; contribution 1, "One guidance term … turns any interpolant-based generative model into a family of posterior samplers."
**Severity** — minor.
**Problem** — The active bullet dropped both the theorem reference and the exactness qualifier that the commented-out variant carried; as written the claim has no stated evidentiary anchor or condition.
**Confidence** — high.
**Fix** — Append "(Theorem 4.1; exact given the exact intermediate likelihood score)."

**Location** — Statements page: "Code … is provided as anonymised supplementary material"; AI use statement.
**Severity** — minor.
**Problem** — Could not verify a supplementary bundle exists; the source retains "VERIFY" scaffolding comments for both the code claim and the AI-use clause, i.e., these factual claims are marked as unconfirmed drafts by the authors themselves.
**Confidence** — high that the comments are present; cannot verify the underlying facts.
**Fix** — Before submission, confirm the code bundle and each AI-statement clause, then delete the VERIFY comments.

**Location** — C18; Introduction, "the samplers differ only in their diffusion coefficient and likelihood-score weighting."
**Severity** — nit.
**Problem** — Per the displayed members and Algorithm 1, they also differ in the initial law (SI-SDE starts at $\bx_0$; DM-SDE/FM-ODE at $\mathcal N(0,\bI)$).
**Confidence** — high.
**Fix** — "…differ only in their source, diffusion coefficient, and likelihood-score weighting."

**Location** — Results 5.1, "only SDA+SURGE also recovers the exact posterior."
**Severity** — nit.
**Problem** — Table 3 shows FlowDAS+SURGE at KL 0.0017 at $M=50$; "only" holds just for robust-across-$M$ recovery, which the next clause explains but the sentence itself overstates.
**Confidence** — high.
**Fix** — "only SDA+SURGE also recovers the exact posterior robustly across step counts."

## Claims with no finding

- C1: supported. C3: supported. C4: supported (conditional on stated theorem assumptions). C5: supported. C6: supported. C9: supported.
- C10–C16: supported (citation-backed background / internally consistent framing).
- C20: supported. C22: supported. C23: supported (Table 3 KL ≈ 1e-3 verified).
- C25: supported — 5%: 6.969/1.731 = 4.0×; 1/64: 8.500/3.854 = 2.2×, matching "2–4×".
- C27: supported — 80.21/19.27 = 4.2×, 62.30/19.79 = 3.1×, matching "3–4×" to rounding.
- C28–C33: supported (limitations accurately reflect Sections 4.2–4.3, Lemma 4.2, and the urban per-step curves).

## Summary

Of the 33 ledger claims, roughly 24 are fully supported; theory-side chain (C3–C6, C13–C16, C20, C22) plus limitations block (C29–C33) in notably good shape. Problems concentrate in: (1) explicitly unfinished ablation table (blocking); (2) abstract/conclusion superlatives contradicted by the paper's own tables ("best accuracy under dense observations" ignores EnKF-1000 at 16², "essential for sparse observations" vs "modest gains in the urban case", "any pretrained … diffusion" demonstrated only via FM-derived score); (3) evidential thinness curable by hedge or table — urban rankings on one trajectory with 0.01-level margins, and tuned hyperparameters (λ, g_0, baseline knobs) promised in Appendix F but present nowhere. All majors fixable with sentence-level edits plus one half-page hyperparameter table; only the urban-trajectory count and the ablation table would require new compute if authors choose evidence over hedging.
