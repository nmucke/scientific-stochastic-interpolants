# Stage 1 — Agent C (Internal consistency) raw report

Status: received, NOT yet verified by orchestrator (Stage 3 pending).
Base directory for all paths: `manuscript/`.

## §1 Introduction

**Location** `sections/introduction.tex:8`, "a deterministic guided ODE, including FIG and OT-ODE" / **Severity** minor / **Problem** A single ODE cannot "include" two methods; the commented predecessor correctly said FIG/OT-ODE are *existing examples* of the deterministic member. Also blurs "guided ODE" (ours, FM-ODE) with the baseline "Guided FM (FIG)". / **Confidence** high / **Fix** "and a deterministic guided ODE, of which FIG and OT-ODE~\citep{yan_fig_2024, pokle_training-free_2024} are existing examples."

**Location** `sections/introduction.tex:19`, "an ensemble-shared Jacobian making the sparse-observation regime feasible" / **Severity** nit / **Problem** Contribution bullet has no finite verb. / **Confidence** high / **Fix** "making" → "make".

## §2 Related work — no findings.

## §3 Preliminaries

**Location** `sections/preliminaries.tex:59–64`, dangling "with" after eq:SI_SDE / **Severity** major / **Problem** Broken sentence visible in compiled paper (p. 3). / **Confidence** high / **Fix** Delete line 63 ("with") so the equation is followed directly by "where the drift can be decomposed…".

**Location** `sections/preliminaries.tex:48–50`, eq ends comma then "Thus" / **Severity** nit / **Fix** Trailing comma → period.

**Location** `sections/preliminaries.tex:56`, eq:DM_SDE ends comma before paragraph break / **Severity** nit / **Fix** Comma → period.

**Location** `sections/preliminaries.tex:45` "$\fmvel(\bx,\bx_0,\tau)$" (also appendix_score_velocity_duality.tex:106, :93; appendix_analytical_case.tex:23,33) / **Severity** nit / **Problem** Mixed argument conventions; line 106 carries τ both as subscript and argument. / **Fix** Normalise to subscript form.

## §4 Methodology

**Location** `sections/methodology.tex:114`, "$A_\tau$ and $c_\tau$ defined in \eqref{eq:SI_score}." / **Severity** minor / **Problem** Missing verb; Eq. (32) lives in Appendix A.1 — main text sends reader to appendix equation without saying so. / **Confidence** high / **Fix** "$A_\tau$ and $c_\tau$ are defined in Eq.~\eqref{eq:SI_score} in Appendix~\ref{appendix:score_reading}."

**Location** `sections/methodology.tex:160`, bare \ref renders "a full trajectory 2" / **Severity** minor / **Fix** "(Figure~\ref{fig:sequential_da})".

**Location** `sections/methodology.tex:170–178`, Algorithm 1 "$\bm c_\tau=\bm 0$" / **Severity** major / **Problem** $\bm c_\tau$ is set (line 171) but never used — the update (line 178) recomputes the correction inline, so at τ=0 it references $\interpolantscorefun$, which the IF-branch never defines. $\bm c_\tau$ also collides with the SI schedule term $c_\tau(\bx,\bx_0)$ of Eq. (32). / **Confidence** high / **Fix** Line 171 → "Set $\interpolantscorefun_\tau^{\interpolantobs_\tau}=\bm 0$" (drops unused colliding symbol, makes line 178 well-defined at τ=0, matches "correction from τ=Δτ onwards").

Checked, consistent: Algorithm 1 line 175 dimensional check OK; Eq. 17 damping matches inactive App. C remark; bold-H vs italic-H bridged in Lemma 4.2.

## §5 Results

**Location** `sections/results.tex:64–66`, Eq. (18) "$-\alpha\vorticity + \varepsilon\,\rd\xi$" / **Severity** minor / **Problem** α (drag) and ε (forcing) defined only in Appendix H; overload α_τ schedule and latent ε; bold v (fluid) shares glyph with generative velocity. / **Confidence** high / **Fix** Extend line 66 defining the symbols inline with "(unrelated to the schedule $\alpha_\tau$ and driver $\latentnoise$)".

**Location** `sections/results.tex:120`, "${\approx}700$,s" and "${>}100$,s" / **Severity** minor / **Fix** `$\,s` both.

**Location** `sections/results.tex:128` (Table 1 caption) vs rows, "equal to it to 2 decimal places" / **Severity** major / **Problem** Bolding applied to *unscaled* values while columns display ×10⁻¹ values, so bolded "ties" differ in the second displayed decimal (32² RMSE 0.638/0.626; 16² CRPS 1.062/1.128; 32² CRPS 0.347/0.333/0.327). §5.2 claim "best CRPS in three of four scenarios" true only under the hidden tie rule — under the caption's literal rule it becomes two of four. / **Confidence** high / **Fix** Caption: "…equal to it to 2 decimal places \emph{before the $\times10^{-1}$ scaling}." (or re-bold on displayed precision and change the claim to "two of four").

**Location** `sections/results.tex:38` vs Table 1 header (line 138), Table 2 header (line 337) / **Severity** minor / **Problem** Prose/figures use percentages; tables use fractions 1/64, 1/128; nothing links them. / **Fix** Line 38: add "($=\!1/64$)" and "($=\!1/128$)".

**Location** `sections/results.tex:38` "of the domain" vs `appendix_urban_case.tex:22` "of fluid cells" / **Severity** minor / **Problem** Urban sensor densities stated over different denominators (domain vs fluid cells). / **Confidence** medium / **Fix** Line 38: "…for the urban case (fractions of fluid cells there)".

**Location** `sections/results.tex:179–190`, stale comment "1.5625% sensor density" but macros use sparse_0_78125 / **Severity** nit / **Fix** Fix comment.

Checked, consistent: urban 145/every-third/49 matches App. I; conclusion numbers match Table 2; "one-tenth the cost" at 32² and "2–4×" sparse gains match Table 1.

## §6 Conclusion

No mismatches; edge case "3–4× lower cost" where Table 2 gives 4.16× at 1/64 — acceptable rounding (nit).

## Appendices A–C (proofs) — no findings.

## Appendix D

**Location** `appendix_implementation.tex:11` "Inerpolant schedules." / nit / Fix typo.
**Location** `appendix_implementation.tex:21` "trajectorires" / nit / Fix typo.

## Appendix E Metrics

**Location** `appendix_metrics.tex:17–21, 36–40`, "Energy-spectrum RMSE" and "Marginal KL … reported separately over observed and unobserved points" / **Severity** minor / **Problem** Neither metric appears in any table or figure of the active paper, yet the KL paragraph claims it is "reported". / **Confidence** high / **Fix** Delete the marginal-KL paragraph (or "can be reported"); delete or reframe spectral-RMSE as context for Fig. 14.

**Location** `appendix_metrics.tex:51`, "both at matched ensemble size" / nit / **Problem** "both" is an orphan from the deleted NFE metric. / **Fix** Delete "both ".

**Location** `appendix_metrics.tex:19` $\mathcal{K}$ shells vs `appendix_ns_case.tex:22` $\mathcal{K}$ forced modes / nit / **Fix** Rename forced-mode set to $\mathcal{K}_f$.

## Appendix F

**Location** `appendix_methods.tex:12,18`, FIG "$k$ … $c(1-\tau)/\tau$", D-Flow "$\lambda\|\bx_0\|^2$" / **Severity** nit / **Problem** k, c, λ reuse the cadence k, SI term c_τ, damping λ of §4.3. / **Fix** Rename to $\lambda_{\mathrm{DF}}$, $k_{\mathrm{FIG}}$, $c_{\mathrm{FIG}}$ or add "(unrelated to §4.3's $k,\lambda$)".

## Appendix G — no findings (all §5.1 numbers match Table 3).

## Appendix H

**Location** `appendix_ns_case.tex:10`, verbless fragment promising "the Jacobian-free companion table" that does not exist / **Severity** minor / **Fix** Rewrite sentence; drop the companion-table clause.

**Location** `appendix_ns_case.tex:192` Fig. 14 caption "one held-out trajectory" vs line 10 "five held-out trajectories" / **Severity** major / **Problem** Direct contradiction about evaluation protocol within the same appendix. / **Confidence** high / **Fix** Whichever is true: caption → "of the five held-out trajectories", or preamble → "five held-out trajectories (spectra: one trajectory)".

**Location** `appendix_ns_case.tex:198,203`, H.3 prose "Table 4 collects the results" vs caption "Run in progress; cells (--) are placeholders" / **Severity** blocking / **Problem** H.3 asserts results the table does not contain. / **Confidence** high / **Fix** Fill Table 4 or delete §H.3 + table (and drop "and the ablations" from line 10).

## Appendix I

**Location** `appendix_urban_case.tex:25`, "$128\times128\times128,\mathrm{m}^3$ … $1,\mathrm{m}$" / **Severity** minor / **Problem** Comma-for-`\,` typo; renders "128,m³", "1,m". / **Fix** `\,` both.

**Location** `results.tex:336` Table 2 "s/step" vs metrics appendix "per assimilation step" / **Severity** minor / **Problem** Urban assimilates every third of 145 forecast steps; "s/step" ambiguous (per forecast step vs per posterior update). / **Confidence** medium / **Fix** Caption: "cost in seconds per assimilation (posterior-update) step" — or "per forecast step", whichever matches.

Checked, consistent: rollout wording agrees across main text/App. I/captions; "final forecast step" (urban) vs "final assimilation step" (NS) both correct for their protocols; Jacobian-refresh phrasing consistent; "FDAS + SURGE" only in commented-out table; no undefined/multiply-defined references in main.log.
