# Stage 1 — Agent F (Presentation) raw report

Status: received, NOT yet verified by orchestrator (Stage 3 pending).
Inspection basis: rendered PDF pages 1–9, 20–22, 26–31, 35–36 checked against source.

---

**Location** Table 4, p. 30, "Run in progress; cells (--) are placeholders."
**Severity** blocking
**Problem** An entirely empty table (every data cell `--`) with a "run in progress" caption is visible in the submitted PDF; Appendix H.3 (p. 21) builds a four-axis ablation narrative around it.
**Confidence** high (rendered PDF + `sections/appendix_ns_case.tex` lines 200–218)
**Fix** Either fill the table before submission or delete the table plus the last sentence of H.3 and soften H.3 to describe the axes studied elsewhere.

**Location** Appendix G.2, p. 20, "together with the convergence of the KL divergence ... and one-dimensional density slices"
**Severity** major
**Problem** Text describes Figure 6 as containing KL-vs-g_τ, KL-vs-M, and 1-D density-slice panels, but the rendered Figure 6 has only the four density heatmaps; the extended figure is commented out (`appendix_analytical_case.tex` lines 92–126).
**Confidence** high (rendered p. 20 vs source line 49)
**Fix** Delete ", together with the convergence of the KL divergence ... and one-dimensional density slices" from line 49.

**Location** Figures 11–13, pp. 26–28 (NS RMSE/CRPS/spread–skill curves)
**Severity** major
**Problem** Three full-page appendix figures are never referenced by number anywhere in the active text; only pointer is vague "Additional convergence, error, and enstrophy results are given in Appendix H" (p. 7).
**Confidence** high (grep over all active `sections/*.tex`)
**Fix** In Appendix H intro add: "Figures~\ref{fig:ns_rmse_curves}--\ref{fig:ns_spread_skill_curves} show RMSE, CRPS, and spread--skill versus $M$ and versus assimilation step."

**Location** Figures 19–21, pp. 35–36 (urban per-step curves)
**Severity** major
**Problem** No active text references `fig:urban_rmse_vs_step`, `fig:urban_crps_vs_step`, or `fig:urban_ss_vs_step`; the referencing paragraph is commented out (`appendix_urban_case.tex` lines 1054–1055). Main text says errors "accumulate (Appendix I)" without naming the figures — and the conclusion leans on these curves.
**Confidence** high
**Fix** Add: "Figures~\ref{fig:urban_rmse_vs_step}--\ref{fig:urban_ss_vs_step} resolve RMSE, CRPS, and spread--skill over the 145-step rollout."

**Location** Figure 21(a)–(b), p. 36, subcaptions "(a) Velocity, sparse 0.78125%."
**Severity** major
**Problem** Subcaptions of the two velocity panels are printed ON TOP OF the "Assimilation step" x-axis labels — an overt visual collision, caused by `\vspace{-17pt}` (temperature panels use `-7pt` and are fine).
**Confidence** high (rendered p. 36 + `appendix_urban_case.tex` lines 1182, 1187)
**Fix** Change both `\vspace{-17pt}` to `\vspace{-7pt}` in the `fig:urban_ss_vs_step` block.

**Location** §4.2, p. 5, "$A_\tau$ and $c_\tau$ defined in (32). The FM/DM score is recovered ... through (31)."
**Severity** major
**Problem** (i) Reader is sent forward to equations (31)–(32) in Appendix A.1 with no indication they are in an appendix (next main-text equation is (18)); (ii) sentence is a fragment (missing "are").
**Confidence** high (rendered p. 5 + `methodology.tex` line 114)
**Fix** "$A_\tau$ and $c_\tau$ are defined in Appendix~\ref{appendix:score_reading}, Eq.~\eqref{eq:SI_score}; the FM/DM score is recovered from the velocity via Eq.~\eqref{eq:fm_score} there."

**Location** §4.1/§4.4, pp. 4–6, "DM-SDE $(g_\tau>0)$"
**Severity** major
**Problem** DM-SDE's actual diffusion schedule never specified in active text: endpoint-vanishing $g_\tau=g_0\sqrt{\sigma_\tau\beta_\tau}$ appears only in Appendix G and a commented-out paragraph; Appendix D omits it, and $g_0$'s value is never given for the field cases. DM-SDE rows of Tables 1–2 not reproducible.
**Confidence** high (grep across active sections + Appendix D read)
**Fix** Add to Appendix D: "The DM-SDE uses the endpoint-vanishing diffusion $g_\tau=g_0\sqrt{\sigma_\tau\beta_\tau}$ with $g_0=\langle value\rangle$ in all experiments," and after the three displayed samplers in §4.1 append "(we use the endpoint-vanishing $g_\tau\propto\sqrt{\sigma_\tau\beta_\tau}$; Appendix D)."

**Location** Table 1 (p. 8) and Table 2 (p. 9) column headers "16² 32² 5% 1/64" / "1/64 1/128"
**Severity** major
**Problem** Captions never define scenario headers, and notation switches units mid-paper: text uses "5% or 1.5625%" / "1.5625% or 0.78125%", tables use "1/64", "1/128". Reader must compute 1/64 = 1.5625% to connect columns to prose and Figures 4–5.
**Confidence** high (rendered pp. 7–9 vs `results.tex`)
**Fix** Add to captions: "random sensors at $5\%$/$\tfrac{1}{64}=1.5625\%$" (Table 1); "sensor densities $\tfrac{1}{64}=1.5625\%$ and $\tfrac{1}{128}=0.78125\%$" (Table 2) — or reuse percentages as headers.

**Location** §4.4, p. 6, "gives a full trajectory 2."
**Severity** minor
**Problem** Bare `\ref{fig:sequential_da}` renders as naked "2".
**Confidence** high (rendered p. 6 + `methodology.tex` line 160)
**Fix** `a full trajectory (Figure~\ref{fig:sequential_da})`.

**Location** §3, p. 3, "in the F"ollmer formulation of Chen et al."
**Severity** minor
**Problem** `F{"o}llmer` missing backslash — PDF prints a literal double quote.
**Confidence** high (rendered p. 3 + `preliminaries.tex` line 59)
**Fix** `F{\"o}llmer`.

**Location** §3, p. 3, "with where the drift can be decomposed into a velocity and score"
**Severity** minor
**Problem** Leftover "with" before "where" (editing artifact); sentence states the same identity twice.
**Confidence** high (rendered p. 3 + `preliminaries.tex` lines 63–64)
**Fix** Delete stray `with` line; "where the drift decomposes as $b_\tau=v_\tau+\tfrac12\gamma_\tau^2 s_\tau$ (Appendix~\ref{appendix:score_reading})."

**Location** §5.2, p. 7, "D-Flow costs ≈700,s per step" and ">100,s per step"
**Severity** minor
**Problem** Literal commas render inside units ("700,s", "100,s").
**Confidence** high (rendered p. 7 + `results.tex` line 120, two occurrences)
**Fix** `700$\,s` and `100$\,s`.

**Location** Figure 3 caption, p. 7, "KL to the exact posterior vs. sampler steps M (five seeds)."
**Severity** minor
**Problem** Caption names neither the system nor the metric definition.
**Confidence** high
**Fix** "Analytical linear–Gaussian case: Gaussian KL divergence to the exact posterior vs. sampler steps $M$ (five seeds; Appendix~\ref{appendix:metrics})."

**Location** Figure 4 caption, p. 8, "All methods and scenarios: Figures 7–10."
**Severity** minor
**Problem** Figures 7–10 show 8 of the 12 methods (DM-SDE variants, plain FlowDAS, plain SDA omitted); unlike the urban appendix no text states this omission for NS.
**Confidence** high (rendered p. 22 vs figure macro)
**Fix** "All scenarios (retained methods): Figures~\ref{fig:ns_fields_32}--\ref{fig:ns_fields_sparse1p5}", and note the omission there.

**Location** Appendix H intro, p. 21, "The data generation and forcing, qualitative reconstructions ..., the Jacobian-free companion table, and the ablations."
**Severity** minor
**Problem** Verbless fragment; promises a "Jacobian-free companion table" that does not exist in Appendix H.
**Confidence** high (`appendix_ns_case.tex` line 10)
**Fix** Rewrite listing what exists; drop the companion-table clause.

**Location** Figure 6, p. 20, panels (a)–(d)
**Severity** minor
**Problem** Panels carry both internal matplotlib titles and duplicate LaTeX subcaptions; (d) renders vertically misaligned; top-level caption adds nothing.
**Confidence** high (rendered p. 20)
**Fix** Regenerate without internal titles (or drop subcaptions); extend main caption.

**Location** Table 2 / §6, p. 9, conclusion wrapped beside wraptable
**Severity** minor
**Problem** wraptable with `\vspace{-50pt}` leaves the Conclusion in a ~0.3-textwidth column of 4–5 words/line for ~10 lines; fragile layout.
**Confidence** high (rendered p. 9 + `results.tex` lines 317–318)
**Fix** Convert to `table*[t]` if slack exists; else minimise the negative vspace and re-check after every §5.3 edit.

**Location** Figures 3, 11–14, 19–21 legends (e.g. p. 26)
**Severity** minor
**Problem** Ten-series line plots rely mainly on hue; pink/magenta and orange/olive pairs merge in greyscale; pink vs magenta hard for deuteranopes. Solid/dashed already separates shared/Jac-free.
**Confidence** med (no greyscale simulation run)
**Fix** Give Guided FM (FIG) a distinct dash pattern or darken it in the palette; no TeX change.

**Location** Appendix D, pp. 16–17, "Inerpolant schedules." / "20 validation trajectorires"
**Severity** nit
**Problem** Typos; also App. H.1 "i.e at every 5000 time step".
**Confidence** high
**Fix** "Interpolant schedules.", "validation trajectories", "i.e.\ every 5000th time step".

**Location** Figures 7, 15–18 colourbar labels (e.g. p. 31)
**Severity** nit
**Problem** Rotated colourbar unit labels render at ~4–5 pt — edge of print legibility.
**Confidence** med
**Fix** Increase colourbar label font in generation scripts when regenerating.

**Location** Abstract, p. 1, final sentence
**Severity** minor
**Problem** Abstract lists testbeds but states no finding — no takeaway from page 1.
**Confidence** high
**Fix** Append e.g.: "With the inflated covariance the samplers recover the exact linear–Gaussian posterior, and on sparse Navier–Stokes observations the shared covariance reduces RMSE by 2–4× over the Jacobian-free variant at competitive cost."

## No-finding areas
- No empty `\IfFileExists` placeholder boxes visible anywhere in rendered PDF (Table 4 is the only rendered placeholder).
- Fig 3 wrapfigure renders cleanly on p. 7.
- Figures 4/5 column labels adequate as rendered.
- Intro structure: problem statement and prior-work failure precede "unified view"; non-specialist gets the claim from pp. 1–2.
- Field-plot colour maps (RdBu, magma): greyscale- and CVD-safe.
- Figures 6–10, 14–18: all referenced in active text; Figures 1–5, Tables 1–3 all referenced.
- Main paper tight at 9 pages; no padded section.
