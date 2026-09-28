# Final review report — Observation-Interpolant DA paper (ICLR, 9 pp., anonymised)

Process: Stage 0 shared map (`paper-map.md`) → Stage 1 eight parallel agents A–H (raw reports in `stage1/`) → Stage 2 targeted verifiers for the one flagged proof gap and the flagged SDA mischaracterisation (`stage2/`) → Stage 3 orchestrator verification against source, tex tables, configs, and result CSVs (`stage3-verification-log.md`). Every blocking and major finding below was re-checked by me against the source or data; confidence notes say when it was not fully checkable.

---

## Verdict

**Needs an experiment** (short of that, a substantive claims-reframe plus data-reporting fixes gets you to "accept after text edits" territory, but not with the current sparse-observation narrative).

**The one driving reason:** the paper's comparative empirical record is contradicted and under-powered by its own data — the repo's frozen NS aggregate contains bare-SDA rows that beat every method reported in Table 1 in both sparse scenarios on RMSE, CRPS, *and* calibration, while the paper reports only the SURGE-wrapped (worse) variants on a justification sentence its own data refutes; on top of that, the ablation table is an empty "run in progress" placeholder and the flagship urban case is a single trajectory with no dispersion anywhere.

---

## Blocking findings (ordered by severity, then cost to fix)

### BL-1. Table 1 omits bare SDA/FlowDAS, which the repo's own data shows dominating the sparse scenarios
- **Location:** §5, `results.tex:36` — "perform better when combined with SURGE, so we report only the SURGE-enhanced versions"; Table 1.
- **Problem:** Verified directly in `paper_experiments/results/navier_stokes/aggregated/all.csv` (same frozen grid, M=100, n=5): bare SDA at sparse 5% has RMSE 1.051, CRPS 0.529, spread–skill 1.19 (×10⁻¹) vs SURGE(SDA) 1.616/1.130/9.385 and Ours SI-SDE (shared) 1.731/0.872/1.965 — bare SDA is best on **all three metrics**, and again at 1/64 (1.784/0.903/0.743). Bare FlowDAS also beats FlowDAS+SURGE in both sparse scenarios. The justification sentence is accurate to the *cited paper* (Stage-2/E verified wei_surge_2026 does claim it) but is contradicted at field scale by the authors' own runs. If code ships as promised, a reviewer can find this in one grep. Note: dense scenarios survive — ours still wins at 16²/32².
- **Confidence:** high (data verified; medium only on whether the authors consider those bare rows final).
- **Fix:** Zero compute: add the two bare rows to Table 1 from the existing aggregate, replace the justification sentence (e.g. "SURGE improves both guidance methods in the exact-posterior regime (Table 3) but degrades them at field scale, where it collapses the ensemble; we report both"), and reframe the sparse-NS claims (abstract "essential", §5.2, conclusion) around what remains true: ours is best-calibrated among the accurate methods at its cost point, and the inflated covariance is what closes the gap *within* the interpolant family. ~Half a day of writing; the narrative decision is yours (see Open questions Q1).

### BL-2. Table 4 (ablations) is an empty placeholder in the compiled PDF
- **Location:** Appendix H.3, `appendix_ns_case.tex:198–218` — caption: "Run in progress; cells (--) are placeholders."
- **Problem:** H.3 promises covariance/k/λ/g_τ/M/E ablations; every cell is `--`. All six agents and the hostile reviewer flagged it; it would be quoted in a meta-review. Agent D inventoried what exists: the **M sweep** (M∈{25,50,100,250}, both covariance modes, all scenarios, 5 traj) is fully present in the aggregate; the covariance axis is Table 1; a partial λ sweep exists (`tuning_lambda_check/`, λ∈{0.9,0.95,1.0}, 2 scenarios); `run_ns_ablation.sh` for k∈{1,5} is written but never run; no g_τ or E sweep exists. Config comments even carry an informal k result (sparse 5%: k=1 → 0.083 vs k=10 → 0.141, +69% RMSE) — showing the cadence choice has real accuracy cost that the paper currently hides.
- **Confidence:** high.
- **Fix:** Cheapest: fill the M and covariance rows from existing aggregates and delete the unrun axes from both the table and the H.3 paragraph (hours, no compute). Fuller: run `run_ns_ablation.sh` on one scenario for k and small g_τ/E sweeps (~35–110 GPU-h). Deleting the whole section is also acceptable — an empty table is strictly worse than no table.

### BL-3. Required AI-use statement is an unverified draft; supplementary-code claim is unverified
- **Location:** `statements.tex:6–8, 14–17, 22–26` — DRAFT/VERIFY comments around the AI-use statement and "provided as anonymised supplementary material".
- **Problem:** The AI statement (required by ICLR 2027) is marked DRAFT in-source; the clause "did not use them to develop the theory … or derive and write their proofs" must be true under a policy where even AI-checking a proof requires disclosure. The reproducibility statement asserts a supplementary bundle that does not exist yet.
- **Confidence:** high that the comments/claims are unresolved; the underlying facts only you can verify.
- **Fix:** Verify each clause factually, edit as needed, delete the comment blocks, and build the anonymised supplement (strip names, paths, git history) before submission. 2–4 h.

---

## Major findings

### MJ-1. No dispersion anywhere; urban case is n=1; bolded winners within noise
- **Location:** Tables 1–3; Table 2 caption "One held-out trajectory"; conclusion sentence 4.
- **Problem:** NS aggregate carries cross-trajectory std that never reaches the paper: SI-SDE (shared) sparse-5% is 1.731±0.198 vs SDA+SURGE 1.616±0.161 — the bold is within 1σ. Urban rankings (DM-SDE best velocity, SI-SDE best temperature, margins 0.01–0.05) rest on one trajectory; NS uses five; analytical Table 3 gives means over 5 seeds without σ.
- **Fix:** Print ±std in Tables 1 and 3 (script tweak, hours); either run urban trajectories 2–5 (`run_urban_grid.sh` already defaults to `TRAJ="1 2 3 4 5"`; ~200–250 GPU-h) or hedge every urban superlative to "on the trajectory tested" including in the conclusion. (A/D/G; verified.)

### MJ-2. Hyperparameters and tuning protocol promised but absent
- **Location:** §5 "details are given in Appendix F"; statement "Appendix F specifies each baseline and its tuning protocol"; the TODO comment at `results.tex:29–32` admits it.
- **Problem:** No tuned values, grids, or budget appear anywhere: ours λ (0.9 superres / 0.95 sparse+urban / 1.0 analytical — and config comments show λ=1.0 *diverges* on sparse, i.e. the method runs one grid point from divergence), DM-SDE g_0 (=1.0), FlowDAS ζ (+J=25), SDA γ_sda, FIG (k,c) (c changes 16× between cases), D-Flow (η, noise, λ_reg, chain settings), urban obs noise (σ=0.05 in *normalised* per-channel space — never stated), EnKF localization (Gaspari–Cohn r=20 on sparse scenarios only, undisclosed, while the E=1000 reference is global), divergence guard, SURGE particle count. Reproducibility from the text is currently impossible and baseline-fairness unassessable.
- **Fix:** One experimental-setup table per case in Appendix D/F transcribed from `paper_experiments/configs/` + 2–3 sentences on the tuning protocol (grid per scenario on a disjoint trajectory) + one sentence on EnKF localization. Writing only, 2–3 h. (A/D; all values verified in configs.)

### MJ-3. Abstract/conclusion overclaims contradicted in-paper
Three separately verified instances (A/D/G):
1. **"essential for sparse observations"** (abstract) — the urban case is sparse and shows 3–5% gains at 3–4× cost ("modest gains", conclusion); after BL-1 the sparse-NS superiority also needs reframing. Also the abstract attributes the effect to inflation per se, but the Jacobian-free variant (Eq. 16) is *also* inflated — what matters is the Jacobian/prior-correlation term. Fix: "…covariance inflated by the model's source covariance, whose Jacobian term carries the prior correlations that are decisive under sparse observations in our Navier–Stokes experiments."
2. **"converts any pretrained flow-based generative model … or diffusion"** — no natively-trained diffusion prior is ever used (App F: constructed from the FM velocity; config `diffusion_from_fm: true`; the native DM checkpoint exists unused, "poorly trained"). Fix: "any generative model built on the affine interpolant path" + one §5 sentence disclosing the FM-derived diffusion prior.
3. **"our methods achieve the best accuracy under dense observations"** (conclusion) — at 16² the E=1000 EnKF reference has RMSE 1.442/CRPS 0.779 vs ours 2.024/1.062; Table 1's bolding silently excludes the reference row without the caption saying so. Fix: "…among methods that do not use the true solver" + caption note excluding the reference row from bolding.

### MJ-4. Theorem 4.1 is under-guarded: the SI-source + g_τ≡0 member is impossible, excluded only by an invisible hypothesis
- **Location:** Theorem 4.1 (`methodology.tex:16–27`) and "every admissible diffusion schedule gives exact posterior samples".
- **Problem:** Stage-2 independently confirmed Agent B: for the point-mass SI source with g=0, no well-posed deterministic flow can produce the non-degenerate marginals p_τ^y; the theorem survives only because its well-posedness/uniqueness hypothesis provably fails there (FP non-uniqueness at δ_{x_0}; velocity ~x/(2τ) at the anchor). Nothing tells the reader; the authors' own remark stating this sits commented out (`appendix_proof_conditioning.tex:66–68`). All other steps of the B.1 → A.3 → Thm 4.1 chain PASS independent verification. Not blocking: the theorem asserts nothing false and the three displayed members avoid the bad combination.
- **Fix (one sentence, after line 27):** "Admissibility is not vacuous: for the point-mass SI source $p_0=\delta_{\bx_0}$ the deterministic choice $\gdiff_\tau\equiv0$ violates the well-posedness hypothesis — a well-posed deterministic flow transports $\delta_{\bx_0}$ to a point mass and cannot realise the non-degenerate marginals $p_\tau^{\obs}$ — so the guided ODE member requires a Gaussian source, and SI priors are used only with $\gdiff_\tau=\gamma_\tau>0$." Optionally reinstate the appendix remark with its "requires g>0" clause reworded to reference the well-posedness hypothesis.

### MJ-5. The Appendix F critique of SDA's guidance weight is mathematically incorrect
- **Location:** `appendix_methods.tex:15`, "conditions trajectories rather than marginals … so the intermediate marginals are not $p_\tau^{\obs}$".
- **Problem:** Stage-2 read the actual SDA paper: SDA samples the **native** VP reverse SDE, and for a native diffusion the paper's own coefficient gives κ_τ = ½g², hence w_τ = g² — the Doob weight and the marginal-path weight coincide, and with an exact tilt the h-transform marginals ARE p_τ^y. (I re-derived this independently; it checks out.) SDA's real bias is the Gaussian–Tweedie surrogate with isotropic Γ — which is, moreover, an *inflated* ΠGDM-style covariance, so "DPS-style" is also off-key, and "learns an all-at-once score over the whole trajectory" is wrong (SDA learns local segment scores and composes them). This matters doubly because the weight analysis is the paper's main theoretical differentiation from SDA and features in the natural rebuttal to the novelty attack.
- **Fix:** Replacement sentences in `stage2/sda-verification.md` §(d) — keeps the true part (the weight does not transfer to non-native/vanishing diffusion; the isotropic covariance carries no source information) and drops the false part. 30 min.

### MJ-6. FIG mischaracterised as flow-matching-only
- **Location:** `related_work.tex:13` "Closest to our work is Yan et al., who interpolate the observations within a flow-matching formulation"; intro "including FIG and OT-ODE".
- **Problem:** FIG's abstract states it uses "a pre-trained diffusion or flow-matching model as a prior", and its official repo ships diffusion variants — so the "we generalize to all three, FIG is FM-only" delta is overstated on the FM/DM axis. The true deltas (derived closed-form moments with bias + source-inflated covariance vs heuristic corrector; the SI member; SDE members with the analytic weight; autoregressive DA) are strong enough without the false one. Confidence: high via abstract + official code; the full PDF was bot-blocked (both attempts), so the specific corrector formula "c(1−τ)/τ, normalised residual" in App F rests on your own reading/implementation — please confirm it.
- **Fix:** "…who interpolate the observations to guide deterministic sampling of flow-matching and diffusion priors in one-shot inverse problems, with a heuristic corrector rather than a derived likelihood covariance." Also change intro "including FIG and OT-ODE" → "of which FIG and OT-ODE are existing deterministic examples" (a single ODE cannot "include" two methods, and FIG's weight is not the family's κ_τ — the paper's own App F says so).

### MJ-7. Missing closely-related work a reviewer will name
- Dou & Song, "…A Filtering Perspective" (FPS), **ICLR 2024 spotlight** — noised **measurement sequence** paired with diffusion states + SMC: the diffusion-side precedent for evolving the observation along the generative path; directly adjacent to the observation-interpolant claim.
- Huang et al., **DiffDA** (ICML 2024) and Manshausen et al., sparse-station generative DA (JAMES 2025; arXiv:2406.16947) — weather-scale generative DA, the same sparse-sensor regime.
- Cardoso et al., **MCGDiff** (ICLR 2024) — asymptotically exact SMC diffusion posterior sampling for exactly the linear–Gaussian setting of your analytical case; belongs next to SURGE.
- **Fix:** three sentences + four bib entries (exact suggested sentences in `stage1/agent-E-relatedwork.md`). ~1 h.

### MJ-8. Internal contradictions and unreferenced material in the appendices
1. Fig 14 caption "one held-out trajectory" vs App H preamble "five held-out trajectories" (`appendix_ns_case.tex:10` vs `:192`) — code says the spectra use one; state which. (C; verified.)
2. Figures 11–13 (pp. 26–28) and 19–21 (pp. 35–36) are **never referenced** in any active text — six full-page figures, including the per-step curves the conclusion's error-accumulation claim leans on; the referencing paragraph is commented out. Fix: two pointer sentences. (F; verified by grep.)
3. App G.2 text promises KL-vs-g_τ, KL-vs-M, and density-slice panels that are commented out of Figure 6 (`appendix_analytical_case.tex:49`). Fix: delete the clause. (F; verified.)
4. Fig 21's two velocity subcaptions are printed on top of the x-axis labels (`\vspace{-17pt}` vs the temperature panels' `-7pt`). Fix: `-7pt`. (F; source verified.)

### MJ-9. Table 1 bolding/tie rule inconsistent with displayed digits; "best CRPS in three of four" depends on it
- **Location:** Table 1 caption vs bolded values (0.638 and 0.626 both bold; 1.062 and 1.128 both bold; 0.347/0.333/0.327 all bold).
- **Problem:** Ties were computed on unscaled values (2 dp before the ×10⁻¹ scaling), which the caption doesn't say; under the caption's literal rule the §5.2 claim "best CRPS in three of four scenarios" becomes two of four.
- **Fix:** Caption: "…equal to it to 2 decimal places before the ×10⁻¹ scaling", or re-bold on displayed precision and write "best or near-best CRPS in three of four". (A/C; verified.)

### MJ-10. DM-SDE's diffusion schedule and amplitude are unspecified in the active text
- **Location:** §4.1 "DM-SDE (g_τ>0)"; Appendix D.
- **Problem:** The endpoint-vanishing g_τ = g_0√(σ_τβ_τ) appears only in Appendix G and a commented-out paragraph; g_0 (=1.0) is nowhere. The DM-SDE rows of Tables 1–2 are irreproducible; also the label "g_τ>0" contradicts g_0=g_1=0 (should be "g_τ>0 on (0,1)"). (F/D/B; verified.)
- **Fix:** One sentence in §4.1 + one in Appendix D with the value.

### MJ-11. Algorithm 1's τ=0 branch is broken as written
- **Location:** `methodology.tex:170–178`.
- **Problem:** Line 171 sets c_τ=0 but c_τ is never used; the update line applies (κ_τ+½g²)ŝ directly, so at τ=0 it references a never-computed ŝ and (for FM) an undefined κ_0. The variable also collides with the SI schedule term c_τ of Eq. (32).
- **Fix:** Either set ŝ=0 in the branch and drop c_τ, or restore the commented line 176 defining c_τ=(κ+½g²)ŝ and use c_τ in the update. Two-line edit. (B/C; verified.)

---

## The hostile review (Agent G) and my assessment

Full text in `stage1/agent-G-adversarial.md`. Summary and adjudication:

**Attack 1 — "The empirical record is not submission-complete"** (empty Table 4; single urban trajectory; no error bars; divergent runs silently dropped from Fig 14). *Lands, fully.* Every factual element verified. G's verdict "fixable only by experiment" is half-right: the M/covariance ablation rows and NS error bars are fillable from existing data; the k/λ/g_τ/E axes and urban replication need compute. The rebuttal G drafts (covariance ablation exists across Tables 1–3, M-convergence in Figs 3/11–13) is partially effective but cannot excuse a table stamped "run in progress".

**Attack 2 — "The central sparse-observation claim is contradicted by the paper's own Table 1, and baselines are reported only in a pathological wrapped form."** *Lands, and is stronger than G knew.* G argued from Table 1 (SDA+SURGE wins sparse RMSE but is collapsed, so "ours best CRPS" is a decent rebuttal). My data verification (BL-1) shows the rebuttal fails against **bare** SDA, which wins RMSE, CRPS, and calibration in both sparse scenarios in the repo's own aggregate. The paper's sparse-NS story must be rebuilt around BL-1's fix; "essential" cannot stand.

**Attack 3 — "The theorem's exactness is vacuous for everything run; the delta over FIG+ΠGDM is incremental; urban FM-ODE(JF) ≡ FIG row-for-row."** *Lands partially.* The FIG-parity rows and the vacuity of the exactness claim for the deployed method are verified facts. But the attack overreaches in calling the theory an assembly with no content: the general-g weight w_τ=κ_τ+½g² has no located precedent (E), the exact-moment lemmas are new, and the analytical case shows FIG collapsing (KL 6×10¹¹) where the derived guidance converges. Two caveats cut the other way: the natural rebuttal G scripts leans on the App F SDA-weight analysis, which Stage-2 showed is itself wrong (MJ-5) — do not use that rebuttal as written; and the FIG-parity defense ("that's only the Jac-free variant; shared wins 2–4×") survives, but after BL-1 the comparison point becomes bare SDA, not FIG. Repositioning per MJ-3/MJ-5/MJ-6 (theorem as organizing result; contribution = derived moments + covariance approximations + calibrated-at-cost samplers) blunts this attack to a weak-reject-at-worst complaint.

G's bottom line (reject, score 3) is what a competent hostile reviewer writes against the current PDF. After the minimal edit list below, Attacks 1 and 2 are defused to the extent the data allows; Attack 3 becomes a matter of taste about framing.

---

## Minimal edit list (ordered; smallest set that moves the verdict up one level)

Doing items 1–8 moves "needs an experiment" → "accept after text edits" (with claims honestly weakened); item 9 is the experiment path that preserves stronger claims.

1. **Fill-or-delete Table 4** (BL-2): fill M+covariance rows from `aggregated/all.csv`, delete unrun axes from table+text. — *hours, no compute.*
2. **Add bare SDA/FlowDAS rows to Table 1 and rewrite the SURGE sentence + sparse-NS claims** (BL-1): abstract "essential" → qualified per MJ-3(1); §5.2 and conclusion reframed around calibration-at-cost. — *half a day, no compute. Needs your decision (Q1).*
3. **Resolve statements.tex + build the anonymised supplement** (BL-3). — *2–4 h.*
4. **Add ±std to Tables 1/3; extend the n=1 caveat to every urban superlative** (MJ-1 cheap path). — *hours.*
5. **Add the experimental-setup/hyperparameter table** (MJ-2), including urban obs-noise and EnKF localization sentences. — *2–3 h, writing only.*
6. **Claims edits** (MJ-3): "any pretrained" scoping + FM-derived-diffusion disclosure; dense-claim qualifier + caption note on the reference row; Table-1 tie-rule caption fix (MJ-9). — *1–2 h.*
7. **Theory guards** (MJ-4, MJ-11, and B's minor items): theorem guard sentence; Algorithm 1 c_τ fix; "g_τ>0 on (0,1)"; τ∈(0,1) qualifiers on Prop B.1/Lemma 4.3; R≻0 in Lemma 4.2. — *1–2 h.*
8. **Related-work corrections** (MJ-5, MJ-6, MJ-7 + minor citation fixes): SDA paragraph replacement, FIG rephrase, four added citations, Tweedie-citation list fix. — *2–3 h.*
9. **Optional experiments that upgrade claims instead of weakening them:** urban trajectories 2–5 (~200–250 GPU-h); k/λ/g_τ/E ablation on one NS scenario (~35–110 GPU-h); (nice-to-have) one natively-trained diffusion prior run to fully back "any pretrained".
10. **Mechanical sweep** (the minor/nit list below): dangling "with", bare `\ref`, "700,s"-type typos, Fig 21 vspace, unreferenced-figure pointer sentences, G.2 clause, A_τ/c_τ appendix pointer, F\"ollmer, acronym expansions, overfull figure grids, `\IfFileExists`→`\includegraphics` before the final build. — *2–3 h total.*

---

## Minor and nit findings (flat list, by section)

**Abstract / Introduction**
- [minor] Abstract states no finding; append one results sentence (F's suggestion in `stage1/agent-F-presentation.md`).
- [minor] Intro "including FIG and OT-ODE" → "of which FIG and OT-ODE are existing examples" (also part of MJ-6).
- [minor] Contribution 1 lost its theorem reference and exactness qualifier; append "(Theorem 4.1; exact given the exact intermediate likelihood score)".
- [minor] Contribution 3 lumps the Jacobian-free covariance into sparse feasibility, but Table 1 shows it failing exactly there; split: "Jacobian-free for dense; ensemble-shared for sparse".
- [minor] "tractable at high dimension" → "at the field scales we consider" (no scaling study; cost grows with N_y by the paper's own limitation).
- [nit] "differ only in their diffusion coefficient and likelihood-score weighting" → add "source,".
- [nit] Contribution bullet verbless: "making" → "make".
- [minor] SDE/ODE never expanded; expand at first use (also CRPS, RMSE, KL, LES, SGLD, EnKF in §5).

**Related work**
- [minor] Tweedie-citation list: drop song_score-based_2021 and martin_pnp-flow_2025 (PnP-Flow avoids backprop), add chung_diffusion_2023.
- [minor] Restore si_latent-ensf_2024 with an "ensemble score filters" clause (entry already in the bib; currently dead).
- [nit] fotiadis_stochastic_2024 is downscaling, not next-state forecasting; widen or drop.
- [nit] "Denoising diffusion corresponds to SDE sampling of an FM-type model" — attach ma_sit_2024 to the correspondence, not ho/song.
- [nit] Two different "closest" works (FIG in §2, FlowDAS in App F); qualify one.

**Preliminaries / Methodology**
- [minor] Dangling "with" after Eq. (7) (`preliminaries.tex:63`); delete.
- [minor] `F{"o}llmer` → `F{\"o}llmer` (renders with a literal quote).
- [minor] "$A_\tau$ and $c_\tau$ defined in (32)" — missing verb + silent forward reference into Appendix A.1; point to the appendix explicitly.
- [minor] Bare `\ref{fig:sequential_da}` renders "a full trajectory 2".
- [nit] Eq. (5) inversion condition should also exclude β_τ=0 (main text).
- [nit] Eq. (6)/(5) trailing-comma punctuation before new sentences.
- [nit] Mixed argument conventions v(x,x_0,τ) vs v_τ(x,x_0); normalise.
- [minor] Prop B.1 and Lemma 4.3 need τ∈(0,1) qualifiers; Lemma 4.2 needs R≻0 (B's items; low-cost).

**Results**
- [minor] "${\approx}700$,s" and "${>}100$,s" → `\,s` (renders "700,s"); same fix in `appendix_urban_case.tex:25` ("128,m³", "1,m"); one Unicode em-dash → `--`.
- [minor] "Classical filters with E=64 are not competitive" — overbroad; EnKF-64 beats several of "ours" at 5%; say "never attain the best accuracy".
- [minor] Table headers switch between percentages (prose) and fractions 1/64, 1/128 (tables) with no bridge; add "=1.5625%"/"=0.78125%" to captions or reuse percentages.
- [minor] Urban densities are fractions "of the domain" in §5 but "of fluid cells" in App I — different denominators; align.
- [minor] Table 2 "s/step" ambiguous (per forecast step vs per posterior update — urban assimilates every third step); specify in caption.
- [minor] Fig 3 caption doesn't name the system or metric; Fig 4 caption "All methods and scenarios: Figs 7–10" — those figures show 8 of 12 methods; say "retained methods" and note the omission.
- [nit] "only SDA+SURGE also recovers the exact posterior" — add "robustly across step counts" (FlowDAS+SURGE matches at M=50).
- [nit] "diverge as M grows at fixed per-step hyperparameters" — ζ was retuned per M for FlowDAS; reword.
- [nit] Stale comment block at `results.tex:179–190` says 1.5625% but macros use 0.78125%.
- [minor] Conclusion wraptable layout (−50pt) is fragile and squeezes the Conclusion to 4–5 words/line; convert to `table*[t]` if any slack exists; re-check p. 9 after every edit.

**Appendices**
- [minor] App H intro: verbless fragment promising a "Jacobian-free companion table" that doesn't exist; rewrite.
- [minor] App E: energy-spectrum RMSE and marginal-KL metrics are defined and claimed "reported" but appear in no table/figure; delete or reword; "both at matched ensemble size" orphan.
- [minor] Figure 6: duplicate internal titles + subcaptions, panel (d) misaligned; regenerate or drop subcaptions.
- [minor] 8 overfull hboxes of 16.6–20.7pt in the Figs 7–10/15–18 grid macros; wrap in `\resizebox{\linewidth}{!}{…}`.
- [minor] Replace `\IfFileExists` figure fallbacks with plain `\includegraphics` before the final build (silent-placeholder hazard) and delete the `\tbd` macro.
- [minor] Line-plot palette: pink/magenta and orange/olive pairs merge in greyscale; give FIG a distinct dash pattern (figure-generation change only).
- [nit] "Inerpolant schedules", "trajectorires", "i.e at every 5000 time step"; `\paragraph` headings in preliminaries missing trailing periods; colourbar labels ~4–5 pt in the 6-column grids.
- [nit] Bib: add volume/pages to rozet/chen_flowdas/wu; normalise arXiv `note` fields; delete 4 dead entries (or restore their citing sentences); "Bastiaan S. Veeling" → "Bas Veeling"; key-year suffix mismatches are cosmetic (no action).
- [nit] No ethics statement (recommended, not required).

---

## Stage 3 accounting

- **Duplicates merged:** ~15 (Table 4 was found by six agents; the SURGE/baseline issue by three; comma-unit typos, bare `\ref`, typos, and the "essential" overclaim each by 2–3). Sharpest formulation kept in each case.
- **Findings discarded:** 0 discarded outright — unusually, every blocking/major finding survived source or data verification (most agents attached file:line pointers and I re-checked the load-bearing ones against the tex, configs, and CSVs; see `stage3-verification-log.md`). Two items were **downgraded/annotated** rather than accepted as written: (i) G's scripted rebuttal to Attack 2 ("ours best CRPS") is invalidated by the bare-SDA data and is marked as unusable; (ii) G's Attack 3 framing ("theory is an assembly") is retained as an attack but not endorsed as a finding — E located no precedent for the general-g weight, and the moment lemmas are new.
- **Agent disagreements:** none unresolved. The one substantive tension — E/Stage-2 showing the paper's SDA critique is wrong while G's rebuttal script relies on it — is stated in the hostile-review assessment rather than silently resolved.
- **Agent quality:** all eight outputs were usable and specific; D was the strongest (found the bare-SDA rows and the EnKF localization); E and H needed one relaunch each after a session-limit failure, with no quality impact. Items flagged as unverifiable remain flagged: FIG's exact corrector formula (PDF bot-blocked), chen_flowdas/kohl venue entries (post-knowledge-cutoff), H's audit-sourced volume/pages.

---

## Open questions for you

1. **Bare SDA/FlowDAS rows (BL-1):** do you consider those aggregate rows final and trustworthy (they postdate the 2026-07-22 SURGE score fix, which touched only the wrapper)? If yes, the honest move is to include them and reframe; if you believe they are invalid for some reason I couldn't see, that reason needs to be in the paper.
2. **Table 4:** fill from existing data + delete unrun axes, or run `run_ns_ablation.sh` (+ small g_τ/E sweeps) for the full ablation? The k-cadence numbers hiding in the config comments (k=10 costs +69% RMSE at sparse 5% vs k=1) suggest a real trade-off worth showing rather than hiding.
3. **Urban replication:** run trajectories 2–5 (~200–250 GPU-h) or hedge all urban superlatives to n=1?
4. **Fig 14 spectra protocol:** one trajectory (what the code does) or five (what the appendix preamble implies) — which do you want the caption to state?
5. **AI-use statement:** are all clauses factually true under the ICLR 2027 policy (including "did not use AI to develop the theory or derive/write proofs" — noting the policy counts AI-checking a proof)? Only you can answer.
6. **FIG corrector formula** "c(1−τ)/τ on the normalised residual": confirm against the FIG paper/code — no agent could access the full text.
7. **Positioning after MJ-5:** with the SDA-weight critique corrected, do you want the differentiation from SDA to rest on (a) generality beyond native diffusion + (b) the source-informed covariance — as the Stage-2 replacement text does — or do you have a different preferred framing?

*Reports referenced: `review/stage1/agent-{A..H}-*.md`, `review/stage2/theorem41-verification.md`, `review/stage2/sda-verification.md`, `review/stage3-verification-log.md`, `review/paper-map.md`.*
