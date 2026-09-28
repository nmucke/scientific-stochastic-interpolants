# Stage 3 — Orchestrator verification log (running)

Each entry: finding → what I checked myself → verdict.

## Verified CONFIRMED (checked against source/data by orchestrator)

1. **Table 4 empty placeholder (A/D/F/G, blocking)** — read `appendix_ns_case.tex:200–218` in Stage 0; caption literally says "Run in progress; cells (--) are placeholders". CONFIRMED.

2. **EnKF-1000 beats "ours" at 16² dense (A, major)** — Table 1 tex: EnKF reference RMSE 1.442 / CRPS 0.779 at 16² vs best Ours 2.024/1.062. Conclusion sentence "best accuracy under dense observations" holds only at 32² unless qualified "among learned-prior methods". CONFIRMED.

3. **Bare SDA beats everything reported in both sparse NS scenarios (D, major→arguably blocking)** — verified directly in `paper_experiments/results/navier_stokes/aggregated/all.csv` (M=100, n_traj=5, same frozen grid):
   - sparse 5%: SDA rmse 0.1051±0.0085, crps 0.0529, spread_skill 0.119 — beats SURGE(SDA) 0.1616/0.1130/0.9385, Ours SI-SDE shared 0.1731/0.0872/0.1965, D-Flow 0.1974/0.0992/0.1151. Bare SDA best on ALL THREE metrics.
   - sparse 1/64: SDA 0.1784/0.0903/0.0743 — again best on all three (D-Flow 0.2313/0.1179/0.0947; ours 0.3854/0.2073/0.1954).
   - Bare FlowDAS also beats FlowDAS+SURGE in both sparse scenarios (0.6937 vs 0.8632; 0.8705 vs 1.1896).
   - Dense: ours still wins (32²: ours 0.0626–0.0667 vs SDA 0.1082; 16²: ours 0.2024 vs SDA 0.2535).
   - Urban aggregate contains NO bare SDA/FlowDAS rows (only SURGE versions were run there).
   CONFIRMED, and worse than Agent D stated: bare SDA also wins CRPS and calibration, so the "SURGE ensembles collapse / ours best CRPS" rebuttal fails against bare SDA. This interacts with G's Attack 2: the sentence "According to wei_surge_2026 … perform better with SURGE" is contradicted by the repo's own data at field scale.

4. **EnKF localization undisclosed (D, minor)** — `run_ns_grid.sh:205–206`: `+enkf_localization_radius=20` on sparse scenarios only; E=1000 reference non-localized. CONFIRMED.

5. **λ (jacobian damping) values absent from paper but in configs (A/D, major)** — `configs/method/si_sde.yaml`: superres 0.9, sparse NS 0.95, urban 0.95, analytical 1.0; comments show λ=1.0 DIVERGES on sparse and 0.95 diverges on superres_32 — i.e. the method sits one grid point from divergence, tuned per scenario; nothing in the manuscript. Config comments also contain an informal k-cadence sweep (sparse 5%: k=1 0.083 vs k=10 0.141, +69% RMSE) — strengthening the "ablation needed / k=10 is a cost choice with real accuracy cost" point. CONFIRMED.

6. **Urban obs-noise spec (D, part of repro set)** — `configs/case/urban.yaml:57–60`: R = σ²I with σ=0.05 in NORMALISED per-channel space; not stated in manuscript. CONFIRMED.

7. **Figures 11–13 and 19–21 never referenced (F, major)** — `grep -rn` over active tex: labels only defined via figure macros, zero `\ref` occurrences. CONFIRMED for all six figures.

8. **Fig 21 velocity-panel \vspace{-17pt} (F, major)** — source confirmed at `appendix_urban_case.tex` (velocity subpanels use −17pt, temperature −7pt); F additionally saw the collision in the rendered PDF. CONFIRMED (source-side; rendering per F).

9. **Fig 14 caption "one held-out trajectory" vs App H preamble "five held-out trajectories" (C, major)** — both strings read in Stage 0. CONFIRMED contradiction. (Note: spectra script averages over 15 steps of one trajectory per D's CSV inspection; the fix should state which.)

10. **Table 1 bolding/tie-rule inconsistency (A/C, major/minor)** — Table 1 tex: 32² RMSE bolds 0.638 AND 0.626; 16² CRPS bolds 1.062 AND 1.128; 32² CRPS bolds 0.347/0.333/0.327 — consistent only with ties computed on unscaled values (2 dp before ×10⁻¹ scaling), contradicting the caption's literal rule as displayed. CONFIRMED.

11. **Preliminaries dangling "with" (C major; F minor)** — `preliminaries.tex:63–64`. CONFIRMED (visible grammar break in compiled p. 3).

12. **Algorithm 1 c_τ set-but-unused + τ=0 undefined reference (B/C, major/minor)** — `methodology.tex:170–178` read in Stage 0: line 171 sets c_τ=0, update line 178 uses ŝ and (κ+½g²) directly. CONFIRMED.

13. **A_τ, c_τ forward-referenced into appendix + missing verb (C/F)** — `methodology.tex:114`. CONFIRMED.

14. **"700,s"/"100,s" comma typos (C/F)** — `results.tex:120`. CONFIRMED. Same pattern `appendix_urban_case.tex:25` ("128,m³", "1,m"). CONFIRMED.

15. **G.2 text promises Figure 6 panels that are commented out (F, major)** — `appendix_analytical_case.tex:49` vs figure block. CONFIRMED in Stage 0 read.

16. **Metrics appendix orphans (C, minor)** — energy-spectrum RMSE and marginal-KL defined but reported nowhere; "both at matched ensemble size" orphan. CONFIRMED from Stage 0 read of appendix_metrics.tex + absence in Tables 1–2.

17. **DM-SDE schedule/g_0 unspecified in active main text/App D (F/D, major)** — g_τ=g_0√(σβ) only in App G + commented block; multiplier 1.0 in configs, never in paper. CONFIRMED via grep + config.

18. **Diffusion prior derived from FM, disclosed only in last ¶ of App F (A/D, major/minor)** — `appendix_methods.tex:27` + `diffusion_from_fm: true` in configs; unused "poorly trained" native DM checkpoint exists. CONFIRMED.

19. **Baseline hyperparameter values/tuning protocol absent (A/D, major)** — grep of manuscript: no ζ, γ_sda, (k,c)_FIG, λ_DF, g_0 values anywhere; `results.tex:29–32` TODO comment admits the planned setup table is missing. CONFIRMED.

20. **Urban n=1 trajectory (A/D/G, major)** — Table 2 caption "One held-out trajectory"; urban aggregate n_traj=1. CONFIRMED. NS tables have no ±std though std exists in aggregate (e.g. 1.731±0.198 vs 1.616±0.161 overlapping). CONFIRMED.

## Pending verification
- Stage-2 SDA-paper read (arXiv:2306.10574) → running; adjudicates E's challenge to the Appendix F SDA critique.

## Verified in later rounds

21. **Theorem 4.1 SI+g=0 (B, major)** — Stage-2 independent agent CONFIRMED: under-guarded, not wrong; all other proof steps PASS; single guard sentence suffices (see stage2/theorem41-verification.md).

22. **SDA weight critique in App F (E, major)** — E's derivation κ_τ = ½g²_native independently re-derived by orchestrator (match a_s=β_{1−s}, v_s=σ²_{1−s}; g² = 2σ²β̇/β − 2σσ̇ = 2κ_τ). For native diffusion sampling the Doob weight equals the marginal-path weight, so "intermediate marginals are not p_τ^y" is wrong as a blanket claim about SDA. Awaiting the SDA-paper read for how SDA actually samples before final acceptance.

23. **Two different "closest" works (E, nit)** — related_work.tex:13 "Closest to our work is yan_fig"; appendix_methods.tex:9 "FlowDAS, the closest prior method". Both strings confirmed in Stage 0 reads. CONFIRMED.

24. **FIG not FM-only (E, major)** — E's evidence: FIG abstract ("a pre-trained diffusion or flow-matching model as a prior") + official repo shipping diffusion checkpoints. Full FIG PDF bot-blocked (OpenReview), so accepted at abstract+repo level; flagged as such in report.

25. **Statements DRAFT/VERIFY comments (H, blocking for submission-readiness)** — confirmed in Stage 0 read of statements.tex. CONFIRMED.

26. **H's bib findings** — H cross-checked against the repo's own INDEPENDENT_REFERENCE_AUDIT.md and the compiled .bbl; 0 BibTeX warnings independently consistent with my Stage-0 log grep (no undefined citations). Accepted.

27. **Overfull hboxes ~16.6–20.7pt in appendix figure grids (H, minor)** — count of 19 overfulls confirmed by orchestrator in Stage 0 (`grep -c` on main.log). Magnitudes per H. Accepted.
- G's attacks: grounded in the same items verified above (Table 4, n=1, "essential", SURGE-only baselines, novelty delta). Attack facts check out; the FIG-parity observation (urban FM-ODE JF ≡ FIG row-for-row) matches Table 2 tex (0.59/0.66, 0.31/0.36, 0.48/0.46 identical rows). CONFIRMED numerically.

## Notes for report
- G's proposed rebuttal to Attack 2 ("SURGE collapse is Pyrrhic; ours best CRPS") is WEAKENED by verification item 3: bare SDA wins CRPS and calibration too in sparse NS. The honest response is to report bare SDA and reframe sparse-NS claims; the paper's sparse-NS advantage narrative does not survive the repo's own bare-SDA rows as-is. Decision point for the user, since re-running is not needed (data exists) but the narrative must change.
- Deferred: whether bare-SDA rows are trusted/final is an author decision (same grid, post-fix; Agent D notes the 2026-07-22 fix touched only the SURGE wrapper).
