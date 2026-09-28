# Stage 1 — Agent H (Compliance and hygiene) raw report

Status: received, NOT yet verified by orchestrator (Stage 3 pending).

## 1. Venue formatting and length

- main.pdf p.9 / **none** — §6 ends on p.9; statements p.10, references pp.10–12. Within the 9-page limit. (high)
- main.tex:5 / **none** — official ICLR style unmodified; `\iclrfinalcopy` commented; running header present. (high)
- `results.tex:317–318` wraptable `\vspace{-50pt}` / **minor** — Rendered p.9 currently clean, but the −50pt lift is fragile; any upstream change can push it into the Figure 5 caption or margin. / **Fix** re-check p.9 after every edit; minimise the negative vspace. (high current / medium fragility)

## 2. Anonymity

- PDF + figure metadata, author block, active TeX / **none** — clean; author block is ICLR dummy; no URLs/paths/acknowledgements. (high)
- mucke self-citations / **none** — third-person phrasing, anonymity-safe. (high)
- `statements.tex:13` "provided as anonymised supplementary material" / **major** — Claim unverified (VERIFY comment remains); no supplementary artifact exists in the repo yet; if code attached, must strip names/paths/git history. / **Fix** build the anonymised supplement (or anonymous repo link), confirm sentence true, delete comment. (high)

## 3. Required statements

- `statements.tex:6–8, 22–25` DRAFT/VERIFY comments / **blocking** — Required AI-use statement explicitly marked DRAFT; clause "did not use them to develop the theory… or derive and write their proofs" must be verified against actual practice (ICLR 2027 policy: even AI-checking a proof requires disclosure). / **Fix** authors confirm every clause factually, edit line 26 if needed, delete comment blocks. (high)
- No ethics statement / **minor** — recommended, not required. (high)

## 4. Reference hygiene

Sources: reused venue verifications from `manuscript/INDEPENDENT_REFERENCE_AUDIT.md` (2026-07-30) after confirming references.bib matches its post-fix state; independently checked key resolution, .bbl, BibTeX warnings, capitalisation, duplicates, uncited entries. (Audit counts stale — bibliography changed since — but per-entry verdicts for surviving keys match.)

- main.bbl/.blg / **none** — all 37 active cite keys resolve, 0 warnings, no duplicates; venue upgrades present and correct (lipman ICLR'23, albergo_building ICLR'23, song_score-based ICLR'21, ho NeurIPS'20, chung ICLR'23, pokle TMLR'24, yan_fig ICLR'25, ma_sit ECCV'24, ben-hamu ICML'24, song_pseudoinverse ICLR'23, martin ICLR'25, lippe NeurIPS'23, gao Nat.Commun.'24, rozet NeurIPS'23; chen_flowdas NeurIPS'25 and kohl Neural Networks '26 post-cutoff — plausible, marked uncertain). Braces protect proper nouns. (high except two marked)
- `rozet_score-based_2023`, `chen_flowdas_2025`, `wu_practical_2023` / **minor** — published-venue entries missing volume/pages (rozet NeurIPS 36 pp.40521–40541; chen_flowdas 38 pp.103061–103102; wu 36 pp.31372–31403). / **Fix** add fields. (medium, audit-sourced)
- `lippe_pde-refiner_2023` "Veeling, Bastiaan S." / **nit** — official listing "Bas Veeling". (medium)
- Uncited entries `ahamed_dawn-si_2024`, `negrel_multitask_2025`; `si_latent-ensf_2024`, `zhang_flow_2025` cited only in commented-out text / **nit** — dead entries; delete or restore citing sentence. (high)
- ~20 `note` fields / **nit** — inconsistent arXiv formats; daras has both arXiv URL and 10.48550 DOI; abstract/keywords/urldate clutter. / **Fix** normalise. (high)
- Key/year mismatches (cheng_generalised_2022→2023, fotiadis 2024→2025, yan_fig 2024→2025, zhang_flow 2025→2024, si 2024→2025, albergo_stochastic 2023→2025, mucke_physics 2025→2026, kohl 2024→2026) / **nit** — cosmetic; rendered labels use year field. No fix needed. (high)

## 5. Acronyms at first use

- `introduction.tex:3` / **minor** — SDE never expanded anywhere; this ODE use precedes its own expansion. / **Fix** expand both at first use. (high)
- `results.tex:36,38,55,172` / **minor** — CRPS, RMSE, KL, LES, SGLD used in main text without expansion; EnKF abbreviation never tied to intro's "ensemble Kalman filters". / **Fix** expand at first use. DA/SI/FM/DM properly defined; MSE/EMA unexpanded = nit; JVP/PF/SMC absent from active text. (high)

## 6. LaTeX issues

- `appendix_ns_case.tex:96–99`, `appendix_urban_case.tex:1038–1041` / **minor** — 8 worst overfull hboxes 16.6–20.7pt: the image-tabular rows of Figures 7–10 and 15–18 exceed \linewidth by ~0.7cm — visible margin protrusion. Remaining 11 overfulls ≤6.1pt. / **Fix** shrink panel width or wrap grids in `\resizebox{\linewidth}{!}{…}`. (high)
- `methodology.tex:160` bare \ref / **minor** — "(Figure~\ref{...})". (high)
- `results.tex:120` "≈700,s" ">100,s"; `appendix_urban_case.tex:25` "128,m³" "1,m" / **minor** — comma-for-`\,`; also a Unicode em-dash where the paper uses `--`. (high)
- `appendix_implementation.tex:11,21` typos / **minor**. (high)
- Table 4 placeholder / **blocking** — as reported by all agents. (high)
- `style/macros.tex:4` \tbd + \figbox IfFileExists fallbacks / **minor** — placeholder infrastructure still live: no placeholder currently renders, but a missing figure file at submission-build time silently prints a grey box instead of failing. / **Fix** replace IfFileExists fallbacks with plain \includegraphics before the final build; delete \tbd. (high)
- `preliminaries.tex:6,20` \paragraph headings lack trailing period used everywhere else / **nit**. (high)
