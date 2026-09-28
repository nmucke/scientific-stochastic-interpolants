# Stage 1H — Compliance and hygiene

Scope: the active `manuscript/main.tex` graph, the 36-page `main.pdf`, `main.log`, `main.aux`, `main.blg`, `main.bbl`, `references.bib`, the bundled ICLR 2027 style, and the code/material presently under `src/` and `paper_experiments/`. Policy snapshot: 2026-08-10. Primary venue sources: [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) and [ICLR 2027 AI Policy for Authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

## Findings

### H1 — The current supplementary tree reveals an author identity

**Location —** Reproducibility statement, p. 10; exact text: “provided as anonymised supplementary material.” Supporting-artifact instances include `pyproject.toml:8`, `README.md:352--372`, `src/scisi/architectures/aurora.py:40`, `src/scisi/bin/main_train.py:1--12`, and multiple `paper_experiments/*.sh`, logs, and Hydra outputs.

**Severity — blocking.**

**Problem —** The current tree is not anonymised: it names Nikolaj T. Mücke and `nmucke@gmail.com`, links `github.com/nmucke/scientific-stochastic-interpolants`, contains `/Users/ntmucke`, `/home/ntmucke`, and `/export/scratch1/ntm` paths in at least 125 files, and its git remote/history identify the same author. ICLR states that identity revealed in the main paper **or supplementary material** causes desk rejection.

**Confidence — high.** Direct string inspection and `git remote`/`git log` confirm the leaks. Confidence would become submission-level only after inspecting the exact archive that will be uploaded.

**Fix —** Do not upload the working tree. Build a fresh allow-list archive containing only required source/configuration files; omit `.git`, root `README.md`, run logs, Hydra metadata, caches, checkpoints, and archived outputs; neutralise the `pyproject.toml` author field and all absolute user/institution paths. Before upload, unpack the archive in a clean directory and scan both filenames and contents for `Nikolaj`, `Mücke`, `nmucke`, the public repository URL, email addresses, home/scratch paths, hostnames, W&B entities, and git metadata.

### H2 — Table 2 exceeds its wraptable width

**Location —** Section 5.3, Table 2, p. 9; exact text: “Urban (uDALES): rollout-averaged velocity RMSE”.

**Severity — minor.**

**Problem —** `main.log` reports a 4.9736 pt overfull box at `results.tex:329--355` and a forced stationary `wraptable`; visual inspection shows the table rule protruding into the right margin. The resize test uses full `\textwidth` even though the containing wraptable is only `0.675\textwidth`.

**Confidence — high.** Confirmed in the log and rendered page. No further check is needed.

**Fix —** In `results.tex:329`, replace `\textwidth` with `\linewidth`: `\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{...}`. Rebuild and require no overfull warning for Table 2.

### H3 — Twelve appendix figures protrude into the margins

**Location —** Appendix H, Figures 7--10, pp. 22--25; exact text: “true vorticity and observations”. Appendix I, Figures 15--21, pp. 31--36; exact text: “true state and observations”.

**Severity — minor.**

**Problem —** The log reports 16.65--20.69 pt overfull boxes for Figures 7--10 and 15--18 and 4.97--6.02 pt boxes for Figures 19--21. Rendered pages 22 and 34 confirm that the rightmost colourbar extends beyond the normal text boundary.

**Confidence — high.** Confirmed in the log and representative rendered pages. Rebuilding after the proposed width constraints would confirm the complete fix.

**Fix —** Reduce both `\nsqualpanel`/`\nscbar` and `\urbqualpanel`/`\urbqualcbar` heights from `0.116\textheight` to about `0.113\textheight`. For `\nscurvepanel` and `\urbcurve`, constrain the image as `\includegraphics[width=\linewidth,height=0.155\textheight,keepaspectratio]{...}` rather than by height alone.

### H4 — Several acronyms are never expanded at first use

**Location —** Section 3, after Eq. (1), p. 3; exact text: “a discretized PDE”. Section 5, p. 6; exact text: “we report RMSE, CRPS”. Section 5, p. 6; exact text: “D-Flow SGLD”. Appendix D, pp. 16--17; exact text: “under an MSE loss” and “adaptive EMA-based gradient clipping”. Appendix F, p. 18; exact text: “6-step RK2”.

**Severity — minor.**

**Problem —** PDE, RMSE, CRPS, KL, SGLD, MSE, EMA, and RK2 appear as unexplained abbreviations. SDE, ODE, FM, SI, DA, and LES are expanded; these are not.

**Confidence — high.** Checked across the active TeX graph. A final acronym linter after editing would raise confidence that none was missed.

**Fix —** Use “partial differential equation (PDE)” on p. 3; “root-mean-square error (RMSE), continuous ranked probability score (CRPS), … Kullback--Leibler (KL) divergence” on p. 6; “D-Flow stochastic-gradient Langevin dynamics (SGLD)” at its first mention; “mean-squared-error (MSE) loss” and “exponential-moving-average (EMA)-based gradient clipping” in Appendix D; and “six-step second-order Runge--Kutta (RK2)” in Appendix F.

### H5 — The PDE-Refiner reference has the wrong author name

**Location —** References, p. 11; exact text: “Phillip Lippe, Bastiaan S. Veeling”.

**Severity — minor.**

**Problem —** The official NeurIPS record names the second author **Bas Veeling**, not “Bastiaan S. Veeling.” Source checked: Phillip Lippe, Bas Veeling, Paris Perdikaris, Richard Turner, and Johannes Brandstetter, *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers* (NeurIPS 2023), [official proceedings/DOI 10.52202/075280-2946](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html).

**Confidence — high.** The official proceedings author line is unambiguous.

**Fix —** In `lippe_pde-refiner_2023`, replace `Veeling, Bastiaan S.` with `Veeling, Bas`.

### H6 — Three published NeurIPS records omit normal proceedings metadata

**Location —** References, pp. 10--12; exact texts: “FlowDAS: A Stochastic Interpolant-based Framework”, “Score-based Data Assimilation”, and “Practical and Asymptotically Exact Conditional Sampling”.

**Severity — minor.**

**Problem —** All three are published proceedings papers, but the bibliography omits pages for FlowDAS and omits volume, pages, DOI, and canonical proceedings URL for the two NeurIPS 2023 papers. This does not break citation resolution, but it leaves peer-reviewed records looking like minimally upgraded preprints.

**Confidence — high** for publication status, authors, venue, volume, and the 2023 records' pages/DOIs; **medium** for FlowDAS pagination because the official HTML/BibTeX endpoint did not expose it in this audit. The official BibTeX record would raise that part to high.

**Fix —** Complete the entries as follows:

- Siyi Chen, Yixuan Jia, Qing Qu, He Sun, and Jeffrey A. Fessler, *FlowDAS: A Stochastic Interpolant-based Framework for Data Assimilation* (NeurIPS 2025), volume 38, pages 103061--103102; use the [official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/951690f8dae19c76394404540b7ca734-Abstract-Conference.html).
- François Rozet and Gilles Louppe, *Score-based Data Assimilation* (NeurIPS 2023), volume 36, pages 40521--40541, DOI `10.52202/075280-1763`; use the [official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html).
- Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, and John P. Cunningham, *Practical and Asymptotically Exact Conditional Sampling in Diffusion Models* (NeurIPS 2023), volume 36, pages 31372--31403, DOI `10.52202/075280-1363`; use the [official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract.html).

### H7 — SURGE is still cited only as an arXiv preprint after ICML acceptance

**Location —** References, p. 12; exact text: “SURGE: Approximation and Training Free Particle Filter”.

**Severity — minor.**

**Problem —** The reference is typed as `@misc`/arXiv only, while ICML's official 2026 program lists the paper. No PMLR proceedings record was found as of 2026-08-10, so claiming final pagination would be premature.

**Confidence — high** for ICML 2026 acceptance; publication pagination remains unchecked because none was available in the official program source. A PMLR record would raise confidence on final metadata.

**Fix —** Change the entry to an in-proceedings/accepted-paper record for Lifu Wei, Yinuo Ren, Naichen Shi, and Yiping Lu, *SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate* (ICML 2026), retaining arXiv:2605.18745 and omitting pages until PMLR posts them. Primary sources: [official ICML 2026 program listing](https://icml.cc/Downloads/2026) and [arXiv:2605.18745](https://arxiv.org/abs/2605.18745).

### H8 — The JAX-CFD software citation does not identify the code version used

**Location —** Section 5.2, p. 7; exact text: “generated with JAX-CFD”. References, p. 11; exact text: “JAX-CFD: Computational fluid dynamics in JAX.”

**Severity — minor.**

**Problem —** The software reference gives only a moving GitHub URL and the year 2021, with no release, commit, or access date. It therefore cannot identify the solver implementation used for the reported data.

**Confidence — high.** The active BibTeX entry contains no version identifier. The executed environment or vendored checkout commit is needed to select the correct value.

**Fix —** Add the exact tag/commit used and an access date to `jax_cfd_2021`, e.g. `note = {GitHub repository, commit <HASH>, accessed 2026-08-10}`; replace `<HASH>` with the actual checked-out revision.

### H9 — A malformed accent command and a dangling word are visible in Section 3

**Location —** Section 3, after Eq. (6), p. 3; exact texts: “F"ollmer formulation” and “with where the drift”.

**Severity — nit.**

**Problem —** `F{"o}llmer` is encoded in source as `F{"o}llmer` without the backslash before the accent command, so the PDF prints `F"ollmer`; the following displayed equation is also followed by the ungrammatical “with where”.

**Confidence — high.** Both are visible in the PDF and trace to `preliminaries.tex:59,63--64`.

**Fix —** Replace the name source with `F{\"o}llmer`; delete the standalone `with` before “where the drift can be decomposed”.

### H10 — Appendix D contains two typographical errors

**Location —** Appendix D, pp. 16--17; exact texts: “Inerpolant schedules” and “20 validation trajectorires”.

**Severity — nit.**

**Problem —** Both words are misspelled.

**Confidence — high.** Direct source/PDF inspection.

**Fix —** Replace them with “Interpolant schedules” and “20 validation trajectories”.

### H11 — Two runtime units render with a comma instead of spacing

**Location —** Section 5.2, p. 8; exact text: “costs ≈700,s per step”.

**Severity — nit.**

**Problem —** `${\approx}700$,s` and `${>}100$,s` render a literal comma before `s`.

**Confidence — high.** Direct source/PDF inspection.

**Fix —** Replace them with `${\approx}700$\,s` and `${>}100$\,s`.

### H12 — The bibliography source header no longer describes the active bibliography

**Location —** `references.bib:1--3`; exact text: “Contains only the 42 entries cited”.

**Severity — nit.**

**Problem —** The file contains 41 entries, of which 37 are active; four (`ahamed_dawn-si_2024`, `negrel_multitask_2025`, `si_latent-ensf_2024`, `zhang_flow_2025`) are uncited. This does not affect the rendered reference list because BibTeX emits only cited records.

**Confidence — high.** Reconciled the BibTeX keys with the 37 `\bibcite` records in `main.aux`.

**Fix —** Replace the header with “Contains the active paper bibliography plus four retained dormant entries,” or remove the four dormant records and state “Contains the 37 entries cited by the built document.”

## Compact compliance checklist

| Check | Status | Evidence / action |
|---|---|---|
| Official ICLR 2027 style, anonymous mode | Pass | Bundled `iclr2027_conference` style is loaded; `\iclrfinalcopy` is commented; PDF header says “Under review as a conference paper at ICLR 2027.” I did not independently byte-compare the bundled style against the current zip. |
| Page size | Pass | US Letter, 612 × 792 pt. |
| Nine-page main-text limit | Pass, exactly at limit | Main text and limitations end on p. 9; statements/references start on p. 10. ICLR excludes the statements and references. |
| Appendix placement/length | Pass | References occupy pp. 10--12; appendices begin p. 13. Unlimited post-reference appendices are allowed. |
| PDF author anonymity | Pass | Anonymous author block, no acknowledgements or identifying repository URL, blank PDF Title/Author/Subject/Keywords metadata. Self-citations are phrased in the third person. |
| Supplementary anonymity | **Fail** | H1; current code/tree would disclose identity and can trigger desk rejection. |
| Required AI-use statement | Present, author verification required | Statement is on p. 10 and discloses implementation, pipelines, figures/tables, result analysis, editing, references, and literature search. The authors must verify its negative theory/proof clause and “remaining categories” clause before submission. |
| AI-use submission form | External action | ICLR also requires disclosure in the submission form; cannot be checked in the manuscript. |
| Ethics statement | Optional / plausibly N/A | Not included. ICLR recommends it only where ethical issues arise; no human/private dataset is described. Authors must still acknowledge the Code of Ethics in the submission system. |
| Reproducibility statement | Pass text; artifact claim currently false | Statement is present, but “anonymised supplementary material” is not true of the current tree (H1). |
| Separate paper checklist | None found | Current ICLR 2027 author guide lists no separate mandatory manuscript checklist beyond the AI statement. |
| Citation/reference resolution | Pass | 37 active entries; `main.blg` has zero warnings; no undefined citations, references, or multiply-defined labels in `main.log`. |
| Bibliography duplicates | Pass for active list | No duplicated active work found by title/DOI/arXiv/OpenReview reconciliation. Metadata issues are H5--H8. |
| Layout warnings | Fail, minor | Table 2 and appendix figures are overfull (H2--H3); no fatal LaTeX errors. |
| Acronyms | Fail, minor | H4. |
| Heading capitalization | Pass | No inconsistent active section-heading capitalization found. |

## Bibliography-hygiene table

| Active record(s) | Status | Smallest action | Primary record |
|---|---|---|---|
| `lippe_pde-refiner_2023` | Wrong author form | `Bastiaan S. Veeling` → `Bas Veeling` | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html) |
| `chen_flowdas_2025` | Missing pages | Add official proceedings pagination after confirming BibTeX | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/951690f8dae19c76394404540b7ca734-Abstract-Conference.html) |
| `rozet_score-based_2023` | Missing volume/pages/DOI/canonical URL | Add 36:40521--40541 and DOI `10.52202/075280-1763` | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html) |
| `wu_practical_2023` | Missing volume/pages/DOI/canonical URL | Add 36:31372--31403 and DOI `10.52202/075280-1363` | [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract.html) |
| `wei_surge_2026` | Stale publication status | Add ICML 2026; retain arXiv until PMLR metadata appears | [ICML 2026 program](https://icml.cc/Downloads/2026), [arXiv](https://arxiv.org/abs/2605.18745) |
| `jax_cfd_2021` | Moving software target | Add exact tag/commit and access date | [official repository](https://github.com/google/jax-cfd) |
| `daras_survey_2024`, `parikh_d-flow_2026` | Current preprints | No hygiene change found as of 2026-08-10 | [arXiv:2410.00083](https://arxiv.org/abs/2410.00083), [arXiv:2602.21469](https://arxiv.org/abs/2602.21469) |
| Remaining 29 active entries | No hygiene finding | No action reported | Checked against their printed DOI/proceedings/OpenReview/publisher records; no duplicate active works found. |

## Required author confirmations

1. **AI disclosure:** I could not verify whether generative AI ever checked a proof, proposed/refined a hypothesis, influenced experimental design/parameters, translated text, or cleaned/reformatted data. The authors must audit those mandatory policy categories and make the negative clause literally true.
2. **Exact upload artifact:** I could not verify anonymity of an archive because no final submission archive exists. H1 applies to the current repository/tree; repeat the scan on the actual uploaded bytes.
3. **JAX-CFD revision:** I could not determine the exact commit used from the manuscript or active citation; the authors must recover it from the executed environment or vendored checkout.
