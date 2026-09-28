# Stage 1D — Experiments and reproducibility

Scope: the complete active paper/PDF, `review/paper-map.md`, active configurations and scripts in `paper_experiments/`, checkpoint configurations, active aggregate/raw results, and the executed observation/prior/posterior/metric code in `paper_experiments/` and `src/scisi/`. Archived result trees were excluded. Figure aesthetics were not assessed.

## Findings

### D1 — The urban experiment directly observes temperature and uses denser sensors than stated

- **Location** — Section 5.3, pp. 8--9; Figure 5; Appendix I, p. 21. Quotes: “temperature is inferred only through its coupling to the observed flow”; “sensor densities of $1.5625\%$ and $0.78125\%$ of fluid cells.”
- **Severity** — blocking.
- **Problem** — The executed urban observation operator samples scalar DOFs from a four-channel `(u,v,w,thl)` mask, so temperature is observed directly: saved headline files contain 256 temperature observations among 1024 at the nominal $1/64$ density and 111 among 512 at $1/128$. The code computes this count as the stated percentage of all $4\times128^2$ grid DOFs and then places it in fluid DOFs; it covers 988/503 unique spatial fluid cells (9.492%/4.832%) and 2.459%/1.230% of four-channel fluid DOFs, neither the stated 1.5625%/0.78125% of fluid cells. This invalidates C26's “temperature inferred from velocity alone” interpretation and means Table 2/Figures 5 and 15--21 are results for a different observation problem.
- **Confidence** — high. `cases/urban/driver.py:307-321` sets the observation shape from all four state channels; `_urban_pipeline.py:83-102` broadcasts the fluid mask across all four; `_urban_pipeline.py:147-170` samples all such DOFs and takes its count from the full-grid `operator.num_dofs`. I also checked `obs_indices` in both saved SI-SDE headline NPZs. No further check is needed to establish the discrepancy.
- **Fix** — If “sensor density” means spatial locations, draw `round(p * n_fluid_cells)` fluid positions and observe all three velocity components at each; otherwise define and report a scalar-DOF density explicitly. In either case exclude temperature, rerun every urban method/variant at both densities, and regenerate Table 2 and Figures 5 and 15--21. Combining this correction with the intended five-trajectory completion is approximately 5--6 GPU-days on one 24-GiB GPU according to `run_urban_grid.sh`; a one-trajectory correction is roughly one fifth of that but would retain D3.

### D2 — “Same trained prior” is false, and Navier--Stokes sampler comparisons are capacity-confounded

- **Location** — Section 5 opening, p. 6; Appendix D, p. 17; Appendix F, p. 19. Quotes: “all using the same trained prior”; “The same trained checkpoint per case serves every sampler”; “all generative methods use the same network.”
- **Severity** — major.
- **Problem** — The field experiments load separate SI and FM checkpoints. In Navier--Stokes the SI network has widths `(8,16,32,64)` and the FM network `(16,32,64,128)`, with different embeddings, attention patch sizes, parameter counts, and learning-rate schedules. SI-SDE/FlowDAS use the SI checkpoint, whereas DM-SDE/FM-ODE/SDA/D-Flow/FIG use the FM checkpoint; cross-family rankings therefore do not isolate the sampler or guidance rule. Urban uses capacity-matched but still separately trained SI and FM networks.
- **Confidence** — high. The separate run names are in `configs/case/navier_stokes.yaml:20-27` and `configs/case/urban.yaml:29-35`; `_ns_pipeline.load_prior` loads both separately; the architecture difference is visible in the two active NS checkpoint configs and is already disclosed in Appendix D. A prior-only held-out comparison would quantify the confound but is not needed to establish it.
- **Fix** — Replace the three quoted claims with: “Methods within each generator family share a checkpoint: SI-SDE and FlowDAS use the SI prior, while DM-SDE, FM-ODE, SDA, D-Flow, and FIG use the FM prior; the Navier--Stokes SI and FM priors have different capacities.” Remove sampler-only causal interpretations of SI-versus-FM rankings. To retain such interpretations, train capacity-matched SI/FM priors and report their held-out one-step and free-rollout errors before rerunning the affected comparisons (one additional NS training pair plus the evaluation grid; multi-GPU-day cost).

### D3 — Urban method rankings rest on one trajectory and one stochastic realization

- **Location** — Table 2 caption, p. 9; Section 5.3, p. 9; Conclusion, p. 9. Quotes: “One held-out trajectory”; “DM-SDE gives the lowest velocity RMSE”; “SI-SDE gives the lowest temperature RMSE.”
- **Severity** — major.
- **Problem** — The active urban aggregate has `n_traj=1` and `std=0` for every row, and the run uses only `seed=0`; differences used to declare winners are often 0.01--0.02. A single flow realization, sensor mask, and posterior-sampling realization cannot establish stable urban rankings.
- **Confidence** — high. This is explicit in Table 2 and `results/urban/aggregated/all.csv`; the master script intends five trajectories but only trajectory 1 is complete (trajectory 2 has an in-progress partial). More completed trajectories are the only way to raise confidence in the ranking.
- **Fix** — Finish trajectories 2--5 for both sensor settings and report mean plus SD or a paired confidence interval over trajectories; keep one common posterior seed per trajectory if compute is constrained, but say so. This is about 4--5 additional GPU-days after the existing trajectory on the script's own budget, and can be combined with D1's mandatory rerun. If no experiment is possible, replace every ranking with “on the single reported rollout” and present Table 2 as a case study rather than comparative evidence.

### D4 — The claimed tuning protocol is neither specified nor uniformly applied

- **Location** — Section 5 opening, p. 6; Reproducibility statement, p. 10; Appendix F, pp. 18--19. Quotes: “Hyperparameters are tuned on separate short trajectories”; “specifies each baseline and its tuning protocol.”
- **Severity** — major.
- **Problem** — Appendix F gives qualitative method descriptions but no search spaces, budgets, tuning trajectory IDs, selection metric, or selected values. The active configs also show unequal treatment: urban SI/DM/FM damping was directly swept at `k=10` for two tuning densities (the headline $0.78125\%$ row reuses the value), but SURGE-FlowDAS and SURGE-SDA explicitly transfer parameters from bare FlowDAS/SDA and say those published combinations were “NOT swept”; sparse NS EnKF uses a hard-coded radius 20, unit inflation, and no recorded selection procedure. Thus “all tuned” is not reproducible and baseline fairness is unverified.
- **Confidence** — high for the missing protocol and the two SURGE transfers; checked `appendix_methods.tex`, `surge_flowdas.yaml`, `surge_sda.yaml`, `enkf.yaml`, and `run_ns_grid.sh:203-210`. I could not verify a common off-repository tuning log or budget; providing one would raise confidence in fairness.
- **Fix** — Add one compact appendix table listing, per method/case, grid, tuning trajectory, horizon, ensemble size, selection metric, and chosen value. Either change the prose to “method-specific tuning or transfer” and label the transferred/fixed settings, or directly tune the two SURGE combinations and EnKF radius/inflation on the same held-out-budget rule. Short direct tuning is roughly a few GPU-hours for SURGE and a few solver-hours for EnKF; any changed winner must then be rerun in the headline grid (amortizable with D1/D3).

### D5 — Headline Navier--Stokes shared-covariance settings were selected in a different regime

- **Location** — Section 5 opening and Table 1, pp. 6--8. Quotes: “shared Jacobian updated every tenth pseudo-step”; “Hyperparameters are tuned on separate short trajectories.”
- **Severity** — major.
- **Problem** — Table 1 uses `k=10`, but the NS damping values were selected on SI-SDE only at `k=1`, one trajectory, `E=8`, `M=50`, and then copied to DM-SDE/FM-ODE and all $M$. The tracked configs explicitly warn that damping and cadence are not independent, that `k=10` can worsen sparse SI-SDE RMSE by 59--69%, and that the DM/FM transfers were not verified. The reported shared-mode rankings and cost--accuracy point are therefore not results at tuned headline settings.
- **Confidence** — high. `si_sde.yaml:31-64`, `dm_sde.yaml:25-48`, `fm_ode.yaml:20-43`, and `run_ns_grid.sh:93-109` record the exact provenance and warnings. Direct $k=10$ sweeps for all three samplers would be needed to determine how much the table changes.
- **Fix** — On one held-out NS trajectory, tune $\lambda$ separately for SI-SDE, DM-SDE, and FM-ODE at `k=10` for the four scenarios (at least $\{0.9,0.95,1.0\}$, $E=8$, $M=100$), then rerun only Table 1 cells whose selected value changes across five trajectories. This is a modest tuning sweep followed by at most the 12 shared headline cells; roughly 1--2 GPU-days. If not run, disclose the transfer and call Table 1 a fixed `k=10` operating point, not a tuned comparison.

### D6 — The claim-critical covariance/scale ablation is unfinished

- **Location** — Appendix H.3 and Table 4, p. 30. Quotes: “we isolate the effect of the components”; “Run in progress”; “cells (`--`) are placeholders.”
- **Severity** — major.
- **Problem** — Every Table 4 result cell is empty, so the paper does not isolate refresh cadence, damping, diffusion strength, step count, or ensemble size despite making the covariance approximation/scalability trade-off a contribution. Main tables compare only Jacobian-free against one damped, stale, shared-Jacobian bundle; they do not separate which component produces the sparse-observation gain.
- **Confidence** — high. The active TeX table is empty and no active ablation aggregate is referenced. The driver contains an ablation entry point, but I could not find a completed active result file that fills Table 4.
- **Fix** — Smallest text-only fix: delete Table 4 and the claim that it “collects the results,” and narrow causal language to the tested bundle. Experimental fix: for one sparse NS scenario run SI-SDE at covariance `{Jac-free, shared}`, $k\in\{1,5,10\}$ with separately selected $\lambda$, $M\in\{25,50,100\}$, and $E\in\{16,64\}$ over the five held-out trajectories; report paired uncertainty. Expect roughly 1--2 GPU-days, with $k=1$ dominating cost.

### D7 — Navier--Stokes “best” rankings omit available uncertainty

- **Location** — Table 1 and Section 5.2, p. 8. Quotes: “achieve the best RMSE and CRPS”; “SI-SDE is therefore the strongest.”
- **Severity** — major.
- **Problem** — Table 1 reports means over five trajectories without SD, confidence intervals, or paired tests, although the active aggregate contains the SD. Several top-row gaps are smaller than between-trajectory variation: at $32^2$, shared FM-ODE RMSE is $0.0626\pm0.0068$, DM-SDE $0.0638\pm0.0068$, and SI-SDE $0.0667\pm0.0059$. The broad dense/sparse gaps can be read from the table, but fine winner declarations are not statistically established.
- **Confidence** — high for the omitted uncertainty and scale comparison; values come from the active M=100 aggregate. A paired per-trajectory bootstrap/test would determine whether the rankings themselves are stable.
- **Fix** — Use the existing five per-trajectory files to add mean$\pm$SD (at least in an appendix companion table) and paired bootstrap intervals for differences underlying “best” claims; soften any interval that crosses zero. No new runs are required; approximately half a day of analysis/typesetting.

### D8 — Urban costs and metrics are normalized per forecast step, not per assimilation step

- **Location** — Section 5 opening, p. 6; Appendix E, pp. 17--18; Table 2, p. 9. Quotes: “runtime per assimilation step”; “All metrics are computed per assimilation step”; “s/step.”
- **Severity** — major.
- **Problem** — Urban assimilates only 49 of 145 generated steps, but `run_assimilation` scores all 145 and divides elapsed time/NFE by 145, explicitly defining a *generated physical step* denominator. The reported Table 2 costs are therefore about $49/145$ of seconds per posterior update, and the accuracy metrics include 96 free-running steps although Appendix E says all metrics are assimilation-step metrics.
- **Confidence** — high. The schedule is stated in Section 5.3 and saved in the NPZ metadata; `_ns_pipeline.py:1214-1225,1227-1242,1293-1297` defines the denominator, and `_urban_pipeline.py:274-300` scores every generated step. No further check is needed.
- **Fix** — Keep the operational metric and replace the global definition with: “Navier--Stokes metrics/costs are per assimilation update; urban accuracy averages all 145 generated steps, and urban cost is seconds per generated physical step under the every-third-step schedule.” Rename Table 2's header to “s/generated step.” If seconds per update are intended instead, recompute each cost as total elapsed/49 (approximately 2.96 times the printed values) and restrict or separately report metrics at update steps. Text-only, under one hour, if the current operational definition is retained.

### D9 — Navier--Stokes convergence figures silently mix four- and five-trajectory aggregates

- **Location** — Appendix H opening and Figures 11--13, pp. 21, 26--28. Quotes: “Throughout: five held-out trajectories”; “as a function of the number of sampler steps $M$.”
- **Severity** — minor.
- **Problem** — The active status file records 18 shared-mode cells at $M=50$ or $250$ with only 4/5 trajectories, while the plots consume the aggregate without exposing `n_traj`. Their curves therefore compare different trajectory cohorts across $M$, contrary to “throughout five.” Table 1's M=100 headline cells are complete and are not affected.
- **Confidence** — high. `results/STATUS.md` enumerates all 18 partial cells and `aggregated/all.csv` records `n_traj`; inspecting a regenerated plot with cohort annotations would only confirm presentation of the already-established input mismatch.
- **Fix** — Either finish the 18 missing trajectory cells, or regenerate all versus-$M$ curves on the common four-trajectory intersection and state `n=4`; alternatively annotate each point's `n`. Completing the missing cells is likely about one GPU-day; the common-cohort edit needs no new runs.

### D10 — The supplementary package does not contain the claimed urban data-generation artefact

- **Location** — Reproducibility statement, p. 10; Appendix I, p. 21. Quote: “Code for the method, baselines, data generation ... is provided.”
- **Severity** — major.
- **Problem** — The active urban case consumes author-provided NetCDF files, and `configs/case/urban.yaml:1-4` explicitly states “no in-repo CFD generator”; I found no uDALES input deck or generation launcher. The prose gives solver class, domain, resolution, and closure, but not enough forcing, boundary/surface, building, and stochastic-initialization parameters to regenerate the 179 trajectories.
- **Confidence** — high that no urban generator/input deck exists in the scoped repository. I could not verify whether the 14-GB local `data/udales/` directory will be distributed separately; a public/anonymised dataset deposit or omitted supplementary bundle could mitigate data access but would not make the quoted code claim true.
- **Fix** — Add the uDALES namelists/input decks, geometry, perturbation seeds, and a generation/preprocessing launcher; otherwise replace the sentence with “Code for the method, baselines, Navier--Stokes generation, and all evaluations is provided; urban trajectories are author-generated uDALES data available at [anonymous data link], with input decks supplied.” Packaging/deposit is roughly half a day if the decks already exist; rerunning CFD is not required.

### D11 — Urban observation noise is an undisclosed, author-unconfirmed normalized-space choice

- **Location** — Section 5 opening, p. 6; Appendix I setup, p. 21. Quote: “with Gaussian observation noise.”
- **Severity** — minor.
- **Problem** — The urban run uses iid $\sigma=0.05$ noise in per-channel standardized space, but neither the main text nor Appendix I states its variance or physical units. The active case config still says “TODO ... confirm the noise level,” so a result-driving setting is neither specified nor author-confirmed.
- **Confidence** — high for the executed value: `configs/case/urban.yaml:57-66` and `driver.py:290-292` set variance 0.0025. Author confirmation is needed to establish that this is the intended physical noise model.
- **Fix** — After author confirmation, add: “Urban observations are corrupted in standardized space by iid $\mathcal N(0,0.05^2)$ noise in every observed channel.” If physically meaningful sensor noise is intended, set per-variable physical variances, rerun the urban grid, and report them; that rerun can be combined with D1.

### D12 — Sparse-sensor robustness is measured for one spatial mask per scenario

- **Location** — Section 5 opening, p. 6; Table 1 and Table 2. Quote: “random sensors covering $5\%$ or $1.5625\%$.”
- **Severity** — minor.
- **Problem** — `mask_seed(case, scenario)` fixes one observation mask for every trajectory and every method; field variability is sampled (five NS trajectories, one urban trajectory), but sensor-placement variability is not. Sparse-observation conclusions can therefore depend on one draw, especially at the two lowest densities.
- **Confidence** — high for the single-mask design; checked both observation builders and the seed construction. Three or more independent mask seeds would raise confidence in generalization across sensor placement.
- **Fix** — State “one fixed random mask per scenario, shared by all methods.” For a robustness check, repeat the M=100 NS SI-SDE shared/Jac-free comparison and the corrected urban M=50 comparison for three mask seeds, reporting mask-to-mask SD; this multiplies only the selected comparison cost by about three, not the full baseline grid.

### D13 — The urban training-set count is off by one

- **Location** — Appendix D, p. 17. Quote: “The urban case uses 150 LES runs for training.”
- **Severity** — nit.
- **Problem** — Both active urban checkpoints specify `files: (0,150)`, and `UDalesDataset` expands this range inclusively, so training uses 151 runs. Validation `(151,170)` contains 20 runs as stated.
- **Confidence** — high; checked both checkpoint configs and `src/scisi/data/datasets.py:660-663`. No further check is needed.
- **Fix** — Replace “150 LES runs” with “151 LES runs (simulations 0--150 inclusive).”

## Manuscript ↔ config ↔ code ↔ result reconciliation

| Item | Manuscript | Active config/script | Executed code | Active result | Reconciliation |
|---|---|---|---|---|---|
| Analytical replication | Five seeds; $E=4096$ in Table 3 | Five-seed analytical driver | Closed-form prior/posterior; no learned checkpoint | Five-seed aggregate at each $M$ | Matches; no scoped finding. |
| NS evidence set | Five trajectories, one $E=64$ table at $M=100$ | Global test trajectories 190--194; `seed=0`; 15 DA steps | Shared truth/obs/mask across methods | M=100 headline has `n_traj=5`; some curve points have 4 | Table 1 matches; D7 and D9 apply. |
| NS prior | “Same trained prior/network” | Separate SI and FM checkpoint names | Family-specific checkpoint dispatch | SI/FM-family rows mix the two | Contradicted; D2. |
| NS shared covariance | $k=10$, scenario damping unstated | $k=10$ headline; damping table copied from SI, $k=1$ tuning | `inflated_shared`, cached Jacobian every tenth step | Variant `shared_jac10` | Executed as labelled, not tuned in that regime; D5. |
| Urban evidence set | One held-out 145-step rollout, 49 updates | Script targets five trajectories but current run completed one; `seed=0` | Foreign IC 8; assimilate every 3 from step 5 | `n_traj=1`, 145 scored forecast steps | Current table matches its one-trajectory caption; D3/D8. |
| Urban observations | Velocity-only; 1/64 and 1/128 of fluid cells | Four-channel data size; scalar normalized variance 0.0025 | Candidate pool is all four fluid channels; count uses full grid | 243/243/282/256 and 136/129/136/111 observations by `(u,v,w,T)` | Contradicted; D1/D11. |
| Urban cost | Seconds per assimilation step | Sparse-in-time DA: 49/145 updates | Elapsed time divided by 145 generated steps | Table values are s/generated step | Mislabelled; D8. |
| Baseline tuning | All tuned on separate short trajectories; Appendix F protocol | Detailed values in YAML, but SURGE urban values transferred from bare methods; EnKF fixed radius/inflation | Values dispatched per scenario/$M$ | Headline rows use those values | Protocol absent and budgets unequal/unverified; D4. |
| Urban data generation | Generation code supplied | Author-provided `.nc`; config says no generator | Loader/preprocessor only | Local 14-GB dataset exists | Generation claim false unless external artefacts are added; D10. |

## Major/blocking code discrepancies for orchestration verification

1. **D1 (blocking):** Verify independently before submission that the urban observation pool includes channel 3 and that the saved headline masks contain 256/111 temperature sensors. This changes the scientific question, not just its description.
2. **D2 (major):** Verify the intended fairness claim: the active NS SI and FM checkpoints are separate and capacity-mismatched, with method families dispatched to different priors.
3. **D4/D5 (major):** Confirm whether off-repository tuning evidence exists. The tracked configs explicitly say urban SURGE combinations were not swept and NS headline $k=10$/DM/FM damping was transferred.
4. **D8 (major):** Decide whether urban `s/step` is meant to be per generated step or per posterior update; the active code implements the former while the paper defines the latter.

## Scoped areas with no additional finding

- **Analytical experiment:** no additional experiments/reproducibility finding; the active aggregate, five-seed description, ensemble size, and table values agree.
- **Truth, observations, data, and metric sharing within a fixed field-case cell:** no discrepancy beyond D1; methods receive the same trajectory, noise realization, mask, ensemble size, and metric implementation.
- **Headline numeric transcription:** active M=100 Navier--Stokes and M=50 urban means/costs agree with Tables 1--2 to displayed precision; the issues are experimental meaning, uncertainty, and provenance rather than copying errors.
