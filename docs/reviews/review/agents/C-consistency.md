# C — Internal consistency review

Scope: the full active `main.tex` input tree and compiled `main.pdf`, checked against `review/paper-map.md`. I treated commented and `\iffalse` material as inactive. I did not assess whether the mathematical arguments are valid or whether the experiments are fair.

## Findings

### C1 — The exact tilt conflates the physical conditioning state with the sampler's initial state

- **Location —** Section 4.1, Eq. (8), and Figure 2 caption; “`\middle|\,\bX_\tau=\bx,\bX_0=\bx_0`” / “initialized at $\state^0$.”
- **Severity —** major.
- **Problem —** The first definition of the tilt conditions on the fixed physical context $\bx_0$, but the equal expression changes this to the pseudo-time random state $\bX_0=\bx_0$. Those objects coincide for SI-SDE but not for FM-ODE/DM-SDE, whose own sampler equations initialise $\bX_0\sim\mathcal N(0,I)$. Figure 2 repeats the conflation by saying the unified SDE/ODE paths are initialized at $\state^0$.
- **Confidence —** high. Eq. (8)'s first expression, Eqs. (9)'s three initial laws, Proposition B.1/Eq. (37), and Figure 1's “$\bx_0$ or noise” source label establish the distinction. No further check is needed.
- **Fix —** In Eq. (8), replace the second conditioning clause with
  `$\E[\conddist{\obs}{\bx_1}\mid \bX_\tau=\bx,\bx_0]$`.
  In Figure 2's caption, replace “initialized at $\state^0$” with “conditioned on $\state^0$ (SI starts there; FM/DM start from Gaussian noise).”

### C2 — Algorithm 1's endpoint branch is dead and contradicts the implementation prose

- **Location —** Section 4.4, Algorithm 1, lines 5--10; “Set the likelihood correction $\bm c_\tau=\bm 0$.”
- **Severity —** major.
- **Problem —** The algorithm sets $\bm c_\tau=0$ at $\tau=0$ but never uses $\bm c_\tau$; the update always inserts $\interpolantscorefun_\tau$, including at $\tau=0$. This contradicts the preceding statement that the first step uses only the prior drift and, for FM, asks the pseudocode to evaluate the score exactly where the text says its representation is singular.
- **Confidence —** high. The mismatch is explicit in adjacent Algorithm 1 lines and Section 4.4's first paragraph. No code inspection is needed to establish the paper-level inconsistency.
- **Fix —** In the `else` branch add
  `\STATE Set $\bm c_\tau=(\vscoef_\tau+\tfrac12\gdiff_\tau^2)\interpolantscorefun_\tau^{\interpolantobs_\tau}$`
  and replace the update drift by `$\drift^{\gdiff}_\tau+\bm c_\tau$`.

### C3 — The stated one-state conditional prior and Algorithm 1 omit the implemented five-state context

- **Location —** Appendix D, Architecture paragraph, versus Eq. (2) and Algorithm 1 input; “the five most recent physical states are supplied as additional input channels.”
- **Severity —** major.
- **Problem —** The formal target, every learned field, and Algorithm 1 condition only on $\bx_0=\state^{n-1}$, while Appendix D says both field priors condition on five recent states. The paper therefore assigns two different conditioning objects—and input dimensions—to the same trained transition prior.
- **Confidence —** high for the textual inconsistency. Inspecting the checkpoint/data-loader tensor shapes would only be needed to decide whether Appendix D or the one-state algorithm describes the executed model.
- **Fix —** Add after Eq. (2): “The learned fields may additionally condition on a fixed history $h^{n-1}=(\state^{n-5},\ldots,\state^{n-1})$, suppressed below; $\bx_0=\state^{n-1}$ remains the interpolant anchor.” Add $h^{n-1}$ to Algorithm 1's input and say it is passed to each network evaluation. This preserves the current derivation because the extra context is fixed during one assimilation step.

### C4 — “Best under dense observations” is false without an $E=64$ qualification

- **Location —** Section 5.2 and Conclusion, Table 1; “our methods achieve the best accuracy under dense observations.”
- **Severity —** major.
- **Problem —** In the $16^2$ dense column, the $E=1000$ EnKF reference has lower RMSE (1.442 vs. 2.024) and CRPS (0.779 vs. 1.062) than the best proposed row. The prose and conclusion can be true only among matched-$E=64$ methods. Table 1's caption also says bold marks the best in each column, yet it bolds the proposed values rather than these lower reference values.
- **Confidence —** high. The conflicting values are in Table 1 itself. No external result check is needed.
- **Fix —** Replace both claims with “Among methods at $E=64$, ours achieve the best dense-observation RMSE and CRPS.” Replace the caption clause with “Bold marks the best $E=64$ value in each column (ties at two decimal places); the $E=1000$ EnKF is a separate reference.”

### C5 — The appendix pointer does not define one of the two analytical metrics

- **Location —** Section 5 opening and Appendix E; “Metric definitions are given in Appendix~\ref{appendix:metrics}.”
- **Severity —** major.
- **Problem —** Sliced-$W_2$ is a headline analytical metric in Section 5.1 and Table 3, but its only definition in Appendix E is commented out. Thus the cross-reference promises a definition that the active paper does not contain.
- **Confidence —** high. The active Appendix E defines RMSE, spectral RMSE, CRPS, spread--skill, KL, and cost; its sliced-$W_2$ paragraph is inactive. The implementation confirms 128 projections, so the existing commented definition matches the recorded metric.
- **Fix —** Uncomment the existing Appendix E paragraph: “Sliced $W_2$ is the 2-Wasserstein distance averaged over 128 random one-dimensional projections …” together with its displayed aggregation formula.

### C6 — The paper says Table 4 contains ablation results, but every result is a placeholder

- **Location —** Appendix H.3, immediately before Table 4; “Table~\ref{tab:ablation} collects the results.”
- **Severity —** major.
- **Problem —** Table 4 is labelled “Run in progress” and all 18 metric cells are `--`. The prose and cross-reference describe a completed result artefact that does not exist in the active paper.
- **Confidence —** high. The contradiction is visible in the active table. Populating it would require checking completed runs; none are represented in this TeX table.
- **Fix —** If no completed sweep is available, delete the final sentence and Table 4 (and remove “and the ablations” from Appendix H's opening inventory). If retained, run and report the listed covariance/$k$/$\lambda$/$g_\tau$/$M$/$E$ sweep; this is a multi-run experiment rather than a textual fix.

### C7 — Appendix G describes panels that are absent from active Figure 6

- **Location —** Appendix G.2, first paragraph, Figure 6; “together with the convergence of the KL divergence … and one-dimensional density slices.”
- **Severity —** minor.
- **Problem —** Active Figure 6 contains only prior, likelihood, exact-posterior, and sampled-posterior panels. The convergence and density-slice panels described by the text are in a commented-out superseded figure block.
- **Confidence —** high. The active subfigures are 6a--6d in both source and `main.aux`.
- **Fix —** Replace the Figure 6 sentence with: “Figure~\ref{fig:analytical_panels} visualises the two-dimensional prior conditional, likelihood, exact posterior, and SI-SDE sampled posterior.” Keep the Table 3 sentence unchanged.

### C8 — The sampler cross-reference points to implementation rather than the sampler definition

- **Location —** Section 5 opening; “The samplers of Section~\ref{subsec:summary_table} are reported as …”
- **Severity —** minor.
- **Problem —** `subsec:summary_table` resolves to Section 4.4, “Implementation”; the three sampler equations are defined in Section 4.1. The stale label name also promises a summary table that does not exist.
- **Confidence —** high. `main.aux` resolves the label to 4.4, while the SI-SDE/DM-SDE/FM-ODE display is in 4.1.
- **Fix —** Replace `\ref{subsec:summary_table}` with `\ref{subsec:posterior_dynamics}`. Optionally rename the stale source label to `subsec:implementation` after updating its references.

### C9 — Figure 4's “all methods” pointer leads to figures that omit methods

- **Location —** Figure 4 caption; “All methods and scenarios: Figures~\ref{fig:ns_fields_32}--\ref{fig:ns_fields_sparse1p5}.”
- **Severity —** minor.
- **Problem —** Figures 7--10 explicitly omit both DM-SDE variants and show only selected baselines; they do cover all scenarios, not all methods. The caption overstates what the cross-reference contains.
- **Confidence —** high. The active column definitions in Appendix H list SI-SDE/FM-ODE plus four baselines and contain no DM-SDE panels.
- **Fix —** Replace with “Additional retained methods and all scenarios: Figures~\ref{fig:ns_fields_32}--\ref{fig:ns_fields_sparse1p5}.”

### C10 — Random paths and their realizations switch case without a definition

- **Location —** Section 3, Eqs. (3)--(4); “$\bx_\tau=\alpha_\tau\bx_0+\sigma_\tau\boldsymbol\varepsilon+\beta_\tau\bx_1$” versus “$\bX_\tau=\bx$.”
- **Severity —** minor.
- **Problem —** Eq. (3) defines lower-case $\bx_\tau$ as the random path, then Eq. (4) conditions on an undefined upper-case $\bX_\tau$; both conventions recur in the proofs. This makes the already-important distinction between $\bx_0$ and $\bX_0$ harder to follow.
- **Confidence —** high. No sentence in the active paper defines the case convention.
- **Fix —** Add: “We write $\bX_\tau$ for the random path and $\bx$ (or $\bx_\tau$) for a realization,” and change the left sides of Eqs. (3), (19), (33), and (41) to $\bX_\tau$.

### C11 — Abstract fields use inconsistent time signatures

- **Location —** Section 3, Eq. (5), and Appendix A, Eq. (32); “$\fmvel(\bx,\bx_0,\tau)$” / “$\drift_\tau(\bx,\bx_0,\tau)$.”
- **Severity —** minor.
- **Problem —** The abstract velocity and drift are otherwise written as $\fmvel_\tau(\bx,\bx_0)$ and $\drift_\tau(\bx,\bx_0)$. The cited equations encode pseudo-time twice, once as a subscript and once as an argument, without identifying a distinct object.
- **Confidence —** high for the notation inconsistency. The parameterised neural fields $\fmvel_\theta(\bx,\state^{n-1},\tau)$ are a separate, explicitly parameterised notation and need not change.
- **Fix —** In Eq. (5) use `$\fmvel_\tau(\bx,\bx_0)$`; in Eq. (32) use `$\drift_\tau(\bx,\bx_0)$`.

### C12 — Core symbols are reused for unrelated physical and generative quantities

- **Location —** Sections 3 and 5.2, Eqs. (4) and (18); “where $\velocity=\nabla^\perp\psi$” and “velocity $\fmvel_\tau$.”
- **Severity —** minor.
- **Problem —** The same bold $\boldsymbol v$ denotes the generative probability-path velocity and the Navier--Stokes fluid velocity. The NS equation also reuses $\alpha$ (path schedule) for drag and $\varepsilon/\xi$ (source/model-noise glyphs) for forcing. These are different types, not merely repeated dummy indices.
- **Confidence —** high. The definitions are explicit in Sections 3 and 5.2/Appendix H.
- **Fix —** Rename the NS-local quantities, e.g. `$\boldsymbol v_{\rm NS}$`, `$\alpha_{\rm d}$`, `$\varepsilon_{\rm f}$`, and `$\xi_{\rm f}$`, in Eq. (18) and Appendix H. This leaves the central method notation untouched.

### C13 — $\mathcal K$ denotes sets with incompatible element types

- **Location —** Appendix E, spectral RMSE, and Appendix H, Eq. (61); “over the shells $\mathcal K$” / “$\mathcal K=\{(6,0),(7,0),(5,5),(8,8)\}$.”
- **Severity —** minor.
- **Problem —** Appendix E uses $\mathcal K$ for radial spectral shells, while Appendix H uses the identical symbol for a set of forced two-dimensional Fourier wavevectors. The element type changes from shells/scalar bins to vectors.
- **Confidence —** high. Both definitions are explicit.
- **Fix —** Rename the forcing set to `$\mathcal K_{\rm force}$` in Eq. (61), or the metric shells to `$\mathcal S$` in Appendix E.

### C14 — The Navier--Stokes appendix's trajectory count conflicts with its spectral figure

- **Location —** Appendix H opening and Figure 14 caption; “Throughout: five held-out trajectories” versus “one held-out trajectory.”
- **Severity —** minor.
- **Problem —** The appendix declares five trajectories “throughout,” while Figure 14 explicitly averages its spectra over the steps of one trajectory. The reader cannot tell which appendix artefacts use five runs and which use one.
- **Confidence —** high for the textual conflict; checking the plot-generation script would be needed only to determine which sentence reflects the generated figure data.
- **Fix —** Replace the opening with: “Headline metrics and step curves aggregate five held-out trajectories; qualitative fields and enstrophy spectra show one representative trajectory. All runs use $E=64$ and 15 assimilation steps.”

### C15 — The theorem drops its fixed conditioning argument in the marginal formula

- **Location —** Theorem 4.1, after Eq. (9); “$p_\tau^{\obs}(\bx\mid\bx_0)\propto\Phi_\tau^{\obs}(\bx,\bx_0)p_\tau(\bx)$.”
- **Severity —** nit.
- **Problem —** The left side, tilt, velocity, and score are all conditional on $\bx_0$, but the prior marginal on the right is written unconditionally. Eq. (37) in the proof uses the consistent `$p_\tau(\bx\mid\bx_0)$`.
- **Confidence —** high. Eq. (37) supplies the intended notation.
- **Fix —** Replace `$p_\tau(\bx)$` with `$p_\tau(\bx\mid\bx_0)$` in Theorem 4.1.

## Cross-reference audit

The compiled active build has no undefined or multiply-defined references or citations in `main.log`, and every active label resolves in `main.aux`. Semantic target checks found the following actionable mismatches; all are findings above.

| Source pointer | Resolved target | Audit result |
|---|---|---|
| “samplers of Section `subsec:summary_table`” | Section 4.4, Implementation | Wrong target; C8 |
| “Metric definitions … Appendix E” | Appendix E | Sliced-$W_2$ definition inactive; C5 |
| Figure 4 “all methods” range | Figures 7--10 | Target omits DM-SDE and several methods; C9 |
| Appendix G prose about Figure 6 | Figure 6a--d | Describes inactive panels; C7 |
| “Table 4 collects the results” | Placeholder Table 4 | Target contains no results; C6 |

All other active theorem/equation/figure/table/appendix references were semantically consistent within this review's scope.

## Notation and dimension mismatch table

| Symbol/expression | First role/type | Conflicting role/type | Finding |
|---|---|---|---|
| $\bx_0$ vs. $\bX_0$ | Fixed physical conditioning state / anchor in $\mathbb R^{N_u}$ | Pseudo-time random sampler state; Gaussian for FM/DM | C1 |
| $\bx_\tau$ vs. $\bX_\tau$ | Random path as written in Eq. (3) | Random variable used in conditioning in Eq. (4) | C10 |
| conditioning input | One state $\state^{n-1}\in\mathbb R^{N_u}$ | Five states / five input channels in Appendix D | C3 |
| $\fmvel_\tau$ / $\drift_\tau$ | Time-subscripted vector fields | Time-subscripted fields with a duplicate $\tau$ argument | C11 |
| $\boldsymbol v$ | Probability-path velocity in $\mathbb R^{N_u}$ | Two-component physical fluid velocity field | C12 |
| $\alpha$, $\varepsilon$, $\xi$ | Interpolant/source/model-noise symbols | NS drag/forcing symbols | C12 |
| $\mathcal K$ | Set of spectral shells | Set of wavevectors in $\mathbb Z^2$ | C13 |

I found no additional dimension mismatch in Eqs. (9), (13)--(17), or Algorithm 1's non-endpoint matrix products: $H\in\mathbb R^{N_y\times N_u}$, $\Sigma^{\rm s}\in\mathbb R^{N_u\times N_u}$, and the displayed guidance is in $\mathbb R^{N_u}$.

## Clean scoped areas

- Abstract, Introduction, and Related Work — no additional internal-consistency finding within this scope.
- Appendix C (moment proofs) — no main/appendix definition drift beyond the path-case issue C10.
- Appendix F (baseline descriptions) — no inconsistent object naming or cross-reference finding.
- Appendix I (urban appendix) — active scenario counts, sensor densities, $E$, $M$, and figure references agree with Section 5.3/Table 2.
- Reproducibility and AI-use statements — no internal cross-reference inconsistency found.
