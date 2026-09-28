# Stage 1E — Related work and positioning

Review date: 2026-08-10. Scope: novelty delta, omitted closely related work, and accuracy of argumentative citation characterisations. I read the complete active manuscript/PDF and `review/paper-map.md`. Bibliography formatting was not reviewed.

## Findings

### E1 — The “closest work” sentence is no longer defensible

- **Location** — Related Work, “Posterior sampling and data assimilation with generative models” (p. 2), no numbered artefact: “Closest to our work is Yan et al. (2025)”
- **Severity** — major
- **Problem** — FIG was plausibly the closest comparator in early 2025, but the statement is false as a current ICLR submission. FlowDPS (ICCV 2025) and FLOWER (ICLR 2026) condition pretrained FM priors without retraining; FAPS (2026 preprint) combines a pretrained function-space FM prior, sparse/noisy observations, and covariance-preconditioned posterior sampling; DAISI (2025 preprint) performs sequential DA with conditional stochastic-interpolant sampling and no per-step prior retraining. These works do not establish the manuscript's exact common equal-marginal ODE/SDE family, but omitting them makes the novelty delta look stale. Multitask Learning with Stochastic Interpolants (NeurIPS 2025), already present in `references.bib` but uncited, also demonstrates posterior sampling without task-specific training after operator-interpolant pretraining.
- **Confidence** — high; I checked the primary papers/abstracts and publication dates. Nothing further is needed to establish the omission; an independent full-paper comparison would only sharpen which one deserves the word “closest.”
- **Fix** — Replace the two sentences beginning “Closest to our work” with: “FIG interpolates measurements within deterministic flow matching, while FlowDPS and FLOWER condition pretrained FM priors and FAPS uses covariance-preconditioned likelihood corrections for function-space FM priors. DAISI applies guided stochastic-interpolant sampling to sequential DA, and operator-valued stochastic interpolants also support posterior sampling after multitask pretraining. Unlike these FM-specific, DA-specific, or specially pretrained constructions, we derive one observation-likelihood correction and equal-marginal ODE/SDE family for conventional pretrained interpolant models.” Add citations to Yan et al. (2025), Kim et al. (2025), Pourya et al. (2026), Shi et al. (2026), Andrae et al. (2025), and Negrel et al. (2025).

### E2 — The model-by-model premise is an overclaim, not the safe novelty delta

- **Location** — Introduction, first paragraph (p. 1), no numbered artefact: “each tied to whether the generator is stochastic or deterministic”
- **Severity** — major
- **Problem** — The categorical “each” is vulnerable to Negrel et al.'s operator-valued stochastic-interpolant framework, which explicitly unifies flow/diffusion dynamics and includes posterior sampling, and to current analyses that place multiple FM inverse solvers in one posterior-transport framework. The defensible novelty is narrower: the manuscript accepts a conventionally trained interpolant prior and derives this particular observation-interpolant likelihood and equal-marginal sampler family.
- **Confidence** — high that the wording is overbroad; medium on absolute priority because proving a negative over the entire literature is impossible. A formal novelty search by the authors could raise the priority confidence.
- **Fix** — Replace the final clause with: “yet most inference rules are presented separately for stochastic and deterministic generators; we seek a common posterior-sampling interface for conventionally trained interpolant priors.” Cite Negrel et al. (2025) when distinguishing its operator-valued/multitask pretraining requirement.

### E3 — The covariance positioning omits prior second-moment Gaussian closures

- **Location** — Related Work, “Posterior sampling and data assimilation with generative models” (p. 2), no numbered artefact: “accounting for covariance inflation and a necessary bias correction”
- **Severity** — major
- **Problem** — This contrast names only DPS and PiGDM, although TMPD derives a Gaussian moment projection with a second-order Tweedie covariance, and SDA Eq. (15) uses an inflated covariance containing a prior-dependent matrix Gamma. The manuscript may still be novel in deriving source conditional moments for a general observation interpolant and reusing them across SI/FM/diffusion, but covariance-aware Gaussian closure itself is not a clean novelty boundary.
- **Confidence** — high; I checked TMPD's method and SDA Eq. (15). Nothing further is needed to establish the missing comparison.
- **Fix** — Replace the final sentence with: “Second-order Tweedie methods such as TMPD and SDA also use model-dependent covariance inflation in Gaussian likelihood approximations. Our distinction is that the biased mean and covariance follow from source moments of an observation interpolant and give one surrogate shared by SI, FM, and diffusion samplers.” Add Boys et al. (2024) and retain Rozet and Louppe (2023).

### E4 — FIG is materially mischaracterised, and is not literally the claimed family member

- **Location** — Appendix F, “Guided FM (FIG)” (p. 18), no numbered artefact: “treats x_tau approximately as x_1, uses no covariance”
- **Severity** — major
- **Problem** — FIG explicitly defines a time-dependent Gaussian measurement-interpolant likelihood, `q_t(y_t|x_t) = N(y_t; A x_t, alpha_t^2 sigma_n^2 I)`, and its corrector follows the gradient of an unnormalised squared residual with a coefficient involving `lambda_t`, `sigma_t`, `alpha_t`, `sigma_n`, and the tuned constant `c`. Its paper gives a probability-path/conditional-likelihood derivation. Therefore “normalised residual,” “magnitude c(1-tau)/tau,” “uses no covariance,” “treats x_tau approximately as x_1,” and “not derived from a probability path” do not accurately describe the published algorithm. The discrete operator-split FIG corrector is related to, but not shown to be identical to, the manuscript's continuous `g_tau=0` member.
- **Confidence** — high; checked FIG Algorithm 1, its conditional likelihood, and its theoretical derivation in the primary ICLR paper. Nothing further is needed to establish the textual inaccuracies; exact equivalence to the manuscript's ODE would require an independent line-by-line derivation.
- **Fix** — Replace the FIG description and limitation with: “FIG alternates a prior-velocity step with `k` gradient steps on the squared measurement-interpolant residual, using the time-dependent Gaussian likelihood `q_t(y_t|x_t)=N(y_t;A x_t,alpha_t^2 sigma_n^2 I)` and a tuned scale `c`. Our `g_tau=0` sampler is a continuous guided-ODE analogue, while FIG is a deterministic FM-specific operator-split scheme.” In the Introduction replace “including FIG and OT-ODE” by “closely related to FIG and OT-ODE.”

### E5 — The SDA description presents an implementation simplification as the published method

- **Location** — Appendix F, “SDA” (p. 18), no numbered artefact: “with isotropic denoiser variance scaled by a tunable factor”
- **Severity** — major
- **Problem** — SDA's published Eq. (15) uses `Sigma_y + sigma(t)^2/mu(t)^2 A Gamma A^T`, where the paper says `Gamma` depends on the eigendecomposition of the prior covariance `Sigma_x`. SDA then permits a constant diagonal simplification and uses `Gamma=10^{-2}I` in its reported experiments. The manuscript presents that simplification as SDA generally and then attributes “no model source information” as a method-level limitation; the cited method is more general, even though its prior-covariance construction is not the manuscript's conditional source-moment construction.
- **Confidence** — high; checked SDA Section 3.2, Eq. (15), and the experimental setting. Nothing further is needed.
- **Fix** — Replace the final clause with: “In the published SDA approximation, `Gamma` may encode prior covariance; our baseline follows its constant-isotropic simplification, whereas our covariance is obtained directly from conditional source moments.” Also change the first sentence to “In our implementation, the denoiser covariance is isotropic and scaled by a tunable factor.”

### E6 — The cited sources do not support the manuscript-specific Doob/marginal conclusion

- **Location** — Appendix F, “SDA” (p. 18), no numbered artefact: “conditions trajectories rather than marginals, so the intermediate marginals are not p_tau^y”
- **Severity** — major
- **Problem** — A conditioned/twisted diffusion has intermediate marginal laws; the relevant distinction is whether those laws equal this manuscript's separately prescribed independent-coupling interpolant marginals. Wu et al. describe an SMC twisted-diffusion sampler targeting conditional distributions, while the accessible Särkkä–Solin source identifies the general SDE text but does not verify this manuscript-specific non-equality. I could not verify this conclusion from either citation, and as written it reads as a category error used to diminish SDA.
- **Confidence** — medium; high that the citations do not themselves establish equality/non-equality with `p_tau^y`, but checking the exact SDA reverse process against the manuscript's path definitions would raise confidence on the mathematical conclusion.
- **Fix** — Replace the quoted clause with: “SDA conditions a fixed prior diffusion through a Doob/twisting correction; its conditional time marginals need not coincide with the independently recoupled interpolant marginals `p_tau^y` prescribed here.” Either prove that narrower claim or remove the asserted non-equality and the two citations.

### E7 — A current posterior-transport paper creates an apparent theoretical conflict that must be distinguished

- **Location** — Introduction, “A unified view” (p. 1), no numbered artefact: “Conditioning this path on observations shifts its target to the posterior”
- **Severity** — major
- **Problem** — Xu et al. (2026) prove that conditioning a fixed deterministic flow transport can be represented by reweighting its source while leaving its velocity unchanged, and analyse FlowDPS/FLOWER/PnP-Flow as approximate correction fields. This does not necessarily contradict the manuscript, which appears to choose a new independently coupled posterior interpolant, but without stating the coupling distinction a hostile reviewer can present the claims as incompatible.
- **Confidence** — medium-high; the Xu et al. result is explicit, but an independent mathematical comparison of its fixed-map coupling with Proposition 1/Theorem 1 would raise confidence that the proposed distinction fully resolves the issue.
- **Fix** — Add to Related Work: “Xu et al. (2026) instead condition a fixed deterministic transport by reweighting its source; our theorem selects a posterior interpolant with an independent source coupling and derives ODE/SDE dynamics realising those prescribed marginals.” If that sentence is not mathematically exact, the authors must state the correct coupling distinction rather than omit the paper.

## No findings within scope

- Abstract: no additional related-work or positioning finding beyond E1–E3's effect on its novelty reading.
- Preliminaries, Methodology, Results, and Conclusion: no additional citation-characterisation finding within this reviewer's scope.
- Appendices A–E and G–I: no finding within scope.
- FlowDAS, OT-ODE, and PnP-Flow: I found no material paper-characterisation error in the active prose; the FlowDAS description agrees with the published algorithm at the level claimed.

## Primary external works inspected

The list distinguishes full/relevant-method inspection from abstract screening. All links are to author/publisher, proceedings, OpenReview, DOI, or arXiv primary records.

### Full paper or relevant method/algorithm inspected

1. Yici Yan, Yichi Zhang, Xiangming Meng, and Zhizhen Zhao (2025), “FIG: Flow with Interpolant Guidance for Linear Inverse Problems,” ICLR 2025. https://openreview.net/forum?id=fs2Z2z3GRx ; direct paper: https://openreview.net/pdf/7aa9caaad83abe3a9ed818e1eaf7a67c75e0df28.pdf
2. François Rozet and Gilles Louppe (2023), “Score-based Data Assimilation,” NeurIPS 2023. https://openreview.net/forum?id=VUvLSnMZdX
3. Siyi Chen, Yixuan Jia, Qing Qu, He Sun, and Jeffrey A. Fessler (2025), “FlowDAS: A Stochastic Interpolant-based Framework for Data Assimilation,” NeurIPS 2025. https://openreview.net/forum?id=1nWqhiulqD
4. Ashwini Pokle, Matthew J. Muckley, Ricky T. Q. Chen, and Brian Karrer (2024), “Training-free Linear Image Inverses via Flows,” TMLR 2024. https://openreview.net/forum?id=PLIt3a4yTm
5. Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye (2025), “FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems,” ICCV 2025. https://arxiv.org/abs/2503.08136 ; DOI: https://doi.org/10.1109/ICCV51701.2025.01146
6. Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebastian Reich, Alan Mosca, and O. Deniz Akyildiz (2024), “Tweedie Moment Projected Diffusions for Inverse Problems,” TMLR 2024 / ICLR 2025 Journal Track. https://openreview.net/forum?id=4unJi0qrTE
7. Hugo Negrel, Florentin Coeurdoux, Michael S. Albergo, and Eric Vanden-Eijnden (2025), “Multitask Learning with Stochastic Interpolants,” NeurIPS 2025. https://arxiv.org/abs/2508.04605
8. Martin Andrae, Erik Larsson, So Takao, Tomas Landelius, and Fredrik Lindsten (2025), “DAISI: Data Assimilation with Inverse Sampling using Stochastic Interpolants.” https://arxiv.org/abs/2512.00252
9. Mehrsa Pourya, Bassam El Rawas, and Michael Unser (2026), “FLOWER: A Flow-Matching Solver for Inverse Problems,” ICLR 2026. https://openreview.net/forum?id=QGd34p02mI
10. Jian Xu, Delu Zeng, John Paisley, and Qibin Zhao (2026), “What Do Flow-Based Inverse Solvers Approximate? A Posterior-Transport View.” https://arxiv.org/abs/2606.24516
11. Yasi Zhang, Peiyu Yu, Yaxuan Zhu, Yingshan Chang, Feng Gao, Ying Nian Wu, and Oscar Leong (2024), “Flow Priors for Linear Inverse Problems via Iterative Corrupted Trajectory Matching,” NeurIPS 2024. https://arxiv.org/abs/2405.18816
12. Julius Erbach, Dominik Narnhofer, Andreas Robert Dombos, Bernt Schiele, Jan Eric Lenssen, and Konrad Schindler (2025), “Solving Inverse Problems with FLAIR,” NeurIPS 2025. https://openreview.net/forum?id=w9xETx7HT1
13. Luhuan Wu, Brian L. Trippe, Christian A. Naesseth, David M. Blei, and John P. Cunningham (2023), “Practical and Asymptotically Exact Conditional Sampling in Diffusion Models,” NeurIPS 2023. https://arxiv.org/abs/2306.17775
14. Simo Särkkä and Arno Solin (2019), “Applied Stochastic Differential Equations.” https://doi.org/10.1017/9781108186735 . I could not verify the cited chapter's full text from the accessible primary page.

### Abstract/metadata screening only

15. Yaozhong Shi, Zachary E. Ross, and Yisong Yue (2026), “Flow Annealing Posterior Sampling for Function-Space Regression and Inverse Problems.” https://arxiv.org/abs/2606.22346
16. Thomas Savary, François Rozet, and Gilles Louppe (2026), “Training-Free Bayesian Filtering with Generative Emulators.” https://arxiv.org/abs/2605.20028
17. Hossein Askari, Yadan Luo, Hongfu Sun, and Fred Roosta (2025), “Latent Refinement via Flow Matching for Training-free Linear Inverse Problem Solving.” https://arxiv.org/abs/2511.06138
18. Alexander Denker, Moshe Eliasof, Zeljko Kereta, and Carola-Bibiane Schönlieb (2026), “Trajectory Stitching for Solving Inverse Problems with Flow-Based Models.” https://arxiv.org/abs/2602.08538
19. Ran Cheng and Lailai Zhu (2026), “FlowDA: Accurate, Low-Latency Weather Data Assimilation via Flow Matching.” https://arxiv.org/abs/2602.06800
20. Shadab Ahamed and Eldad Haber (2026), “DAWN-FM: Data-aware and Noise-informed Flow Matching for Solving Inverse Problems,” Foundations of Data Science. https://doi.org/10.3934/fods.2026005
21. Segolene Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl (2025), “PnP-Flow: Plug-and-Play Image Restoration with Flow Matching,” ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/708e58b0b99e3e62d42022b4564bad7a-Abstract-Conference.html

## Alleged mischaracterisations requiring Stage-2 independent primary-paper reading

1. **FIG (E4):** independently verify Algorithm 1, the conditional Gaussian `q_t(y_t|x_t)`, the residual scaling, and its probability-path derivation; then compare the discrete FIG update line by line with the manuscript's `g_tau=0` ODE.
2. **SDA covariance (E5):** independently verify SDA Eq. (15), the definition of `Gamma`, and the `Gamma=10^{-2}I` experimental simplification.
3. **SDA Doob/marginal claim (E6):** read Wu et al. and the cited Särkkä–Solin section, then compare their conditioned/twisted process with this manuscript's prescribed `p_tau^y` path. I could not verify the manuscript-specific non-equality from the cited sources.
4. **Posterior-transport relation (E7):** independently read Xu et al. (2026) and determine whether “fixed deterministic transport/source reweighting” versus “new independently coupled posterior interpolant” is the exact distinction needed to reconcile the results.
