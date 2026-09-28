# Paper map

Source mapped: the active `manuscript/main.tex` build and every file it inputs, checked against `manuscript/main.pdf` (36 pages, built 2026-08-10). Supporting context inspected: the active experiment configurations and result aggregates in `paper_experiments/`, and the prior, likelihood, posterior-sampler, observation-operator, metric, and aggregation code in `src/scisi/` and `paper_experiments/`. Archived/old TeX and archived experiment results are not part of the active paper and are excluded from the inventories below.

Page convention: PDF pages, not source line numbers. The ICLR-limited main paper occupies pages 1--9; statements and references begin on page 10; appendices begin on page 13.

## Section-by-section outline

### Front matter

- **Abstract (p. 1):** States the posterior-sampling problem, the common observation-interpolant construction, its exact idealised dynamics, its practical Gaussian/covariance approximations, and the three evaluation cases.

### Main paper

- **1 Introduction (pp. 1--2):** Motivates generative data assimilation, frames SI, FM, and diffusion as one posterior-sampler family, and lists the unified sampler, observation-interpolant, and scalable-covariance contributions.
- **2 Related work (p. 2):** Positions the work against physical generative forecasting and training-free posterior/DA guidance methods, with emphasis on how their likelihood scores and covariances differ.
- **3 Preliminaries (p. 3):** Defines the one-step Bayesian filtering target and the affine Gaussian interpolant path, then relates its velocity, score, FM/DM sampler, and Föllmer SI sampler.
  - **Bayesian inverse problems and data assimilation (p. 3):** Defines the Markov transition, observation model, one-step posterior, and autoregressive rollout.
  - **Interpolation-based generative modelling (p. 3):** Defines the common path, velocity--score identity, FM/DM SDE lift, and native SI SDE.
- **4 Methodology (pp. 3--6):** Derives the exact equal-marginal posterior family, replaces its intractable tilt by a Gaussian observation-interpolant surrogate, gives scalable covariance approximations, and provides the shared assimilation algorithm.
  - **4.1 Posterior sampling (p. 4):** Defines the likelihood tilt and states the unified posterior-sampler theorem and its SI-SDE, DM-SDE, and FM-ODE members.
  - **4.2 Closed-form likelihood score from observation interpolation (pp. 4--5):** Defines the observation interpolant, derives its exact conditional moments, and introduces a stopped-covariance Gaussian score approximation.
  - **4.3 Practical approximations (p. 5):** Describes stopped covariance gradients, the Jacobian-free covariance, the ensemble-shared Jacobian, and empirical Jacobian damping.
  - **4.4 Implementation (pp. 5--6):** Specifies endpoint handling, solver choices, autoregressive feedback, and Algorithm 1.
- **5 Results (pp. 6--9):** Compares the method and baselines on an analytical control, stochastic Navier--Stokes, and urban airflow.
  - **5.1 Simple analytical test case (p. 7):** Uses a closed-form linear--Gaussian posterior to isolate discretisation and likelihood-covariance bias.
  - **5.2 Stochastic Navier--Stokes (pp. 7--8):** Evaluates dense super-resolution and sparse-sensor filtering with learned priors, conventional filters, calibration, and cost.
  - **5.3 Urban airflow over a building array (pp. 8--9):** Evaluates a long, sparse-in-time rollout from a mismatched initial state, including the unobserved temperature field.
- **6 Conclusion (p. 9):** Summarises which sampler/covariance choices win in each case and states the exactness, observation-model, covariance-cost, and autoregressive-filtering limitations.

### End matter

- **Reproducibility statement (p. 10):** Points to theory, implementation, metrics, baseline, and case appendices and claims anonymised supplementary code is provided.
- **AI use statement (p. 10):** Discloses AI assistance for code, experiments, analysis, editing, formatting, and literature search, while excluding theory/proof development.
- **References (pp. 10--12):** Lists 37 cited works.

### Appendices

- **A Velocity--score duality and marginal-preserving SDE family (pp. 13--15):** Proves Tweedie identities, velocity--score duality, and marginal preservation, then specialises score recovery to FM and SI networks.
  - **A.1 Reading velocity and score from the trained network (pp. 14--15):** Gives the FM and SI network-to-score formulas used by the samplers.
- **B Conditioning the interpolant path: proofs (pp. 15--16):** Proves that observation conditioning retargets the same path and proves Theorem 4.1.
  - **B.1 Posterior path (p. 15):** Establishes the conditional path and score/velocity shifts.
  - **B.2 Proof of Theorem 4.1 (p. 16):** Applies marginal preservation to the conditioned path and handles the initial law.
- **C Proof of Lemmas 4.2 and 4.3 (p. 16):** Derives the observation-interpolant decomposition and source conditional moments.
- **D Implementation details (pp. 16--17):** Records schedules, integrators, U-Net architectures, training data, optimisation, and checkpoint reuse.
- **E Evaluation metrics (pp. 17--18):** Defines ensemble-mean RMSE, spectral RMSE, CRPS, spread--skill, pointwise Gaussian KL, analytical Gaussian KL, and runtime.
- **F Method descriptions and comparison to alternatives (pp. 18--19):** Describes FlowDAS, FIG, SDA, D-Flow SGLD, SURGE, EnKF/PF, and the FM-derived diffusion prior.
- **G Analytical test case (pp. 19--20):** Specifies the exact Gaussian prior/posterior, analytic SI/FM fields, and detailed convergence/cost results.
  - **G.1 Setup (pp. 19--20):** Defines the two-dimensional plotted case and its exact posterior.
  - **G.2 Additional results (p. 20):** Gives posterior panels, full method table, and the covariance-ablation interpretation.
- **H Navier--Stokes test case (pp. 21--30):** Specifies the forcing/data generation and supplies per-scenario fields, step curves, spectra, and an unfinished ablation table.
  - **H.1 Forcing and data generation (p. 21):** Defines the forced vorticity process, numerical grid, time integration, and dataset construction.
  - **H.2 Enstrophy spectra (p. 21; figure on p. 29):** Defines and plots the reconstruction-scale diagnostic.
  - **H.3 Ablations (pp. 21, 30):** Enumerates covariance, Jacobian cadence/damping, diffusion, step-count, and ensemble-size ablations.
- **I Urban airflow test case (pp. 21, 30--36):** Specifies the uDALES setting and gives qualitative fields and per-step RMSE/CRPS/spread--skill curves.

## Claim ledger

The ledger includes every externally checkable or contribution-level claim in the abstract, introduction, and conclusion. Pure statements of paper organisation are omitted.

| ID | Location | Claim | Intended evidence in the paper |
|---|---|---|---|
| C1 | Abstract, sentence 1 | High-dimensional, non-Gaussian posterior sampling in DA remains difficult. | Introduction p. 1 and DA references `evensen_data_2022`, `carrassi_data_2018`; no theorem/figure/table. |
| C2 | Abstract, sentence 2 | The observation-interpolant mechanism can convert any pretrained SI, FM, or diffusion generator into a posterior sampler without retraining. | Common path Eq. (3), Theorem 4.1/Eq. (9), Algorithm 1; implementation Appendix D. |
| C3 | Abstract, sentence 3 | Conditioning adds one shared likelihood-score correction to score and velocity. | Proposition B.1, Eqs. (34)--(35). |
| C4 | Abstract, sentence 3 | The resulting marginal-preserving SDE is exact when the intermediate likelihood score is known. | Theorem 4.1 and Proposition A.3. |
| C5 | Abstract, sentence 4 | The framework contains native SI and denoising-diffusion SDEs and a deterministic guided ODE. | Eq. (9) and the three displayed members immediately after Theorem 4.1; Eqs. (6)--(7). |
| C6 | Abstract, sentence 5 | A closed-form Gaussian surrogate uses a bias-corrected mean and source-inflated covariance. | Lemmas 4.2--4.3, Eqs. (13)--(15). |
| C7 | Abstract, sentence 5 | Source-covariance inflation is essential for sparse observations. | Table 1, especially SI-SDE shared vs Jacobian-free at 5% and 1/64; Appendix G/Table 3 for bias control. |
| C8 | Abstract, sentence 6 | Jacobian-free and ensemble-shared approximations make the method tractable at high state dimension. | Section 4.3 complexity discussion; Tables 1--2 runtime columns; implementation in `InterpolantGaussianLikelihood`. |
| C9 | Abstract, sentence 7 | The paper evaluates linear--Gaussian, stochastic 2D Navier--Stokes, and urban airflow cases. | Sections 5.1--5.3; Figures 3--5; Tables 1--3. |
| C10 | Introduction, paragraph 1 | EnKF is efficient but Gaussian, while particle filters are general but sample-hungry. | `evensen_data_2022`; analytical and NS EnKF/PF rows in Tables 1 and 3 are illustrative, not general evidence. |
| C11 | Introduction, paragraph 1 | Learned forward surrogates are typically deterministic and require hard-to-specify model-error distributions. | Citations `mucke_deep_2024`, `cheng_generalised_2022`, `chen_reduced-order_2023`; no internal theorem/figure/table. |
| C12 | Introduction, paragraph 2 | SI and FM are the two dominant flow-based families used here. | SI/FM citations; no internal quantitative evidence for “dominate.” |
| C13 | Introduction, paragraph 2 | Denoising diffusion corresponds to SDE sampling of an FM-type model. | Eqs. (5)--(6), Proposition A.3, and `ho_denoising_2020`, `song_score-based_2021`. |
| C14 | Introduction, paragraph 2 | Existing posterior-conditioning methods were developed model by model and tied to deterministic/stochastic generator type. | Related work Section 2 and Appendix F; no table/theorem. |
| C15 | Introduction, “unified view” | SI, FM, and diffusion priors arise from one interpolant path. | Eq. (3), Appendix A, and `albergo_stochastic_2023`, `lipman_flow_2023`, `ma_sit_2024`, `holderrieth_generator_2024`. |
| C16 | Introduction, “unified view” | Conditioning retargets that path and shifts prior score/velocity through a shared likelihood-score term. | Proposition B.1. |
| C17 | Introduction, “unified view” | Equal-marginal dynamics give native SI-SDE, FM-derived DM-SDE, and deterministic guided ODE, with FIG/OT-ODE as deterministic precedents. | Theorem 4.1/Eq. (9), Appendix A, and `yan_fig_2024`, `pokle_training-free_2024`. |
| C18 | Introduction, “unified view” | The three practical samplers use one Gaussian surrogate and differ only in diffusion and likelihood-score weighting. | Eq. (9), Eq. (15), Figure 1, and the displayed sampler specialisations. |
| C19 | Introduction, contribution 1 | Weight (w_\tau=\kappa_\tau+\tfrac12g_\tau^2) turns any interpolant-based generator into a posterior-sampler family. | Theorem 4.1/Eq. (9); scope is conditioned on the theorem assumptions. |
| C20 | Introduction, contribution 2 | Observation interpolation gives a practical closed-form substitute with exact biased mean and inflated covariance. | Lemmas 4.2--4.3 and Gaussian closure Eq. (15). |
| C21 | Introduction, contribution 3 | Jacobian-free and ensemble-shared covariance approximations make sparse field-scale assimilation feasible. | Section 4.3, Tables 1--2 runtime columns, and supporting implementation/configuration; no dedicated scaling table. |
| C22 | Conclusion, sentence 1 | The work introduces one observation-interpolant posterior framework for SI, FM, and diffusion. | Theorem 4.1, Eq. (15), Algorithm 1. |
| C23 | Conclusion, sentence 2 | With full covariance, all three samplers recover the exact linear--Gaussian posterior. | Figure 3 and Table 3 (KL about (10^{-3})); Appendix G. |
| C24 | Conclusion, sentence 3 | On dense Navier--Stokes observations, the proposed methods achieve the best accuracy. | Table 1, 32² and 16² RMSE/CRPS columns; Figure 4 is sparse and not evidence for the dense part. |
| C25 | Conclusion, sentence 3 | Shared covariance substantially improves SI-SDE for sparse Navier--Stokes observations. | Table 1, SI-SDE shared vs Jacobian-free at 5% and 1/64. |
| C26 | Conclusion, sentence 4 | Urban DM-SDE is best for observed velocity/calibration, while SI-SDE best limits error in unobserved temperature. | Table 2 and Appendix-I per-step curves, Figures 19--21. |
| C27 | Conclusion, sentence 5 | Urban Jacobian-free performance remains close to shared at 3--4× lower cost. | Table 2 runtime and accuracy columns. |
| C28 | Conclusion, sentence 6 | The best sampler/covariance depends on the problem; prior correlations matter strongly in sparse NS but modestly in urban. | Tables 1--2 and Appendices H--I. |
| C29 | Limitations, sentence 1 | The exact theorem assumes exact prior/likelihood score; practice uses Gaussian closure, stopped covariance gradients, and approximate Jacobians. | Theorem 4.1 assumptions and Sections 4.2--4.3. |
| C30 | Limitations, sentence 2 | These approximations can bias results, visible in analytical and sparse NS Jacobian-free results. | Figure 3, Tables 1 and 3. |
| C31 | Limitations, sentence 3 | The current method assumes linear Gaussian observations. | Lemma 4.2 and Eq. (15); experiment observation models. |
| C32 | Limitations, sentence 3 | Shared covariance becomes costly with observation dimension. | Section 4.3 complexity statement; implementation forms an (N_y\times N_y) system and observation-column JVPs. |
| C33 | Limitations, sentence 4 | One-step autoregressive filtering does not reweight past states or use future observations, and error can accumulate. | Eq. (2), Algorithm 1, Figure 2; urban per-step curves in Figures 19--21 support accumulation empirically. |

## Notation table

“First defined” means the first active formal definition; if a symbol appears earlier pictorially or rhetorically, that is noted.

### Data-assimilation and probability notation

| Symbol | Meaning | Type / dimension | First defined |
|---|---|---|---|
| (n\) | Physical-time index | Integer (1,\ldots,N) | Sec. 3, Eq. (1), p. 3 |
| (N\) | Number of physical steps in the abstract DA setup | Positive integer | Sec. 3, after Eq. (1), p. 3 |
| (\state^n\) | Physical state at step (n) | (\mathbb R^{N_u}) | Sec. 3, after Eq. (1), p. 3 |
| (\obs^n\) | Observation at step (n) | (\mathbb R^{N_y}) | Sec. 3, after Eq. (1), p. 3 |
| (N_u\) | State dimension | Positive integer | Sec. 3, after Eq. (1), p. 3 |
| (N_y\) | Observation dimension | Positive integer | Sec. 3, after Eq. (1), p. 3 |
| (\obsoperator\) | Observation operator | Map (\mathbb R^{N_u}\to\mathbb R^{N_y}) | Sec. 3, after Eq. (1), p. 3 |
| (H\) | Matrix of a linear observation operator | (\mathbb R^{N_y\times N_u}) | Sec. 4.2, Lemma 4.2, p. 4 |
| (\obsnoise^n\) | Observation noise | Random vector in (\mathbb R^{N_y}) | Eq. (1b), p. 3 |
| (R\) | Observation-noise covariance | (\mathbb R^{N_y\times N_y}), positive (semi)definite; experiments use scalar multiples of (I) | Lemma 4.2, p. 4 |
| (p(\cdot),p_\tau(\cdot)\) | Density/law, optionally at pseudo-time (\tau) | Scalar density or probability law | Sec. 3, Eqs. (1)--(2), p. 3 |
| (p(\state^n\mid\state^{n-1})\) | One-step transition prior | Conditional law on (\mathbb R^{N_u}) | Eq. (1a), p. 3 |
| (p(\state^n\mid\obs^n,\state^{n-1})\) | One-step filtering posterior | Conditional law on (\mathbb R^{N_u}) | Eq. (2), p. 3 |
| (\mathbb E,\operatorname{Cov},\operatorname{Var}\) | Expectation/covariance/variance | Operators | Used from Sec. 3, Eq. (4), p. 3 |
| (\delta_{\bx_0}\) | Point-mass source at the SI anchor | Probability measure | Appendix F, p. 18 (also implicit in Eq. (7)) |

### Interpolant path and sampler notation

| Symbol | Meaning | Type / dimension | First defined |
|---|---|---|---|
| (\bx_0\) | Deterministic anchor/previous state | (\mathbb R^{N_u}); equals (\state^{n-1}) | Sec. 3, before Eq. (3), p. 3 |
| (\bx_1\) | Terminal/next state | (\mathbb R^{N_u}); equals (\state^n) | Sec. 3, before Eq. (3), p. 3 |
| (\bx_\tau,\bX_\tau\) | Path state / random path state | (\mathbb R^{N_u})-valued | Eq. (3), p. 3 |
| (\tau\) | Pseudo-time, unrelated to physical time | Scalar in ([0,1]) | Eq. (3), p. 3 |
| (\alpha_\tau,\beta_\tau,\sigma_\tau\) | Anchor, target, and Gaussian-source schedules | Differentiable scalar functions of (\tau) | Eq. (3), p. 3 |
| (\dot\alpha_\tau,\dot\beta_\tau,\dot\sigma_\tau\) | Pseudo-time derivatives of schedules | Scalars | Eq. (4), p. 3 |
| (\latentnoise\) | Standard-normal path driver | (\mathbb R^{N_u}), (\mathcal N(0,I)) | Eq. (3), p. 3 |
| (I\) | Identity matrix | (\mathbb R^{N_u\times N_u}), or context-dependent identity in observation space | Eq. (3), p. 3 |
| (\fmvel_\tau(\bx,\bx_0)\) | Marginal velocity / FM field | Map (\mathbb R^{N_u}\times\mathbb R^{N_u}\to\mathbb R^{N_u}) | Eq. (4), p. 3 |
| (\genscore_\tau(\bx,\bx_0)\) | Prior path score (\nabla_\bx\log p_\tau(\bx\mid\bx_0)) | Vector in (\mathbb R^{N_u}) | After Eq. (4), p. 3 |
| (\kappa_\tau\) | Velocity--score coefficient | Scalar schedule | Eq. (5), p. 3 |
| (\gamma_\tau\) | Native SI diffusion/noise schedule | Nonnegative scalar function | Eq. (6), p. 3 (SI meaning specialised in Eq. (7)) |
| (\bW_\tau\) | Standard Wiener process | (\mathbb R^{N_u})-valued | Eq. (6), p. 3 |
| (\drift_\tau\) | Native SI drift | Vector field (\mathbb R^{N_u}\to\mathbb R^{N_u}) conditional on (\bx_0) | Eq. (7), p. 3 |
| (g_\tau\) | Free diffusion coefficient of equal-marginal sampler family | Measurable nonnegative scalar, (L^2[0,1]) in Theorem 4.1 | Theorem 4.1, p. 4 |
| (\drift^{g}_\tau\) | Prior drift for chosen (g_\tau) | Vector in (\mathbb R^{N_u}) | Eq. (9), p. 4 |
| (w_\tau\) | Likelihood-guidance weight (\kappa_\tau+\tfrac12g_\tau^2) | Scalar | Eq. (9), p. 4; used earlier in Introduction p. 2 |
| (\bX^*_\tau\) | Posterior-sampler state | (\mathbb R^{N_u})-valued stochastic process | Eq. (9), p. 4 |
| (p_\tau^{\obs}\) | Posterior-path marginal | Density/law on (\mathbb R^{N_u}) | Theorem 4.1, p. 4 |
| (\Phi_\tau^{\obs}\) | Exact likelihood tilt (p(\obs\mid\bX_\tau=\bx,\bx_0)) | Positive scalar function | Eq. (8), p. 4 |
| (\scorefun_\tau^{\obs}\) | Exact tilt score (\nabla_\bx\log\Phi_\tau^{\obs}) | Vector in (\mathbb R^{N_u}) | After Eq. (8), p. 4 |
| (\genscore_\tau^{\obs},\fmvel_\tau^{\obs}\) | Conditional path score and velocity | Vector fields in (\mathbb R^{N_u}) | Proposition B.1, p. 15 |

### Observation-interpolant and approximation notation

| Symbol | Meaning | Type / dimension | First defined |
|---|---|---|---|
| (\interpolantobs_\tau\) | Interpolated observation (\alpha_\tau H\bx_0+\beta_\tau\obs) | (\mathbb R^{N_y}) | Eq. (10), p. 4 |
| (\bar\Phi_\tau^{\interpolantobs_\tau}\) | Surrogate intermediate likelihood | Positive scalar density/function | Sec. 4.2, after Eq. (10), p. 4 |
| (\interpolantscorefun_\tau^{\interpolantobs_\tau}\) | Surrogate likelihood score | Vector in (\mathbb R^{N_u}) | Sec. 4.2, after Eq. (10), p. 4 |
| (\postdrift_\tau\) | Practical posterior drift | Vector in (\mathbb R^{N_u}) | Eq. (11), p. 4 |
| (\src_\tau\) | Zero-mean Gaussian source process (\sigma_\tau\latentnoise) | (\mathbb R^{N_u})-valued | Lemma 4.2, p. 4 |
| (\interpolantobsnoise_\tau\) | Effective observation-interpolant noise (\beta_\tau\obsnoise-H\src_\tau) | (\mathbb R^{N_y})-valued | Eq. (12), p. 4 |
| (\srcmean_\tau\) | Conditional mean (\mathbb E[\src_\tau\mid\bx_\tau,\bx_0]) | (\mathbb R^{N_u}) | Lemma 4.2, after Eq. (13), p. 5 |
| (\srccov_\tau\) | Conditional covariance of (\src_\tau) | (\mathbb R^{N_u\times N_u}) | Lemma 4.2, after Eq. (13), p. 5 |
| (\bar\mu_\tau\) | Conditional mean of (\interpolantobs_\tau) | (\mathbb R^{N_y}) | Eq. (13), p. 4 |
| (\bar\Sigma_\tau\) | Conditional/effective likelihood covariance | (\mathbb R^{N_y\times N_y}) | Eq. (13), p. 4 |
| (\nabla_\bx\genscore_\tau\) | Score Jacobian / log-density Hessian | (\mathbb R^{N_u\times N_u}) | Eq. (14), p. 5 |
| (\nabla_\bx\bar\mu_\tau\) | Jacobian of surrogate likelihood mean | (\mathbb R^{N_y\times N_u}) | Eq. (15), p. 5 |
| (A_\tau\) | SI score-recovery schedule factor | Scalar | Appendix A, Eq. (32), p. 15; used earlier in Sec. 4.2 p. 5 |
| (c_\tau(\bx,\bx_0)\) | SI affine score-recovery term | (\mathbb R^{N_u}) | Appendix A, Eq. (32), p. 15; used earlier in Sec. 4.2 p. 5 |
| (J_{\drift_\tau}\) | Jacobian of learned SI drift | (\mathbb R^{N_u\times N_u}) | Eq. (17), p. 5 |
| (\lambda\) | Empirical learned-Jacobian damping | Scalar in ([0,1]) as stated in the paper | Sec. 4.3, p. 5 |
| (E\) | Posterior ensemble size | Positive integer | Sec. 4.3, p. 5 |
| (k\) | Shared-Jacobian refresh cadence | Positive integer pseudo-steps | Sec. 4.3, p. 5 |
| (M\) | Number of pseudo-time integration steps | Positive integer | Sec. 4.4/Algorithm 1, p. 6 |
| (\Delta\tau\) | Uniform pseudo-time step (1/M) | Positive scalar | Algorithm 1, p. 6 |
| (\bz\) | Fresh Gaussian Euler--Maruyama increment | (\mathbb R^{N_u}), (\mathcal N(0,I)) | Algorithm 1, p. 6 |
| (\bm c_\tau\) | Algorithm-local likelihood correction placeholder | (\mathbb R^{N_u}) | Algorithm 1 line 5, p. 6 |

### Proof-only notation

| Symbol | Meaning | Type / dimension | First defined |
|---|---|---|---|
| (\bm m_\tau\) | Noise-free affine component (\alpha_\tau\bx_0+\beta_\tau\bx_1) | (\mathbb R^{N_u})-valued | Lemma A.1, p. 13 |
| (q_\tau\) | Candidate SDE density in Fokker--Planck proof | Scalar density on (\mathbb R^{N_u}) | Proposition A.3 proof, p. 14 |
| (\theta\) | Neural-network parameters (subscript on learned field) | Parameter vector, dimension architecture-dependent | Appendix A.1, p. 14 |

### Experimental-system notation

| Symbol | Meaning | Type / dimension | First defined |
|---|---|---|---|
| (\mathcal T^2\) | Periodic Navier--Stokes domain ([0,2\pi]^2) | Two-dimensional torus | Sec. 5.2, p. 7 |
| (t,\rd t\) | Physical continuous time and increment | Scalars | Eq. (18), p. 7 |
| (\omega\) | Vorticity field | Scalar field on (\mathcal T^2) | Eq. (18), p. 7 |
| (\boldsymbol v\) | Fluid velocity | Two-component vector field on (\mathcal T^2) | Eq. (18), p. 7 |
| (\psi\) | Streamfunction | Scalar field on (\mathcal T^2) | After Eq. (18), p. 7 |
| (\nu\) | Kinematic viscosity | Positive scalar, (10^{-3}) in NS experiment | Eq. (18), p. 7 |
| (\alpha\) | Linear drag coefficient in NS equation | Scalar, (0.1) in Appendix H | Eq. (18), p. 7; overloads path schedule (\alpha_\tau) |
| (\varepsilon\) | NS forcing amplitude | Scalar, (1) in Appendix H | Eq. (18), p. 7; glyph overloads latent (\boldsymbol\varepsilon) |
| (\xi\) | Temporally white NS forcing | Random field/process | Eq. (18), p. 7 |
| (\mathcal K\) | Forced Fourier-mode set | Set of four wavevectors in (\mathbb Z^2) | Eq. (61), p. 21 |
| (\boldsymbol k\) | Fourier wavevector | Element of (\mathbb Z^2) | Eq. (61), p. 21 |
| (W_t^{\boldsymbol k,c},W_t^{\boldsymbol k,s}\) | Independent forcing Wiener processes | Scalar Wiener processes | Eq. (61), p. 21 |
| (Z(k)\) | Radially averaged enstrophy spectrum | Nonnegative scalar by radial bin | Eq. (62), p. 21 |
| (u,v,w\) | Three urban velocity components | Scalar grid fields | Sec. 5.3, p. 8 |
| (\theta_\ell\) | Urban potential-temperature field | Scalar grid field | Sec. 5.3/Appendix I captions, pp. 8, 32 |

### Evaluation notation

| Symbol | Meaning | Type / dimension | First defined |
|---|---|---|---|
| (\{\bx^{(e)}\}_{e=1}^E\) | Posterior ensemble | (E) vectors/fields in (\mathbb R^{N_u}) | Appendix E, p. 17 |
| (\bx^*\) | Truth/reference state | (\mathbb R^{N_u}) | Appendix E, p. 17 |
| (\mathcal I\) | Set of evaluated grid indices | Finite index set | Appendix E, p. 17 |
| (\bar\bx\) | Ensemble mean | (\mathbb R^{N_u}) | Appendix E, p. 17 |
| (E_k(\cdot)\) | Radially averaged kinetic-energy spectrum | Nonnegative scalar per shell | Appendix E, p. 17 |
| (\mathcal K\) | Non-empty spectral shells in metric definition | Finite set; overloads forced-mode set in Appendix H | Appendix E, p. 17 |
| (s_i\) | Ensemble standard deviation at point (i) | Nonnegative scalar | Appendix E, p. 17 |
| (m_i,s_i,m_i^r,s_i^r\) | Method/reference marginal Gaussian moments | Scalars at grid point (i) | Appendix E, p. 18 |
| (\mu_p,\Sigma_p,\mu_q,\Sigma_q\) | Sample and exact posterior Gaussian moments | Vectors in (\mathbb R^d), matrices in (\mathbb R^{d\times d}) | Appendix E, p. 18 |
| (d\) | Analytical-state dimension | Positive integer | Appendix E, p. 18 |
| (K\) | Kalman gain | (\mathbb R^{N_u\times N_y}); (\tfrac12I) in the analytical case | Appendix G, p. 19 |
| (\mu_1(\bx_0),\Sigma_1\) | Conditional Gaussian target mean/covariance | (\mathbb R^{N_u}), (\mathbb R^{N_u\times N_u}) | Appendix G, p. 19 |

## Artefact inventory

### Stated results

| Kind | Number / label | Title or function | Page |
|---|---|---|---|
| Theorem | 4.1, `theorem:unified_posterior` | Unified posterior sampler | 4 |
| Lemma | 4.2, `lem:interpolated_likelihood` | Interpolated-observation likelihood | 4 |
| Lemma | 4.3, `lem:source_moments` | Source conditional moments | 5 |
| Lemma | A.1, `lem:tweedie` | Tweedie identities for the interpolant path | 13 |
| Proposition | A.2, `prop:score_velocity` | Velocity--score identity | 13 |
| Proposition | A.3, `prop:marginal_preservation` | Marginal-preserving SDE family | 14 |
| Proposition | B.1, `prop:conditional_path` | Posterior path | 15 |

There are no active environments of kind Assumption, Definition, Corollary, or Remark in the compiled paper.

### Algorithms

| Number / label | Caption | Page |
|---|---|---|
| Algorithm 1, `alg:unified` | Unified posterior assimilation step | 6 |

### Figures

| Number / label | Caption/topic | Page |
|---|---|---|
| Figure 1, `fig:visual_abstract` | Unified posterior-sampling schematic | 2 |
| Figure 2, `fig:sequential_da` | Sequential assimilation with unified sampler | 6 |
| Figure 3, `fig:analytical` | Analytical KL versus sampler steps | 7 |
| Figure 4, `fig:ns_fields_main` | Navier--Stokes 5% sparse qualitative comparison | 8 |
| Figure 5, `fig:urban_fields_main` | Urban 0.78125% temperature qualitative comparison | 9 |
| Figure 6, `fig:analytical_panels` | Analytical prior, likelihood, exact and sampled posterior (subfigures 6a--6d) | 20 |
| Figure 7, `fig:ns_fields_32` | NS 32²→128² qualitative grid | 22 |
| Figure 8, `fig:ns_fields_16` | NS 16²→128² qualitative grid | 23 |
| Figure 9, `fig:ns_fields_sparse5` | NS 5% sparse qualitative grid | 24 |
| Figure 10, `fig:ns_fields_sparse1p5` | NS 1.5625% sparse qualitative grid | 25 |
| Figure 11, `fig:ns_rmse_curves` | NS RMSE versus (M) and assimilation step | 26 |
| Figure 12, `fig:ns_crps_curves` | NS CRPS versus (M) and assimilation step | 27 |
| Figure 13, `fig:ns_spread_skill_curves` | NS spread--skill versus (M) and assimilation step | 28 |
| Figure 14, `fig:ns_enstrophy_spectra` | NS full and high-wavenumber enstrophy spectra | 29 |
| Figure 15, `fig:urban_fields_vel_0p7` | Urban velocity, 0.78125% sensors | 31 |
| Figure 16, `fig:urban_fields_temp_0p7` | Urban temperature, 0.78125% sensors | 32 |
| Figure 17, `fig:urban_fields_vel_1p5` | Urban velocity, 1.5625% sensors | 33 |
| Figure 18, `fig:urban_fields_temp_1p5` | Urban temperature, 1.5625% sensors | 34 |
| Figure 19, `fig:urban_rmse_vs_step` | Urban per-step velocity/temperature RMSE | 35 |
| Figure 20, `fig:urban_crps_vs_step` | Urban per-step velocity/temperature CRPS | 35 |
| Figure 21, `fig:urban_ss_vs_step` | Urban per-step velocity/temperature spread--skill | 36 |

### Tables

| Number / label | Caption/topic | Page |
|---|---|---|
| Table 1, `tab:ns_accuracy` | NS RMSE, CRPS, spread--skill, and runtime for four observation scenarios | 8 |
| Table 2, `tab:urban_accuracy_M50` | Urban velocity/temperature RMSE, CRPS, spread--skill, and runtime | 9 |
| Table 3, `tab:analytical_accuracy` | Analytical KL, sliced-(W_2), and cost | 20 |
| Table 4, `tab:ablation` | Placeholder NS ablation matrix (“run in progress”) | 30 |

### Appendix sections

| Number / label | Title | First page |
|---|---|---|
| A, `appendix:score_velocity` | Velocity--score duality and the marginal-preserving SDE family | 13 |
| A.1, `appendix:score_reading` | Reading velocity and score from the trained network | 14 |
| B, `appendix:proof_conditioning` | Conditioning the interpolant path: proofs | 15 |
| B.1 | Posterior path | 15 |
| B.2 | Proof of Theorem 4.1 | 16 |
| C, `appendix:proof_interpolant_likelihood` | Proof of Lemmas 4.2 and 4.3 | 16 |
| D, `appendix:implementation` | Implementation details | 16 |
| E, `appendix:metrics` | Evaluation metrics | 17 |
| F, `appendix:methods` | Method descriptions and comparison to alternatives | 18 |
| G, `appendix:analytical_case` | Analytical test case: setup and additional results | 19 |
| G.1, `appendix:analytical_setup` | Setup | 19 |
| G.2, `appendix:analytical_results` | Additional results | 20 |
| H, `appendix:ns_case` | Navier--Stokes test case: setup and additional results | 21 |
| H.1, `subsec:ns_data` | Forcing and data generation | 21 |
| H.2, `subsec:ns_spectra` | Enstrophy spectra | 21 |
| H.3, `subsec:results_ablations` | Ablations | 21 |
| I, `appendix:urban_case` | Urban airflow test case: setup and additional results | 21 |

## Supporting-material map

- **Core path/schedules:** `src/scisi/models/interpolations.py`; defines linear FM and quadratic SI schedules and velocity--score conversion.
- **Prior models:** `src/scisi/models/flow_matching_model.py`, `src/scisi/models/follmer_stochastic_interpolant.py`.
- **Posterior samplers:** `src/scisi/posterior_models/stochastic_interpolant_posterior.py`, `flow_matching_posterior.py`, and shared rollout logic in `base_posterior.py`.
- **Observation-interpolant likelihood:** `src/scisi/likelihood_models/gaussian_likelihood.py`; implements inflated, inflated-shared, and Jacobian-free modes and the FlowDAS baseline likelihood.
- **Observation operators and metrics:** `src/scisi/likelihood_models/observation_operators.py`, `src/scisi/metrics/*.py`.
- **Experiment entry points:** `paper_experiments/cases/{analytical,navier_stokes,urban}/driver.py` plus `_ns_pipeline.py`/`_urban_pipeline.py`.
- **Declared configurations:** `paper_experiments/configs/case/*.yaml`, `configs/method/*.yaml`, and `configs/scenario/*.yaml`.
- **Executed result aggregates:** `paper_experiments/results/{analytical,navier_stokes,urban}/aggregated/*.csv`; NS aggregate has five trajectories, while the currently aggregated urban table has one trajectory.
- **Table/figure generation:** `paper_experiments/aggregate_*.py`, `make_tables.py`, `make_*_figures.py`, and `common/latex_tables.py`.
- **Urban active run protocol:** the working-tree `run_urban_grid.sh` uses 145 generated physical steps, a foreign initial condition, and assimilation every third step; this protocol is reflected in the main paper, whereas several older README/config defaults still describe 15 or 50 every-step updates.

## Parts not parsed or resolved

- The exact intended ICLR 2027 policy/format version was not independently fetched during Stage 0; the build uses the bundled `iclr2027_conference` style, and the user-specified body limit is 9 pages.
- The active TeX compiles cleanly enough to produce all references, but source comments and old specification/status files contain stale plans and pending-result statements that conflict with the active PDF/result tree; reviewers should use the active TeX, current aggregates, and current run script.
- Table 4 is explicitly a placeholder with every data cell `--`; there is no completed active ablation result to map.
- The currently aggregated urban result has `n_traj=1`; files/scripts describe a five-trajectory target, but the remaining headline cells were not present in the active aggregate at mapping time.
- Several variables are overloaded by context: (\alpha) is both an interpolant schedule and NS drag, (\varepsilon) is both a latent-noise glyph and NS forcing amplitude, (\mathcal K) denotes both metric shells and the forced-mode set, and (\boldsymbol v) denotes both generative velocity and fluid velocity.
- The paper’s “exact likelihood score” (\scorefun_\tau^{\obs}) and practical observation-interpolant score (\interpolantscorefun_\tau) are distinct; the same base glyph `s` is also used for the prior score, so prose context is required to disambiguate.
- The proof assumptions refer to well-posedness/uniqueness rather than deriving sufficient coefficient conditions; no stronger regularity certificate is present in the supporting code or paper.
- No author-provided “do not touch” list was supplied beyond the instruction not to rewrite or edit the manuscript; later verification should therefore remove only out-of-scope findings, not findings by section.
