# Literature scan: few-step Itô-map samplers for probabilistic PDE/SPDE forecasting

Date of scan: 2026-11-23. Method: web search plus arXiv abstract/HTML/PDF reads.

Verification legend used throughout:

- **[V]** arXiv abstract page (or full text) fetched in this session; id, authors and date confirmed.
- **[S]** id and title seen in search results in this session, abstract not opened.
- **[M]** cited from memory, id not re-checked in this session. Check before citing.

Search summaries were produced by a small model reading result snippets, so descriptions marked [S] are leads, not verified summaries.

---

## 0. Pan et al., "Itô maps for any-step SDEs" (arXiv:2606.11156) [V, full HTML v2 read]

Zhengkai Pan, Peter Potaptchik, Wenxi Yao, Michael S. Albergo, Jakiw Pidstrigach. v1 9 Jun 2026, v2 2 Nov 2026. Listed as a NeurIPS 2026 poster.

**Parametrisation.** For the generative SDE `dX = [b_t + (σ_t²/2)∇log p_t] dt + σ_t dW`, the map is
`X̂_{s,t}(x,W) = x + (t−s) Ĝ_{s,t}(x,W) + (M_t − M_s)`, `M_t = ∫_0^t σ_u dW_u`.
The network outputs only the average drift `Ĝ`; the martingale part is computed from `W` outside the network. On the diagonal `Ĝ_{t,t}` is the instantaneous drift and is independent of `W`. Noise schedule `σ_t = sqrt(2(1−t))`.

**Path features.** Karhunen–Loève coefficients of the Brownian path, K=5 (K=10 on ImageNet-256), i.i.d. Gaussians. They report that KL coefficients on `[s,1]` work better than on `[s,t]`. `M_t − M_s` is deliberately not a network input: feeding it gave low consistency loss but wrong samples (shortcut learning). On ImageNet the coefficients enter through a path-attention branch before each transformer block, with a zero-path response subtracted and a scale that vanishes on the diagonal.

**Losses.** Diagonal: stochastic-interpolant regression `E‖Ĝ_{t,t}(I_t) − (X_1 − 2X_0)‖²`. Off-diagonal: Lagrangian self-distillation (LSD), `‖Ĝ_{s,t} + (t−s)∂_t Ĝ_{s,t} − sg(Ĝ_{t,t}(X_t))‖²`, or progressive self-distillation (PSD), a two-leg semigroup target with stop-gradient. Proposition 2: the naive Eulerian loss admits zero-loss solutions that ignore `W`; a similar pathological solution is noted for PSD.

**Experiments.** 1D/2D Gaussian mixtures (same-path tracking, posterior steering), MNIST (same-path pixel MSE 0.0191 / 0.0143 / 0.0091 at 1 / 2 / 4 steps), ImageNet-256 with SiT-XL (FID 6.43 / 2.10 / 1.41 at 2 / 4 / 16 steps), reward steering against Meta Flow Maps and DPS. No forecasting, PDE or dynamics experiments.

**Stated limitations.** Only one: the best ImageNet steering results need a pretrained Decoupled MeanFlow diagonal teacher, and the self-distilled model lags strong deterministic flow maps. No K ablation, no empirical `[s,1]` vs `[s,t]` ablation, no strong-error metric at scale, no discussion of scoring rules.

**Open discrepancy.** Blasingame et al. (2610.29605) state that Pan et al. "empirically found it necessary to add the local increment `W_t − W_s`" to the KL coefficients. The v2 HTML I read says the increment is withheld from the network. This may be a v1/v2 difference or a difference between network input and the additive martingale term. Not resolved.

### Follow-ups and concurrent work

| Paper | Id | Status | What it does |
|---|---|---|---|
| Strong Stochastic Flow Maps (SSFM). McCallum, Blasingame, Herschell, Rindtorff, Tong, Foster | 2606.01086, 31 May 2026 | [V, HTML read] | Concurrent with Pan et al. Learns the strong solution map of additive-noise SDEs: `Ψ = x + f_θ(x, I)(t−s) + g_θ(I)(W_t − W_s)`. Path features are shifted-Legendre coefficients `I^(n)_{s,t}` (N=3 images, N=4 toy), independent Gaussians with variance `(t−s)/(2n+1)` and closed-form Chen relations. Loss is diagonal matching plus semigroup distillation. CIFAR-10, CelebA-64, alanine dipeptide, chignolin. Limitation: additive noise only. |
| Improved Training of Strong Stochastic Flow Maps (iSSFM). Blasingame, Franca, Miñán, Koke, Rindtorff, Tong | 2610.29605, 21 Oct 2026 | [V, PDF pp. 1–10 read] | The most relevant follow-up. (a) Compares path representations: shifted Legendre beats global KL, KL plus increment, and dyadic increments in strong RMSE; KL lacks Chen relations and carries information outside `[s,t]`. (b) Wong–Zakai objective: replace the SDE by a random ODE driven by the smooth polynomial path, which permits Eulerian / improved-MeanFlow derivative losses. Doss–Sussmann variant (`Z = X − M`) fails to use the coefficients at 1 NFE; Lagrangian variants unstable. (c) Exponential-integrator lift `d[α_t X_t] = 2α_t u_t dt + α_t ν_t dW`, closed-form martingale, drift-matching diagonal. (d) Fine-tunes a pretrained DMF-XL/2+ into a strong map; about 3x better FD-DINOv2 than SSFM at 4 NFE on ImageNet-256. Feynman–Kac correctors as an application. |
| Kolmogorov-Chapman Maps. Hassan, Skreta, Beaini, Neklyudov | 2611.12880, 9 Nov 2026 | [V, abstract] | Weak (kernel-level) map for a physical SDE with known drift and diffusion, trained trajectory-free from marginal samples using the Chapman–Kolmogorov equation. Peptide molecular dynamics. |
| Stochastic Self-Distillation for Flow-Based VLA Models (VLA-SSD). Guo, Russo, Bose, Ziakas | 2611.20460, 16 Nov 2026 | [V, abstract] | Distils a flow-matching robot action head into a one-step stochastic policy conditioned on a Brownian path, for likelihood-ratio RL. A search snippet says it uses Pan et al.'s Itô maps; the abstract does not name them. |
| Sig-DEG. Ge, Jiang, Yang, ..., Ni | 2508.16939, v2 13 Oct 2026 | [V, abstract] | Distils diffusion SDEs on a coarse grid using partial Brownian signatures that can be sampled exactly, with a recurrent generator; supervised on fine teacher trajectories. Includes a Schrödinger bridge example. |
| Flow-Map GRPO | 2607.00535 | [S] | RL post-training of deterministic flow-map generators; cites SSFM. |
| Stochastic Flow Map for Count Data | 2609.23290 | [S] | Title only. |
| Geometry and Stochastic Analysis of Flow and Diffusion Models: A Survey | 2611.12107 | [S] | Title only. |

**None of the follow-ups found targets forecasting, PDEs, SPDEs, a point-mass (Föllmer) source, scoring-rule training, or autoregressive rollouts.**

---

## 1. Stochastic / SDE flow maps and consistency models for SDEs and bridges

Strong (pathwise) maps: Pan et al., SSFM, iSSFM, Sig-DEG (table above).

Weak (kernel-level) maps:

- **Meta Flow Maps.** Potaptchik, Saravanan, Mammadov, Prat, Albergo. arXiv:2601.14430 [S]. One-step posterior sampling from any noisy state; reward steering on ImageNet. Main baseline in Pan et al.
- **Diamond Maps.** Holderrieth et al. arXiv:2602.05993, ICML 2026 [S]. Stochastic flow maps for reward alignment; "posterior" variant distilled from GLASS flows, "weighted" variant renoises an off-the-shelf flow map.
- **GLASS Flows.** Holderrieth et al., 2025/ICLR 2026. Id not verified.
- **Stochastic Transition-Map Distillation.** Rapakoulias, Garud, Zhu, Tsiotras. arXiv:2605.07661, 8 May 2026 [V]. Teacher-free distillation of the SDE transition map with a conditional MeanFlow; Wasserstein bounds; MNIST/CIFAR/CelebA.
- **Stochastic Few-step Models.** Passaro, Blasingame, Bronstein, Tong, 2026. Maps the conditional SDE to an ODE with matching marginals. **Id not verified**: a search result suggested 2604.05181, which I fetched and is an unrelated protein-design paper.
- **Neural Stochastic Flows.** Kiyohara, Johns, Li. arXiv:2510.25769, NeurIPS 2025 [S]. Conditional normalising flows for SDE transition laws.
- **Kolmogorov-Chapman Maps.** 2611.12880 [V], above.

Bridges:

- **DBIM.** Zheng, He, Chen, Bao, Zhu. arXiv:2405.15885, ICLR 2025 [S]. Training-free non-Markovian sampler for DDBMs (the DDIM analogue).
- **CDBM.** He, Zheng, Chen, Bao, Zhu. arXiv:2410.22637, NeurIPS 2024 [S]. Consistency function of the bridge probability-flow ODE; distillation and direct training. Deterministic given the first noisy state.

**Not done (as far as found).** Strong or weak stochastic flow maps for a *conditional forecasting SDE started from a point mass* (Föllmer process), where all randomness comes from `W` and there is no random `X_0`. All existing strong maps are for noise-to-data diffusion SDEs with a Gaussian source. No consistency / flow-map distillation of the Chen et al. (2403.13724) forecaster was found. No Itô-map treatment of state-dependent (multiplicative) diffusion; SSFM flags this as needing rough-path machinery.

---

## 2. Parametrisation of flow / consistency maps

- **MeanFlow.** Geng, Deng, et al. arXiv:2505.13447, NeurIPS 2025 [S]. Average-velocity parametrisation with a JVP identity.
- **Improved MeanFlow (iMF).** Geng, Lu, et al., 2025. Id from memory 2512.02012 [M]. Replaces the conditional velocity in the target by a learned marginal velocity. Used by iSSFM as its best loss.
- **AlphaFlow.** ICLR 2026 [S, proceedings link]. Id from memory 2510.20771 [M]. Unifies trajectory flow matching, shortcut and MeanFlow with a curriculum.
- **Align Your Flow.** Sabour, Fidler, Kreis. arXiv:2506.14603, NeurIPS 2025 [S]. Continuous-time Eulerian / Lagrangian flow-map distillation; reduces to MeanFlow under the Euler parametrisation.
- **Shortcut models.** Frans et al. Id from memory 2410.12557 [M].
- **Pixel MeanFlow.** Lu et al. arXiv:2601.22158, ICML 2026 [S]. Network outputs x-prediction, loss in average-velocity space; reports u-prediction degrades as dimension grows.
- **Riemannian MeanFlow.** arXiv:2602.07744 [S]. Snippet reports x1-prediction stays stable where velocity prediction fails in high dimension.
- **Decoupled MeanFlow.** Lee et al., 2026. Id not verified. Converts a pretrained flow model to a flow map through architecture; teacher for both Pan et al. and iSSFM.
- **Terminal velocity matching.** Zhou et al. arXiv:2511.19797 (id as quoted by Pan et al.).
- **iSSFM.** 2610.29605 [V]. Exponential-integrator lift for strong stochastic maps (see section 0). This is the only variation-of-constants form for an SDE map found.
- **x-Prediction Is All You Need.** Peng, Gao. arXiv:2607.06114 [V]. Training-free early stopping of the ODE using the endpoint decoder. Marginal relevance.

**Not done.** A data-prediction (x1 / denoiser-average) parametrisation of a *strong stochastic* map: Pan et al. and SSFM/iSSFM all predict an average drift. No analysis of parametrisations for a point-mass-source interpolant, where the SDE drift contains a `1/(1−t)`-type term toward the endpoint and the noise schedule vanishes at both ends. The iSSFM lift addresses the `α̇/α` stiffness for Gaussian-source diffusion only. One search snippet notes that a flow-map endpoint and a posterior-mean denoiser are different functions, which is a caveat for any denoiser-average form.

---

## 3. Training few-step generators with proper scoring rules

- **Distributional Diffusion Models with Scoring Rules.** De Bortoli et al. arXiv:2502.02483 [S]. Learns the posterior `p(x_0 | x_t)` with generalised energy / kernel scores, enabling few-step sampling.
- **Proper Scoring Rule-based Diffusion for Probabilistic Weather Forecasting.** Park, Nam, Lee, Cho, Park, Lee. arXiv:2609.38632, 29 Sep 2026 [V]. Extends CRPS-trained one-pass forecasters with auxiliary denoising tasks across noise levels (distributional-diffusion style); CRPS training is the special case.
- **AIFS-CRPS.** Lang et al. arXiv:2412.15832 [S]; npj AI 2026. Almost-fair CRPS, noise-injected one-pass ensemble members.
- **Multi-scale loss for proper score optimisation.** Lang, Leutbecher, Maciel. arXiv:2506.10868 [V]. Multi-scale afCRPS to constrain small-scale variability.
- **FGN.** Alet et al. "Skillful joint probabilistic weather forecasting from marginals". arXiv:2506.10772 [S]. Parameter-space noise, per-marginal CRPS.
- **Swift.** Stock, Arcomano, Kotamarthi. arXiv:2509.25631, NeurIPS 2025 [S]. Consistency model, then autoregressive CRPS fine-tuning.
- **Tyche.** Xu et al. arXiv:2605.06916 [V]. One-step average-velocity flow, then rollout fine-tuning with curriculum CRPS.
- **CRPS-LAM.** arXiv:2510.09484 [S]. Regional CRPS-trained ensemble.
- **TORNADO.** Mifsut-Benet et al. arXiv:2611.10251, NeurIPS 2026 [V]. Latent SDE with state-dependent diffusion between VAE posteriors, plus an energy-distance regulariser.
- **Kastor.** arXiv:2608.06107 [S]. CRPS fine-tuning of a deterministic PDE foundation model into a generative one.
- **Inductive Moment Matching.** Zhou, Ermon, Song. arXiv:2503.07565 [S]. MMD-based distribution-level self-consistency along an interpolant.
- **Drifting models.** Deng, Li, Li, Du, He. arXiv:2602.04770, NeurIPS 2026 [S]. One-step generator trained by a drifting field on the pushforward. Follow-up 2602.20463 [S] reinterprets it as a long-short flow-map factorisation.
- **Engression.** Shen, Meinshausen. Id from memory 2307.00835 [M].
- **Signature kernel scores for neural SDE/SPDE.** Issa et al. arXiv:2305.16274, NeurIPS 2023 [S]. **FDM** (finite-dimensional matching) arXiv:2410.03973, ICLR 2025 [S]. Path-space scoring rules for neural SDEs.
- Others seen: ESP energy-score policy 2610.07696 [S]; one-step world model candidates 2611.11887 [S]; EnScale 2509.26258 [S].

**Not done.** No paper found that trains a *Brownian-path-conditioned* map with a scoring rule. The scoring-rule line treats noise as an unstructured latent with no meaning shared across step counts (weak consistency only). The strong-map line uses pathwise regression / self-distillation and never a distributional loss. Specific open combinations: (i) energy-score fine-tuning of an Itô map while keeping same-path (strong) agreement with the multi-step SDE; (ii) using the strong coupling as a variance-reduction or regulariser for a scoring-rule loss; (iii) IMM-style distributional semigroup consistency for a conditional forecasting kernel. Swift and Tyche show the two-stage recipe (self-consistency, then rollout CRPS) works at weather scale, but with a deterministic-ODE one-step generator.

---

## 4. Brownian path representations for learned SDE solvers

- **SSFM** 2606.01086 [V]: shifted Legendre coefficients, Chen relations, pathwise convergence proof, strong error decreasing in N with diminishing returns.
- **iSSFM** 2610.29605 [V]: direct comparison of shifted Legendre, global KL, KL plus increment, and dyadic increments. Legendre wins on strong RMSE at 1/2/4 NFE and on ImageNet (semigroup loss, 4 NFE: Legendre FID 14.68 vs KL+increment 24.92). Argues KL has no closed-form Chen relations and its coefficients carry information from outside `[s,t]`.
- **Pan et al.** 2606.11156 [V]: KL on `[s,1]`, K=5–10, no ablation.
- **Sig-DEG** 2508.16939 [V]: exactly-sampleable partial signatures with a recurrent generator.
- **Foster, Lyons, Oberhauser**, polynomial expansion of Brownian motion (2020); **Foster et al.** on space-time Lévy area solvers (2024). Cited by SSFM/iSSFM; ids not checked.
- **Foster**, "Approximating the signature of Brownian motion for high order SDE simulation". arXiv:2409.10118 [S]. Survey-style chapter.
- **LévyGAN.** Jelinčič, Tao, Turner, Cass, Foster, Ni. arXiv:2308.02452, SIMODS 2025 [S]. Generates Lévy area conditional on increments.
- **Virtual Brownian tree** (Li et al. 2020; Kidger et al. 2021, arXiv:2105.13493 [S]): reproducible Brownian queries; not used as network conditioning in anything found.

**Not done.** Everything is for finite-dimensional or image-shaped noise with independent per-pixel coefficients. No work on (i) path features for *spatially correlated / Q-Wiener* noise in SPDEs; (ii) spatial compression of the path features, since per-pixel N coefficients multiply input channels; (iii) path features under autoregressive rollout, where each physical step consumes a fresh interpolant path; (iv) Lévy-area features, which only matter for non-additive noise that no strong map yet handles. The KL-vs-Legendre question itself is now answered for images by iSSFM, so it is not a novel contribution on its own; a replication in the point-mass forecasting setting would be.

---

## 5. Long autoregressive rollout stability for probabilistic emulators

- **PDE-Refiner.** Lippe et al. Id from memory 2308.05732 [M].
- **Thermalizer.** Pedersen, Zanna, Bruna. arXiv:2503.18731, ICML 2025 [S]. Diffusion-based correction pulling rollouts back to the invariant measure.
- **Invariant-measure regularisation.** Jiang et al. 2023 (id from memory 2306.01187 [M]); DySLIM, Schiff et al. (id from memory 2402.04467 [M]).
- **Energy-consistent stochastic interpolants.** Mücke, Sanderse. arXiv:2504.05852 [V]. Learnable interpolant coefficients embedding energy consistency; longer stable rollouts than standard interpolants on Kolmogorov flow.
- **Forced-dissipative stochastic interpolants.** Massy, Shamekh, Falasca, Blanke. arXiv:2611.09317 [V]. Structural constraints giving a Foster–Lyapunov drift condition and geometric ergodicity; two-scale Lorenz-96.
- **Memory-conditioned flow matching.** Armegioiu. arXiv:2602.06689 [S]. Per-step conditional-law error governs rollout stability; Wasserstein stability of the kernel.
- **Bias-constrained diffusion schedules with unrolled training.** arXiv:2604.08357 [S].
- **Seq-Flow.** Huang, Govil, Dai, Li. arXiv:2610.10440 [V]. Flow from previous forecast distribution to the next; self-rollout training with an EMA model; stable beyond 400 updates when trained on 4.
- **Aligning latent rollout.** arXiv:2610.21370 [S]. Predictions reaching the same target from different rollout origins must agree (a Chapman–Kolmogorov-type regulariser).
- **Continuous ensemble forecasting.** Andrae et al. arXiv:2410.05431 [S]. Lead-time-conditioned diffusion with shared noise across lead times.
- **Elucidated Rolling Diffusion.** arXiv:2506.20024, NeurIPS 2025 [S].
- **Swift** 2509.25631 [S], **Tyche** 2605.06916 [V]: multi-step CRPS fine-tuning of one-step generators; Swift reports 75-day stability.
- **StocBench.** Pfister, Holzschuh, Thuerey. arXiv:2608.22309 [V, HTML read]. Stochastically forced 2D Kolmogorov flow; 50-step rollouts from the invariant measure, enstrophy-spectrum error. Distilled one-step models (consistency distillation, adversarial distillation) preserve the invariant measure well on the stochastic task but are worst on the deterministic control task. Flags non-monotone convergence of distillation.
- **MeLISA.** Yang, Xue. arXiv:2605.05540 [V]. Pixel MeanFlow block-wise forecaster with a time-increment consistency loss across lags.
- From memory, not re-checked: GenCast 2312.15796, Diffusion Forcing 2407.01392, ArchesWeatherGen 2412.12971, pushforward trick (Brandstetter et al. 2202.03376).

**Not done.** No rollout study of any Brownian-path-conditioned map. No multi-step scoring-rule fine-tuning of an SDE-based (as opposed to ODE-based) few-step forecaster. No use of the two-time structure of the map to enforce Chapman–Kolmogorov consistency in *physical* time alongside semigroup consistency in interpolant time. StocBench does not include flow maps, MeanFlow, Itô maps or scoring-rule methods, so those are unbenchmarked on stochastic fluid dynamics.

---

## 6. Physics / constraint consistency, especially few-step

Multi-step, sampling-time:

- **PCFM.** Utkarsh et al. arXiv:2506.04171, NeurIPS 2025 [S]. Zero-shot projection onto nonlinear constraint manifolds during flow sampling.
- **ECI sampling.** Cheng et al. Id from memory 2412.01786 [M]. Gradient-free extrapolate–correct–interpolate.
- **SNAP-FM** 2607.00095 [S]; **Energy-guided transport** 2611.06711 [S]; **Constraint-aware flow matching** 2605.12754 [S].
- From memory: PIDM 2403.14404, DiffusionPDE 2406.17763, CoCoGen 2312.10527.

Few-step / one-step:

- **Stabilizing physics-informed consistency models.** Chang, Dai, Lin, Lai, Lai. arXiv:2602.09303, KDD 2026 [V]. Physics residual pushes consistency models to trivial solutions; two-stage training with a frozen decoder and a two-step residual objective.
- **Physics-informed one-step generation on long-short flow maps.** Li, Chen, He, Zhu. arXiv:2611.01651 [V]. Long map plus a closed-form short map near the endpoint with physics potentials, giving direct targets and removing the Jensen gap; compares PI-consistency, PI-MeanFlow, PI-shortcut on Darcy flow.
- **PMosFM.** arXiv:2609.40287 [S]. One-step physics-constrained generation; title only.
- **Physics is the Best Teacher.** Chiang, Yao, Lin, Anandkumar. arXiv:2610.04108 [V]. Deterministic time-invariant evolution operator trained with differential and compositional consistency against known physics.
- **Spatio-temporal MeanFlow PDE solver.** arXiv:2605.08915 [S]. MeanFlow identity with the PDE right-hand side in place of the velocity.

Structure in the noise or interpolant:

- **Energy-consistent stochastic interpolants** 2504.05852 [V]; **divergence-free diffusion**, Genuist et al. arXiv:2601.19368 [S, not opened]; **Colored Noise Diffusion Sampling**, Davidson, Issachar, Benaim, arXiv:2605.30332 [S], image generation only.

**Not done.** All few-step physics-informed work is for steady or deterministic problems (Darcy, deterministic operators) with hard residuals. Nothing found on (i) constraints for a *stochastic* few-step forecaster, where the deterministic PDE residual is not zero per sample; (ii) weak-form, moment or generator-based (Dynkin, Stein, energy-budget) losses for stochastically forced dynamics, for which my searches returned no hits at all; (iii) constraint-preserving path features, for example divergence-free or mean-zero Brownian coefficients so that the additive martingale term `M_t − M_s` itself respects the constraint. The Jensen-gap problem identified in 2611.01651 and 2602.09303 applies directly to an average-drift map and would need to be handled.

---

## 7. Noise-space / path-space inference for one-step generators

- **D-Flow.** Ben-Hamu et al. arXiv:2402.14017 [S]. Optimise source noise by differentiating through the flow.
- **Posterior sampling by source-space MCMC via few-step transport maps.** Luu, Hartmann, Wang. arXiv:2610.01034, 1 Oct 2026 [V]. Parallel-tempered pCN (plus split HMC) in the Gaussian source space of an improved-MeanFlow map; Wasserstein bounds in terms of training loss; synthetic and CLIP-guided ImageNet.
- **Consistency Posterior Sampling.** Purohit et al., CVPR 2025 [S]. Langevin dynamics in the noise space of a consistency model.
- **Consistency models as plug-and-play priors** 2509.22736 [S]; **MACS** 2510.02208 [S]; **SNaP** 2609.34071 [S]; **FLINT** 2611.05945 [S] (surveys source-space methods).
- **Meta Flow Maps** 2601.14430, **Diamond Maps** 2602.05993, **Pan et al.** 2606.11156: one-step posterior samples used for value-function / reward-gradient estimators (Pan et al. use a Malliavin / Bismut–Elworthy–Li estimator).
- **Unified Bayesian data assimilation with observation interpolants.** Mücke, Sanderse. arXiv:2610.03396 [V]. Guidance-type posterior sampling with pretrained multi-step interpolant / flow / diffusion models; stochastic 2D Navier–Stokes.
- Latent 4D-Var with a learned decoder: arXiv:2502.02884 [S].

**Not done.** No inference over the *Brownian path coefficients* of a strong map. For an Itô map with a point-mass source the path features are the entire latent, they are i.i.d. Gaussian, and the map is cheap and differentiable, which is exactly the setting pCN and D-Flow need. No data assimilation or filtering paper using a flow map, consistency model or Itô map as the forecast kernel was found; 2610.01034 is static image priors only.

---

## 8. Few-step generative forecasting in weather and fluids, 2025–2026

| Paper | Id | Status | Few-step mechanism | Calibration / rollout handling |
|---|---|---|---|---|
| Swift | 2509.25631 | [S] | Consistency model, 1 step | Autoregressive CRPS fine-tuning; 75-day stability |
| Tyche | 2605.06916 | [V] | Average-velocity flow, 1 step, JVP-regularised | Rollout fine-tuning with curriculum CRPS |
| RecFM | 2605.26535 | [V] | Self-consistency across discretisation scales, 1–4 steps | Not stated in abstract; headline metric is MSE |
| PixelFlowCast (Zhu, Shi, Feng, Niu) | 2605.10046 | [S] | Pixel MeanFlow, two-stage, precipitation nowcasting on SEVIR | Not checked |
| MeLISA | 2605.05540 | [V] | Pixel MeanFlow block kernel | Time-increment consistency loss; spectra and TKE statistics |
| StocBench | 2608.22309 | [V] | Benchmarks consistency and adversarial distillation vs multi-step | Energy distance, mean/std error, enstrophy spectrum over 50 steps |
| TORNADO | 2611.10251 | [V] | Latent SDE, few-step | Energy-distance regulariser; state-dependent diffusion |
| Seq-Flow | 2610.10440 | [V] | Flow from previous forecast | Self-rollout training; CRPS evaluation |
| Scale-recursive rectified flows | 2610.02611 | [S] | Rectified flow, precipitation | CRPS reported |
| Variational Streaming Flow | 2610.00976 | [V] | Velocity field in physical time plus latent variable | Rollouts beyond 1000 steps |
| CoReFlow | 2611.05715 | [S] | Consistency-relaxed flow maps, traffic | CRPS reported |
| Proper scoring rule-based diffusion | 2609.38632 | [V] | One-pass CRPS predictor with denoising tasks | Better calibration at long horizons |

**Pattern.** The working recipe is one-step ODE-based generator (consistency or MeanFlow) followed by multi-step CRPS fine-tuning. Calibration comes from the fine-tuning stage, not from distillation.

**Not done.** No weather or fluid forecaster built on a stochastic flow map or Itô map. No few-step version of the Föllmer / stochastic-interpolant forecaster of Chen et al. Nobody reports strong (same-noise) agreement between the few-step and multi-step forecaster, which is the property that would let ensemble members be refined by adding steps without changing the member.

---

## Cross-cutting gaps, ranked by how clearly open they appear

1. Itô / strong stochastic flow maps for conditional forecasting from a point mass. No paper found. Every existing strong map is for Gaussian-source image or molecular diffusion.
2. Scoring-rule (energy score / CRPS) training or fine-tuning of a path-conditioned map, and the interplay between strong consistency and distributional calibration. No paper found.
3. Rollout behaviour of path-conditioned maps, and joint semigroup (interpolant time) plus Chapman–Kolmogorov (physical time) consistency. No paper found.
4. Inference over Brownian path coefficients (pCN, noise optimisation) for assimilation with an Itô-map forecast kernel. Building blocks exist separately (2610.01034, 2610.03396).
5. Physics constraints for stochastic few-step maps, including constraint-preserving path features and statistical or weak-form losses. No paper found; deterministic analogues exist (2602.09303, 2611.01651).
6. Data-prediction parametrisation for a strong map and parametrisation analysis for the point-mass interpolant. Partly anticipated by iSSFM's exponential lift.
7. Path-feature choice. Largely settled for images by iSSFM in favour of shifted Legendre with Chen relations; open only for spatially correlated noise and for the forecasting setting.

## Caveats

- arXiv ids marked [M] were not re-checked; ids marked [S] were seen in search results but the abstracts were not opened.
- "Not done" verdicts mean "not found in roughly 45 searches and fetches". The field moved fast between June and November 2026, and the strong-map literature in particular had three papers in five months.
- The Pan et al. v1-vs-v2 discrepancy on feeding the increment to the network is unresolved.
- Passaro et al. "Stochastic Few-step Models", GLASS Flows and Decoupled MeanFlow exist per multiple citing papers, but I could not confirm their arXiv ids.
