# Conditional Itô maps for probabilistic forecasting: methodology research

Research date: 2026-10-07. Research agent: methodology. This note records derivations and implementation recommendations, not empirical findings. The manuscript and code must identify which recommendations were actually implemented.

## Final implementation decision: exact local bridge queries

The implemented driver (`src/scisi/sampling/brownian_path.py`) supersedes the fixed-grid recommendation below. It represents M as B_q and lazily samples its Gaussian bridges at arbitrary requested times. Given cached l<t<r, the conditional mean is M_l + V(l,t)/V(l,r) (M_r−M_l), and conditional variance is V(l,t)V(t,r)/V(l,r). Initialization samples M_0=0 and M_1~N(0,a²/3). Subsequent values retain the exact joint Gaussian law at any finite set of query times. The variance is evaluated by the factored difference of cubes to avoid cancellation near t=1.

Each map receives K chronological, raw martingale increments over equal subdivisions of its own interval, concatenated as K*C spatial channels. All PSD calls share one lazy path instance. This avoids feeding values beyond the queried interval as model features. Because features sum to the full increment, the explicit martingale residual is fully represented in the input. Local feature compression still means the nonlinear average drift is approximate; exact Gaussian finite-query laws do not make the learned map exact. The global/grid choices below are research alternatives, not claims about final code.

Validation uses a simulation-based frozen-teacher rollout discrepancy when a teacher exists, even when training uses simulation-free PSD. This is intentional external supervision for checkpoint selection and must be labeled in the paper. `tests/test_brownian_path.py` verifies covariance, independent increments, telescope identities, repeated queries, diagonal/zero-noise behavior, and tiny-interval stability. `tests/test_map_trainer.py` verifies analytic constant and time-dependent ODE/SDE teacher rollout, PSD, frozen teacher gradients, EMA updates, offline checkpoints and reproducible validation. At this checkpoint all fourteen tests pass on CPU.

## Primary sources and scope

1. Chen, Goldstein, Hua, Albergo, Boffi, and Vanden-Eijnden, **Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes**, arXiv:2403.13724v2 (2024), https://arxiv.org/html/2403.13724v2. Core references: Theorems 3.1–3.3; equations (2)–(16); Appendix B.1; Appendix C.2 (Navier–Stokes).
2. Pan, Potaptchik, Yao, Albergo, and Pidstrigach, **Itô maps for any-step SDEs**, arXiv:2606.11156v1 (2026), https://arxiv.org/html/2606.11156v1. Core references: equations (7)–(21); Algorithm 3; Theorem 2/Corollary 1; Appendix 7 Brownian simulation and teacher guidance.
3. Boffi, Albergo, and Vanden-Eijnden, **Flow Map Matching**, arXiv:2406.07507, https://arxiv.org/abs/2406.07507. Deterministic map framework; separately researched by the repository/flow-map agent.

Full HTML and extracted text of sources 1 and 2 are stored in `../sources/2403.13724v2.{html,txt}` and `../sources/2606.11156v1.{html,txt}`. HTML preserves all math; the text extractor can omit some multiline equation containers, so consult HTML for exact equations.

Our proposed contribution is the conditional forecasting adaptation and its evaluation, not inventing Itô maps, the diagonal regression, or the semigroup identity. New analytical observations below are explicitly derived here. No claim of quality superiority is justified before measurement.

## The two clocks and the conditional distribution

Let physical time index be k, lead time Δ, history H_k, most recent state c=x_k, and next observed state Y=x_{k+Δ}. A conditional generative model approximates the transition distribution P(Y|H_k). If the system is Markov and fully observed, H_k can be c; otherwise the history is essential. Static fields and physical parameters are included in H throughout.

Generative time t∈[0,1] is separate from physical lead time Δ. All map calls within one forecast keep H fixed. After producing a physical forecast, shift the history once, then draw a new independent driving path for the next physical transition. Repeated forecasts approximate a learned transition kernel; they are not automatically exact physical dynamics.

## Existing interpolation and its correct diagonal target

For a fixed history h with latest state c, define

    I_t = α(t)c + β(t)Y + γ(t) B_t,
    B_t ~ N(0,t I), independent of (h,Y).

The repository uses α(t)=1−t, β(t)=t or t², γ(t)=a(1−t). Its `wiener_process=True` realizes B_t as √t z. Use the existing interpolation's `forward` and `forward_diff` methods.

The regression target is

    R_t = α'(t)c + β'(t)Y + γ'(t)√t z.
    b(t,x;h) = E[R_t | I_t=x,H=h].

For the quadratic choice, R_t=−c+2tY−a√t z. For the linear choice, R_t=−c+Y−a√t z. These coefficients are bounded. The resulting SDE is

    dX_t = b(t,X_t;h)dt + γ(t)dW_t,   X_0=c.

An Itô-formula calculation shows its marginal density equals the conditional interpolant density, hence X_1 has law P(Y|h), under well-posedness assumptions. This conditional extension follows by fixing h in the original proof.

**Do not differentiate √t in the target.** Although √t z produces the right one-time marginal, treating the same z as a differentiable path gives the probability-flow velocity, not this SDE drift. In particular, `sigma` in `AffineGaussianPathMixin` is γ√t, whereas the sampler's diffusion is `gamma`=γ. These are different quantities.

**Do not copy the Itô-map paper's X1−2X0 target.** That formula assumes its own Gaussian-source linear interpolant and diffusion √(2(1−t)). Our base is a conditional point mass and the source paper's conditional forecast target must be retained.

The interpolation Brownian value √t z used to draw I_t is independent of the future generative Brownian increments used in an off-diagonal map call. Equal interpolant/SDE marginals do not imply that a paired interpolation path is an SDE sample path. Regression directly from paired (I_s,I_t) does not train the intended Markov Itô map.

## Map parameterization

For the additive-noise SDE, set

    M_t = ∫_0^t γ(r)dW_r,
    Φ(s,t,x;h,W) = x + (t−s)G(s,t,x;h,W) + M_t−M_s.

G is the average accumulated drift along the SDE solution. In particular G(t,t,x;h)=b(t,x;h), independent of future noise. The neural map uses a finite feature vector Ψ(W), with dimensions per spatial field, and should additionally receive M_t−M_s explicitly. This is a direct specialization of the source Itô map to a conditional interpolant SDE.

One map call over (0,1) is one neural evaluation per physical forecast. K calls over a partition permit an accuracy/cost tradeoff. Gaussian random-number generation and feature construction also cost time, so report measured latency in addition to NFE. A map network can have a slightly larger input layer than its teacher; equal NFE alone does not mean equal FLOPs.

At s=t, return x exactly. The drift need not be evaluated for a zero-length sampling interval. Ensure Brownian conditioning disappears on the diagonal, either architecturally or by explicitly supplying zero features and enforcing the boundary.

## Diagonal supervision and progressive self-distillation

For each data pair (h,Y), draw a random r and interpolant noise z, then train

    Ldiag = E ||Gθ(r,r,I_r;h,0) − R_r||².

A pretrained repository FSI teacher provides a lower-noise target b_teacher(r,I_r;h) instead of R_r. This distills the learned teacher, whose error remains; it is not an exact data-distribution oracle. Mixing raw regression with teacher supervision is possible, but report weights. Keep the teacher frozen and in evaluation mode.

To learn off-diagonal intervals draw s<u<t, one I_s from its exact conditional marginal, and one Brownian realization. Write λ=(u−s)/(t−s). The semigroup identity implies

    X_u = I_s + (u−s)G(s,u,I_s;h,W) + (M_u−M_s),
    G(s,t,I_s;h,W)
      = λG(s,u,I_s;h,W) + (1−λ)G(u,t,X_u;h,W).

The weighted-average-velocity PSD objective trains the left side against a detached right side. An EMA copy θbar may supply the right side. All three calls must refer to the same path and fixed history. The additive increments must satisfy ΔM_st=ΔM_su+ΔM_ut exactly. Independent draws for these terms are mathematically wrong.

Pseudocode (batch and spatial dimensions suppressed):

```python
# Diagonal: reuse interpolation and pretrained drift.
r, z = sample_time(), randn_like(target)
I_r = interpolation.forward(base, target, r, z)
diag_target = teacher.drift(I_r, r, history, field_cond, pars_cond).detach()
L_diag = mse(map.average_drift(I_r, r, r, zeros, history), diag_target)

# Off diagonal: choose grid-aligned interval triple and coherent path.
s, u, t = sample_ordered_grid_times(include_endpoint_intervals=True)
I_s = interpolation.forward(base, target, s, randn_like(target))
path = sample_weighted_brownian_path(diffusion=interpolation.gamma)
psi = path.features()
with torch.no_grad():
    G_su = target_map.average_drift(I_s, s, u, psi, path.increment(s,u), history)
    X_u = I_s + expand(u-s)*G_su + path.increment(s,u)
    G_ut = target_map.average_drift(X_u, u, t, psi, path.increment(u,t), history)
    target_G = expand((u-s)/(t-s))*G_su + expand((t-u)/(t-s))*G_ut
G_st = map.average_drift(I_s, s, t, psi, path.increment(s,t), history)
L_psd = mse(G_st, target_G)
loss = w_diag*L_diag + ramp(step)*w_psd*L_psd
```

The source's map-displacement PSD and average-velocity PSD have the same zero set but differ by the interval weight (t−s)². They are not numerically identical training objectives. The velocity form avoids automatically downweighting short intervals. Record which is used.

Useful necessary stabilization choices, not guarantees: initialize shared backbone from the teacher; make new time/noise projections initially zero; first verify diagonal equivalence; warm up diagonal regression or use frozen teacher supervision; ramp PSD weight; clip gradients; retain endpoint intervals s=0,t=1 with positive probability; include some short intervals to anchor the diagonal; monitor per-loss terms and ensemble variance. Do not silently describe direct numerical teacher-trajectory regression as simulation-free PSD. That is a different optional distillation baseline and entails teacher solver cost.

At an ideal optimum with exact diagonal, uncompressed driving paths, sufficient support and regularity, PSD identifies the SDE Itô map. A practical compressed network, discretized time grid, stop-gradient optimization, finite data, and finite training do not inherit an unconditional exactness guarantee. Collapsed or inaccurate fixed points must be checked empirically.

## Brownian implementation with exact grid increments

For additive deterministic γ, M is a Gaussian process with independent increments and variance clock

    q(t) = ∫_0^t γ(r)²dr.

It can be generated directly as Brownian motion at q(t), without separately approximating W. For γ(t)=a(1−t),

    q(t) = a²(t−t²+t³/3),
    V(s,t) = q(t)−q(s) = a²[(1−s)³−(1−t)³]/3.

Thus ΔM_st=√V(s,t) ε has the exact law. For a fine grid t_j,

    ΔM_j = √V(t_j,t_{j+1}) ε_j,   ε_j iid N(0,I),
    M_j = cumulative_sum(ΔM_j).

Grid increments and values are exact in distribution; evaluating the drift between these grid points remains a discretization/compression issue. For PSD pick all three times on this grid, avoiding unacknowledged linear interpolation bias in Brownian variance. Start with N=32 or 64; full-grid random-number generation is cheap relative to UNet evaluations. If arbitrary query times are required, sample Brownian bridges rather than linearly interpolating the Brownian path and claiming exact variance.

**Practical global feature choices.** A small vector of normalized coarse block sums of ΔM is simple and interpretable. Disjoint block sums divided by √block-variance are exactly iid N(0,I), and their sums resolve the coarse path. Alternatively, use discrete orthonormal Haar/cosine projections of the standardized ε; orthonormality yields exactly standard-normal feature coefficients. These are finite discrete Gaussian path projections; call them that unless implementing the source's actual KL integral. Prefix-sum values provide exact ΔM for each queried interval. Supply the same global feature vector to all PSD calls.

For true continuous time-changed Brownian KL features on q∈[0,Q], Q=q(1), use

    ξ_k = √(2/Q) ∫_0^Q cos((k−1/2)πv/Q) dB_v.

Approximate stochastic-integral coefficients on the fine variance grid, or construct an exact discrete orthonormal basis. The corresponding KL expansion is a representation of B_q, not of the physical-time W_t; because γ is deterministic this is sufficient for our additive-noise map.

A local dyadic representation is also legitimate. Generate a single sufficiently refined path over [s,t], then extract features of its restrictions [s,u] and [u,t]. Do not draw the three local feature sets independently. With a fixed feature count per interval, the children resolve finer detail than the parent, so PSD may have an irreducible conditional-projection residual. This is compression error, not a defect in the continuous semigroup law.

A truncated sinusoidal path alone generally changes Var(M_t−M_s). Keeping the exact increment as a separate residual and network input is particularly valuable: with b=0, the sampler is exactly Gaussian on grid intervals regardless of the learned feature truncation. It still cannot capture all nonlinear dependence of drift accumulation on unresolved Brownian bridges.

The feature dimension is per field coordinate, not five scalars for an entire 2D field. Flatten feature mode and physical-channel dimensions into spatial conditioning channels, preserving all ordinary field conditioning and history channels.

## Relation to deterministic flow maps

For a zero-noise ODE dX=v(t,X;h)dt, set M=0 and omit Brownian features. The same map parameterization and PSD loss become deterministic flow-map matching. This is the correct shared `BaseMapModel` abstraction.

A deterministic flow from a point mass remains a point mass. Therefore a probabilistic FlowMapModel should reuse the repository FlowMatchingModel's Gaussian base Z~N(0,I), conditional on h, rather than use c as its only source of randomness. For the linear FM path I_t=(1−t)Z+tY, diagonal target is Y−Z. Use the existing deterministic interpolation methods, or the pretrained FM velocity as teacher. The map is deterministic given (Z,h), but its samples are random after drawing Z. Setting γ=0 in a point-source conditional FSI cannot by itself yield a nondegenerate forecast distribution.

Eulerian/MeanFlow identities for ordinary ODE flows must not be transplanted to stochastic Itô maps: the source paper explains the pathwise initial-time obstruction. PSD is shared by both settings and avoids temporal autodifferentiation.

Lagrangian self-distillation is a valid optional alternative:

    G(s,t,x;h,W)+(t−s)∂_t G(s,t,x;h,W) = b(t,Φ(s,t,x;h,W);h).

The derivative holds the underlying Brownian realization fixed. If input features themselves depend on the query endpoint, their path-dependent changes must be treated consistently; naively freezing interval-normalized noise while differentiating t does not implement the stated pathwise derivative. PSD is easier to audit with existing UNets.

## Naming: repository FSI versus strict KL-optimal Föllmer process

The repository class is called `FollmerStochasticInterpolant`, but its default diffusion is γ(t), the original conditional stochastic-interpolant SDE. In the forecasting source, a particular *different* diffusion g_F together with a score-corrected drift is KL-optimal and identified as a Föllmer process. Preserve class/API names requested by the user, but state which sampler is implemented.

In the notation here,

    score(t,x;h) = A_t [β_t b(t,x;h) − c_t(x,c)],
    A_t = [tγ_t(β'_tγ_t−β_tγ'_t)]^−1,
    c_t(x,c)=β'_t x+(β_t α'_t−β'_t α_t)c,
    b^g = b + 0.5(g²−γ²)score.

The source's optimal amplitude is

    g_F² = |2tγ(β'/β·γ−γ')−γ²|.

For α=1−t, γ=a(1−t):

- β=t: g_F²=a²(1−t²), integrated variance q_F(t)=a²(t−t³/3).
- β=t²: g_F²=a²(1−t)(3−t), q_F(t)=a²(3t−2t²+t³/3).

The corresponding exact weighted-noise variance is q_F(t)−q_F(s). A strict Föllmer Itô map must train its diagonal against b^g_F and use M=∫g_F dW. Swapping only the diffusion after learning an off-diagonal map is not correct: the entire average drift/path relation changes. Unlike the original drift model, a learned map is tied to its driving schedule unless the schedule is explicitly conditioned during training.

Endpoint score expressions require the source's limiting/first-step treatment; the quadratic schedule's g_F(0)=√3 a differs from γ(0)=a. Simply evaluating a generic score quotient with an arbitrary epsilon and declaring exact Föllmer behavior is unjustified. Initial implementation should retain the well-understood repository gamma schedule and make the limitation explicit; strict Föllmer is an ablation only after endpoints are correctly handled.

## Novel, elementary error interpretation for the adaptation

Let Φ be the exact conditional SDE endpoint map, Φ_h a numerical teacher using Brownian resolution h, and Φθ the learned map coupled to the same Gaussian path. By the triangle inequality in L²,

    (E||Φθ−Φ||²)^1/2
      ≤ (E||Φθ−Φ_h||²)^1/2 + (E||Φ_h−Φ||²)^1/2.

Each term upper-bounds a corresponding Wasserstein-2 discrepancy through this coupling. Separately, the exact learned-drift teacher may differ from the ideal data transition. Compression, map training, numerical reference, and teacher/model error are distinct. This is an elementary diagnostic decomposition, not a new strong convergence theorem. It does not guarantee a finite-step map beats its teacher.

At a fixed finite feature representation Ψ and state x, squared-error regression for the drift accumulation targets its conditional expectation given (x,Ψ,ΔM). The unexplained conditional variance is an irreducible same-path error. More Brownian modes can reduce this approximation bottleneck but also increase model cost. Assess a small ablation (e.g. 1,4,8 modes) only after the main comparison works.

## Experiment requirements and interpretive safeguards

The main comparison should contain: existing FSI teacher at several Euler–Maruyama NFEs; its Itô-map counterpart at 1,2,4,8 map NFEs; existing FM teacher plus deterministic flow map, at matched backbone scale and training-data split. Compare quality versus measured latency as well as NFE. Include at least one high-step SDE reference to check discretization saturation.

Use held-out entire trajectories/initial conditions, never overlapping windows from a training trajectory. Fit normalization on training data only. Freeze checkpoint selection using validation data. If pretrained baselines have unknown or unequal training budgets, state that comparison transparently rather than claim fair from-scratch compute.

Distributional diagnostics: ensemble CRPS or energy score; ensemble-mean error; empirical spread, spread/skill, and interval coverage; enstrophy/energy spectrum; short physical rollouts and distributional stability. A single target per context is sufficient to estimate proper scoring rules averaged across contexts but not the true context-specific conditional density. Spatial independence assumptions should not be hidden in metric labels.

For acceleration, state both the denominator (e.g. 100-step SI baseline) and actual measured map latency. For sample quality, report uncertainty over test trajectories and preferably independently trained seeds. Bootstrap individual highly correlated pixels or windows is invalid evidence of broad generalization. Check whether the apparent gain is denoising toward the mean accompanied by collapsed spread.

A finite pilot can establish integration correctness, numerical speed, and preliminary quality. It does not justify the desired conclusion automatically. If quality falls, report that and continue methodological diagnosis rather than shaping the paper around an unsupported headline.

## Immediate implementation acceptance checks

1. Diagonal map with zero noise agrees with a copied teacher at random t.
2. Φ(s,s,x)=x exactly.
3. Zero drift map has mean x and empirical covariance V(s,t)I; test several intervals including (0,1).
4. Shared-path increments telescope to numerical precision; deterministic zero-noise map has no stochastic residual.
5. PSD target uses one fixed history, the intermediate *generated* X_u, and shared Brownian realization, never the observed Y in the off-diagonal RHS except through drawing I_s.
6. Gaussian base is resampled for FlowMapModel, while FSI starts at latest observed field; both generate diverse ensembles.
7. Model checkpoint save/load and existing trajectory APIs retain shapes, masks, static fields and parameter conditioning.
8. Map NFE is actually K, excluding optional diagnostics/teacher calls; no teacher solver executes during inference.
