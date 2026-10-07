# Handover status

Updated 2026-10-07 during initial implementation session.

## Branch and scope
Branch `research/ito-map-forecasting`. User authorized long research, manuscript, implementation, tests and experiments with delegated agents. Work remains local; no submission/publication is authorized or performed.

## Complete
- Full-text research on three requested papers, broader prior-art/novelty review.
- ICML2026 official template and guidelines checked;2027 requirements unverified.
- Compiling manuscript `paper/main.tex`; proposal/methods plus planned experiments, no forecast superiority asserted.
- BaseMapModel, FollmerItoMapModel, FlowMapModel, MapConditionedNetwork.
- Exact shared weighted Brownian bridge queries, local path features.
- MapTrainer (PSD and teacher rollout), offline checkpointing, exact epoch-boundary resume for deterministic datasets.
- Hydra train/test entrypoints and Navier–Stokes configs.
- Reused existing dataset, preprocessing, U-Net, optimizer/clipping and metric functions.
-63 tests passed in latest full suite (CPU), real GPU train/evaluate smoke succeeded.
- Validation solver baseline measurements on trajectories160:164, origins5/45/85, ensemble8; not test data.
- Analytic/Monte Carlo OU path-compression diagnostic, CPU only.

## Running
First Itô PSD pilot: `checkpoints/stochastic_navier_stokes/ito_psd_seed0`, 20 epochs x100 batches x8, teacher32-step validation on4 fixed-noise shuffled batches. Full160trajectory training split and20trajectory validation split. Source command in `experiments/train_psd_pilots.sh`; stdout `results/ito_psd_seed0_train.log` (ignored as generated log). After completion train matching `flow_psd_seed0` sequentially on single RTX3090. Epoch8 validation objective improved from0.343 to0.055; this is teacher-map error, not forecast CRPS.

## Next actions
1. Finish PSD pilots; evaluate forecast quality on validation using `experiments/compare.py`.
2. If needed, improve main methodology/training (e.g. labeled same-path teacher-rollout distillation stage, more iterations or path-feature ablation); do not tune on test.
3. Freeze selected configurations, evaluate all20 held-out test trajectories with matched ensembles/budgets and physical autoregressive rollout.
4. Record complete raw scores, checkpoint hashes, configs/training histories and costs; report uncertainty over trajectories.
5. Update figures/tables/manuscript with actual results, rebuild PDF, audit claims against evidence.

## Important constraints and findings
- Repository default FollmerStochasticInterpolant gamma diffusion is distinct from the strict KL-optimal Föllmer schedule.
- Raw diagonal target excludes derivative of sqrt(t); for supplied teachers actual diagonal training uses frozen teacher drift.
- Training teacher rollout uses exact integrated additive-noise increments plus Euler drift; baseline original solver uses Euler–Maruyama. Explain this numerical distinction.
- Finite path features can shrink spread; exact Gaussian queried increments do not make the neural map exact.
- Original archive physical dt/forcing provenance is incomplete; label lead times in stored snapshot intervals.
- Teachers are existing pretrained models; original training budgets/seed history unavailable.
- New work must show more than first few-step scientific forecasting (close prior art StocBench/RecFM/Swift).
