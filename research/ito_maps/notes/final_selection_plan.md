# Selection and confirmation plan

Recorded 2026-10-07 before the K16 forecasting validation and before any held-out
test forecasts. The development runs used only validation trajectories160–163,
origins5/45/85, ensemble8. They do not estimate held-out test performance.

## Remaining development choice

Compare the K4 and K16 Itô students after equal 500-update teacher-rollout
refinement, starting from the same selected K4 PSD weights. The K16 initialization
splits each original projection weight over its four child increments so that
the initial function is preserved on a common path. Select the representation
with lower **one-step fair CRPS** on the existing development cases. Use the same
selected checkpoint for every inference budget; do not choose separate weights
for individual budgets. The deterministic flow comparator is its completed
500-update refinement. Preserve the K4/K16 result as an ablation either way.

The K16 ablation was motivated by a fixed-history nested Monte Carlo diagnostic:
K4 omits approximately9.7% of the finite-teacher endpoint variance while K16
omits approximately0.66%. This is one validation history, not a general guarantee.

Check 128-step solver validation against64 steps to measure reference-budget
sensitivity. Do not silently call a 64-step solver converged.

## Frozen held-out evaluation

- All20 test trajectories180–199. No student tuning on their forecasts.
- One-step physical forecasts: origins5/45/85, ensemble16, random seed20261007.
- Seed0 student curves:1/2/4/8 map evaluations; teacher curves:1/4/16/64/128
  solver steps. Record actual drift network evaluations and latency.
- Autoregression: origin45,10 physical leads, ensemble16, same20 trajectories.
  Student budgets1/8, teacher64. Maintain separate generated member histories.
- Repeat the complete two-stage student training with seeds1 and2, fixed teacher
  checkpoints, selected feature count and otherwise identical settings. Evaluate
  their1/8-step forecasts with the same test cases. The replication count is fixed
  independently of the resulting scores. Report any incomplete run explicitly.
- Preselected qualitative case: trajectory180,origin45,first ensemble member;
  shared plot scales, no selection by appearance.

## Interpretation

Report measured quality differences and speed ratios, including failures of the
desired quality-retention hypothesis. Paired trajectory bootstrap quantifies
test-case variation for fixed trained models; training-seed variation is separate.
Existing teacher pretraining history and seed variation are unavailable. These
experiments cannot establish generality beyond this benchmark and teacher pair.
No formal equivalence margin has been preregistered. Avoid turning a small point
estimate difference into a statistical claim of equal forecast distributions.
