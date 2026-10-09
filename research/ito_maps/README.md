# Itô maps for probabilistic Navier–Stokes forecasting

Few-step conditional forecasting with Brownian-path-conditioned Itô maps, distilled from the repository's forecasting stochastic interpolant ([Chen et al.](https://arxiv.org/abs/2403.13724)). The map construction follows [Pan et al.](https://arxiv.org/abs/2606.11156), and a deterministic [flow map](https://arxiv.org/abs/2406.07507) serves as comparator. **Scope: the stochastically forced Navier–Stokes (NS) case only.** Current status and next steps are in [STATUS.md](STATUS.md).

## Layout

| Path | Contents |
|---|---|
| [STATUS.md](STATUS.md) | What is done, what is pending, next steps |
| [notes/key_findings.md](notes/key_findings.md) | **Single consolidated note.** Method and math decisions, implementation pitfalls, review outcomes, NS provenance gaps, initial-stage lessons, TODOs for the new experiments |
| [paper/](paper/) | ICML-template manuscript. Method complete; all NS results are red `\pending{}` placeholders. See [paper/README.md](paper/README.md) and [paper/notes/results_plan.md](paper/notes/results_plan.md) |
| [experiments/](experiments/) | NS training, evaluation, long-horizon and diagnostic scripts. See [experiments/README.md](experiments/README.md); several defaults still name initial-stage runs |
| [sources/](sources/) | Primary paper downloads and extracted text |
| `results/` | *(absent until the new experiments run)* Default output location of the experiment scripts |
| `archive/` | Initial-stage material: old notes and reviews, results, paper build scripts and figures, video case, old README/STATUS. **Git-ignored** by the root `.gitignore` (`archive/`), so it exists on disk only; tracked originals remain in git history |

## Code

- Models: `src/scisi/models/{base_map_model,follmer_ito_map_model,flow_map_model}.py`. Brownian driver: `src/scisi/sampling/brownian_path.py`. Two-time/path conditioning: `src/scisi/architectures/map_conditioning.py`.
- Training: `src/scisi/training/map_trainer.py`, which provides progressive self-distillation with an optional teacher diagonal, and `teacher_rollout`. The two modes are described in [paper/appendix/training_modes.tex](paper/appendix/training_modes.tex).
- Entry points: `scisi.bin.main_map_model_train`, `scisi.bin.main_map_model_test`. Configs: `config/follmer_ito_map_stochastic_navier_stokes.yaml`, `config/flow_map_stochastic_navier_stokes.yaml`, `config/map_stochastic_navier_stokes_common.yaml`.

## Environment and data

- Project `.venv` (Python 3.13, PyTorch 2.9 + CUDA 12.8). One RTX 3090 (24 GiB); run one GPU job at a time.
- Data `data/stochastic_navier_stokes/data.npz`: key `state`, shape `(200, 100, 128, 128)`, float32. Trajectory split: train 0–159, validation 160–179, test 180–199. Five history fields.
- Teachers: `checkpoints/stochastic_navier_stokes/{stochastic_interpolant,stochastic_interpolant_small,flow_matching}`. Their pretraining records are unavailable.
- Build the paper with `make -C research/ito_maps/paper`. Do not run bare `latexmk`; see paper/README.md.
