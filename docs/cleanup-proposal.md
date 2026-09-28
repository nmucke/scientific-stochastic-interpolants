# Repository cleanup proposal

Audit date: 2026-09-28. Prepared from a read-only review by the primary agent and
three GPT-6 Sol agents covering experiments, submission artifacts, and the library.
This document is the only change made during the review. No experiments, tests,
cleanup commands, moves, or deletions were run.

## Recommended direction

Keep this repository as the maintained research library plus the full paper
experiment harness. Preserve the submitted code as a frozen release artifact,
and move bulky research artifacts into separately managed storage after recording
their provenance. Start by keeping `src/`, `config/`, and `paper_experiments/` at
their existing paths; widespread renaming would break paths without much benefit.

The directory described as `submission_code/` is actually
`paper_submission_code/` in this checkout. It is a deliberately reduced,
independently corrected package, not a duplicate that can replace the research tree.
Do not maintain both copies as competing development branches. Review and port
useful submission fixes individually into the maintained tree.

## What is here

Sizes below are approximate local disk usage (`du -h`), not Git repository sizes.
The checkout occupies about **349 GiB**. Git tracks 316 paths; tracked files still
present occupy only about **7.5 MB**, and `.git/` occupies about 613 MiB.

| Area | Local size | Recommended treatment |
| --- | ---: | --- |
| `src/`, `tests/`, `config/`, `docs/` | Small | Keep as the maintained research project; repair documentation and packaging. |
| `paper_experiments/` | 108 GiB | Keep active code/configuration; inventory and separate published results from historical runs. |
| `data/` | 88 GiB | Preserve original data and split/preprocessing metadata; relocate only after path configuration is portable. |
| `cache/` | 84 GiB | Main space-recovery candidate, once regeneration from preserved inputs has been checked. |
| `external_libs/` | 41 GiB | Keep needed source; almost all size is `jax_cfd_lib/paper_results/`, not library code. Archive those outputs separately. |
| `checkpoints/` | 19 GiB | Preserve paper weights with their `config.yaml`; inventory old training checkpoints before archiving. |
| `.venv/` | 8.9 GiB | Rebuildable only after a working environment specification has been preserved and verified. |
| `archive/` | 2.5 GiB | Consolidate into a documented historical archive; mixed tracked and ignored content needs explicit handling. |
| `manuscript/`, `manuscript_final/`, root paper PDF | About 373 MiB combined | Preserve final sources, bibliography, figures, and submitted PDF together; identify which exact source produced the uploaded paper. |
| `paper_submission_code/`, ZIP, checksum | About 135 MiB combined | Freeze and store together with release metadata. |
| `paper_submission_notes/` | 166 MiB | Retain reports and validation provenance; duplicate extracted validation packages are later cleanup candidates. |
| `review/`, `review_chatgpt/`, presentation figures, `root/` | Small | Consolidate paper reviews and presentation assets into the paper archive. |

Moving artifacts within this checkout improves organization but does not recover
disk space. Space is recovered only by removing proven-regenerable material or
moving verified copies to storage outside this filesystem.

## First: preserve the submitted and current states

There are 40 tracked paths with existing modifications/deletions, including
sampler code and experiment drivers. The submission package, notes, and
`manuscript_final/` are untracked. A tag of the current HEAD alone would not
preserve this state. Review and snapshot current source changes separately from
cleanup; preserve ignored research assets through an artifact manifest and backup.

The submission agent verified the ZIP SHA-256, ZIP CRC, and internal package
checksums. The ZIP has 112 files, is 66,223,772 bytes, and has SHA-256:

```text
bb120a414cb4a47f46c43959476d0a974e5505d9a3e406cd8b4ff7e2834a0451
```

Only 42 package files byte-match their same-path working-tree counterparts;
55 differ and 15 have no corresponding root file. The package has corrected
analytical results, compact Navier–Stokes data, its own lockfile, and reduced source.
See `paper_submission_notes/SUBMISSION_REVIEW.md` and the package README for the
scope, corrections, and validation limits. Verify that this local ZIP is the one
actually uploaded before calling it the authoritative submitted release.

A normal `git add paper_submission_code/` would omit essential assets because
root ignore rules exclude `*.lock`, `*.npz`, and `checkpoints/`. Either retain the
complete ZIP in durable artifact storage with its checksum and a tracked pointer,
or explicitly verify every file if versioning the extracted package. Git status
and Git tags alone do not account for ignored artifacts.

## Code and documentation cleanup

1. **Retain the full experiment pipeline.** Keep `run.py`, `cases/`, `configs/`,
   `common/`, schemas, grid scripts, aggregators, and figure tools. Keep the NS
   reference and ablation workflows, timing tools, and their inputs. The submission
   package omits urban experiments, training, external solvers, and published
   ablation outputs, so it cannot serve as the full replacement.
2. **Resolve a broken regeneration path.** `aggregate_ns.py` documents and uses
   results produced by `compute_ns_reference_metrics.py`, which is currently
   deleted in the working tree. Existing `reference/metrics.csv` allows current
   aggregation, but regeneration needs that script or an explicitly documented
   replacement. Review the deletion rather than automatically restoring it.
3. **Move historical utilities together.** Candidates include
   `temp_make_analytical_figures.py` (explicitly a throwaway presentation copy),
   `chain_ns_ablation.sh` (a watcher with a historical hardcoded PID), and the
   corrective `rerun_ns_surge_sda.sh`, `strip_method_rows.py`, and
   `RERUN_SURGE_SDA_STEPS.md`. Retain the correction records and associated backups
   until the resulting measurements are accounted for. Update `run_status.sh`,
   which still refers to deleted launchers.
4. **Rewrite the short entry-point documentation.** Root `README.md` lists a
   nonexistent `paper_scripts/` and links deleted `DESIGN_NOTES.md`.
   `paper_experiments/README.md`, `RUN_STATUS.md`, and ignored result READMEs contain
   historical assumptions. Document the final pipeline, data/checkpoint locations,
   published result sources, and supported commands. Move dated progress notes
   into the paper archive. Treat old status summaries as historical evidence,
   not as proof that the final grid is complete or incomplete.
5. **Make paths configurable before moving artifacts.** Active launchers have
   absolute `cd /export/scratch1/...` commands, and root YAMLs contain machine-local
   data/cache paths. Derive the repository root from the launcher and expose
   explicit data, checkpoint, cache, result, and figure roots. Some experiment
   loaders obtain data paths from checkpoint configuration, so updating the case
   YAML alone is insufficient. Figure tools also target `manuscript/`, whereas
   the final source is under `manuscript_final/`; use explicit output destinations.
6. **Repair environment documentation and metadata.** Preserve and track an
   intentionally maintained root `uv.lock` (currently ignored by `*.lock`).
   README commands use `--extra dev-cpu/dev-gpu`, but these are declared dependency
   groups. Audit direct runtime dependencies: `einops` and `pandas` are directly
   imported but absent from root dependencies; LSIM paths additionally use SciPy
   and imageio. Align formatter/type-checker versions and set isort's first-party
   package to `scisi`. Avoid folding dependency upgrades into structural cleanup.
7. **Separate legacy source from runtime code carefully.**
   `src/scisi/posterior_models/archive.py` appears unreferenced and contains loose
   functions without required imports; it is an archival candidate.
   `external_libs/torch_cfd_lib` has no observed imports outside itself, so review
   its standalone workflows before archiving. Keep `jax_cfd_lib`: the NS EnKF
   baseline imports it. Aurora is imported by current data/architecture modules;
   making it optional requires a deliberate change to those import paths.
8. **Fix ignore rules after preservation.** Ignore generated logs, Python/tool
   caches, and LaTeX auxiliaries. Explicitly remove already-tracked generated files
   from the index only after preserving any useful provenance. The ignored
   `archive/` still contains tracked manuscript sources and build products; an
   ignore rule does not untrack them. Avoid broad rules that hide release inputs.

Keep behavior changes separate from moves and documentation. For example,
`src/scisi/metrics/lsim.py` loads weights at import from the cwd-relative path
`src/scisi/metrics/LSIM/LSiM.pth`; fixing installed-package behavior deserves its
own small change and verification. Apply submission correctness fixes with the
same care, particularly sample counts, aggregation, failure handling, and seeds.

## Results and storage policy

`paper_experiments/results/` contains 296 NPZ files totaling about **100.4 GiB**,
but its 1,163 CSV files total only about **50 MiB**. Preserve the compact raw
metric CSVs, per-step CSVs, published aggregates, timing overrides, reference
metrics, and resolved configurations. Their storage cost is negligible compared
with saved ensembles, and they make later revisions much easier.

Create a tracked manifest mapping each paper table/figure to its input paths,
run parameters, seeds/trajectory IDs, checkpoint and data hashes, and reproduction
command. Store large ensembles in durable artifact storage and record their
locations/checksums. Distinguish published inputs from regenerated outputs.
In particular:

- Preserve `results/timing/seconds_per_step.csv` and NS `reference/metrics.csv`:
  `aggregate_ns.py` uses these to replace timings and add reference rows.
- Preserve published analytical summaries separately. The submission review
  documents corrected five-seed, 100,000-sample summaries and an older root
  aggregate that is not equivalent. Do not silently replace published values.
- Keep `urban_dasteps_15/`, `urban_every1_physsteps50/`, and
  `urban_foreign_ic_1p5_every3/` separate. They encode different horizons/cadences;
  `run_urban_grid.sh` explicitly warns that current aggregation keys do not capture
  the horizon, so combining these directories can silently average unlike runs.
- Treat `results_full/` (6.5 GiB) as a historical snapshot to compare and archive,
  not as a proven duplicate. Retain NS reference states and ablation inputs.
- Retain states needed for qualitative panels. `make_sequential_da_tiles.py`
  hardcodes a specific `states/traj11/` NPZ. A directory-level rule such as
  “keep only traj1” would therefore be unsafe.
- Preserve checkpoint `config.yaml` and `model.pth` together, including preprocessing
  and dataset slice metadata. Keep original data unless a complete, verified
  replacement and acquisition/preprocessing recipe exist.

After those records exist, the first storage cleanup should target `cache/`
(84 GiB: approximately 49 GiB uDALES and 35 GiB KNMI). Dataset code can generate
these window caches, but verify the inputs and regeneration settings first;
several configurations assume an existing cache. Next archive the approximately
41 GiB of `external_libs/jax_cfd_lib/paper_results/` outputs and superseded run
trees. Neither their names nor age prove they are expendable.

Small straightforward candidates, after archival checks, are Python/tool caches,
LaTeX build auxiliaries, duplicate validation extractions under
`paper_submission_notes/validation_20260925/`, and obsolete presentation copies.
Keep useful validation logs and the submitted manuscript build inputs.

## End state and implementation order

The maintained root should expose `src/`, `tests/`, `config/`, `paper_experiments/`,
`external_libs/` source still in use, `docs/`, and project metadata. Add a small
release manifest pointing to the frozen submission, final manuscript, and paper
artifact archive. Local `data/`, `checkpoints/`, `cache/`, and results paths can
remain as configured working locations during migration. Consolidate paper reviews
and old drafts in the archive rather than leaving multiple root-level collections.

Suggested independently reviewable stages:

1. Preserve current source changes and exact submission/manuscript artifacts;
   inventory published inputs and record checksums/locations.
2. Fix README/ignore rules, retain the lockfile, and consolidate historical notes
   and one-off scripts. Resolve the missing reference-metric generator.
3. Make storage/output paths configurable and verify the supported workflows in
   a temporary output directory. Archive bulky artifacts with verified copies.
4. Port selected submission fixes and address package dependency/import issues in
   separate changes. Consider larger module/config refactors only afterward.

Validation before completing implementation: run existing metric and relevant
JAX localization tests; check declared environments and imports from outside the
repository; run small analytical and NS smoke cases; regenerate aggregates and
representative figures into a temporary directory and compare numerical inputs
against the frozen baseline. Compare table values and row/seed coverage rather
than relying on PDF byte equality. Verify moved checkpoint/data loading and retain
the submission checksum unchanged. Full expensive experiment grids are unnecessary
for moves/docs alone; changes affecting scientific behavior need targeted checks.

The audit used filesystem/Git inspection, static source review, and submission
integrity verification. It did not establish dataset backups, actual upload
identity, final run completeness, or a passing runtime test suite. Those are
implementation checks, not assumptions supporting deletion.
