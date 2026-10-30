# Agent notes

This file exists only on `dev`; it is removed from every release to `main`.
`CLAUDE.md` is a symlink to it, so edit `AGENTS.md`.

## Branches

- `main` is the GitHub default branch but only receives releases. All other
  work goes through `dev`.
- Workflow: branch off `origin/dev`, develop, open a PR into `dev`
  (`gh pr create --base dev`). Never open a PR into `main` from a feature
  branch; a GitHub Action retargets such PRs to `dev` anyway.
- Never merge or push `dev` into `main` directly.

## Dev-only paths

[.github/dev-only-paths](.github/dev-only-paths) lists paths that live on `dev`
and must never reach `main` (currently `research/`, this file and `CLAUDE.md`). The
`guard-main` workflow fails any PR into, or push to, `main` that contains them.

- Put new dev-only material (notes, experiments, papers, their tests) under
  `research/`, or add its path to `.github/dev-only-paths`.
- Outside the dev-only paths, `dev` and `main` must stay identical. Do not make
  dev-only edits to shared files (root `README.md`, `.gitignore`, `config/`,
  `src/`, `tests/`); code in `src/` and `tests/` must not depend on `research/`.
  Tests for research scripts go in `research/ito_maps/tests/`.

## Releasing dev to main

```bash
.github/scripts/make-release-branch.sh [name]   # release/<name>: dev minus dev-only paths
git push -u origin release/<name>
```

Open a PR from `release/<name>` into `main` and merge it with a merge commit,
not a squash, so later releases keep a clean merge base.

## Syncing main into dev

Only needed when `main` received commits that did not come from `dev` (e.g. a
hotfix). A plain merge would delete the dev-only paths on `dev`, so use:

```bash
.github/scripts/sync-main-into-dev.sh   # sync/main-into-dev-<date>, PR into dev
```
