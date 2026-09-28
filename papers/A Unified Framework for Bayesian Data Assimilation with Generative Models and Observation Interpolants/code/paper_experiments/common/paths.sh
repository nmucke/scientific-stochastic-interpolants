#!/usr/bin/env bash
# Shared locations for experiment shell entry points. Source this file from a
# script; it does not change the caller's working directory.
PE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PE_REPO_ROOT="$PE_SCRIPT_DIR"
while [[ ! -f "$PE_REPO_ROOT/pyproject.toml" || ! -d "$PE_REPO_ROOT/src/scisi" ]]; do
  if [[ "$PE_REPO_ROOT" == / ]]; then
    printf 'Cannot locate repository containing src/scisi from %s\n' "$PE_SCRIPT_DIR" >&2
    return 1
  fi
  PE_REPO_ROOT="$(dirname -- "$PE_REPO_ROOT")"
done
PE_PYTHON="${PY:-$PE_REPO_ROOT/.venv/bin/python}"
