#!/usr/bin/env bash
# Create sync/main-into-dev-<date> from origin/dev with origin/main merged in,
# keeping dev's version of the paths in .github/dev-only-paths. A plain merge
# would apply main's removal of those paths to dev.
#
#   .github/scripts/sync-main-into-dev.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit or stash your changes first." >&2
  exit 1
fi

branch="sync/main-into-dev-$(date +%Y-%m-%d)"
git fetch origin
git switch -c "$branch" --no-track origin/dev

mapfile -t paths < <(grep -Ev '^[[:space:]]*(#|$)' .github/dev-only-paths)
git merge --no-commit --no-ff origin/main || true
git checkout HEAD -- "${paths[@]}"

if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
  echo "Resolve the remaining conflicts, then run: git commit" >&2
  git diff --name-only --diff-filter=U >&2
  exit 1
fi
git commit -q --no-edit

echo "Created $branch. Push it and open a PR into dev:"
echo "  git push -u origin $branch"
