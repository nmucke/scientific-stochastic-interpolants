#!/usr/bin/env bash
# Create release/<name> from origin/dev with the paths in .github/dev-only-paths
# removed, ready for a PR into main. Merge that PR with a merge commit, not a
# squash, so the next release has a clean merge base.
#
#   .github/scripts/make-release-branch.sh [name]   # name defaults to today's date
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Commit or stash your changes first." >&2
  exit 1
fi

branch="release/${1:-$(date +%Y-%m-%d)}"
git fetch origin
git switch -c "$branch" --no-track origin/dev

mapfile -t paths < <(grep -Ev '^[[:space:]]*(#|$)' .github/dev-only-paths)
git rm -r -q --ignore-unmatch -- "${paths[@]}"
git commit -q -m "Release dev to main without dev-only paths"

echo "Created $branch. Push it and open a PR into main:"
echo "  git push -u origin $branch"
