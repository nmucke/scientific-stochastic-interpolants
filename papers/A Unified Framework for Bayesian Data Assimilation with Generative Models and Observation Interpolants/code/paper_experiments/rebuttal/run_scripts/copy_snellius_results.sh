#!/usr/bin/env bash
# Pull the DelftBlue result tree (staged on Snellius) to /export/scratch2, then md5-verify.
# Resumable: rerun the script. Copy-only, never deletes.
set -euo pipefail
DST=/export/scratch2/ntm/delftblue_results/results
mkdir -p "$DST"
echo "== rsync start $(date -Is)"
rsync -aH --partial --info=progress2,stats2 snellius:/scratch-shared/nmucke1/results/ "$DST/"
echo "== rsync done $(date -Is)"
cd "$DST"
echo "== md5 check $(date -Is)"
md5sum -c --quiet MANIFEST.md5 && echo "MD5 OK: every file in MANIFEST.md5 matches"
echo "== COPY DONE $(date -Is)"
