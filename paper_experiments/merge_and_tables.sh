#!/bin/bash
# =============================================================================
# Merge headline / stepbench CSVs into all_results.csv and regenerate LaTeX.
#
# Usage: bash paper_experiments/merge_and_tables.sh
#
# Sources (whichever exist):
#   paper_experiments/results/analytical_results.csv
#   paper_experiments/results/headline/*.csv           <- new headline grid
#   paper_experiments/results/stepbench/csv/*.csv      <- step-count benchmark
#   paper_experiments/results/navier_stokes_results.csv  (old Ours only, may be stale)
# =============================================================================
set -u
cd /export/scratch1/ntm/postdoc/scientific-stochastic-interpolants
PY=.venv/bin/python
RES=paper_experiments/results

# Locate a header row (any CSV will do).
HEADER_SRC=""
for f in "$RES/analytical_results.csv" "$RES/headline"/*.csv "$RES/navier_stokes_results.csv"; do
    [ -f "$f" ] && HEADER_SRC="$f" && break
done
if [ -z "$HEADER_SRC" ]; then
    echo "[merge] ERROR: no source CSVs found in $RES" >&2; exit 1
fi

echo "[merge] header from $HEADER_SRC"
head -1 "$HEADER_SRC" > "$RES/all_results.csv"

# Append all data rows (skip headers).
for f in \
    "$RES/analytical_results.csv" \
    "$RES/headline"/*.csv \
    "$RES/navier_stokes_results.csv"; do
    [ -f "$f" ] && tail -q -n +2 "$f" >> "$RES/all_results.csv"
done

NROWS=$(wc -l < "$RES/all_results.csv")
echo "[merge] all_results.csv: $NROWS rows (incl header)"

# Regenerate LaTeX tables.
mkdir -p paper_experiments/generated
$PY paper_experiments/make_tables.py \
    --results "$RES/all_results.csv" \
    --out paper_experiments/generated

echo "[merge] Done. Recompile the paper:"
echo "   cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex"
