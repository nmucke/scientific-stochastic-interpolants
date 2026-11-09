#!/usr/bin/env bash
# Forecast-residual correction switch for the grid scripts (FORECAST_CORRECTION.md).
# Source AFTER PY / ROOT / PE_SCRIPT_DIR are set:   source common/fc_grid.sh <case>
# Sets FC_ARGS (run.py overrides; expand as ${FC_ARGS[@]+"${FC_ARGS[@]}"}, since
# bash < 4.4 treats an empty array as unbound under `set -u`) and FC_TAG (file suffix).
#
#   FC=off (default)  nothing is added: the commands and files are the plain ones.
#   FC=on             case.forecast_correction.enabled=true plus FC_OVERRIDES, a
#                     space-separated list of <key>=<value> for keys of the case
#                     YAML block (no spaces inside a value), e.g.
#                       FC=on FC_OVERRIDES="forecast_num_steps=10 gain_scale=0.5"
#                       FC=on FC_OVERRIDES="localisation_radius={default:8,superres_16:12}"
#                     Measure only: FC_OVERRIDES="gain_scale=0 forecast_seed=0".
#   FC_DIAG           per-step diagnostics (default true; never changes the numbers).
# FC_TAG is "__fc<steps>_<hash8>", computed by common/forecast_correction.py from the
# composed config -- the same function that builds the rows' `variant`.
FC="${FC:-off}"
FC_ARGS=(); FC_TAG=""
case "$FC" in
  off) ;;
  on)
    FC_ARGS+=("case.forecast_correction.enabled=true")
    read -r -a _fc_kv <<< "${FC_OVERRIDES:-}"
    for _kv in ${_fc_kv[@]+"${_fc_kv[@]}"}; do FC_ARGS+=("case.forecast_correction.$_kv"); done
    _fc_tag=$(cd "$PE_SCRIPT_DIR" && "$PY" -m common.forecast_correction tag "$1" "${FC_ARGS[@]}") \
      || { echo "[grid] FATAL: bad FC_OVERRIDES='${FC_OVERRIDES:-}'" >&2; exit 1; }
    FC_TAG="__${_fc_tag}"
    FC_ARGS+=("case.forecast_correction.diagnostics=${FC_DIAG:-true}"
              "case.forecast_correction.diagnostics_root=$ROOT/forecast_correction")
    ;;
  *) echo "[grid] FATAL: FC must be off|on, got '$FC'" >&2; exit 1;;
esac
