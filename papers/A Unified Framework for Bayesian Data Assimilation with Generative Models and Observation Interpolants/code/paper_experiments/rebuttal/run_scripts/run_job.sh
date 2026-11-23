#!/usr/bin/env bash
# usage: run_job.sh <name> <command...>   -> logs to _runs/<name>_<ts>.log, exit code to .exit
source "$(dirname "$0")/env.sh"
name=$1; shift
log="$PE/results/_runs/${name}_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date -Is)] start: $*" > "$log"
( eval "$@" ) >> "$log" 2>&1
rc=$?
echo "[$(date -Is)] exit $rc" >> "$log"
echo $rc > "$log.exit"
