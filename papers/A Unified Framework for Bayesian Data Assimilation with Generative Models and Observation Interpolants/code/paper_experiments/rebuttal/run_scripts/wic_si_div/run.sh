#!/usr/bin/env bash
# Local RTX 3090 reruns of the poor-IC SI-SDE (shared k=10) cells that diverged on DelftBlue,
# same settings as the DelftBlue manifests (main cells_*.tsv, ext job 1281559 r=8 / 1305160 r=20,
# SURGE job 1281559): seed 0, ic_index 15, NP=40, E=64, M=100. Resumable.
set -u
source "$(dirname "$0")/../env.sh"
cd "$REPO"
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
D="$PE/results/_runs/wic_si_div"
mkdir -p "$D"/{metrics,per_step,states/traj11,forecast_correction,hydra,logs}
declare -A SCN=([32_2_128_2]="32^2->128^2" [16_2_128_2]="16^2->128^2" [sparse_5]="sparse 5%" [sparse_1_5625]="sparse 1.5625%")
while IFS='|' read -r SS N KIND METHOD; do
  [ -z "$SS" ] && continue
  V=(likelihood_mode=inflated_shared +jacobian_refresh_every=10 +variant_override=shared_jac10)
  FC=(case.forecast_correction.enabled=true case.forecast_correction.forecast_num_steps=25
      case.forecast_correction.diagnostics=true "case.forecast_correction.diagnostics_root=$D/forecast_correction")
  case $KIND in
    main)  ID="${SS}__M100__traj${N}__ours_shared_jac10__si_sde"; X=() ;;
    r8)    ID="${SS}__M100__traj${N}__ours_shared_jac10__fc25_70a3134d__si_sde"; X=("${FC[@]}") ;;
    r20)   ID="${SS}__M100__traj${N}__ours_shared_jac10__fc25_8797d2ae__si_sde"; X=("${FC[@]}" case.forecast_correction.localisation_radius=20) ;;
    surge) ID="${SS}__M100__traj${N}__surge_ours_shared_jac10__si_sde"; X=() ;;
  esac
  [ -f "$D/metrics/$ID.csv" ] && { echo "[div] SKIP $ID"; continue; }
  if [ "$N" = 11 ]; then ST=(+save_states=true "+states_root=$D/states/traj11"); else ST=(+save_states=false "+states_root=$D/states/_unused"); fi
  echo "[div] RUN $ID $(date -Is)"
  python -u "$PE/run.py" case=navier_stokes seeds=[0] case.require_weights=true case.device=cuda \
    "+test_index=$N" +divergence_rmse_threshold=10.0 "results_file=$D/.tmp_$ID.csv" "hydra.run.dir=$D/hydra/$ID" \
    +save_per_step=true "+per_step_file=$D/per_step/$ID.csv" \
    "+ns_methods=[\"$METHOD\"]" "+ns_scenarios=[\"${SCN[$SS]}\"]" +ic_index=15 case.num_physical_steps=40 \
    +skip_kl_reference=true ensemble_size=64 num_steps=100 "${V[@]}" "${X[@]}" "${ST[@]}" \
    > "$D/logs/$ID.log" 2>&1 && mv "$D/.tmp_$ID.csv" "$D/metrics/$ID.csv" && echo "[div] OK $ID $(date -Is)" \
    || echo "[div] FAIL $ID $(date -Is)"
done < "$D/cells.txt"
echo "[div] ALL DONE $(date -Is)"
