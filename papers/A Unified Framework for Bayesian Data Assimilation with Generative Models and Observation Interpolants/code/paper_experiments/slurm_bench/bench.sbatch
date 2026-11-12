#!/bin/bash
#SBATCH --job-name=b_bench
#SBATCH --partition=gpu-a100-small
#SBATCH --account=research-ceg-gse
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=7900M
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench/bench_%j.out
B=/scratch/ntmucke/scientific-stochastic-interpolants/slurm_bench; cd $B
mapfile -t U < <(nvidia-smi -L | grep -oE 'MIG-[0-9a-f-]+')
echo "slices: ${U[*]}  CVD=$CUDA_VISIBLE_DEVICES"
SC='+ns_scenarios=["32^2->128^2"]'
K(){ echo "+jacobian_refresh_every=$1 +variant_override=$2"; }
(
export CUDA_VISIBLE_DEVICES=${U[0]}
./cell.sh si_jacfree 1200 '+ns_methods=["Ours (SI-SDE)"]' "$SC" num_steps=100 case.num_physical_steps=7 likelihood_mode=dps_jacobian_free
./cell.sh dm_k10 2400 '+ns_methods=["Ours (DM-SDE)"]' "$SC" num_steps=100 case.num_physical_steps=7 likelihood_mode=inflated_shared +jacobian_refresh_every=10 +variant_override=shared_jac10
./cell.sh si_k1 3000 '+ns_methods=["Ours (SI-SDE)"]' "$SC" num_steps=100 case.num_physical_steps=6 likelihood_mode=inflated_shared +jacobian_refresh_every=1 +variant_override=shared_jac1
./cell.sh dm_k1 5400 '+ns_methods=["Ours (DM-SDE)"]' "$SC" num_steps=100 case.num_physical_steps=6 likelihood_mode=inflated_shared +jacobian_refresh_every=1 +variant_override=shared_jac1
) &
(
export CUDA_VISIBLE_DEVICES=${U[1]:-${U[0]}}
./cell.sh enkf64 1500 '+ns_methods=["EnKF"]' "$SC" num_steps=50 case.num_physical_steps=7
./cell.sh pf64 1500 '+ns_methods=["Particle filter"]' "$SC" num_steps=50 case.num_physical_steps=7
./cell.sh dflow 4500 '+ns_methods=["D-Flow SGLD"]' "$SC" num_steps=100 case.num_physical_steps=6 likelihood_mode=dps_jacobian_free
E=1000 ./cell.sh enkf1000 3000 '+ns_methods=["EnKF"]' "$SC" num_steps=50 case.num_physical_steps=7 case.reference_ensemble_size=8
) &
wait
