True-solver EnKF with Gaspari-Cohn r=20 on the super-resolution scenarios (16^2, 32^2), wrong-IC grid
settings (ic_index=15, NP=40, E=64, traj 10-14), from the frozen copy code_ns_wrong_ic plus its obs_coords
fallback (README_FROZEN.txt there; same change committed as 948fa40). Production array 1305653 on gpu-v100
(DEVICE=gpu, JAX matmul precision highest). Results: production/{metrics,per_step,logs,states/traj11}.
Smokes (2 assimilation steps, traj 10): smoke_mig/ (A100 MIG, passed), smoke/ (CPU, cancelled).
