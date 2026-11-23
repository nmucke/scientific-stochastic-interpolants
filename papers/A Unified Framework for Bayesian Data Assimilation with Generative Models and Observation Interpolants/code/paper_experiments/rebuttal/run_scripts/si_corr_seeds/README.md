# Re-run: poor IC, 32^2, Ours SI-SDE (shared_jac10) + corr. (fc25_8797d2ae), traj 11
Original (DelftBlue job 1305160, V100, seed 0) diverged at the first assimilation step (RMSE 16.2).
Same settings as ext_cells.tsv (IC 15, NP=40 -> 35 assimilation steps, E=64, M=100, lambda=0.9, FC r=20, 25 forecast steps),
current checkout (only additive changes since ac8aa01), RTX 3090.
- smoke/: seed 0, 1 assimilation step: RMSE 0.926 (did NOT diverge on this GPU; normalised innov^2 19.6 vs 16.0 originally).
- full/: seeds 1, 2, 3, then 0, full 35 steps, sequential (other user's process holds ~19 GB of the GPU). States in full/states/traj11.
Launch: tmux session rb_sicorr (overnight.sh), started 2026-11-16 evening. Done marker: full.done.
seed-1 finished; stopped seeds 2,3,0 at Mon Nov 16 09:47:21 PM CET 2026
